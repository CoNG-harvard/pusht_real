import cv2
import numpy as np

from shapely.strtree import STRtree
from shapely.geometry import LineString, box
import numpy as np
from itertools import combinations

from skimage.draw import line as skimage_line
from scipy.ndimage import binary_dilation


class TBlockTracker:
    # hardcoded color limits..
    color_u = np.array([250, 140, 85])
    color_l = np.array([90, 0, 0]) 

    def __init__(self, mode='bgr'):
        self.mode = mode

    def get_obj(self, img, min_cnt_area=1000):
        img = img.copy()
        mask = np.all((self.color_l <= img) & (img <= self.color_u), axis=-1).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = tuple(
            contour for contour in contours if cv2.contourArea(contour) > min_cnt_area)
        contours = tuple(cv2.approxPolyDP(contour, 1, True) for contour in contours)
        if len(contours) == 0:
            print('no contours')
            return None

        mask = cv2.drawContours(
            np.zeros(img.shape[:-1]), contours, -1, color=255, thickness=-1)
        obj_only = img.copy()
        obj_only[mask == 0] = 0
        return obj_only
    
    def get_lines(self, obj_only, ang_tol=15):

        ang_tol = np.radians(ang_tol)
        tempo = obj_only.copy()
        # get lines
        edges = cv2.Canny(tempo, threshold1=10, threshold2=40)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=40, minLineLength=1, maxLineGap=20)
        if lines is None:
            print('no lines')
            return None
        
        lines = lines[:, 0]
        angles = np.array([np.arctan2(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in lines])
        
        # choose longest line
        lens = [np.hypot((x2 - x1), (y2 - y1)) for (x1, y1, x2, y2) in lines]
        ang_longest = angles[np.argmax(lens)]
        l = lines[np.argmax(lens)]

        # de-rotate to get perpendicular
        obj_pts = np.argwhere(np.any(obj_only, axis=-1))
        rotation_matrix = np.array([
            [np.cos(-ang_longest), -np.sin(-ang_longest)],
            [np.sin(-ang_longest),  np.cos(-ang_longest)]
        ])

        rot = obj_pts @ rotation_matrix.T

        # Note: We are trying to find if the longest line is the stem or the top
        # works most of the time!
        sym1, sym2 = symmetry_score(rot[:, 0]), symmetry_score(rot[:, 1])
        if max(sym1, sym2) < 15:
            perc = percent_points_on_segment(obj_pts, l[:2], l[2:])
            if perc > 80:
                # chosen line is at top
                print('top, no symmetry')
                top = True
                rot_ang = ang_longest + np.pi/2
                x = rot[:, 1]
            else:
                print('stem, no symmetry')
                top = False
                rot_ang = ang_longest
                x = rot[:, 0]
        else:
            if sym1 < sym2:
                # chosen line is at stem
                print('top')
                top = True
                rot_ang = ang_longest + np.pi/2
                x = rot[:, 1]
            else:
                # chosen line is at top
                print('stem')
                top = False
                rot_ang = ang_longest
                x = rot[:, 0]

        # T is upside down
        if np.mean(x < (x.max() + x.min())/2) < 0.5:
            rot_ang += np.pi

        # group lines
        perp_lines = [line for angle, line in zip(angles, lines) if abs(
            abs(angle - ang_longest) - np.pi/2) < ang_tol]
        par_lines = [line for angle, line in zip(angles, lines) if abs(
            angle_diff_deg(angle, ang_longest)) < ang_tol]

        # assign top and stem
        top_lines, stem_lines = (par_lines, perp_lines) if top else (perp_lines, par_lines)
        top_lines, stem_lines = np.array(top_lines), np.array(stem_lines)

        return l, rot_ang, top_lines, stem_lines
    
    def get_lines_extcut(self, top_lines, stem_lines, obj_only, dx=100, dy=100, tol=3):
        min_x, min_y = -dx, -dy
        max_x, max_y = obj_only.shape[1] + dx, obj_only.shape[0] + dy

        top_lines_ext = extend_lines(min_x, max_x, min_y, max_y, top_lines)
        stem_lines_ext = extend_lines(min_x, max_x, min_y, max_y, stem_lines)

        mask = np.any(obj_only, axis=-1)
        top_lines_cut, stem_lines_cut = compute_joined_lines(top_lines_ext, stem_lines_ext, mask, tol=tol)
        top_lines_cut, stem_lines_cut = np.array(
            top_lines_cut), np.array(stem_lines_cut)

        return top_lines_cut, stem_lines_cut
    
    def reduce_lines(self, top_lines, stem_lines, top_lines_cut, stem_lines_cut, obj_only, rot_ang, 
                     stem_max_mul=3, stem_d=0.7, min_len_top_up=75, min_len_top_down=35):
        
        h, w = obj_only.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -np.degrees(rot_ang), 1.0)

        mask = np.any(obj_only, axis=-1)
        stem_lens = np.array([np.hypot((x2 - x1), (y2 - y1)) for (x1, y1, x2, y2) in stem_lines])
        stem_cut_lens = np.array([np.hypot((x2 - x1), (y2 - y1)) for (x1, y1, x2, y2) in stem_lines_cut])
        mul = stem_cut_lens / stem_lens

        # Note: the stem lines are usually mostly visible so even if they are extended they dont
        # get much extended
        stem_lines_cut = stem_lines_cut[mul < stem_max_mul]
        if len(stem_lines_cut) < 2:
            print('not enough stem lines')
            return None
        if len(stem_lines_cut) > 2:
            order, normal = compute_common_normal_and_sort(stem_lines_cut)
            res1 = count_mask_hits_between_lines(mask, stem_lines_cut[order], normal, d=stem_d, tol=3)
            res2 = count_mask_hits_between_lines(mask, stem_lines_cut[order][::-1], -normal, d=stem_d, tol=3)[::-1]
            idx = np.argmax(np.minimum(res1, res2))
            stem_lines_cut = stem_lines_cut[order][idx:idx+2]
        # By now we should be left with 2 two lines in the stem.
        sl1, sl2 = stem_lines_cut
        sl1_mid = np.mean([M @ (pt[0], pt[1], 1) for pt in [sl1[:2], sl1[2:]]], axis=0)[0]
        sl2_mid = np.mean([M @ (pt[0], pt[1], 1) for pt in [sl2[:2], sl2[2:]]], axis=0)[0]

        stem_lines_filtered = np.array([sl1, sl2]) if sl1_mid < sl2_mid else np.array([sl2, sl1])

        # ------ remove lines that dont intersect stem -----
        new, new_filtered = [], []
        for l, l_orig in zip(top_lines_cut, top_lines):
            i1 = get_intersection_point(l[:2], l[2:], sl1[:2], sl1[2:], extend=5)
            i2 = get_intersection_point(l[:2], l[2:], sl2[:2], sl2[2:], extend=5)
            if i1 is not None and i2 is not None:
                new.append(l_orig)
                new_filtered.append(l)
        top_lines = np.array(new)
        top_filtered = np.array(new_filtered)

        if len(top_filtered) < 2:
            print('not enough top lines')
            return None
        
        # ------ remove lines that are in bottom -----
        # Note: I am fairly certain this will work for the most cases
        percy = 0.8
        miny_bound = 0
        for l in stem_lines_filtered:
            b, e = l[:2], l[2:]
            by = (M @ np.array([b[0], b[1], 1]))[1]
            ey = (M @ np.array([e[0], e[1], 1]))[1]
            maxy = max(by, ey)
            miny = min(by, ey)
            miny_bound = max((maxy - miny) * percy + miny, miny_bound)

        new, new_filtered = [], []
        for l, l_orig in zip(top_filtered, top_lines):
            b, e = l[:2], l[2:]
            by = (M @ np.array([b[0], b[1], 1]))[1]
            ey = (M @ np.array([e[0], e[1], 1]))[1]
            if max(by, ey) < miny_bound:
                new.append(l_orig)
                new_filtered.append(l)
        top_lines = np.array(new)
        top_filtered = np.array(new_filtered)
        assert len(top_filtered) == len(new_filtered)

        if len(top_filtered) < 2:
            print('not enough top lines')
            return None
        
        # ----- Select lines for up and down for top -----
        if len(top_filtered) >= 2:
            beg_pts_y = np.array([M @ (x1, y1, 1)
                                 for (x1, y1, x2, y2) in top_filtered])[:, 1]
            end_pts_y = np.array([M @ (x2, y2, 1)
                                for (x1, y1, x2, y2) in top_filtered])[:, 1]

            beg_pts_y_arg = np.argsort(beg_pts_y)
            end_pts_y_arg = np.argsort(end_pts_y)

            beg_mid = np.argmax(np.diff(beg_pts_y[beg_pts_y_arg])) + 1
            end_mid = np.argmax(np.diff(end_pts_y[end_pts_y_arg])) + 1

            up = list(set(end_pts_y_arg[:end_mid]).intersection(set(beg_pts_y_arg[:beg_mid])))
            down = list(set(end_pts_y_arg[end_mid:]).intersection(set(beg_pts_y_arg[beg_mid:])))

            # get the original lines and the filtered lines
            up_orig = top_lines[up]
            down_orig = top_lines[down]
            up = top_filtered[up]
            down = top_filtered[down]

            # for up simply pick the lowest one that is not too short, this is usually the one
            if len(up) > 1:
                lens_orig = np.array([np.hypot((x2 - x1), (y2 - y1)) for (x1, y1, x2, y2) in up_orig])
                filtr = lens_orig > min_len_top_up
                if np.sum(filtr) > 0:
                    up = up[filtr]
                beg_pts_y = np.array([M @ (x1, y1, 1)
                                      for (x1, y1, x2, y2) in up])[:, 1]
                end_pts_y = np.array([M @ (x2, y2, 1)
                                    for (x1, y1, x2, y2) in up])[:, 1]
                mid_y = np.mean([beg_pts_y, end_pts_y], axis=0)
                up = up[np.argmax(mid_y)]
            else:
                up = up[0]

            # for down the task is more difficult...
            if len(down) > 1:
                mid_pts_x = np.array([M @ (x1, y1, 1)
                                   for (x1, y1, x2, y2) in down])[:, 0]
                mid_pts_stem_x = np.array([M @ (x1, y1, 1)
                                            for (x1, y1, x2, y2) in stem_lines_filtered])[:, 0]
                mid_pts_stem_x = np.mean(mid_pts_stem_x)
                down_left = down[mid_pts_x < mid_pts_stem_x]
                down_left_orig = down_orig[mid_pts_x < mid_pts_stem_x]
                down_right = down[mid_pts_x > mid_pts_stem_x]
                down_right_orig = down_orig[mid_pts_x > mid_pts_stem_x]

                if len(down_left) == 0 or len(down_right) == 0:
                    not_empty = down_right if len(down_left) == 0 else down_left
                    not_empty_orig = down_right_orig if len(down_left) == 0 else down_left_orig
                    lens_orig = np.array(
                        [np.hypot((x2 - x1), (y2 - y1)) for (x1, y1, x2, y2) in not_empty_orig])
                    filtr = lens_orig > min_len_top_down
                    if np.sum(filtr) > 0:
                        not_empty = not_empty[filtr]
                    # return the upper one
                    beg_pts_y = np.array([M @ (x1, y1, 1)
                                            for (x1, y1, x2, y2) in not_empty])[:, 1]
                    end_pts_y = np.array([M @ (x2, y2, 1)
                                        for (x1, y1, x2, y2) in not_empty])[:, 1]
                    mid_y = np.mean([beg_pts_y, end_pts_y], axis=0)
                    down = not_empty[np.argmin(mid_y)]
                elif len(down_left) == 1 and len(down_right) == 1:
                    # should we pick both or just one?
                    close = are_lines_close(down_left[0], down_right[0], 10)
                    if close:
                        down = np.array([down_left[0], down_right[0]])
                    else:
                        dl, dr = down_left[0], down_right[0]
                        dl_orig, dr_orig = down_left_orig[0], down_right_orig[0]
                        dl_orig_len = np.hypot((dl_orig[2] - dl_orig[0]), (dl_orig[3] - dl_orig[1]))
                        dr_orig_len = np.hypot((dr_orig[2] - dr_orig[0]), (dr_orig[3] - dr_orig[1]))
                        ratio = max(dl_orig_len, dr_orig_len) / min(dl_orig_len, dr_orig_len)
                        if ratio > 2:
                            # pick the longest one
                            down = dl if dl_orig_len > dr_orig_len else dr
                        else:
                            down = np.array([dl, dr])
                elif len(down_left) == 1 or len(down_right) == 1:
                    single, multi = (down_left, down_right) if len(down_left) == 1 else (down_right, down_left)
                    single_orig, multi_orig = (down_left_orig, down_right_orig) if len(down_left) == 1 else (down_right_orig, down_left_orig)
                    multi_orig_lens = np.array([np.hypot((x2 - x1), (y2 - y1)) for (x1, y1, x2, y2) in multi_orig])
                    filtr = multi_orig_lens > min_len_top_down
                    if np.sum(filtr) > 0:
                        multi = multi[filtr]
                        multi_orig = multi_orig[filtr]

                        # return the upper one
                        beg_pts_y = np.array([M @ (x1, y1, 1)
                                            for (x1, y1, x2, y2) in multi])[:, 1]
                        end_pts_y = np.array([M @ (x2, y2, 1)
                                            for (x1, y1, x2, y2) in multi])[:, 1]
                        mid_y = np.mean([beg_pts_y, end_pts_y], axis=0)
                        down_side = multi[np.argmin(mid_y)]
                        down_side_orig = multi_orig[np.argmin(mid_y)]

                        # check if the single line is close to the multi line
                        close = are_lines_close(single[0], down_side, 10)
                        down_side_orig_len = np.hypot((down_side_orig[2] - down_side_orig[0]), (down_side_orig[3] - down_side_orig[1]))
                        single_orig_len = np.hypot((single_orig[0][2] - single_orig[0][0]), (single_orig[0][3] - single_orig[0][1]))
                        ratio = max(down_side_orig_len, single_orig_len) / min(down_side_orig_len, single_orig_len)
                        if close and ratio < 2:
                            down = np.array([single[0], down_side]) if len(down_left) == 1 else np.array([down_side, single[0]])
                        else:
                            down = down_side
                    else:
                        down = single[0]
                else:
                    # both have >= 2 lines...
                    sides, origs = []
                    for down_side, down_side_orig in zip([down_left, down_right], [down_left_orig, down_right_orig]):
                        down_side_orig_lens = np.array([np.hypot((x2 - x1), (y2 - y1)) for (x1, y1, x2, y2) in down_side_orig])
                        filtr = down_side_orig_lens > min_len_top_down
                        if np.sum(filtr) > 0:
                            down_side = down_side[filtr]
                            down_side_orig = down_side_orig[filtr]

                            # return the upper one
                            beg_pts_y = np.array([M @ (x1, y1, 1)
                                                for (x1, y1, x2, y2) in down_side])[:, 1]
                            end_pts_y = np.array([M @ (x2, y2, 1)
                                                for (x1, y1, x2, y2) in down_side])[:, 1]
                            mid_y = np.mean([beg_pts_y, end_pts_y], axis=0)
                            sides.append(down_side[np.argmin(mid_y)])
                            origs.append(down_side_orig[np.argmin(mid_y)])
                    down_left, down_right = sides
                    down_left_orig, down_right_orig = origs
                    # check if the single line is close to the multi line
                    close = are_lines_close(down_left, down_right, 10)
                    down_left_orig_len = np.hypot((down_left_orig[2] - down_left_orig[0]), (down_left_orig[3] - down_left_orig[1]))
                    down_right_orig_len = np.hypot((down_right_orig[2] - down_right_orig[0]), (down_right_orig[3] - down_right_orig[1]))
                    ratio = max(down_left_orig_len, down_right_orig_len) / min(down_left_orig_len, down_right_orig_len)
                    if close and ratio < 2:
                        down = np.array([down_left, down_right])
                    else:
                        down = down_left if down_left_orig_len > down_right_orig_len else down_right
            else:
                down = down[0]

            top_filtered = np.concatenate([np.atleast_2d(up), np.atleast_2d(down)], axis=0)  
                
        return top_filtered, stem_lines_filtered

    def detect_corners(self, img, min_cnt_area=1000, ang_tol=15, dx=100, dy=100, tol=3, stem_max_mul=3, stem_d=0.7, min_len_top_up=75, min_len_top_down=35):
        # get object
        obj_only = self.get_obj(img, min_cnt_area)
        if obj_only is None:
            return None
        
        # get lines
        res = self.get_lines(obj_only, ang_tol)
        if res is None:
            return None
        l, rot_ang, top_lines, stem_lines = res

        # TODO: For now I assume I see at least 2 lines for each
        # Maybe later I can do something when we see fewer lines e.g. use temporal info..
        if len(top_lines) < 2:
            print('not enough top lines')
            return None

        if len(stem_lines) < 2:
            print('not enough stem lines')
            return None

        # get extcut lines
        top_lines_cut, stem_lines_cut = self.get_lines_extcut(top_lines, stem_lines, obj_only, 
                                                              dx=dx, dy=dy, tol=tol)

        res = self.reduce_lines(top_lines, stem_lines, top_lines_cut, stem_lines_cut, obj_only, rot_ang,
                                stem_max_mul=stem_max_mul, stem_d=stem_d, min_len_top_up=min_len_top_up, min_len_top_down=min_len_top_up)
        if res is None:
            print('not enough lines after reduction')
            return None
        
        top_filtered, stem_lines_filtered = res

        tl = get_intersection_point(top_filtered[0][:2], top_filtered[0][2:], stem_lines_filtered[0][:2], stem_lines_filtered[0][2:], extend=5)
        tr = get_intersection_point(top_filtered[0][:2], top_filtered[0][2:], stem_lines_filtered[1][:2], stem_lines_filtered[1][2:], extend=5)

        if len(top_filtered) == 2:
            down = top_filtered[1]
            bl = get_intersection_point(down[:2], down[2:], stem_lines_filtered[0][:2], stem_lines_filtered[0][2:], extend=5)
            br = get_intersection_point(down[:2], down[2:], stem_lines_filtered[1][:2], stem_lines_filtered[1][2:], extend=5)
        elif len(top_filtered) > 2:
            down_l, down_r = top_filtered[1], top_filtered[2]
            bl = get_intersection_point(down_l[:2], down_l[2:], stem_lines_filtered[0][:2], stem_lines_filtered[0][2:], extend=5)
            br = get_intersection_point(down_r[:2], down_r[2:], stem_lines_filtered[1][:2], stem_lines_filtered[1][2:], extend=5)

        return np.array([tl, bl, br, tr])


# ----------- Utility functions -----------
def angle_diff_deg(theta1, theta2):
    """
    Compute minimal angle difference (in degrees) between two angles.
    Result is in [-180, 180].

    Args:
        theta1, theta2: angles in degrees

    Returns:
        float: signed angle difference
    """
    diff = (theta2 - theta1 + np.pi) % (2*np.pi) - np.pi
    return diff


def percent_points_on_segment(points, A, B):
    """
    Return the percentage of points whose projections fall between A and B.

    Args:
        points: (N, 2) numpy array
        A, B: endpoints of the line segment

    Returns:
        float: percentage (0–100) of projected points lying between A and B
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)

    AB = B - A
    AB_norm = np.linalg.norm(AB)
    AB_unit = AB / AB_norm

    AP = points - A
    scalars = np.dot(AP, AB_unit)

    # Count how many fall within [0, length]
    in_range = (scalars >= 0) & (scalars <= AB_norm)
    percent = 100.0 * np.sum(in_range) / len(points)

    return percent


def extend_lines(min_x, max_x, min_y, max_y, lines):
    """
    Extend each line in `lines` so it reaches the bounding box and update it in-place.

    Parameters:
        min_x, max_x, min_y, max_y: float
            Bounds of the box.
        lines: list of lists or tuples (mutable)
            Each is [(x1, y1), (x2, y2)] representing a line segment.
    """
    bounds = box(min_x, min_y, max_x, max_y)
    ext_lines = []
    for i in range(len(lines)):
        (x1, y1), (x2, y2) = lines[i][:2], lines[i][2:]
        dx = x2 - x1
        dy = y2 - y1
        factor = 1e5

        # Create a long line in the same direction
        x1_ext = x1 - factor * dx
        y1_ext = y1 - factor * dy
        x2_ext = x2 + factor * dx
        y2_ext = y2 + factor * dy

        long_line = LineString([(x1_ext, y1_ext), (x2_ext, y2_ext)])
        clipped = long_line.intersection(bounds)

        if clipped.geom_type == "LineString":
            coords = list(clipped.coords)
        elif clipped.geom_type == "MultiLineString":
            coords = list(max(clipped.geoms, key=lambda g: g.length).coords)
        else:
            continue  # Skip invalid lines

        # Overwrite the original line with extended endpoints
        ext_lines.append([*coords[0], *coords[-1]])
    return ext_lines


def symmetry_score(points, center_type='median'):
    """
    Compute a symmetry score for 1D points by reflecting about the center.

    Args:
        points: array-like of shape (N,)
        center_type: 'median' or 'mean'

    Returns:
        float: lower is more symmetric
    """
    points = np.sort(points)
    N = len(points)

    if center_type == 'median':
        center = np.median(points)
    elif center_type == 'mean':
        center = np.mean(points)
    else:
        raise ValueError("center_type must be 'median' or 'mean'")

    reflected = 2 * center - points[::-1]  # reflect sorted points
    return np.mean(np.abs(points - reflected))  # or use np.square(...) for


def line_to_points(x1, y1, x2, y2, num=100):
    """Generate points along a line."""
    return np.linspace([x1, y1], [x2, y2], num=num)


def mask_value_at(mask, point, tol=3):
    x, y = int(round(point[0])), int(round(point[1]))
    h, w = mask.shape
    return 0 <= y < h and 0 <= x < w and mask[y-tol:y+tol, x-tol:x+tol].any()


def find_relevant_points(lines, other_lines, other_lines_tree, mask, tol=3):
    relevant_lines = []
    for (x1, y1, x2, y2) in lines:
        pts = line_to_points(x1, y1, x2, y2, num=200)
        line_geom = LineString(pts)

        # Check which points lie on a True mask value
        pts_on_mask = [tuple(p)
                       for p in pts if mask_value_at(mask, p, tol=tol)]

        # Check if the line intersects any of the other lines
        intersections = []
        for other in other_lines_tree.query(line_geom):
            other = other_lines[other]
            inter = line_geom.intersection(other)
            if inter.is_empty:
                continue
            if inter.geom_type == "Point":
                intersections.append((inter.x, inter.y))
            elif inter.geom_type == "MultiPoint":
                intersections.extend([(p.x, p.y) for p in inter.geoms])

        # Combine and deduplicate
        all_pts = pts_on_mask + intersections
        if len(all_pts) >= 2:
            # Pick extremal points (e.g., first and last after sorting)
            sorted_pts = sorted(all_pts, key=lambda p: (p[0], p[1]))
            p1, p2 = sorted_pts[0], sorted_pts[-1]
            relevant_lines.append((p1[0], p1[1], p2[0], p2[1]))

    return relevant_lines


def compute_joined_lines(group_a, group_b, mask, tol=3):
    group_a_geoms = [LineString([(x1, y1), (x2, y2)])
                     for (x1, y1, x2, y2) in group_a]
    group_b_geoms = [LineString([(x1, y1), (x2, y2)])
                     for (x1, y1, x2, y2) in group_b]

    tree_a = STRtree(group_a_geoms)
    tree_b = STRtree(group_b_geoms)

    new_lines_a = find_relevant_points(group_a, group_b_geoms, tree_b, mask, tol=tol)
    new_lines_b = find_relevant_points(group_b, group_a_geoms, tree_a, mask, tol=tol)

    return new_lines_a, new_lines_b


def extend_linestring(line: LineString, n: float) -> LineString:
    coords = np.array(line.coords)

    # Get direction vectors for start and end
    start_dir = coords[0] - coords[1]
    end_dir = coords[-1] - coords[-2]

    # Normalize
    start_dir = start_dir / np.linalg.norm(start_dir)
    end_dir = end_dir / np.linalg.norm(end_dir)

    # Extend
    new_start = coords[0] + n * start_dir
    new_end = coords[-1] + n * end_dir

    # Create new coordinates
    new_coords = np.vstack([new_start, coords[1:-1], new_end])
    return LineString(new_coords)


def get_intersection_point(p1, p2, q1, q2, extend=0):
    l1 = LineString([p1, p2])
    l2 = LineString([q1, q2])
    if extend > 0:
        l1 = extend_linestring(l1, extend)
        l2 = extend_linestring(l2, extend)

    inter = l1.intersection(l2)
    if inter.is_empty:
        return None
    return np.array(inter.coords[0])


# ----------------- Grouping similar lines -----------------

def are_lines_close(line1, line2, N):
    s1, e1 = line1[:2], line1[2:]
    s2, e2 = line2[:2], line2[2:]
    return (np.linalg.norm(s1 - s2) <= N and np.linalg.norm(e1 - e2) <= N) or \
           (np.linalg.norm(s1 - e2) <= N and np.linalg.norm(e1 - s2) <= N)


def align_line(ref_line, line):
    s_ref, e_ref = ref_line[:2], ref_line[2:]
    s, e = line[:2], line[2:]
    direct = np.linalg.norm(s_ref - s) + np.linalg.norm(e_ref - e)
    flipped = np.linalg.norm(s_ref - e) + np.linalg.norm(e_ref - s)
    return (s, e) if direct <= flipped else (e, s)


def average_line_group(lines, group):
    ref_line = lines[group[0]]
    aligned_lines = [align_line(ref_line, lines[i]) for i in group]

    starts = np.array([l[0] for l in aligned_lines])
    ends = np.array([l[1] for l in aligned_lines])

    avg_start = np.mean(starts, axis=0)
    avg_end = np.mean(ends, axis=0)

    return (*avg_start, *avg_end)


def group_lines(lines, N):
    from collections import defaultdict, deque

    # Build graph
    adj = defaultdict(list)
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            if are_lines_close(lines[i], lines[j], N):
                adj[i].append(j)
                adj[j].append(i)

    # BFS to find components
    visited = set()
    groups = []

    for i in range(len(lines)):
        if i not in visited and adj[i]:
            group = []
            queue = deque([i])
            visited.add(i)
            while queue:
                node = queue.popleft()
                group.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            groups.append(group)

    # for groups pick right orientation for each line and average
    grouped = np.array([average_line_group(lines, group) for group in groups])
    if grouped.size == 0:
        grouped = np.zeros((0, 4))

    # add unreferenced lines and return
    groups_flat = [idx for group in groups for idx in group]
    ungrouped_idx = np.setdiff1d(np.arange(len(lines)), groups_flat)

    return grouped, ungrouped_idx

# ----------------- For eliminating side surfaces -----------------
def compute_common_normal_and_sort(lines):
    """
    Args:
        lines: list of tuples [(p1, p2), ...] where p1 and p2 are 2D points (np.array of shape (2,))
        
    Returns:
        sorted_lines: lines sorted according to projection onto the common normal
        normal: the common normal vector (unit length)
    """
    # Step 1: Compute direction vectors
    directions = [l[2:] - l[:2] for l in lines]
    directions = np.array([d / np.linalg.norm(d)
                          for d in directions])  # normalize each

    # Step 2: Compute mean direction
    mean_dir = np.mean(directions, axis=0)
    mean_dir /= np.linalg.norm(mean_dir)

    # Step 3: Compute normal (perpendicular vector)
    normal = np.array([-mean_dir[1], mean_dir[0]])  # 90° rotation in 2D
    normal /= np.linalg.norm(normal)

    # Step 4: Project each line's midpoint onto the normal
    projections = []
    for i, l in enumerate(lines):
        p1, p2 = l[:2], l[2:]
        midpoint = (p1 + p2) / 2
        proj_value = np.dot(midpoint, normal)
        projections.append(proj_value)

    return np.argsort(projections), normal


def count_mask_hits_between_lines(mask, sorted_lines, normal, d, tol=3):
    """
    Args:
        mask: 2D binary numpy array
        sorted_lines: list of lines [(p1, p2), ...] sorted by projection along the common normal
        normal: 2D numpy array, the common normal (unit vector)
        d: float, threshold between 0 and 1 (e.g., 0.5 for 50%)
    
    Returns:
        counts: list of counters, one per pair of adjacent lines
    """
    kernel = np.ones((tol*2+1, tol*2+1),
                     dtype=bool)  # 2-pixel radius (5x5 square)
    dilated_mask = binary_dilation(mask.copy(), structure=kernel)

    height, width = mask.shape
    counts = []

    for i in range(len(sorted_lines) - 1):
        l0, l1 = sorted_lines[i], sorted_lines[i + 1]
        p1a, p1b = l0[:2], l0[2:]
        p2a, p2b = l1[:2], l1[2:]

        # Compute how far to shift in normal direction (in pixels)
        mid1 = (p1a + p1b) / 2
        mid2 = (p2a + p2b) / 2
        dist = np.dot(mid2 - mid1, normal)
        num_steps = int(np.ceil(abs(dist)))

        count = 0
        for step in range(num_steps):
            offset = normal * step
            q1 = (p1a + offset).astype(int)
            q2 = (p1b + offset).astype(int)

            # Clamp to mask bounds
            if np.any(q1 < 0) or np.any(q1 >= [width, height]) or \
               np.any(q2 < 0) or np.any(q2 >= [width, height]):
                continue

            # Rasterize the line into pixel indices
            rr, cc = skimage_line(q1[1], q1[0], q2[1],
                                  q2[0])  # row = y, col = x
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            rr, cc = rr[valid], cc[valid]

            if len(rr) == 0:
                continue

            values = dilated_mask[rr, cc]
            ratio = np.sum(values) / len(values)

            if ratio >= d:
                count += 1

        counts.append(count)

    return counts

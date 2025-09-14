import cv2
import numpy as np

from shapely.strtree import STRtree
from shapely.geometry import LineString, box
import numpy as np
from itertools import combinations

from skimage.draw import line as skimage_line
from scipy.ndimage import binary_dilation
import math
from matplotlib.path import Path



class TBlockTracker:
    # hardcoded color limits..
    color_u = np.array([250, 140, 85])
    color_l = np.array([110, 0, 0]) 
    # harcoded real scale
    real_scale = 30
    
    def __init__(self, mode='bgr'):
        self.mode = mode
        temp_pts_real = get_template_pts(scale=self.real_scale)
        self.temp_pts_real = np.concatenate([temp_pts_real, np.zeros((4, 1))], axis=1)

    def get_obj(self, img, min_cnt_area=100):
        img = img.copy()
        mask = img.astype(np.float32)
        mask = ((mask[..., 0] - mask[..., 1] > 50) & (mask[..., 0] - mask[..., 2] > 50)).astype(np.uint8)
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
    
    def get_main_line(self, obj_only, pts, pts_none, normal_check_w=70, pair_check_w=2):
        edges = cv2.Canny(obj_only, threshold1=50, threshold2=70)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=50, minLineLength=1, maxLineGap=30)
    
        if lines is None:
            print('no lines')
            return None
        
        lines = lines[:, 0]
        lens = np.array([np.hypot((x2 - x1), (y2 - y1)) for (x1, y1, x2, y2) in lines])
        idx = np.argmax(lens)
        l = lines[idx]
        angle = math.atan2((l[3] - l[1]), (l[2] - l[0]))
        mid = np.array([(l[0] + l[2]) / 2, (l[1] + l[3]) / 2])

        normal = np.array([-math.sin(angle), math.cos(angle)])
        tmp1 = count_points_in_orthogonal_rectangle(l[:2], l[2:], normal, normal_check_w, pts)
        tmp2 = count_points_in_orthogonal_rectangle(l[:2], l[2:], -normal, normal_check_w, pts)
        if tmp2 > tmp1:
            normal = -normal

        pair = self.get_pair(obj_only, pts_none, normal, mid, angle, pair_check_w)
        if pair is None:
            print('no pair for first')
            return None
        return l, pair, angle, normal

    def get_pair(self, obj_only, pts_none, normal, mid, angle, ang_tol=3, pair_check_w=2):
        # get pair
        edges = cv2.Canny(obj_only, threshold1=10, threshold2=30)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi /
                                180, threshold=10, minLineLength=1, maxLineGap=30)
        lines = lines[:, 0]
        angles2 = np.array([math.atan2((y2 - y1), (x2 - x1))
                        for (x1, y1, x2, y2) in lines])
        mids2 = np.array([((x1 + x2) / 2, (y1 + y2) / 2)
                        for (x1, y1, x2, y2) in lines])
        ls = []
        for _l, _angle, _mid in zip(lines, angles2, mids2):
            mid_diff_norm = np.dot(normal, _mid - mid)
            if abs(angle_diff_rad(angle, _angle)) < np.radians(ang_tol) and 20 <= mid_diff_norm <= 80:
                if count_points_in_orthogonal_rectangle(_l[:2], _l[2:], normal, pair_check_w, pts_none) > 0:
                    ls.append(_l)
        if len(ls) == 0:
            return None
        lens2 = np.array([np.hypot((x2 - x1), (y2 - y1)) for (x1, y1, x2, y2) in ls])
        l2 = ls[np.argmax(lens2)]
        return l2
    
    def get_second_line(self, obj_only, masked, pts, pts_none, angle, ang_tol_perp=20, normal_check_w=70, pair_check_w=2):
        ang_tol_perp = np.radians(ang_tol_perp)
        edges = cv2.Canny(masked, threshold1=10, threshold2=30)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                threshold=20, minLineLength=1, maxLineGap=30)
        
        if lines is None:
            print('No second line found')
            return None

        lines = lines[:, 0]
        angles = np.array([math.atan2((y2 - y1), (x2 - x1)) for (x1, y1, x2, y2) in lines])
        ls = []
        for l, _angle in zip(lines, angles):
            if abs(angle_diff_rad(angle - np.pi/2, _angle)) < ang_tol_perp or \
                    abs(angle_diff_rad(angle + np.pi/2, _angle)) < ang_tol_perp:
                _normal = np.array([-math.sin(_angle), math.cos(_angle)])
                tmp1 = count_points_in_orthogonal_rectangle(l[:2], l[2:], _normal, normal_check_w, pts)
                tmp2 = count_points_in_orthogonal_rectangle(l[:2], l[2:], -_normal, normal_check_w, pts)
                if tmp2 > tmp1:
                    _normal = -_normal
                if count_points_in_orthogonal_rectangle(l[:2], l[2:], -_normal, pair_check_w, pts_none) > 0:
                    ls.append((l, _normal))
        if len(ls) == 0:
            print('no second line')
            return None

        lens = np.array([np.hypot((x2 - x1), (y2 - y1)) for ((x1, y1, x2, y2), _) in ls])
        l, _normal = ls[np.argmax(lens)]
        mid = np.array([(l[0] + l[2]) / 2, (l[1] + l[3]) / 2])
        angle = math.atan2((l[3] - l[1]), (l[2] - l[0]))
        pair = self.get_pair(obj_only, pts_none, _normal,
                             mid, angle, pair_check_w)
        if pair is None:
            return l, _normal
        return l, pair, _normal




    def detect_corners(self, img, min_cnt_area=1000, fat = 10):
        # get object
        obj_only = self.get_obj(img, min_cnt_area)
        if obj_only is None:
            return None

        mask = np.any(obj_only, axis=-1)
        pts = np.argwhere(mask)[:, ::-1]
        pts_none = np.argwhere(~mask)[:, ::-1]

        # get main line
        main = self.get_main_line(obj_only, pts, pts_none)
        if main is None:
            return None
        main, main_pair, angle, normal = main

        # mask between main and its pair
        h, w = obj_only.shape[:2]
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        masked = obj_only.copy()
        lfat = np.array([main[0] - fat * normal[0], main[1] - fat * normal[1],
                        main[2] - fat * normal[0], main[3] - fat * normal[1]])

        l2fat = np.array([main_pair[0] + fat * normal[0], main_pair[1] + fat * normal[1],
                         main_pair[2] + fat * normal[0], main_pair[3] + fat * normal[1]])

        mask = points_between_lines((lfat[:2], lfat[2:]), (l2fat[:2], l2fat[2:]), np.concatenate(
            [X[..., None], Y[..., None]], axis=-1).reshape(-1, 2))
        mask = mask.reshape(h, w)
        masked[mask] = 0


        # get second line
        second = self.get_second_line(obj_only, masked, pts, pts_none, angle)
        if second is None:
            return None
        elif len(second) == 3:
            second, second_pair, _ = second
        else:
            # maybe also shift till
            second, normal_second = second
            mid_main = np.array([(main[0] + main[2]) / 2, (main[1] + main[3]) / 2])
            mid_pair = np.array([(main_pair[0] + main_pair[2]) / 2, (main_pair[1] + main_pair[3]) / 2])
            main_dist = np.dot(mid_pair - mid_main, normal)
            second_pair = np.concatenate([second[:2] + main_dist * normal_second, second[2:] + main_dist * normal_second])

        corners = []
        for l in [main, main_pair]:
            for l2 in [second, second_pair]:
                corner = get_intersection_point(
                    l[:2], l[2:], l2[:2], l2[2:], extend=500)
                if corner is None:
                    print('no corner')
                    return None
                corners.append(corner)

        corners = np.array(corners)
        corners = order_points_clockwise(corners)

        return corners


# ----------- Utility functions -----------
def angle_diff_rad(theta1, theta2):
    """
    Compute minimal angle difference (in radians) between two angles.
    Result is in [-np.pi, np.pi].

    Args:
        theta1, theta2: angles in radians

    Returns:
        float: signed angle difference
    """
    diff = (theta2 - theta1 + np.pi) % (2*np.pi) - np.pi
    return diff


def count_points_in_orthogonal_rectangle(A, B, normal, width, points):
    A = np.array(A)
    B = np.array(B)
    n = np.array(normal)  # Assumed orthogonal and unit length
    points = np.array(points)

    # Rectangle corners
    P1 = A
    P2 = B
    P3 = B + width * n
    P4 = A + width * n
    rectangle = np.array([P1, P2, P3, P4])

    # Path for point-in-polygon
    path = Path(rectangle)
    inside = path.contains_points(points)

    return np.sum(inside)


def line_from_points(p1, p2):
    # Returns (a, b, c) for the line equation ax + by + c = 0
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, c


def point_side(line, point):
    a, b, c = line
    x, y = point
    return a * x + b * y + c

def points_between_lines(line1_pts, line2_pts, points):
    line1 = line_from_points(*line1_pts)
    line2 = line_from_points(*line2_pts)

    sides1 = points[:, 0] * line1[0] + points[:, 1] * line1[1] + line1[2]
    sides2 = points[:, 0] * line2[0] + points[:, 1] * line2[1] + line2[2]

    # Points between the lines will have opposite signs with respect to the lines
    return (sides1 * sides2) < 0


def get_intersection_point(p1, p2, q1, q2, extend=500):
    l1 = LineString([p1, p2])
    l2 = LineString([q1, q2])
    if extend > 0:
        l1 = extend_linestring(l1, extend)
        l2 = extend_linestring(l2, extend)

    inter = l1.intersection(l2)
    if inter.is_empty:
        return None
    return np.array(inter.coords[0])


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


def order_points_clockwise(pts):
    """
    Orders 2D points in clockwise order.
    pts: (N, 2) array-like (should be 4 points ideally)
    Returns: list of 4 points in clockwise order
    """
    pts = np.array(pts)
    center = np.mean(pts, axis=0)

    # Compute angle from center to each point
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

    # Sort by angle in clockwise order (use -angles for clockwise)
    ordered = pts[np.argsort(-angles)]

    return ordered

def get_template_pts(scale=0.5):
    pts = np.array([[0, 0], [scale, 0], [scale, scale], [0, scale]], dtype=np.float32)
    return pts
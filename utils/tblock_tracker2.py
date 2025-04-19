import cv2
import numpy as np
from itertools import combinations


class TBlockTracker:
    # hardcoded color limits..
    color_u = np.array([250, 140, 85])
    color_l = np.array([90, 0, 0]) 

    def __init__(self, table_bb, approx_contours=True, mode='bgr'):
        self.table_bb = table_bb
        self.approx_contours = approx_contours
        self.mode = mode

    def get_lines(self, img, min_cnt_area=1000):
        mx, my, mw, mh = self.table_bb
        img = img[my: my+mh, mx: mx+mw]
        if self.mode == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = np.all((self.color_l <= img) & (img <= self.color_u), axis=-1).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = tuple(
            contour for contour in contours if cv2.contourArea(contour) > min_cnt_area)
        if self.approx_contours:
            contours = tuple(cv2.approxPolyDP(contour, 1, True)
                             for contour in contours)
        
        mask = cv2.drawContours(
            np.zeros(img.shape[:-1]), contours, -1, color=255, thickness=-1)
        img_copy = img.copy()
        img_copy[mask == 0] = 0
        
        gray = cv2.cvtColor(img_copy.copy(), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=30, threshold2=100)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                                minLineLength=1, maxLineGap=50)
        lines = lines[:, 0]
        if lines is None:
            return None
        
        angles = np.array([np.arctan2(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in lines])
        group1, group2 = [], []
        min_ang, max_ang = angles.min(), angles.max()
        for i, angle in enumerate(angles):
            d1 = abs(angle - min_ang)
            d2 = abs(angle - max_ang)
            if d1 < d2:
                group1.append(lines[i])
            else:
                group2.append(lines[i])
        return mask, group1, group2, contours
    
    def get_rectangles(self, group1, group2):
        ls = []
        for (l1a, l1b) in combinations(group1, 2):
            for (l2a, l2b) in combinations(group2, 2):
                c1 = get_intersection_point(*l1a, *l2a)
                c2 = get_intersection_point(*l1a, *l2b)
                c3 = get_intersection_point(*l1b, *l2a)
                c4 = get_intersection_point(*l1b, *l2b)
                cs = [c1, c2, c3, c4]
                is_none = np.array([c is None for c in cs])
                if is_none.sum() == 0:
                    ls.append(((l1a, l1b, l2a, l2b), (cs)))
        return ls
    
    def detect_single_pose(self, img, tol=5):
        mask, group1, group2, _ = self.get_lines(img)
        if len(group1) < 2 or len(group2) < 2:
            print('Not enough lines detected, need at least 2 vertical and 2 horizontal lines')
            return None
        
        # TODO: maybe extend infinitely for the failure case?, maybe try both
        try:
            ext_group1 = [extend_line((x1, y1), (x2, y2), mask, tol=tol)
                          for (x1, y1, x2, y2) in group1]
            ext_group2 = [extend_line((x1, y1), (x2, y2), mask, tol=tol)
                        for (x1, y1, x2, y2) in group2]
            rectangles = self.get_rectangles(ext_group1, ext_group2)
        except:
            ext_group1 = [extend_line((x1, y1), (x2, y2), mask, tol=tol*5)
                          for (x1, y1, x2, y2) in group1]
            ext_group2 = [extend_line((x1, y1), (x2, y2), mask, tol=tol*5)
                          for (x1, y1, x2, y2) in group2]
            rectangles = self.get_rectangles(ext_group1, ext_group2)

        if len(rectangles) == 0:
            print('No rectangles found')
            return None
        
        squares, areas = [], []
        for i, (_ls, cs) in enumerate(rectangles):
            l1a, l1b, l2a, l2b = _ls
            c1, c2, c3, c4 = cs

            h = max(np.linalg.norm(c1 - c2), np.linalg.norm(c3 - c4))
            w = max(np.linalg.norm(c1 - c3), np.linalg.norm(c2 - c4))
            rat = min(h, w) / max(h, w)
            area = h * w
            if rat > 0.75 and area > 2_000:
                squares.append(((l1a, l1b, l2a, l2b), (cs)))
                areas.append(area)
        
        if len(squares) > 0:
            # TODO: Is this really the best way to choose the square?
            square = squares[np.argmin(areas)]
            return square

        
        # TODO: maybe we see side surface?
        print('No squares found')
        return None

def get_intersection_point(p1, p2, q1, q2):
    def det(a, b):
        return a[0]*b[1] - a[1]*b[0]

    xdiff = (p1[0] - p2[0], q1[0] - q2[0])
    ydiff = (p1[1] - p2[1], q1[1] - q2[1])

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Lines are parallel or colinear

    d = (det(p1, p2), det(q1, q2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # Check if the point is on both segments
    def on_segment(a, b, p):
        return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

    intersection = (x, y)
    if on_segment(p1, p2, intersection) and on_segment(q1, q2, intersection):
        return np.array(intersection)
    return None

def extend_line(p1, p2, mask, max_len=1000, tol=5):
    def in_mask(x, y):
        if tol == 0:
            val = mask[y, x] > 0
        else:
            val = mask[y-tol:y+tol, x-tol:x+tol].sum() > 0
        return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and val

    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    norm = np.hypot(dx, dy)
    dx, dy = dx / norm, dy / norm  # unit direction

    def step(p, dx, dy, sign=1):
        for i in range(max_len):
            xi = int(round(p[0] + sign * i * dx))
            yi = int(round(p[1] + sign * i * dy))
            if not in_mask(xi, yi):
                return (int(round(p[0] + sign * (i - 1) * dx)), int(round(p[1] + sign * (i - 1) * dy)))
        return (xi, yi)

    new_p1 = step(p1, dx, dy, -1)
    new_p2 = step(p2, dx, dy, 1)
    return new_p1, new_p2


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

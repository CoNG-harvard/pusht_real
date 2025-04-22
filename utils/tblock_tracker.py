import pickle
import cv2
import numpy as np
from sklearn.cluster import KMeans

class TBlockTracker:
    # hardcoded color limits..
    color_u = np.array([220, 80, 80])
    color_l = np.array([140, 10, 10]) # 90
    # harcoded real scale
    real_scale = 50

    def __init__(self, template_scale=30, template_w=180, template_h=180,
                 template_rot_angle_step=1, allowed_angle_diff_per_frame=60, mode='bgr'):
        self.temp_img, self.temp_pts, self.temp_keypoint = generate_template(scale=template_scale, 
                                                                             w=template_w, 
                                                                             h=template_h)
        # hardcoded
        self.temp_pts_real, self.temp_keypoint_real = get_template_pts(scale=self.real_scale)
        # 3d space
        self.temp_pts_real = np.concatenate([self.temp_pts_real, np.zeros((self.temp_pts_real.shape[0], 1))], axis=1)
        
        self.allowed_angle_diff_per_frame = allowed_angle_diff_per_frame
        self.template_rot_angle_step = template_rot_angle_step
        self.mode = mode
        self.temp_dct = self.generate_template_bank(angle_step=self.template_rot_angle_step)
        self.first = True

    def generate_template_bank(self, angle_step=1):
        template_bank = []
        keypoints_all = []
        rotated_pts_all = []
        angles = np.arange(0, 360, angle_step)
        for angle in angles:
            rotated, rotated_pts, keypoints = rotate_image(self.temp_img, self.temp_pts, self.temp_keypoint, angle)
            rotated_pts_all.append(rotated_pts)
            template_bank.append(rotated)
            keypoints_all.append(keypoints)
        temp_dct = {angle: (template, rotated_pts, keypoint) for angle, (template, keypoint, rotated_pts) in zip(
            angles, zip(template_bank, keypoints_all, rotated_pts_all))}
        return temp_dct
    
    def detect_block_pose(self, image, morph_sz=11):
        if self.first:
            self.score = 0.0
            angle_last = 180
            allowed_ang_diff = 180
        else:
            angle_last = self.curr_ang
            allowed_ang_diff = self.allowed_angle_diff_per_frame

        score, curr_pts, curr_kp, curr_ang = self.detect_block_pose_single(image, morph_sz=morph_sz, 
                                                                           angle_last=angle_last, 
                                                                           allowed_ang_diff=allowed_ang_diff)
        

        if (self.score - score < 0.35):
            self.curr_pts = curr_pts
            self.curr_kp = curr_kp
            self.curr_ang = curr_ang
            self.score = score
        
        self.first = False
        return self.score, self.curr_pts.astype(np.int32), self.curr_kp.astype(np.int32), self.curr_ang

    def detect_block_pose_single(self, image, angle_last=180, allowed_ang_diff=180):
        if self.mode == 'bgr':
            color_u, color_l = self.color_u[::-1], self.color_l[::-1]
        else:
            color_u, color_l = self.color_u, self.color_l

        temp_dct = {k: v for k, v in self.temp_dct.items() if 
                    ((k-angle_last) % 360 <= allowed_ang_diff) or ((angle_last - k) % 360 <= allowed_ang_diff)}
        try:
            results = match_template_bank(image, temp_dct, color_u, color_l, use_bb=True)
        except:
            results = match_template_bank(image, temp_dct, color_u, color_l, use_bb=False)

        if len(results) == 0:
            return None
        
        best_result = max(results, key=lambda x: x["score"])
        x, y = best_result['top_left']

        _, pts, keypoint = self.temp_dct[best_result['angle']]
        keypoint = keypoint + np.array([x, y])
        pts = pts + np.array([x, y])
        return best_result['score'], pts, keypoint, best_result['angle']
    
def match_template_bank(image, temp_dct, color_u, color_l, mask_thk=80, method=cv2.TM_CCOEFF_NORMED, use_bb=True):

    # Find contours of the color
    mask = np.all((color_l <= image) & (image <= color_u), axis=-1).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = tuple(
        contour for contour in contours if cv2.contourArea(contour) > 500)
    contours = tuple(cv2.approxPolyDP(contour, 1, True)
                    for contour in contours)
    mask = cv2.drawContours(
        np.zeros(image.shape[:-1]), contours, -1, color=255, thickness=-1)
    cpy = image.copy()
    cpy[mask == 0] = 0

    # surface clearing
    red_mask = cpy[..., 0]
    red_mask_flat = red_mask.flatten()
    red_mask_flat = red_mask_flat[red_mask_flat > 0]
    kmeans = KMeans(n_clusters=2).fit(red_mask_flat.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()
    mx, mn = (max(centers), min(centers))

    diff1 = abs(red_mask - mx)
    diff2 = abs(red_mask - mn)
    top_surf = diff1 < diff2

    kernel = np.ones((5, 5), np.uint8)
    top_surf = cv2.morphologyEx(top_surf.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    top_surf = cv2.morphologyEx(top_surf, cv2.MORPH_CLOSE, kernel)
    mask = top_surf


    results = []

    if use_bb:
        h, w = mask.shape
        x, y, wb, hb = cv2.boundingRect(mask.astype(np.uint8))
        start_y, start_x = max(0, y-mask_thk), max(0, x-mask_thk)
        end_y, end_x = min(h, y+hb+mask_thk), min(w, x+wb+mask_thk)
        mask = mask[start_y: end_y, start_x: end_x]
        start = np.array([start_x, start_y])
        
    for angle, (template, _, _) in temp_dct.items():
        res = cv2.matchTemplate(mask, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if use_bb:
            max_loc += start

        score = max_val if method in [
            cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED] else -min_val
        results.append({
            "score": score,
            "angle": angle,
            "top_left": max_loc,
            "template": template
        })
    return results

def get_template_pts(scale=0.5):
    length = 4*scale
    dl = (length - scale) / 2
    pts = np.array([[0, 0], [length, 0],
                    [length, scale], [length - dl, scale],
                    [length - dl, length], [dl, length],
                    [dl, scale], [0, scale]], dtype=np.float32)
    # pts = np.array([[-length / 2, 0], 
    #                 [length / 2, 0],
    #                 [length / 2, -scale],
    #                 [length / 2 - dl, -scale],
    #                 [length / 2 - dl, -length],
    #                 [-length / 2 + dl, -length],
    #                 [-length / 2 + dl, -scale],
    #                 [-length / 2, -scale]], dtype=np.float32)
    kp = np.array([length / 2, scale / 2], dtype=np.float32)
    return pts, kp

def generate_template(scale=30, w=200, h=200):
    pts, kp = get_template_pts(scale=scale)
    length = 4*scale
    tlx, tly = (h - length) / 2, (w - length) / 2
    
    pts += np.array([tlx, tly])
    kp += np.array([tlx, tly])

    template = np.zeros((w, h))
    template = cv2.fillPoly(
        template, [pts.astype(np.uint64)], color=255)

    return template, pts, kp


def rotate_image(image, pts, keypoint, angle):
    """Rotate image around center without cropping."""
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_NEAREST).astype(np.uint8)

    keypoint = M @ np.array([keypoint[0], keypoint[1], 1])

    pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
    rotated_pts = pts @ M.T

    return rotated, rotated_pts, keypoint


def angle_in(angle, low, high):
    """Check if angle is in range [low, high)"""
    if low < high:
        return low <= angle < high
    else:
        return low <= angle or angle < high


def get_kalman_filter(init_state, dt=1):
    # 6 state vars: x, y, θ, dx, dy, dθ; 3 measurements: x, y, θ
    kf = cv2.KalmanFilter(6, 3)

    kf.statePost = init_state.astype(np.float32)

    # Transition matrix (constant velocity model)
    kf.transitionMatrix = np.array([
        [1, 0, 0, dt, 0, 0],
        [0, 1, 0, 0, dt, 0],
        [0, 0, 1, 0, 0, dt],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(6, dtype=np.float32)

    return kf

import pickle
import cv2
import numpy as np

class TBlockTracker:
    # hardcoded color limits..
    color_u = np.array([220, 80, 80])
    color_l = np.array([90, 10, 10])

    def __init__(self, template_len=4, template_scale=30, template_w=180, template_h=180,
                 template_rot_angle_step=1, allowed_angle_diff_per_frame=24, mode='bgr'):
        self.temp_img, self.temp_pts, self.temp_keypoint = generate_template(length=template_len, 
                                                                             scale=template_scale, 
                                                                             w=template_w, 
                                                                             h=template_h)
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
    
    def detect_block_pose(self, image, morph_sz=11, use_kf=True):
        if self.first:
            angle_last = 180
            allowed_ang_diff = 180
        else:
            angle_last = self.curr_ang
            allowed_ang_diff = self.allowed_angle_diff_per_frame

        self.curr_pos, self.curr_ang = self.detect_block_pose_single(image, morph_sz=morph_sz, 
                                                                     angle_last=angle_last, allowed_ang_diff=allowed_ang_diff)
        if use_kf and self.curr_pos is not None:
            if self.first:
                init = np.array([self.curr_pos[0], self.curr_pos[1], self.curr_ang, 0, 0, 0 ], dtype=np.float32)
                self.kf = get_kalman_filter(init_state=init, dt=1)
            else:
                self.kf.predict()
                self.kf.correct(np.array([self.curr_pos[0], self.curr_pos[1], self.curr_ang], dtype=np.float32))
                self.curr_pos = np.array([self.kf.statePost[0], self.kf.statePost[1]], dtype=np.int32)
                self.curr_ang = self.kf.statePost[2]
        
        self.first = False
        return self.curr_pos, int(self.curr_ang)

    def detect_block_pose_single(self, image, morph_sz=11, angle_last=180, allowed_ang_diff=180):
        if self.mode == 'bgr':
            color_u, color_l = self.color_u[::-1], self.color_l[::-1]
        else:
            color_u, color_l = self.color_u, self.color_l

        temp_dct = {k: v for k, v in self.temp_dct.items() if 
                    (k-angle_last) % 360 <= allowed_ang_diff or (angle_last - k) % 360 <= allowed_ang_diff}
        results = match_template_bank(image, temp_dct, color_u, color_l, morph_sz=morph_sz)
        if len(results) == 0:
            return None
        
        best_result = max(results, key=lambda x: x["score"])
        x, y = best_result['top_left']

        _, _, keypoint = self.temp_dct[best_result['angle']]
        kpx, kpy = keypoint
        kpx = kpx + y
        kpy = kpy + x
        return np.array([kpx, kpy]), best_result['angle']
    
def match_template_bank(image, temp_dct, color_u, color_l, mask_thk=80, morph_sz=11, method=cv2.TM_CCOEFF_NORMED):

    mask = np.all((color_l <= image) & (image <= color_u), axis=-1).astype(np.uint8)
    kernel = np.ones((morph_sz, morph_sz), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    results = []
    x, y, wb, hb = cv2.boundingRect(mask.astype(np.uint8))

    h, w = mask.shape
    start_y, start_x = max(0, y-mask_thk), max(0, x-mask_thk)
    end_y, end_x = min(h, y+hb+mask_thk), min(w, x+wb+mask_thk)
    mask = mask[start_y: end_y, start_x: end_x]

    start = np.array([start_x, start_y])
    for angle, (template, _, _) in temp_dct.items():
        res = cv2.matchTemplate(mask, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
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


def generate_template(length=4, scale=30, w=200, h=200):
    length *= scale

    tlx, tly = (h - length) / 2, (w - length) / 2
    dl = (length - scale) / 2
    pts = np.array([[tlx, tly], [tlx + length, tly],
                          [tlx + length, tly + scale], [tlx + length - dl, tly + scale],
                          [tlx + length - dl, tly + length], [tlx + dl, tly + length],
                          [tlx + dl, tly + scale], [tlx, tly + scale]])
    kp = np.array([tly + scale / 2, tlx + length / 2], dtype=np.int32)
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

    Minv = cv2.getRotationMatrix2D(center, -angle, 1.0)
    keypoint = Minv @ np.array([keypoint[0], keypoint[1], 1])
    keypoint = keypoint.astype(np.int32)

    pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
    rotated_pts = pts @ M.T
    rotated_pts = rotated_pts.astype(np.int32)

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

    kf.statePost = init_state

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

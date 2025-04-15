import pickle
import cv2
import numpy as np

class TBlockTracker:
    def __init__(self, template_path):
        self.load_template(template_path)

    def load_template(self, template_path):
        with open(template_path, 'rb') as f:
            dct = pickle.load(f)
            self.template = dct['template']
            self.temp_keypoint = dct['keypoint']
        self.generate_template_bank()

    def generate_template_bank(self, angle_step=5):
        template_bank = []
        keypoints = []
        angles = np.arange(0, 360, angle_step)
        for angle in angles:
            rotated, keypoint = rotate_image(
                self.template, self.temp_keypoint, angle)
            template_bank.append(rotated)
            keypoints.append(keypoint)
        keypoints = np.array(keypoints, dtype=np.int32)
        self.template_bank = template_bank
        self.angles = angles
        self.template_dct = {angle: (template, keypoint) for angle, template, keypoint in zip(angles, template_bank, keypoints)}

    def detect_block_pose(self, image):
        u = np.array([220, 80, 80])[::-1]
        l = np.array([90, 10, 10])[::-1]

        mask = np.all((l <= image) & (image <= u), axis=-1).astype(np.uint8)

        sz = 11
        kernel = np.ones((sz, sz), np.uint8)
        mask = mask.copy()
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        results = self.match_template_bank(mask)
        if len(results) == 0:
            return None
        best_result = max(results, key=lambda x: x["score"])
        x, y = best_result['top_left']

        _, keypoint = self.template_dct[best_result['angle']]
        kpx, kpy = keypoint
        kpx = kpx + y
        kpy = kpy + x
        return np.array([kpx, kpy]), best_result['angle']
    
    def match_template_bank(self, mask, method=cv2.TM_CCOEFF_NORMED):
        results = []
        for template, angle in zip(self.template_bank, self.angles):
            res = cv2.matchTemplate(mask, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            score = max_val if method in [
                cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED] else -min_val
            results.append({
                "score": score,
                "angle": angle,
                "top_left": max_loc,
                "template": template
            })
        return results


def rotate_image(image, keypoint, angle):
    """Rotate image around center without cropping."""
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_NEAREST).astype(np.uint8)

    tmp = np.zeros_like(image)
    tmp[keypoint[0], keypoint[1]] = 255
    tmp = cv2.warpAffine(
        tmp, M, (w, h), flags=cv2.INTER_NEAREST).astype(np.uint8)
    rotated_keypoint = np.argmax(tmp)
    rotated_keypoint = np.unravel_index(rotated_keypoint, tmp.shape)

    return rotated, rotated_keypoint


    

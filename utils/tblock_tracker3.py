import cv2
import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans


class TBlockTracker:
    # hardcoded color limits..
    color_u = np.array([250, 140, 85])
    color_l = np.array([90, 0, 0]) 

    def __init__(self, mode='bgr'):
        self.mode = mode

    def get_surface(self, img, min_cnt_area=1000):
        img = img.copy()
        mask = np.all((self.color_l <= img) & (img <= self.color_u), axis=-1).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = tuple(
            contour for contour in contours if cv2.contourArea(contour) > min_cnt_area)
        contours = tuple(cv2.approxPolyDP(contour, 1, True)
                        for contour in contours)

        mask = cv2.drawContours(
            np.zeros(img.shape[:-1]), contours, -1, color=255, thickness=-1)
        obj_only = img.copy()
        obj_only[mask == 0] = 0

        red_mask = obj_only[..., 0]
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

        return obj_only, top_surf

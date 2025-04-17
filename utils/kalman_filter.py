import numpy as np

import cv2

def get_kalman_filter():
    kf = cv2.KalmanFilter(6, 3)  # 6 state vars: x, y, θ, dx, dy, dθ; 3 measurements: x, y, θ
    dt = 1 / 30
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

    # kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
    # kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
    # kf.errorCovPost = np.eye(6, dtype=np.float32)
    kf.processNoiseCov = np.diag([1, 1, 1e-1, 1e-1, 1e-1, 1e-3]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([5, 5, 1e-1]).astype(np.float32)
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 1
    
    return kf
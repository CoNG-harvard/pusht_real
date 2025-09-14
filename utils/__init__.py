import numpy as np

window_size = 512

CENTER = np.array([-0.151 - 0.25, -0.125 - 0.25])

# cameraMatrix = np.load(osp.join(pkg_dir, "cameraMatrix.npy"))
# distortionCoeffs = np.load(osp.join(pkg_dir, "distortions.npy"))

# camera_tcp = [-0.05785834, 0.01470036, 0.02680225, -0.19765198, 0.15008495, -1.55158033, ]
# CAMERA_TCP = [-0.07119155, 0.03392638, 0.0302255, -0.20909042, 0.21550955, -1.53705897]
CAMERA_TCP = [0.00985436, -0.06754846, 0.03888817, 0.0444753, 0.37508808, 2.88841112]
ROD_TCP = [0.0, 0.0, 0.2002, 0.0, 0.0, -1.57079633]

CAMERA_POSE = [-0.67209632, 0.5332222, 0.6231013, -2.07117477, 1.64185512, -0.53794638]
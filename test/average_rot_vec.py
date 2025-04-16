import numpy as np
from scipy.spatial.transform import Rotation as R

# Original rotation vector (axis-angle) – example
rot_vec = np.array([0.1, 0.2, 0.3])  # 3D vector (radians)

# Define noise parameters
noise_std = 0.05  # Standard deviation of the Gaussian noise
num_samples = 100

# Generate noisy rotation vectors
noisy_rot_vecs = rot_vec + np.random.normal(scale=noise_std, size=(num_samples, 3))

# Convert noisy vectors to rotation matrices
rot_mats = R.from_rotvec(noisy_rot_vecs).as_matrix()

# Average rotation matrices using rotation averaging
# We'll use the method of averaging via rotation matrices and converting back
# to a single average rotation

# Step 1: Sum the rotation matrices
rot_mat_avg = np.mean(rot_mats, axis=0)

# Step 2: Use SVD to project the mean matrix back onto SO(3)
U, _, Vt = np.linalg.svd(rot_mat_avg)
R_avg = U @ Vt

# Ensure it's a valid rotation matrix (det(R) = +1)
if np.linalg.det(R_avg) < 0:
    U[:, -1] *= -1
    R_avg = U @ Vt

# Convert back to rotation vector
rot_vec_avg = R.from_matrix(R_avg).as_rotvec()

print("Average rotation vector:", rot_vec_avg)

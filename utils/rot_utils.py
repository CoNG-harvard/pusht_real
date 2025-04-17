import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

def average_rotations(rotation_vectors):
    # # Assume you have a list of rotation vectors (N x 3)
    # rotation_vectors = np.array([
    #     [0.1, 0.2, 0.1],
    #     [0.12, 0.18, 0.11],
    #     [0.11, 0.21, 0.09],
    #     # ... more vectors
    # ])

    # Convert to Rotation objects
    rotations = R.from_rotvec(rotation_vectors)

    # Convert to quaternions
    quaternions = rotations.as_quat()  # shape (N, 4)

    # Average the quaternions (naive mean and normalize)
    mean_quat = np.mean(quaternions, axis=0)
    mean_quat /= np.linalg.norm(mean_quat)

    # Convert back to rotation vector
    mean_rotation = R.from_quat(mean_quat).as_rotvec()

    return mean_rotation


import numpy as np
from scipy.spatial.transform import Rotation as R

def get_z_inverted_rotvec(marker_world):
    marker_tvec_world, marker_rvec_world = marker_world[:3], marker_world[3:]
    
    # Convert to Rotation objects
    rotations = R.from_rotvec(marker_rvec_world)
    
    # Step 1: Define your target z-direction
    z_target = -1 * rotations.as_matrix()[:, -1]
    z_axis = z_target / np.linalg.norm(z_target)

    # Step 2: Define reference x-axis
    x_reference = np.array([0, -1, 0])

    # Step 3: Project x_reference onto plane orthogonal to z_axis
    x_axis = x_reference - np.dot(x_reference, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    # Step 4: Get y-axis to complete right-handed system
    y_axis = np.cross(z_axis, x_axis)

    # Step 5: Build rotation matrix
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # Step 6: Convert rotation matrix to rotation vector
    rot = R.from_matrix(rotation_matrix)
    rotation_vector = rot.as_rotvec()

    return rotation_vector


def rodrigues_to_matrix(rvec, tvec):
    """Convert Rodrigues rotation vector to 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def reverse_transformation(T):
    """Reverse a transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

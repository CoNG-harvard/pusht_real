import numpy as np
import cv2


center = np.array([-0.151 - 0.25, -0.125 - 0.25])
fixed_camera_pose = [-0.65697878, -0.14177968,
                     0.6323136, -1.83416353,  2.11347714, -0.60475937]


def rodrigues_to_matrix(rvec, tvec):
    """Convert Rodrigues rotation vector to 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def pose_tran_fixed_camera(marker_rvec, marker_tvec, fixed_cam_pose):
    T_marker_to_camera = rodrigues_to_matrix(marker_rvec, marker_tvec)
    fixed_cam_pose = np.array(fixed_cam_pose)
    fixed_cam_rvec, fxied_cam_tvec = fixed_cam_pose[3:], fixed_cam_pose[:3]
    T_camera_to_world = rodrigues_to_matrix(fixed_cam_rvec, fxied_cam_tvec)
    T_marker_to_world = T_camera_to_world @ T_marker_to_camera
    R = T_marker_to_world[:3, :3]
    tvec = T_marker_to_world[:3, 3]
    rvec = cv2.Rodrigues(R)[0]
    return np.concatenate([tvec, rvec.flatten()])


def marker_to_pose(marker_world):
    marker_tvec_world, marker_rvec_world = marker_world[:3], marker_world[3:]
    R, _ = cv2.Rodrigues(np.array(marker_rvec_world))
    z = R[:, -1]
    x = R[:, 0]
    y = R[:, 1]
    target_pos = marker_tvec_world + 0.026 * y
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([target_pos[0], target_pos[1], - yaw])


def real_to_env(pose_real: np.ndarray) -> np.ndarray:
    """transform real world to env coordinate.

    Args:
        pose_real (np.ndarray): real pose
        x_env = 256 + 0.6 * 1000 * (x_real - center[0])
        y_env = 256 - 0.6 * 1000 * (y_real - center[1])

    Returns:
        _type_: _description_
    """

    x = 256 + 0.6 * 1000 * (pose_real[0] - center[0])
    y = 256 - 0.6 * 1000 * (pose_real[1] - center[1])
    yaw = pose_real[2]
    return np.array([x, y, yaw])


def marker_to_env(marker_world):
    pose_real = marker_to_pose(marker_world)
    pose_env = real_to_env(pose_real)
    return pose_env


def render_env(env, rvec, tvec):
    T_leftupper_to_camera = rodrigues_to_matrix(rvec, tvec)
    T_origin_to_leftupper = rodrigues_to_matrix(
        np.array([np.pi, 0., 0.,]), np.array([100, 0.0, 0.0]))
    T_origin_to_camera = T_leftupper_to_camera @ T_origin_to_leftupper
    rvec = cv2.Rodrigues(T_origin_to_camera[:3, :3])[0]
    tvec = T_origin_to_camera[:3, 3]
    marker_world = pose_tran_fixed_camera(rvec, tvec / 1000., fixed_camera_pose)
    tblk_env = marker_to_env(marker_world)

    obs, _ = env.reset(tblk_env, np.zeros(2))
    return env._render_frame('rgb_array')

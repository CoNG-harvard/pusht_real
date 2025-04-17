import numpy as np
import cv2
from scipy.optimize import least_squares
import os.path as osp
import pyrealsense2 as rs

pkg_dir = osp.dirname(osp.dirname(__file__))
import sys
sys.path.append(pkg_dir)
from utils.marker_util import Marker, MarkerReader, ARUCO_DICT
import rtde_control
import rtde_receive


# Initial guess of fixed camera pose (rvec, tvec): [-1.83416353  2.11347714 -0.60475937] [-0.65697878 -0.14177968  0.6323136 ]



# ============== Robot Configuration ==============
rtde_c = rtde_control.RTDEControlInterface("192.168.1.10")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.10")
from utils import CAMERA_TCP
rtde_c.setTcp(CAMERA_TCP)

# ====== Parameters ======
MARKER_SIZE = 0.038  # Size of the ArUco marker in meters (e.g., 5cm)
CAMERA_MATRIX_D405 = np.load(osp.join(pkg_dir, "data", "cameraMatrix.npy"))  # Load your camera matrix
DIST_COEFFS_D405 = np.load(osp.join(pkg_dir, "data", "distortions.npy"))  # Replace with your distortion coefficients

# =============== Initialize cameras ==================
# Configure both cameras
pipeline_d405 = rs.pipeline()
config1 = rs.config()
config1.enable_device('126122270638')  # Replace with your camera's serial number
config1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline_d435 = rs.pipeline()
config2 = rs.config()
config2.enable_device('233722070172')  # Replace with your camera's serial number
config2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start both pipelines
profile_d405 = pipeline_d405.start(config1)
profile_d435 = pipeline_d435.start(config2)

color_stream_d435 = profile_d435.get_stream(rs.stream.color)  # Get color stream
color_stream_d405 = profile_d405.get_stream(rs.stream.color)  # Get color stream
intr_d435 = color_stream_d435.as_video_stream_profile().get_intrinsics()

# Print camera matrix and distortion coefficients
cameraMatrix_d435 = np.array([
    [intr_d435.fx, 0, intr_d435.ppx],
    [0, intr_d435.fy, intr_d435.ppy],
    [0, 0, 1]
])

distortionCoeffs_d435 = np.array(intr_d435.coeffs)  # [k1, k2, p1, p2, k3]

intr_d405 = color_stream_d405.as_video_stream_profile().get_intrinsics()

# Print camera matrix and distortion coefficients
cameraMatrix_d405 = np.array([
    [intr_d405.fx, 0, intr_d405.ppx],
    [0, intr_d405.fy, intr_d405.ppy],
    [0, 0, 1]
])

distortionCoeffs_d405 = np.array(intr_d405.coeffs)  # [k1, k2, p1, p2, k3]

# ============== Set up camera parameters ==============
device = profile_d405.get_device()
color_sensor = device.query_sensors()[0]  # Usually the second sensor is the RGB camera

# Turn off auto-exposure
# if color_sensor.supports(rs.option.enable_auto_exposure):
#     color_sensor.set_option(rs.option.enable_auto_exposure, 0)
#     # color_sensor.set_option(rs.option.enable_auto_exposure, 1)
# # Set manual exposure (in microseconds)
# if color_sensor.supports(rs.option.exposure):
#     color_sensor.set_option(rs.option.exposure, 1000)  # Try values like 50–500 depending on your lighting
# Optionally set gain
if color_sensor.supports(rs.option.gain):
    color_sensor.set_option(rs.option.gain, 48)  # Default is 16; lower = less bright

# ============== Initialize marker reader ==============
markerId=0
markerDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_4X4_50"])    
markerReader_d405 = MarkerReader(markerId, markerDict, 38, cameraMatrix_d405, distortionCoeffs_d405)
markerReader_d435 = MarkerReader(markerId, markerDict, 38, cameraMatrix_d435, distortionCoeffs_d435)

# ====== Helper Functions ======
def detect_markers(image):
    """Detect ArUco markers and return corners and IDs."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict)
    return corners, ids

def estimate_marker_pose(corners, camera_matrix, dist_coeffs, marker_size):
    """Estimate marker pose relative to the camera."""
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_size, camera_matrix, dist_coeffs
    )
    return rvec, tvec  # Rotation (Rodrigues) and translation

def rodrigues_to_matrix(rvec, tvec):
    """Convert Rodrigues rotation vector to 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

# ====== Bundle Adjustment ======
def bundle_adjustment_residuals(params, observations, robot_poses, marker_poses):
    """
    Residuals for bundle adjustment.
    Args:
        params: [tx, ty, tz, rx, ry, rz] (fixed camera pose)
        observations: List of (marker_corners_img, marker_id)
        robot_poses: List of T_cam_to_base (4x4) for each observation
    """
    # Convert fixed camera pose from params to 4x4 matrix
    rvec = params[3:]
    tvec = params[:3]
    T_fixed_to_base = rodrigues_to_matrix(rvec, tvec)

    residuals = []
    for (corners, marker_id), T_cam_to_base in zip(observations, robot_poses):
        # Get 3D marker corners in marker frame (assuming 1 marker for simplicity)
        marker_corners_3d = np.array([
            [-MARKER_SIZE/2, MARKER_SIZE/2, 0],
            [MARKER_SIZE/2, MARKER_SIZE/2, 0],
            [MARKER_SIZE/2, -MARKER_SIZE/2, 0],
            [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
        ])

        # Transform marker corners to fixed camera frame
        T_marker_to_fixed = rodrigues_to_matrix(marker_poses)  # From marker detection in fixed camera
        T_marker_to_base = T_fixed_to_base @ T_marker_to_fixed
        marker_corners_base = (T_marker_to_base[:3, :3] @ marker_corners_3d.T + T_marker_to_base[:3, [3]]).T

        # Project marker corners into robot camera
        projected_corners, _ = cv2.projectPoints(
            marker_corners_base,
            cv2.Rodrigues(T_cam_to_base[:3, :3])[0],
            T_cam_to_base[:3, 3],
            CAMERA_MATRIX_D405,
            DIST_COEFFS_D405
        )

        # Compute residuals (observed - projected)
        residuals.extend((corners.reshape(-1, 2) - projected_corners.reshape(-1, 2)).flatten())
    
    return np.array(residuals)

def optimize_fixed_camera_pose(initial_pose, observations, robot_poses):
    """Run bundle adjustment to refine fixed camera pose."""
    result = least_squares(
        bundle_adjustment_residuals,
        x0=initial_pose,
        args=(observations, robot_poses),
        method='lm',
        verbose=2
    )
    return result.x

# ====== Main Pipeline ======
if __name__ == "__main__":
    # Example data: Replace with real observations
    observations = []  # List of (corners_in_fixed_cam, marker_id)
    robot_poses = []   # List of T_cam_to_base (4x4) for robot camera

    # (1) Collect data: Loop over multiple robot poses and detect markers
    while True:
        # Get frames from both cameras
        frames1 = pipeline_d405.wait_for_frames()
        frames2 = pipeline_d435.wait_for_frames()
        
        # Get color frames
        color_frame1 = frames1.get_color_frame()
        color_frame2 = frames2.get_color_frame()
        

        
        if not color_frame1 or not color_frame2:
            continue
        
        color_frame2 = np.asanyarray(color_frame2.get_data())
        color_frame1 = np.asanyarray(color_frame1.get_data())
        
        key = cv2.waitKey(1) & 0xFF
        
        # if key == ord('w'):
        #     move_z(rtde_c, rtde_r, -20, 0.01)
            # time.sleep(2.0)
        if key == ord('r'):
        # Capture image from fixed camera and detect markers
            fixed_cam_image = color_frame1  # Your fixed camera image
            corners, ids = detect_markers(fixed_cam_image)
            if ids is not None:
                for (corner, id) in zip(corners, ids):
                    if id == 0:
                        observations.append((corner, id))  # Assuming 1 marker

            # Capture image from robot camera (pose known via kinematics)
            robot_cam_image = color_frame2
            found, robot_cam_image, marker = markerReader_d435.detectMarkers(robot_cam_image, markerDict)
            # rvec, tvec = estimate_marker_pose(corners, cameraMatrix_d435, distortionCoeffs_d435, MARKER_SIZE)
            T_marker_to_cam = rodrigues_to_matrix(marker.rvec[0], marker.tvec[0] / 1000)
            pose_cam = np.array(rtde_r.getActualTCPPose())  # From robot FK
            T_cam_to_base = rodrigues_to_matrix(pose_cam[3:], pose_cam[:3])
            robot_poses.append(T_cam_to_base)
            print("current poses", len(robot_poses), len(observations))
            
        if key == ord('q'):
            break  
        
        # Combine images horizontally
        combined = np.vstack((color_frame1, color_frame2))
        
        # Show images
        cv2.namedWindow('RealSense Cameras', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense Cameras', combined)

    # (2) Initial guess for fixed camera pose (e.g., from first observation)
    # Initial guess of fixed camera pose (rvec, tvec): [-1.83416353  2.11347714 -0.60475937] [-0.65697878 -0.14177968  0.6323136 ]
    initial_pose = np.array([-0.65697878, -0.14177968, 0.6323136, -1.83416353, 2.11347714, -0.60475937]) # tvec, rvec

    # (3) Refine with bundle adjustment
    optimized_pose = optimize_fixed_camera_pose(initial_pose, observations, robot_poses)
    print("Optimized fixed camera pose (tvec, rvec):", optimized_pose)

    # Convert back to 4x4 matrix
    T_fixed_to_base = rodrigues_to_matrix(optimized_pose[3:], optimized_pose[:3])
    print("Fixed camera pose (4x4):\n", T_fixed_to_base)
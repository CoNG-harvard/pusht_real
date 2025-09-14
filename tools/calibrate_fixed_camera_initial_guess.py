import numpy as np
import cv2
from scipy.optimize import least_squares
import os.path as osp
import pyrealsense2 as rs

pkg_dir = osp.dirname(osp.dirname(__file__))
import sys
sys.path.append(pkg_dir)
from utils.marker_util import Marker, MarkerReader, ARUCO_DICT, get_average_rot_tran
import rtde_control
import rtde_receive



# ============== Robot Configuration ==============
rtde_c = rtde_control.RTDEControlInterface("192.168.0.191")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.0.191")
from utils import CAMERA_TCP
rtde_c.setTcp(CAMERA_TCP)

# ====== Parameters ======
MARKER_SIZE = 0.097  # Size of the ArUco marker in meters (e.g., 5cm)
# CAMERA_MATRIX_D405 = np.load(osp.join(pkg_dir, "data", "cameraMatrix.npy"))  # Load your camera matrix
# DIST_COEFFS_D405 = np.load(osp.join(pkg_dir, "data", "distortions.npy"))  # Replace with your distortion coefficients
markerId=0

# =============== Initialize cameras ==================
# Configure both cameras
pipeline_d405 = rs.pipeline()
config1 = rs.config()
config1.enable_device('234422060060')  # Replace with your camera's serial number
config1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline_d435 = rs.pipeline()
config2 = rs.config()
config2.enable_device('337322073528')  # Replace with your camera's serial number
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

markerDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_4X4_50"])    
markerReader_d405 = MarkerReader(markerId, markerDict, MARKER_SIZE * 1000, cameraMatrix_d405, distortionCoeffs_d405)
markerReader_d435 = MarkerReader(markerId, markerDict, MARKER_SIZE * 1000, cameraMatrix_d435, distortionCoeffs_d435)

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

def reverse_transformation(T):
    """Reverse a transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

# ====== Main Pipeline ======
if __name__ == "__main__":
    # Example data: Replace with real observations
    observations = []  # List of (corners_in_fixed_cam, marker_id)
    robot_poses = []   # List of T_cam_to_base (4x4) for robot camera

    try: 
        
        # (1) Collect data: Loop over multiple robot poses and detect markers
        while True:
            # Get frames from both cameras
            frames_d405 = pipeline_d405.wait_for_frames()
            frames_d435 = pipeline_d435.wait_for_frames()
            
            # Get color frames
            color_frame1 = frames_d405.get_color_frame()
            color_frame2 = frames_d435.get_color_frame()
            

            
            if not color_frame1 or not color_frame2:
                continue
            
            color_frame1 = np.asanyarray(color_frame1.get_data())
            color_frame2 = np.asanyarray(color_frame2.get_data())
            
            (found, color_frame2, marker) = markerReader_d435.detectMarkers(color_frame2, markerDict)
            (found, color_frame1, marker) = markerReader_d405.detectMarkers(color_frame1, markerDict)

            
            key = cv2.waitKey(1) & 0xFF
            
            # if key == ord('w'):
            #     move_z(rtde_c, rtde_r, -20, 0.01)
                # time.sleep(2.0)
            if key == ord('p'):
                if found:
                    rvec, tvec = get_average_rot_tran(markerReader_d435, markerDict, pipeline_d435, num=30)
                    marker_world = rtde_c.poseTrans(rtde_r.getActualTCPPose(), 
                                        np.concatenate([tvec / 1000, rvec]))
                    print("Found marker by d435 in world", marker_world)
                    marker_world = np.array(marker_world)
                    T_marker_to_world = rodrigues_to_matrix(marker_world[3:], marker_world[:3])
            
            if key == ord('i'):
                if found:
                    rvec, tvec = get_average_rot_tran(markerReader_d405, markerDict, pipeline_d405, num=30)
                    
                    T_marker_to_fixed = rodrigues_to_matrix(rvec, tvec / 1000)
                    print("Found marker at d405 in relative", T_marker_to_fixed)
                    T_fixed_to_marker = reverse_transformation(T_marker_to_fixed)
                    print("reverse transformation", T_fixed_to_marker)
                
            if key == ord('q'):
                break  
            
            # Combine images horizontally
            combined = np.vstack((color_frame1, color_frame2))
            
            # Show images
            cv2.namedWindow('RealSense Cameras', cv2.WINDOW_NORMAL)
            cv2.imshow('RealSense Cameras', combined)

        T_initial_guess = T_marker_to_world @ T_fixed_to_marker
        initial_guess_rvec = cv2.Rodrigues(T_initial_guess[:3, :3])[0]
        initial_guess_tvec = T_initial_guess[:3, 3]
        print("Initial guess of fixed camera pose (tvec, rvec):", np.concatenate([initial_guess_tvec.flatten(), initial_guess_rvec.flatten()]))
    
        # Get initial guess of the fixed camera pose
    finally:
        # Stop streaming
        pipeline_d405.stop()
        pipeline_d435.stop()
        cv2.destroyAllWindows()
        rtde_c.disconnect()
        rtde_r.disconnect()
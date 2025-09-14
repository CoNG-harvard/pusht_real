import pyrealsense2 as rs
import numpy as np
import cv2

import rtde_receive
import rtde_control

import numpy as np
import os.path as osp

pkg_dir = osp.dirname(osp.dirname(__file__))
import sys
sys.path.append(pkg_dir)
print(pkg_dir)
from utils.marker_util import Marker, MarkerReader, ARUCO_DICT
from utils.rot_utils import get_z_inverted_rotvec
from utils.robot_control import move_z
import time
from PIL import Image
from utils.marker_util import MultiMarkerObjectTracker

import pygame
import datetime
import pickle
import os
from utils.tblock_tracker import TBlockTracker
import yaml

window_size = 512

center = np.array([-0.155 - 0.256, -0.132 - 0.256])

# cameraMatrix = np.load(osp.join(pkg_dir, "cameraMatrix.npy"))
# distortionCoeffs = np.load(osp.join(pkg_dir, "distortions.npy"))
fixed_camera_pose = [-0.61720919, -0.12040074,  0.61971925, -1.87933148,  2.25229305, -0.69381239] # average
camera_tcp = [-0.07119155, 0.03392638, 0.0302255, -0.20909042, 0.21550955, -1.53705897]
rod_tcp = [0.0, 0.0, 0.2002, 0.0, 0.0, -1.57079633]
block_name = 'tee'
with open("./data/marker_layout.yaml", "r") as f:
    block_config = yaml.safe_load(f)
    
for s in block_config:
    if s['shape'] == block_name:
        block_config = s
        break
# Create pipeline


pipeline = rs.pipeline()

# Configure the pipeline
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

#config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
#config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)


# Define the red color range in RGB
red_lower = np.array([100, 0, 0])  # Lower bound for red (R > 100, G < 50, B < 50)
red_upper = np.array([255, 50, 50])  # Upper bound for red

rtde_c = rtde_control.RTDEControlInterface("192.168.1.10")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.10")
rtde_c.setTcp(camera_tcp)


def move_to_markers(rtde_c, rtde_r, marker_world, dist=0.05,):
    marker_tvec_world, marker_rvec_world = marker_world[:3], marker_world[3:]
    rtde_c.setTcp(rod_tcp)
    R, _ = cv2.Rodrigues(np.array(marker_rvec_world))
    z = R[:, -1]
    x = R[:, 0]
    y = R[:, 1]
    print(z)
    target_pose = marker_tvec_world + z * 0.05 + x * 0.07 + y * - 0.07
    print(target_pose)
    current_pose = rtde_r.getActualTCPPose()
    
    current_pose[0] = target_pose[0]
    current_pose[1] = target_pose[1]
    current_pose[2] = target_pose[2]
    rot_vec = get_z_inverted_rotvec(marker_world)
    current_pose[3] = rot_vec[0]
    current_pose[4] = rot_vec[1]
    current_pose[5] = rot_vec[2]
    rtde_c.moveL(current_pose, speed=0.05)
    rtde_c.setTcp(camera_tcp)
    
def make_contact(rtde_c):
    speed = [0, 0, -0.009, 0, 0, 0]
    rtde_c.moveUntilContact(speed, acceleration=0.1)
    marker_tvec_world, marker_rvec_world = marker_world[:3], marker_world[3:]
    R, _ = cv2.Rodrigues(np.array(marker_rvec_world))
    z = R[:, -1]
    x = R[:, 0]
    y = R[:, 1]
    target_pos = marker_tvec_world + 0.026 * y
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([target_pos[0], target_pos[1], - yaw])


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



# Start streaming
profile = pipeline.start(config)
# Get stream profile and camera intrinsics
color_stream = profile.get_stream(rs.stream.color)  # Get color stream
intr = color_stream.as_video_stream_profile().get_intrinsics()
print(intr.coeffs)

# Print camera matrix and distortion coefficients
cameraMatrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
])

distortionCoeffs = np.array(intr.coeffs)  # [k1, k2, p1, p2, k3]
marker_world = None
# rtde_c.moveL(seek_pose, speed=0.2)

## setup to detect marker 0

markerId=0
markerDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_4X4_50"])
marker = Marker()        
markerReader = MarkerReader(markerId, markerDict, 38, cameraMatrix, distortionCoeffs)

window = None

folder = 'images'

shape_tracker = MultiMarkerObjectTracker(block_config, cameraMatrix, distortionCoeffs)

tracker = TBlockTracker()

os.makedirs(folder, exist_ok=True)
if not os.path.exists(f'{folder}/intrinsic.pkl'):
    with open(f'{folder}/intrinsic.pkl', 'wb') as f:
        pickle.dump({'cameraMatrix':cameraMatrix, 'distortionCoeffs': distortionCoeffs}, f)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Reverse image
        # depth_image = depth_image[::-1,::-1]
        # color_image = color_image[::-1,::-1]
        
        # parameters = cv2.aruco.DetectorParameters()
        
        # detector = cv2.aruco.ArucoDetector(markerDict, parameters)
        # corners, ids, _ = detector.detectMarkers(color_image)
        # print(ids)
        
        color_image_orig = color_image.copy()
        
        found, rvec, tvec, color_image = shape_tracker.detect(color_image, 
                                                              fixed_camera_pose[3:], 
                                                              fixed_camera_pose[:3])
        

        # Show images
        cv2.imshow('RealSense Color', color_image)
        # cv2.imshow('RealSense Depth', depth_image)
        
        key = cv2.waitKey(1) & 0xFF
        
        if window is None:
            pygame.init()
            pygame.display.init()
            window = pygame.display.set_mode(
                (window_size, window_size))
        
        if key == ord('s'):
            print('saving')
            name = folder + '/' + str(datetime.datetime.now())
            img = Image.fromarray(color_image_orig, 'RGB')
            img.save(name + '.png')
            cv2.imwrite(name + '-depth.png', depth_image)

            
            if found:    
                marker_world = np.concatenate([tvec, rvec])
                tblk_env = marker_to_env(marker_world)
                with open(name+'-vecs.pkl', 'wb') as f:
                    pickle.dump({'rvec':rvec, 'tvec': tvec, 'marker_world': marker_world, 'tblk_env': tblk_env}, f)
            else:
                print('couldnt find marker')
            
        if key == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
    rtde_c.disconnect()

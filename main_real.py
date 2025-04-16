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
# from utils.robot_control import move_z, move_real_speed
import time
from pathlib import Path
import jax
from relax.utils.persistence import PersistFunction
import pickle

import pygame
from envs import PushTRealEnv

window_size = 512

center = np.array([-0.151 - 0.25, -0.125 - 0.25])

# cameraMatrix = np.load(osp.join(pkg_dir, "cameraMatrix.npy"))
# distortionCoeffs = np.load(osp.join(pkg_dir, "distortions.npy"))

# camera_tcp = [-0.05785834, 0.01470036, 0.02680225, -0.19765198, 0.15008495, -1.55158033, ]
camera_tcp = [-0.07119155, 0.03392638, 0.0302255, -0.20909042, 0.21550955, -1.53705897]

rod_tcp = [0.0, 0.0, 0.2002, 0.0, 0.0, -1.57079633]

# Create pipeline
pipeline = rs.pipeline()

# Configure the pipeline
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)


# Define the red color range in RGB
red_lower = np.array([100, 0, 0])  # Lower bound for red (R > 100, G < 50, B < 50)
red_upper = np.array([255, 50, 50])  # Upper bound for red

rtde_c = rtde_control.RTDEControlInterface("192.168.1.10")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.10")
rtde_c.setTcp(camera_tcp)


def move_real_speed(rtde_c,rtde_r, d):
    ''' 
    d: speed command in millimeters
    In the simulation, kp is 350
    real: simu = 200:120
    so the kp will change to ~ 600,
    means action 1 approximate to 600mm/s, 0.6
    if 100 Hz then 600 / 50 = 12 mm, maybe too large
    '''
    
    tcp_pose = rtde_r.getActualTCPPose()
    target_pose = tcp_pose.copy()
    target_pose[0] += d[0]*0.1
    target_pose[1] -= d[1]*0.1

    # When doing position-based force control, the speed has to be extremely slow for stable results.
    return rtde_c.servoL(target_pose, 0.0, 0.0, 
                         0.3, # time
                         0.2, #look_ahead time
                         350 # gain
                         )
    # return rtde_c.moveL(target_pose, 
    #                      np.linalg.norm(d), # speed
    #                      )
    # return rtde_c.speedL(target_pose, 
    #                      np.linalg.norm(d), # speed
    #                      )


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
    
def marker_to_pose(marker_world):
    marker_tvec_world, marker_rvec_world = marker_world[:3], marker_world[3:]
    R, _ = cv2.Rodrigues(np.array(marker_rvec_world))
    z = R[:, -1]
    x = R[:, 0]
    y = R[:, 1]
    print(np.arctan2(R[1, 0], R[0, 0]))
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

def get_agent_env_pos(rtde_r, rtde_c):
    rtde_c.setTcp(rod_tcp)
    rod_end_pose = rtde_r.getActualTCPPose()
    rod_end_xy = rod_end_pose[:2]
    x = 256 + 0.6 * 1000 * (rod_end_xy[0] - center[0])
    y = 256 - 0.6 * 1000 * (rod_end_xy[1] - center[1])
    rtde_c.setTcp(camera_tcp)
    return np.array([x, y])

def move_to_higher_center(rtde_c, rtde_r):
    rtde_c.setTcp(camera_tcp)
    center_pose = [center[0], center[1], 0.5, -3.1415730563016435, 0, 0]
    rtde_c.moveL(center_pose, speed=0.05)
    
def move_to_lower_push(rtde_c, rtde_r):
    rtde_c.setTcp(rod_tcp)
    center_pose = [center[0], center[1], 0.03, -3.1415730563016435, 0, 0]
    rtde_c.moveL(center_pose, speed=0.05)
    
def rotate_camera(rtde_c, rtde_r, direction):
    current_pose = rtde_r.getActualTCPPose()
    if direction == "w":
        rotvec = [-3.1415730563016435, 0, 0]
    elif direction == "a":
        rotvec = [2.22144147, 2.22144147, 0]
    elif direction == "s":
        rotvec = [3.1415730563016435, 0, 0]
    elif direction == "d":
        rotvec = [2.22144147, -2.22144147, 0]
    target_pose = current_pose[:3] + rotvec
    rtde_c.moveL(target_pose, speed=0.05)
    

# Start streaming
profile = pipeline.start(config)
# Get stream profile and camera intrinsics
color_stream = profile.get_stream(rs.stream.color)  # Get color stream
intr = color_stream.as_video_stream_profile().get_intrinsics()

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

env = PushTRealEnv()

### load policy

policy_root = Path('/home/mht/PycharmProjects/DACER-Diffusion-with-Online-RL/logs/pushtcurriculum-v1/sdac_2025-03-15_16-28-36_s100_mask_multitask')

obs = None

policy = PersistFunction.load(policy_root / "deterministic.pkl")
@jax.jit
def policy_fn(policy_params, obs):
    return policy(policy_params, obs).clip(-1, 1)

policy_path = "policy-4000000-800000.pkl"
with open(policy_root / policy_path, "rb") as f:
    policy_params = pickle.load(f)
    
## warm up
policy_fn(policy_params, np.zeros((10)))

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
        
        (found, color_image, marker) = markerReader.detectMarkers(color_image, markerDict)
        
        
        tvec = np.array(marker.tvec[0]) / 1000
        rvec = np.array(marker.rvec[0])

        # Show images
        cv2.imshow('RealSense Color', color_image)
        # cv2.imshow('RealSense Depth', depth_image)
        
        key = cv2.waitKey(1) & 0xFF
        
        # if key == ord('w'):
        #     move_z(rtde_c, rtde_r, -20, 0.01)
            # time.sleep(2.0)
        if key == ord('v'):
            rtde_c.servoStop()
            # time.sleep(2.0)
        # if key == ord('s'):
        #     move_z(rtde_c, rtde_r, -0.05, 0.025)
        # if key == ord('m'):
            # if marker_world is not None:
            #     move_to_markers(rtde_c, rtde_r, marker_world)
                
        # if key == ord('c'):
        #     if marker_world is not None:
        #         make_contact(rtde_c)
        if key == ord('r'):
            marker_world = rtde_c.poseTrans(rtde_r.getActualTCPPose(), 
                                    np.concatenate([tvec, rvec]))
            obs, _ = env.reset(marker_to_env(marker_world), get_agent_env_pos(rtde_r, rtde_c))
            env.render()
            
        if key == ord('s'):
            if obs is None:
                print("Please press r to reset the env first")
                continue
            action = policy_fn(policy_params, obs)
            print("action", action)
            obs, _, _, _, _ = env.step(action)
            env.render()
        
        if key == ord('a'):
            if action is None:
                print("Please press s to get actions first")
                continue
            move_real_speed(rtde_c, rtde_r, 600 / 1000 * action)
            action = None
            
                
        if key == ord('p'):
            if found:
                marker_world = rtde_c.poseTrans(rtde_r.getActualTCPPose(), 
                                    np.concatenate([tvec, rvec]))
                print("Found marker at", marker_world)
                print("t block env pos", marker_to_env(marker_world))
                print("agent_pos", get_agent_env_pos(rtde_r, rtde_c))
        if key == ord('h'):
            move_to_higher_center(rtde_c, rtde_r)
        if key == ord('b'):
            move_to_lower_push(rtde_c, rtde_r)
        if key == ord('i'):
            rotate_camera(rtde_c, rtde_r, "w")
        if key == ord('j'):
            rotate_camera(rtde_c, rtde_r, "a")
        if key == ord('k'):
            rotate_camera(rtde_c, rtde_r, "s")
        if key == ord('l'):
            rotate_camera(rtde_c, rtde_r, "d")
            # print(marker.tvec)
            # print(marker.rvec)
        # Break loop on 'q' key press
        # if key == ord('a'):
        #     move_real_speed(rtde_c, rtde_r, 600 / 1000 * np.array([-1.0, 0.0]))
        # if key == ord('d'):
        #     move_real_speed(rtde_c, rtde_r, 600 / 1000 * np.array([1.0, 0.0]))
        if key == ord('q'):
            # move_z(rtde_c, rtde_r, 0.08, 0.04)
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
    rtde_c.disconnect()
    rtde_r.disconnect()
    env.close()

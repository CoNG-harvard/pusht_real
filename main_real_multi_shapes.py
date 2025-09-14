"""

Main file with multi shapes.

Shape information stored in data/marker_layout.yaml.
"""

import pyrealsense2 as rs
import numpy as np
import cv2

import rtde_receive
import rtde_control

import numpy as np
import os.path as osp
import os
import yaml

pkg_dir = osp.dirname(__file__)
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
from utils.rot_utils import rodrigues_to_matrix
from utils.kalman_filter import get_kalman_filter
from utils.marker_util import MultiMarkerObjectTracker
from utils.csv_logger import CSVLogger, ActionCSVLogger
import pygame
from envs import PushTRealEnv

import datetime

use_kalman = False

window_size = 512

center = np.array([-0.155 - 0.256, -0.132 - 0.256])
target_z = 0.02

video_frames = []

# cameraMatrix = np.load(osp.join(pkg_dir, "cameraMatrix.npy"))
# distortionCoeffs = np.load(osp.join(pkg_dir, "distortions.npy"))

# camera_tcp = [-0.05785834, 0.01470036, 0.02680225, -0.19765198, 0.15008495, -1.55158033, ]

# tvec, rvec
# fixed_camera_pose = [-0.65697878, -0.14177968,  0.6323136, -1.83416353,  2.11347714, -0.60475937] # first d405
# fixed_camera_pose = [-0.61743599, -0.11876692,  0.60051386, -1.86847763,  2.24117025, -0.70505939] # second d435
# fixed_camera_pose = [-0.62894439, -0.11924051,  0.59263195, -1.8921129,   2.21835316 , -0.72856077] # third d435
# Final average rvec, tvec: [-1.87933148  2.25229305 -0.69381239] [-0.61720919 -0.12040074  0.61971925]
fixed_camera_pose = [-0.61720919, -0.12040074,  0.61971925, -1.87933148,  2.25229305, -0.69381239] # average
camera_tcp = [-0.07119155, 0.03392638, 0.0302255, -0.20909042, 0.21550955, -1.53705897]
rod_tcp = [0.0, 0.0, 0.2002, 0.0, 0.0, -1.57079633]
block_name = 'tee'
with open("./data/marker_layout.yaml", "r") as f:
    block_configs = yaml.safe_load(f)
    
for s in block_configs:
    if s['shape'] == block_name:
        block_config = s
        break

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
rtde_c.setTcp(rod_tcp)

kalman = get_kalman_filter()


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
    if target_pose[0] > -0.23:
        target_pose[1] = np.clip(target_pose[1], -0.755, -0.15)
    if target_pose[1] > -0.23:
        target_pose[0] = np.clip(target_pose[0], -0.555, -0.155)
    else:
        target_pose[0] = np.clip(target_pose[0], -0.755, 0.0)
    target_pose[2] = target_z

    # When doing position-based force control, the speed has to be extremely slow for stable results.
    return rtde_c.servoL(target_pose, 0.0, 0.0, 
                         0.3, # time
                         0.2, #look_ahead time
                         350 # gain
                         )
    
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

def get_agent_env_pos(rtde_r, rtde_c):
    rtde_c.setTcp(rod_tcp)
    rod_end_pose = rtde_r.getActualTCPPose()
    rod_end_xy = rod_end_pose[:2]
    x = 256 + 0.6 * 1000 * (rod_end_xy[0] - center[0])
    y = 256 - 0.6 * 1000 * (rod_end_xy[1] - center[1])#  - 7.5
    return np.array([x, y])

def move_to_higher_center(rtde_c, rtde_r):
    rtde_c.setTcp(camera_tcp)
    center_pose = [center[0], center[1], 0.5, -3.1415730563016435, 0, 0]
    rtde_c.moveL(center_pose, speed=0.05)
    rtde_c.setTcp(rod_tcp)
    
def move_to_lower_push(rtde_c, rtde_r):
    rtde_c.setTcp(rod_tcp)
    # center_pose = [-0.25, -0.25, target_z, -3.1415730563016435, 0, 0]
    center_pose = [center[0], center[1], target_z, -3.1415730563016435, 0, 0]
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

# markerId=0
# markerDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_4X4_50"])
# marker = Marker()        
# markerReader = MarkerReader(markerId, markerDict, 38, cameraMatrix, distortionCoeffs)
shape_tracker = MultiMarkerObjectTracker(block_config, cameraMatrix, distortionCoeffs)

window = None

env = PushTRealEnv()

### load policy

policy_root = Path('/home/mht/PycharmProjects/pusht_real/policy/sdac_2025-04-23_17-55-35_s100_test_use_atp1')
# policy_root = Path('/home/mht/PycharmProjects/pusht_real/policy/sac_2025-04-24_15-26-11_s100_test_use_atp1')

obs = None
recording = False

policy = PersistFunction.load(policy_root / "deterministic.pkl")
@jax.jit
def policy_fn(policy_params, obs):
    return policy(policy_params, obs).clip(-1, 1)

policy_path = "policy-4000000-800000.pkl"
with open(policy_root / policy_path, "rb") as f:
    policy_params = pickle.load(f)
    
## warm up
policy_fn(policy_params, np.zeros((10)))
kalman_initialized = False
last_rot = None
logger = None
step = 0

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
        
        found, rvec, tvec, color_image = shape_tracker.detect(color_image, 
                                                              fixed_camera_pose[3:], 
                                                              fixed_camera_pose[:3])
        
        if found:
            marker_world = np.concatenate([tvec, rvec])
            tblk_env = marker_to_env(marker_world)
            if use_kalman:
                if last_rot is None:
                    last_rot = tblk_env[2]
                else:
                    if np.abs(tblk_env[2] - last_rot)>0.5: # handling noise
                        pass
                    else:
                        measurement = np.array([[tblk_env[0]], [tblk_env[1]], [tblk_env[2]]], dtype=np.float32)
                        kalman.correct(measurement)
                        last_rot = tblk_env[2]
                
                predicted = kalman.predict().flatten()[:3]
            # print(predicted[:3])
            else:
                predicted = tblk_env
            
            env_block = block_name if block_name != 'thin_tee' else 'tee'
            obs, _ = env.reset(predicted,# predicted[:3], 
                               get_agent_env_pos(rtde_r, rtde_c),
                               options={"shape_type": env_block})
            img = env.render()
        

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
                
        if key == ord('c'):
            block_name = 'cee'
            for s in block_configs:
                if s['shape'] == block_name:
                    block_config = s
                    break
            shape_tracker = MultiMarkerObjectTracker(block_config, cameraMatrix, distortionCoeffs)
        if key == ord('c'):
            block_name = 'cee'
            for s in block_configs:
                if s['shape'] == block_name:
                    block_config = s
                    break
            shape_tracker = MultiMarkerObjectTracker(block_config, cameraMatrix, distortionCoeffs)
        #     if marker_world is not None:
        #         make_contact(rtde_c)
        if key == ord('r'):
            strtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = osp.join(pkg_dir,
                                 "real_logs",
                                 f"real_exps/{block_name}_{strtime}")
            os.makedirs(save_dir, exist_ok=True)
            recording = True
            configuration = {'block_name': block_name,
                             'init_pos_agent': get_agent_env_pos(rtde_r, rtde_c).tolist(),
                             'init_pos_block': predicted.tolist()}
            with open(osp.join(save_dir, 'configure.yaml'), 'w') as file:
                yaml.dump(configuration, file, default_flow_style=False)
            logger = CSVLogger(osp.join(save_dir, 'log.csv'))  
            action_logger = ActionCSVLogger(osp.join(save_dir, 'action_log.csv'))  
            print("Start recording")
            
        if key == ord('t'):
            # env.stop_recording()
            recording = False
            import imageio
            imageio.mimsave(osp.join(save_dir, 'video.mp4'), video_frames, fps=30)
            video_frames = []
            step = 0
            print("Stop recording")
            
            
        if key == ord('s'):
            if obs is None:
                print("Please press r to reset the env first")
                continue
            action = policy_fn(policy_params, obs)
            th = env.goal_pose[-1]
            rotmat_blk = np.array([[np.cos(th), -np.sin(th)],
                                    [np.sin(th), np.cos(th)]])  
            action = np.reshape(action, [2, 1])
            action = rotmat_blk @ action
            move_real_speed(rtde_c, rtde_r, 600 / 1000 * action.flatten())
            print("action", action)
            if recording:
                action_logger.log_data(step, action)
            if not found:
                obs, _, _, _, info = env.step(action)
                predicted = info['block_pose']
                img = env.render()
                print("Using step to guess the next obs")
                
        if recording:   
            video_frames.append(img)
            
        if logger is not None and recording:
            logger.log_data(step, get_agent_env_pos(rtde_r, rtde_c),
                            predicted)
            step += 1
            
        
        
        # if key == ord('a'):
        #     if action is None:
        #         print("Please press s to get actions first")
        #         continue
        #     move_real_speed(rtde_c, rtde_r, 600 / 1000 * action)
        #     action = None
            
                
        if key == ord('p'):
            # if found:
            #     marker_world = rtde_c.poseTrans(rtde_r.getActualTCPPose(), 
            #                         np.concatenate([tvec, rvec]))
            #     print("Found marker at", marker_world)
            #     print("t block env pos", marker_to_env(marker_world))
            #     print("agent_pos", get_agent_env_pos(rtde_r, rtde_c))
            if tblk_env is not None:
                env.set_goal_pose(tblk_env)
                print("Set goal pose to current", tblk_env)
        if key == ord('h'):
            move_to_higher_center(rtde_c, rtde_r)
        if key == ord('b'):
            move_to_lower_push(rtde_c, rtde_r)
        # if key == ord('i'):
        #     rotate_camera(rtde_c, rtde_r, "w")
        # if key == ord('j'):
        #     rotate_camera(rtde_c, rtde_r, "a")
        # if key == ord('k'):
        #     rotate_camera(rtde_c, rtde_r, "s")
        # if key == ord('l'):
        #     rotate_camera(rtde_c, rtde_r, "d")
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
        time.sleep(1 / 30.0)
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
    rtde_c.disconnect()
    rtde_r.disconnect()
    env.close()

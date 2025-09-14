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
from utils.rot_utils import rodrigues_to_matrix
from utils.kalman_filter import get_kalman_filter

import pygame
from utils.tblock_tracker6 import TBlockTracker
from envs import PushTRealEnv

window_size = 512
center = np.array([-0.151 - 0.25, -0.125 - 0.25])
home = np.array([-0.151 - 0.1, -0.125 - 0.1])

# cameraMatrix = np.load(osp.join(pkg_dir, "cameraMatrix.npy"))
# distortionCoeffs = np.load(osp.join(pkg_dir, "distortions.npy"))

# camera_tcp = [-0.05785834, 0.01470036, 0.02680225, -0.19765198, 0.15008495, -1.55158033, ]

# tvec, rvec
fixed_camera_pose = np.array([-0.61720919, -0.12040074,  0.61971925, -1.87933148,  2.25229305, -0.69381239]) # average
camera_tcp = [-0.07119155, 0.03392638, 0.0302255, -0.20909042, 0.21550955, -1.53705897]
rod_tcp = [0.0, 0.0, 0.2002, 0.0, 0.0, -1.57079633]

# Create pipeline
pipeline = rs.pipeline()

# Configure the pipeline
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)


# Define the red color range in RGB
red_lower = np.array([100, 0, 0])  # Lower bound for red (R > 100, G < 50, B < 50)
red_upper = np.array([255, 50, 50])  # Upper bound for red


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

def env_to_real(pose_env: np.ndarray) -> np.ndarray:
    """transform env coordinate to real world.

    Args:
        pose_env (np.ndarray): env pose
        x_real = (x_env - 256) / 600 + center[0]
        y_real = (256 - y_env) / 600 + center[1]

    Returns:
        _type_: _description_
    """
    
    x = (pose_env[0] - 256) / 600 + center[0]
    y = (256 - pose_env[1]) / 600 + center[1]
    yaw = pose_env[2]
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
    center_pose = [home[0], home[1], 0.03, -3.1415730563016435, 0, 0]
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

tracker = TBlockTracker(camera_matrix=cameraMatrix, fixed_camera_pose=fixed_camera_pose, mode='bgr')

src_pts = np.array([[150,100], [1050,75], [1200, 700], [150,700]], dtype=np.float32)
height, width = 700, 700
dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
H = cv2.getPerspectiveTransform(src_pts, dst_pts)

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
env2 = PushTRealEnv()

### load policy

policy_root = Path('/home/mht/PycharmProjects/DACER-Diffusion-with-Online-RL/logs/pushtcurriculum-v1/sdac_2025-03-15_16-28-36_s100_mask_multitask')


policy = PersistFunction.load(policy_root / "deterministic.pkl")
@jax.jit
def policy_fn(policy_params, obs):
    return policy(policy_params, obs).clip(-1, 1)

policy_path = "policy-4000000-800000.pkl"
with open(policy_root / policy_path, "rb") as f:
    policy_params = pickle.load(f)
    
## warm up
policy_fn(policy_params, np.zeros((10)))
obs = None
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame or not depth_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Reverse image
        # depth_image = depth_image[::-1,::-1]
        # color_image = color_image[::-1,::-1]
        
        #img_table = cv2.warpPerspective(color_image, H, (width, height))
    
        
        #(found, color_image, marker) = markerReader.detectMarkers(color_image, markerDict)
        

        #     marker_world = pose_tran_fixed_camera(rvec, tvec, fixed_camera_pose)
        #     tblk_env = marker_to_env(marker_world)
        #     # if not kalman_initialized:
        #     #     kalman.statePost = np.array([[tblk_env[0]], [tblk_env[1]], [tblk_env[2]], [0], [0], [0]], dtype=np.float32)
        #     #     kalman_initialized = True
        #     # else:
        #     if last_rot is None:
        #         last_rot = tblk_env[2]
        #     else:
        #         if np.abs(tblk_env[2] - last_rot)>0.5: # handling noise
        #             pass
        #         else:
        #             measurement = np.array([[tblk_env[0]], [tblk_env[1]], [tblk_env[2]]], dtype=np.float32)
        #             kalman.correct(measurement)
        #             last_rot = tblk_env[2]

        

        # Show images
        cv2.imshow('RealSense Color', color_image)
        # cv2.imshow('RealSense Depth', depth_image)
        
        key = cv2.waitKey(1) & 0xFF
        
        # if key == ord('w'):
        #     move_z(rtde_c, rtde_r, -20, 0.01)
            # time.sleep(2.0)
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
            loss, pose = tracker.estimate(color_image, depth_image)
            env_pose = real_to_env(pose)
            env.render()
            #obs, _ = env.reset(env_pose, get_agent_env_pos(rtde_r, rtde_c))

            # pass
            # on arm camera
            # marker_world = rtde_c.poseTrans(rtde_r.getActualTCPPose(), 
            #                         np.concatenate([tvec, rvec]))
            # fixed camera
            # if found:
            #     marker_world = pose_tran_fixed_camera(rvec, tvec, fixed_camera_pose)
            #     predicted = kalman.predict().flatten()
            #     obs, _ = env.reset(predicted[:3], get_agent_env_pos(rtde_r, rtde_c))
            #     env.render()
            
        if key == ord('s'):
            if obs is None:
                print("Please press r to reset the env first")
                continue
        
            res = tracker.track(color_image, depth_image, last_pose=env_to_real(obs), safe=False)
            if res is not None:
                loss, pose = res
                env_pose = real_to_env(pose)
                #obs, _ = env2.reset(env_pose,)
                    
            env.render()

  
        
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
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
    env.close()

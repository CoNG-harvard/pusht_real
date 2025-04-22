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

import pygame
import datetime
import pickle
import os
from utils.tblock_tracker import TBlockTracker


window_size = 512

# cameraMatrix = np.load(osp.join(pkg_dir, "cameraMatrix.npy"))
# distortionCoeffs = np.load(osp.join(pkg_dir, "distortions.npy"))

home_pose = [-0.5126198706787246,
 0.027845369496906615,
 0.23745981307202224,
 -1.0901105741573625,
 2.946376920000866,
 3.4564257722096603e-05]

seek_pose = [-0.27493117962596614,
 -0.06506990006878725,
 0.6307633915258241,
 2.8002145894588053,
 -0.0057063949190226,
 -0.16493417235262048]

camera_tcp = [-0.05785834, 0.01470036, 0.02680225, -0.19765198, 0.15008495, -1.55158033, ]
rod_tcp = [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]

# Create pipeline
pipeline = rs.pipeline()

# Configure the pipeline
config = rs.config()
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)


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

with open('data/homo2.pkl', 'rb') as f:
    dct = pickle.load(f)
    all_corners = np.array(dct['allCorners'])[:, 0]

rect = cv2.minAreaRect(all_corners.reshape(-1, 2))
src_pts = cv2.boxPoints(rect)
height, width = 400, 300
dst_points = np.float32([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
])
H = cv2.getPerspectiveTransform(src_pts, dst_points)

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
        
        (found, color_image_marked, marker) = markerReader.detectMarkers(color_image, markerDict)
        
        thk = 5
        if False and (res := tracker.detect_block_pose_single(color_image_orig)):
            score, pts, kp, orient = res
            pts, kp = pts.astype(int), kp.astype(int)
            color_image_marked[kp[1]-thk:kp[1]+thk, kp[0]-thk:kp[0]+thk] = [255, 0, 0]
            for px, py in pts:
                color_image_marked[py-thk:py+thk, px-thk:px+thk] = [0, 255, 0]
        
        tvec = np.array(marker.tvec[0]) / 1000
        rvec = np.array(marker.rvec[0])

        # Show images
        color_image = cv2.warpPerspective(color_image.copy(), H, (width, height))
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
                img_marked = Image.fromarray(color_image_marked, 'RGB')
                img_marked.save(name+'-marked.png')
                with open(name+'-vecs.pkl', 'wb') as f:
                    pickle.dump({'rvec':rvec, 'tvec': tvec}, f)
            else:
                print('couldnt find marker')
            
        if key == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
    rtde_c.disconnect()

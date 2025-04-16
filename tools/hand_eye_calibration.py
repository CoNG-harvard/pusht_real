import cv2
import numpy as np

import sys
import tty
import termios
import select
import time
import os.path as osp
import pyrealsense2 as rs
from datetime import datetime

pkg_dir = osp.dirname(osp.dirname(__file__))
import sys
sys.path.append(pkg_dir)
print(pkg_dir)
from utils.marker_util import Marker, MarkerReader, ARUCO_DICT
from scipy.spatial.transform import Rotation as R

# cameraMatrix = np.load(osp.join(pkg_dir, "cameraMatrix.npy"))
# distortionCoeffs = np.load(osp.join(pkg_dir, "distortions.npy"))

# Create pipeline
pipeline = rs.pipeline()

# Configure the pipeline
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
markerId=0

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


# Robot listeners
import rtde_receive

rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.10")

def is_key_pressed():
    # Check if there's keyboard input ready
    return select.select([sys.stdin], [], [], 0)[0]


def hand_eye_calibration(R_gripper2base,
                         t_gripper2base,
                         R_target2cam,
                         t_target2cam,):
    # Collect rotation and translation pairs for each pose


    # Calibrate hand-eye
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # Combine to get full transformation matrix
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.ravel()
    rvec, _ = cv2.Rodrigues(R_cam2gripper)
    print(rvec)
    print(t_cam2gripper)
    np.save(osp.join(pkg_dir, 'data', f"rvec_hand_eye_calibration_" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + ".npy"), rvec)
    np.save(osp.join(pkg_dir, "data", f"tvec_hand_eye_calibration_" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + ".npy"), t_cam2gripper)
    

def get_average_rot_tran(markerReader, markerDict, pipeline, num=30):
    rotmats = []
    tvecs = []
    while len(rotmats) < num:
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        (found, color_image, marker) = markerReader.detectMarkers(color_image, markerDict)
        rot_mat = R.from_rotvec(marker.rvec[0]).as_matrix()
        rotmats.append(rot_mat)
        tvecs.append(marker.tvec[0])
        time.sleep(0.03)
    rot_mat_avg = np.mean(np.array(rotmats), axis=0)
    tvec_avg = np.mean(np.array(tvecs), axis=0)

    # Step 2: Use SVD to project the mean matrix back onto SO(3)
    U, _, Vt = np.linalg.svd(rot_mat_avg)
    R_avg = U @ Vt

    # Ensure it's a valid rotation matrix (det(R) = +1)
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt

    # Convert back to rotation vector
    rot_vec_avg = R.from_matrix(R_avg).as_rotvec()
    return rot_vec_avg, tvec_avg
    



def main():
    R_gripper2base = []  # list of rotation matrices (3x3)
    t_gripper2base = []  # list of translation vectors (3x1)

    R_target2cam = []    # from marker detection
    t_target2cam = []
    print("Press any key (press 'q' to quit)...")

    # Save current terminal settings
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        # Set terminal to cbreak mode (raw input without Enter)
        tty.setcbreak(fd)

        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            # depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Reverse image
            # depth_image = depth_image[::-1,::-1]
            # color_image = color_image[::-1,::-1]

            markerDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_4X4_50"])

            marker = Marker()        
            markerReader = MarkerReader(markerId, markerDict, 40, cameraMatrix, distortionCoeffs)
            (found, color_image, marker) = markerReader.detectMarkers(color_image, markerDict)
            # print(marker.tvec)
            # print(marker.rvec)
            # Show images
            cv2.imshow('RealSense Color', color_image)
            # Do other stuff if needed (like in a ROS loop)
            time.sleep(0.1)
            # Break loop on 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
              
                print("Recording")
                pose = np.array(rtde_r.getActualTCPPose())
                print(pose)
                marker_rvec_avg, marker_tvec_avg = get_average_rot_tran(markerReader, markerDict, pipeline, num=30)
                print(marker_rvec_avg, marker_tvec_avg)
                robot_tvec = pose[:3]
                robot_rvec = pose[3:]
                # R, _ = cv2.Rodrigues(pose[3:])
                R_gripper2base.append(np.array(robot_rvec))
                t_gripper2base.append(np.array(robot_tvec))
                
                # t_target2cam.append(np.array(marker.tvec[0])/1000)
                # R_target2cam.append(np.array(marker.rvec[0]))
                t_target2cam.append(marker_tvec_avg/1000)
                R_target2cam.append(marker_rvec_avg)
                    
                print(f"Total points: {len(R_gripper2base)}")

                # if key == 'q':
                #     print("Finish, implement calibration")
                #     break
                
            # Break loop on 'q' key press
            elif key == ord('q'):
                break

            
            
        hand_eye_calibration(np.array(R_gripper2base), 
                             np.array(t_gripper2base), 
                             np.array(R_target2cam), 
                             np.array(t_target2cam))

    finally:
        # Restore the terminal to original state
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        
# if __file__ == '__main__':
main()



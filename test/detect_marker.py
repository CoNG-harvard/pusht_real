import pyrealsense2 as rs
import numpy as np
import cv2

import numpy as np
import os.path as osp

pkg_dir = osp.dirname(osp.dirname(__file__))
import sys
sys.path.append(pkg_dir)
print(pkg_dir)
from utils.marker_util import Marker, MarkerReader, ARUCO_DICT

# cameraMatrix = np.load(osp.join(pkg_dir, "cameraMatrix.npy"))
# distortionCoeffs = np.load(osp.join(pkg_dir, "distortions.npy"))
# print(cameraMatrix)
# print(distortionCoeffs)

# Create pipeline
pipeline = rs.pipeline()

# Configure the pipeline
config = rs.config()
config.enable_device('126122270638')  # Replace with your camera's serial numbers
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)


# Define the red color range in RGB
red_lower = np.array([100, 0, 0])  # Lower bound for red (R > 100, G < 50, B < 50)
red_upper = np.array([255, 50, 50])  # Upper bound for red


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
        markerDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_4X4_50"])
        
        marker = Marker()        
        markerReader = MarkerReader(markerId, 
                                    markerDict, 
                                    40, cameraMatrix, distortionCoeffs)
        (found, color_image, marker) = markerReader.detectMarkers(color_image, markerDict)
        if found:
            print(markerReader.ids)
            print(marker.tvec[0])

        # Show images
        cv2.imshow('RealSense', color_image)
        # cv2.imshow('RealSense Depth', depth_image)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

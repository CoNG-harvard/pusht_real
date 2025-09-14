import pyrealsense2 as rs
import numpy as np
import cv2
import os.path as osp

pkg_dir = osp.dirname(osp.dirname(__file__))
import sys
sys.path.append(pkg_dir)

# Create pipeline
pipeline = rs.pipeline()

# Configure the pipeline
config = rs.config()
# config.enable_device('233722070172')  # Replace with your camera's serial numbers 126122270638 or 233722070172 337322073528
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Create colorizer for depth visualization
colorizer = rs.colorizer()

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

print("Camera Matrix:")
print(cameraMatrix)
print("Distortion Coefficients:")
print(distortionCoeffs)

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
        
        # Reverse image if needed
        # depth_image = depth_image[::-1,::-1]
        # color_image = color_image[::-1,::-1]
        
        # Define the red color range in HSV for better color detection
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # Define range for red color in HSV
        # Red has two ranges in HSV due to the circular nature of hue
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Create colorized depth image
        colorized_depth = colorizer.colorize(depth_frame)
        colorized_depth_image = np.asanyarray(colorized_depth.get_data())
        
        # Apply red mask to the colorized depth image
        # Set non-red areas to black (0, 0, 0)
        masked_depth = colorized_depth_image.copy()
        masked_depth[red_mask == 0] = [0, 0, 0]  # Set non-red pixels to black
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(masked_depth, f'Colorized Depth - Red Objects Only', (10, 30), font, 1, (255, 255, 255), 2)
        
        # Show only the masked colorized depth image
        cv2.imshow('RealSense - Colorized Depth (Red Objects Only)', masked_depth)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

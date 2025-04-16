import pyrealsense2 as rs
import numpy as np
import cv2

# Configure both cameras
pipeline1 = rs.pipeline()
config1 = rs.config()
config1.enable_device('126122270638')  # Replace with your camera's serial number
config1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline2 = rs.pipeline()
config2 = rs.config()
config2.enable_device('233722070172')  # Replace with your camera's serial number
config2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start both pipelines
pipeline1.start(config1)
pipeline2.start(config2)

try:
    while True:
        # Get frames from both cameras
        frames1 = pipeline1.wait_for_frames()
        frames2 = pipeline2.wait_for_frames()
        
        # Get color frames
        color_frame1 = frames1.get_color_frame()
        color_frame2 = frames2.get_color_frame()
        
        if not color_frame1 or not color_frame2:
            continue
            
        # Convert images to numpy arrays
        color_image1 = np.asanyarray(color_frame1.get_data())
        color_image2 = np.asanyarray(color_frame2.get_data())
        
        # Resize images if needed (to make them the same size for concatenation)
        height, width = color_image1.shape[:2]
        color_image2 = cv2.resize(color_image2, (width, height))
        
        # Combine images horizontally
        combined = np.vstack((color_image1, color_image2))
        
        # Show images
        cv2.namedWindow('RealSense Cameras', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense Cameras', combined)
        
        # Break loop with ESC key
        if cv2.waitKey(1) == 27:
            break
            
finally:
    # Stop streaming
    pipeline1.stop()
    pipeline2.stop()
    cv2.destroyAllWindows()
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

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

        # Isolate red pixels (assuming red is dominant in the R channel)
        # You can adjust the threshold to better match your definition of "red"
        red_mask = cv2.inRange(color_image, (0, 0, 150), (100, 100, 255))

        # Apply the mask to the color image to get only red pixels
        red_pixels = cv2.bitwise_and(color_image, color_image, mask=red_mask)

        # Create a depth mask for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Apply the red mask to the depth colormap
        red_depth = cv2.bitwise_and(depth_colormap, depth_colormap, mask=red_mask)

        # Stack images horizontally for visualization
        images = np.hstack((red_pixels, red_depth))

        # Show images
        cv2.imshow('Red Pixels and Depth', images)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
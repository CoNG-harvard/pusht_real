import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

def save_depth_images():
    """
    Capture and save original and masked depth images for offline analysis
    """
    # Create output directory
    output_dir = "saved_depth_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Camera setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    
    # Create colorizer for depth visualization
    colorizer = rs.colorizer()
    
    try:
        profile = pipeline.start(config)
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Camera intrinsic parameters
        cameraMatrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ])
        
        print("Camera Matrix:")
        print(cameraMatrix)
        print("Press 's' to save current frame, 'q' to quit")
        
        frame_count = 0
        
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Create red mask
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            # Create visualizations
            colorized_depth = colorizer.colorize(depth_frame)
            colorized_depth_image = np.asanyarray(colorized_depth.get_data())
            masked_depth = colorized_depth_image.copy()
            masked_depth[red_mask == 0] = [0, 0, 0]
            
            # Show images
            cv2.imshow('Original Color', color_image)
            cv2.imshow('Original Depth (Colorized)', colorized_depth_image)
            cv2.imshow('Masked Depth (Red Objects Only)', masked_depth)
            cv2.imshow('Red Mask', red_mask)
            
            # Add info overlay
            info_image = color_image.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(info_image, f'Frame: {frame_count}', (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(info_image, f'Red pixels: {np.sum(red_mask > 0)}', (10, 60), font, 0.7, (255, 255, 255), 2)
            cv2.putText(info_image, f'Depth range: {depth_image.min()}-{depth_image.max()} mm', (10, 90), font, 0.7, (255, 255, 255), 2)
            cv2.putText(info_image, "Press 's' to save, 'q' to quit", (10, 120), font, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Info', info_image)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                # Save images with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save original images
                cv2.imwrite(f"{output_dir}/color_{timestamp}.png", color_image)
                cv2.imwrite(f"{output_dir}/depth_colorized_{timestamp}.png", colorized_depth_image)
                
                # Save masked images
                cv2.imwrite(f"{output_dir}/masked_depth_{timestamp}.png", masked_depth)
                cv2.imwrite(f"{output_dir}/red_mask_{timestamp}.png", red_mask)
                
                # Save raw depth data (actual depth values in mm)
                np.save(f"{output_dir}/depth_raw_{timestamp}.npy", depth_image)
                
                # Save masked raw depth (only red pixels)
                masked_raw_depth = depth_image.copy()
                masked_raw_depth[red_mask == 0] = 0  # Set non-red pixels to 0
                np.save(f"{output_dir}/masked_depth_raw_{timestamp}.npy", masked_raw_depth)
                
                # Save camera parameters
                np.savetxt(f"{output_dir}/camera_matrix_{timestamp}.txt", cameraMatrix)
                
                # Save additional info
                with open(f"{output_dir}/info_{timestamp}.txt", 'w') as f:
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Image shape: {color_image.shape}\n")
                    f.write(f"Depth range: {depth_image.min()}-{depth_image.max()} mm\n")
                    f.write(f"Red pixels: {np.sum(red_mask > 0)}\n")
                    f.write(f"Camera matrix:\n{cameraMatrix}\n")
                
                print(f"Data saved with timestamp: {timestamp}")
                print(f"  - color_{timestamp}.png")
                print(f"  - depth_colorized_{timestamp}.png")
                print(f"  - masked_depth_{timestamp}.png")
                print(f"  - red_mask_{timestamp}.png")
                print(f"  - depth_raw_{timestamp}.npy (raw depth values)")
                print(f"  - masked_depth_raw_{timestamp}.npy (masked raw depth)")
                print(f"  - camera_matrix_{timestamp}.txt")
                print(f"  - info_{timestamp}.txt")
                
                frame_count += 1
                
            elif key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"Images saved in directory: {output_dir}")

if __name__ == "__main__":
    save_depth_images()

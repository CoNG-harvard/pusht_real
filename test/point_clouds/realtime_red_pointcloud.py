#!/usr/bin/env python3
"""
Real-time Red Depth to Point Cloud Visualization
Converts red objects from depth images to 3D point clouds and visualizes them at 10Hz
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
import threading
from typing import Tuple, Optional

class RealTimeRedPointCloud:
    def __init__(self, target_fps: int = 10):
        """
        Initialize the real-time red point cloud visualizer
        
        Args:
            target_fps: Target update rate in Hz (default: 10)
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_update_time = 0
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        
        # Get camera intrinsics
        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Create camera matrix
        self.camera_matrix = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # Create align object to align depth to color
        self.align = rs.align(rs.stream.color)
        
        # Initialize Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Real-time Red Point Cloud", width=1200, height=800)
        
        # Create point cloud object
        self.pcd = o3d.geometry.PointCloud()
        self.pcd_added = False
        
        # Red color detection parameters (HSV)
        # self.lower_red1 = np.array([0, 50, 50])
        # self.upper_red1 = np.array([10, 255, 255])
        # self.lower_red2 = np.array([170, 50, 50])
        # self.upper_red2 = np.array([180, 255, 255])
        self.lower_red1 = np.array([100, 50, 50])
        self.upper_red1 = np.array([130, 255, 255])
        self.lower_red2 = np.array([100, 50, 50])
        self.upper_red2 = np.array([130, 255, 255])
        # Statistics
        self.frame_count = 0
        self.point_count_history = []
        
        print("Real-time Red Point Cloud Visualizer initialized")
        print(f"Camera Matrix:\n{self.camera_matrix}")
        print(f"Target FPS: {target_fps}")
        
    def create_red_mask(self, color_image: np.ndarray) -> np.ndarray:
        """
        Create a mask for red objects using HSV color space
        
        Args:
            color_image: BGR color image
            
        Returns:
            Binary mask for red pixels
        """
        # Convert to HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv_image, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv_image, self.lower_red2, self.upper_red2)
        
        # Combine masks
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        return red_mask
    
    def depth_to_point_cloud(self, depth_image: np.ndarray, color_image: np.ndarray, 
                           red_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth image to 3D point cloud, filtering for red objects only
        
        Args:
            depth_image: Raw depth image in millimeters
            color_image: BGR color image
            red_mask: Binary mask for red pixels
            
        Returns:
            Tuple of (points, colors) arrays
        """
        height, width = depth_image.shape
        points = []
        colors = []
        
        # Get valid depth values (non-zero) and red pixels
        valid_mask = (depth_image > 0) & (red_mask > 0)
        y_coords, x_coords = np.where(valid_mask)
        
        for y, x in zip(y_coords, x_coords):
            depth = depth_image[y, x]
            
            # Convert pixel coordinates to 3D coordinates
            # Keep depth in millimeters
            z = depth
            x_3d = (x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
            y_3d = (y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
            
            points.append([x_3d, y_3d, z])
            colors.append(color_image[y, x] / 255.0)  # Normalize colors to 0-1
        
        return np.array(points), np.array(colors)
    
    def should_update(self) -> bool:
        """
        Check if enough time has passed for the next update based on target FPS
        
        Returns:
            True if should update, False otherwise
        """
        current_time = time.time()
        if current_time - self.last_update_time >= self.frame_interval:
            self.last_update_time = current_time
            return True
        return False
    
    def update_statistics(self, point_count: int):
        """
        Update and display statistics
        
        Args:
            point_count: Number of points in current frame
        """
        self.point_count_history.append(point_count)
        if len(self.point_count_history) > 100:  # Keep last 100 frames
            self.point_count_history.pop(0)
        
        avg_points = np.mean(self.point_count_history)
        print(f"Frame {self.frame_count}: {point_count} red points (avg: {avg_points:.1f})")
    
    def run(self):
        """
        Main loop for real-time visualization
        """
        print("Starting real-time visualization...")
        print("Press 'q' in the Open3D window to quit")
        
        try:
            while True:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                
                # Align depth to color
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Create red mask
                red_mask = self.create_red_mask(color_image)
                
                # Convert to point cloud
                points, colors = self.depth_to_point_cloud(depth_image, color_image, red_mask)
                
                # Update statistics
                self.update_statistics(len(points))
                
                # Only update visualization at target FPS
                if self.should_update() and len(points) > 0:
                    # Update point cloud
                    self.pcd.points = o3d.utility.Vector3dVector(points)
                    self.pcd.colors = o3d.utility.Vector3dVector(colors)
                    
                    # Add to visualizer if not already added
                    if not self.pcd_added:
                        self.vis.add_geometry(self.pcd)
                        self.pcd_added = True
                    
                    # Update visualization
                    self.vis.update_geometry(self.pcd)
                    self.vis.poll_events()
                    self.vis.update_renderer()
                
                self.frame_count += 1
                
                # Check if visualizer window is closed
                if not self.vis.poll_events():
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping visualization...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Clean up resources
        """
        print("Cleaning up...")
        self.pipeline.stop()
        self.vis.destroy_window()
        print("Cleanup complete")

def main():
    """
    Main function
    """
    print("=== Real-time Red Depth to Point Cloud Visualizer ===")
    print("This script converts red objects from depth images to 3D point clouds")
    print("and visualizes them in real-time at 10Hz")
    print()
    
    # Create and run the visualizer
    visualizer = RealTimeRedPointCloud(target_fps=10)
    visualizer.run()

if __name__ == "__main__":
    main()

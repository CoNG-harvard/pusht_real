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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, Optional

from utils import CAMERA_POSE

def rodrigues_to_matrix(rvec, tvec):
    """Convert Rodrigues rotation vector to 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

class RealTimeBluePointCloud:
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
        
        # NEW: Create camera-to-world matrix.
        # self.fixed_camera_pose = np.array([-0.14163252, 0.769735,  0.67708028, 0.16289344, -2.74628015, 0.2818546 ])
        self.fixed_camera_pose = np.array(CAMERA_POSE)
        self.T_camera_to_world = rodrigues_to_matrix(self.fixed_camera_pose[3:], self.fixed_camera_pose[:3])
        
        # Create align object to align depth to color
        self.align = rs.align(rs.stream.color)
        
        # Initialize Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Real-time Red Point Cloud", width=1200, height=800)
        
        # Create point cloud object
        self.pcd = o3d.geometry.PointCloud()
        self.pcd_added = False
        
        # Initialize 2D matplotlib visualization
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-2, 0)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_title('2D Point Cloud Visualization (Top-Down View)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        self.scatter = None
        
        # Red color detection parameters (HSV)
        self.lower_blue = np.array([100, 50, 50])
        self.upper_blue = np.array([130, 255, 255])
        # Statistics
        self.frame_count = 0
        self.point_count_history = []
        
        print("Real-time Red Point Cloud Visualizer initialized")
        print(f"Camera Matrix:\n{self.camera_matrix}")
        print(f"Target FPS: {target_fps}")
        
    def create_blue_mask(self, color_image: np.ndarray) -> np.ndarray:
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
        blue_mask = cv2.inRange(hsv_image, self.lower_blue, self.upper_blue)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        
        return blue_mask
    
    
    def pixels_to_point_cloud(self, color_image: np.ndarray, 
                           blue_mask: np.ndarray, T_camera_to_world,
                           camera_matrix, z_world=0.0) -> np.ndarray:
        """
        Convert depth image to 3D point cloud, filtering for red objects only
        
        Args:
            depth_image: Raw depth image in millimeters
            color_image: BGR color image
            blue_mask: Binary mask for red pixels
            
        Returns:
            Tuple of (points, colors) arrays
        """
        
        img_obj_only = color_image.copy()
        img_obj_only[blue_mask == 0] = 0

        a, b, c, d = T_camera_to_world[-2]
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        H, W, _ = color_image.shape
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)
        
        num = z_world - d
        denom = a*(u-cx)/fx + b*(v-cy)/fy + c
        Z_cam_img = num / denom
        Y_cam_img = (v-cy)/fy * Z_cam_img
        X_cam_img = (u-cx)/fx * Z_cam_img
        
        coord_cam_img = np.stack([X_cam_img, Y_cam_img, Z_cam_img, np.ones_like(Z_cam_img)], axis=-1)
        coord_world_img = T_camera_to_world[None, None] @ coord_cam_img[..., None]
        coord_world_img = coord_world_img[..., :3, 0]
        
        points_table = coord_world_img[np.any(img_obj_only, axis=-1)]
        points_table[:, 0], points_table[:, 1] = -points_table[:, 1].copy(), points_table[:, 0].copy()
        print(points_table[:5, :])
        return points_table
    
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
    
    def update_2d_visualization(self, points: np.ndarray):
        """
        Update the 2D matplotlib visualization with new points
        
        Args:
            points: 3D points array (N, 3) - will project to X-Y plane
        """
        if len(points) == 0:
            return
            
        # Extract X and Y coordinates for 2D visualization
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Clear previous scatter plot
        if self.scatter is not None:
            self.scatter.remove()
        
        # Create new scatter plot
        self.scatter = self.ax.scatter(x_coords, y_coords, c='blue', s=2, alpha=0.7)
        
        # Update plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
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
        print(f"Frame {self.frame_count}: {point_count} blue points (avg: {avg_points:.1f})")
    
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
                blue_mask = self.create_blue_mask(color_image)
                
                # Convert to point cloud
                points = self.pixels_to_point_cloud(color_image, blue_mask, self.T_camera_to_world, self.camera_matrix, z_world=-0.02)
                
                # Update statistics
                self.update_statistics(len(points))
                
                # Only update visualization at target FPS
                if self.should_update() and len(points) > 0:
                    # Update 3D point cloud
                    self.pcd.points = o3d.utility.Vector3dVector(points)
                    
                    # Add to visualizer if not already added
                    if not self.pcd_added:
                        self.vis.add_geometry(self.pcd)
                        self.pcd_added = True
                    
                    # Update 3D visualization
                    self.vis.update_geometry(self.pcd)
                    self.vis.poll_events()
                    self.vis.update_renderer()
                    
                    # Update 2D visualization
                    self.update_2d_visualization(points)
                    
                    cv2.namedWindow('RealSense Cameras', cv2.WINDOW_NORMAL)
                    cv2.imshow('RealSense Cameras', color_image)
                
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
        plt.close(self.fig)
        print("Cleanup complete")

def main():
    """
    Main function
    """
    print("=== Real-time Blue Depth to Point Cloud Visualizer ===")
    print("This script converts blue objects from depth images to point clouds")
    print("and visualizes them in both 3D (Open3D) and 2D (matplotlib) at 10Hz")
    print("2D visualization shows top-down view (X-Y plane)")
    print()
    
    # Create and run the visualizer
    visualizer = RealTimeBluePointCloud(target_fps=10)
    visualizer.run()

if __name__ == "__main__":
    main()

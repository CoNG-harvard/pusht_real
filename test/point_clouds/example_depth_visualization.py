#!/usr/bin/env python3
"""
Example script demonstrating how to visualize depth images from saved_depth_images folder
"""

import sys
import os.path as osp

# Add the current directory to the path
sys.path.append(osp.dirname(__file__))

from visualize_point_cloud import visualize_depth_images_from_saved, list_available_depth_images, show_images_and_point_cloud

def main():
    """
    Example usage of depth image visualization
    """
    print("=== DEPTH IMAGE VISUALIZATION EXAMPLE ===\n")
    
    # List all available depth image sets
    print("1. Listing available depth image sets:")
    available_files = list_available_depth_images("saved_depth_images")
    
    if not available_files:
        print("No depth images found in saved_depth_images folder!")
        return
    
    print(f"\nFound {len(available_files)} depth image sets")
    
    # Visualize the first image set (now shows both 2D images and 3D point cloud together)
    print("\n2. Visualizing the first depth image set:")
    print("   This will show both 2D images (color, depth, masked, mask) and 3D point cloud simultaneously")
    visualize_depth_images_from_saved("saved_depth_images", image_index=0)
    
    print("\n3. You can also visualize other image sets by changing the image_index:")
    print("   visualize_depth_images_from_saved('saved_depth_images', image_index=1)")
    print("   visualize_depth_images_from_saved('saved_depth_images', image_index=2)")
    print("   etc.")
    print("\n4. The visualization now shows:")
    print("   - 2D images: Original color, depth (colorized), masked depth, and red mask")
    print("   - 3D point cloud: Interactive 3D visualization of the red object")
    print("   - Both are displayed simultaneously for better analysis")

if __name__ == "__main__":
    main()

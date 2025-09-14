#!/usr/bin/env python3
"""
Test script for the real-time red point cloud visualizer
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from realtime_red_pointcloud import RealTimeRedPointCloud

def test_visualizer():
    """
    Test the real-time red point cloud visualizer
    """
    print("Testing Real-time Red Point Cloud Visualizer...")
    print("Make sure you have red objects in front of the camera!")
    print("Press 'q' in the Open3D window to quit")
    print()
    
    try:
        # Create visualizer with 10Hz update rate
        visualizer = RealTimeRedPointCloud(target_fps=10)
        
        # Run the visualization
        visualizer.run()
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Make sure your RealSense camera is connected and working properly.")

if __name__ == "__main__":
    test_visualizer()

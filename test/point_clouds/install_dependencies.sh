#!/bin/bash

# Installation script for Real-time Red Point Cloud Visualizer dependencies

echo "Installing dependencies for Real-time Red Point Cloud Visualizer..."
echo "================================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

# Install required Python packages
echo "Installing Python packages..."

# Install pyrealsense2
echo "Installing pyrealsense2..."
pip3 install pyrealsense2

# Install OpenCV
echo "Installing OpenCV..."
pip3 install opencv-python

# Install Open3D
echo "Installing Open3D..."
pip3 install open3d

# Install NumPy (usually comes with other packages, but just in case)
echo "Installing NumPy..."
pip3 install numpy

echo ""
echo "Installation complete!"
echo ""
echo "To test the installation, run:"
echo "  python3 test_realtime_red_pointcloud.py"
echo ""
echo "To run the main visualizer, run:"
echo "  python3 realtime_red_pointcloud.py"
echo ""
echo "To run the demo with different configurations, run:"
echo "  python3 demo_red_pointcloud.py"
echo ""
echo "Note: Run these commands from the test/point_clouds directory"
echo ""
echo "Make sure your Intel RealSense camera is connected before running!"

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a real-world robotic manipulation project focused on the "Push-T" task using computer vision and robotics control. The system integrates Intel RealSense cameras, ArUco marker detection, Universal Robots (UR) arm control, and reinforcement learning policies to perform block pushing tasks with T-shaped objects.

## Core Architecture

### Main Execution Scripts
- `main_real_marker.py` - Primary execution script using ArUco marker-based tracking
- `main_real_icp.py` - Uses Iterative Closest Point (ICP) for object tracking  
- `main_real_pnp*.py` - Perspective-n-Point (PnP) based tracking variants
- `main_real_marker_config.py` - Configuration-driven marker-based approach

### Key Components

**Environment & Simulation (`envs.py`)**
- `PushTRealEnv` class implementing the core environment interface
- Physics simulation using PyMunk
- Shape definitions for T-blocks: 'tee', 'cee', 'lee', 'eye'
- Coordinate transformations between simulation and real-world spaces

**Tracking & Vision (`utils/`)**
- `marker_util.py` - ArUco marker detection and pose estimation (ARUCO_DICT mapping)
- `tblock_tracker*.py` - Multiple versions of T-block tracking algorithms
- `rot_utils.py` - Rotation matrix utilities and transformations
- `kalman_filter.py` - Kalman filtering for pose smoothing

**Robot Control**
- `utils/robot_control.py` - UR robot control interface using ur_rtde
- TCP (Tool Center Point) configurations for different end-effectors

**Data Processing**
- Point cloud processing and red color detection
- Camera calibration and coordinate transformations
- Real-time depth image processing

## Dependencies & Setup

**Installation:**
```bash
./install_dependencies.sh
```

**Key Dependencies:**
- `pyrealsense2` - Intel RealSense camera interface
- `ur_rtde` - Universal Robots real-time data exchange
- `cv2==4.6.0` - OpenCV for computer vision
- `open3d` - 3D data processing
- `numpy`, `scipy` - Mathematical operations
- `pygame`, `pymunk` - Simulation environment
- `jax` - Policy execution framework

## Configuration System

**Primary Configuration:** `data/real_marker_config.yaml`
- Camera parameters (RealSense D435 settings)
- Robot IP and movement parameters
- ArUco marker configuration (DICT_4X4_50, marker size 38mm)
- Policy paths and model files
- Coordinate transformation parameters

**Additional Configs:**
- `data/cameras.yaml` - Multi-camera setup
- `data/marker_layout.yaml` - Marker positioning
- `data/marker_on_table.yaml` - Table-mounted markers

## Development Workflow

**Testing Structure:**
- `test/` directory contains component-specific tests
- `test/detect_marker*.py` - Marker detection validation
- `test/test_camera.py` - Camera functionality testing
- Point cloud processing tests in `test/point_clouds/`

**Camera Calibration:**
```bash
python tools/calibrate_fixed_camera.py
python tools/calibrate_fixed_camera_initial_guess.py
```

**Tracking Algorithm Development:**
- Multiple tracker versions (`tblock_tracker.py` through `tblock_tracker6.py`)
- Template matching and color-based detection
- ICP registration for 3D pose estimation

## Key Coordinate Systems

- **Simulation Space:** PyMunk physics coordinates
- **Camera Space:** RealSense depth/color coordinates  
- **Robot Space:** UR arm base frame coordinates
- **Marker Space:** ArUco marker reference frames

**Transformation Chain:**
Camera → Marker → Robot Base → TCP (Tool Center Point)

## Policy Integration

**Trained Models:** Located in `/home/mht/PycharmProjects/DACER-Diffusion-with-Online-RL/logs/`
- SAC-based policies for manipulation tasks
- Deterministic execution modes
- Multi-task curriculum learning support

**Action Space:** 2D push directions with speed control
**Observation Space:** Object pose, robot state, visual features

## Hardware Integration

**Intel RealSense D435:** 1280x720 @ 30fps (color + depth)
**Universal Robots:** TCP control via Ethernet (192.168.0.191)
**ArUco Markers:** 38mm DICT_4X4_50 markers for pose reference

## Common Debugging Areas

- Camera-robot calibration accuracy (`fixed_camera_pose` parameters)
- Marker detection reliability under varying lighting
- Coordinate transformation precision between spaces
- Real-time performance optimization for control loop
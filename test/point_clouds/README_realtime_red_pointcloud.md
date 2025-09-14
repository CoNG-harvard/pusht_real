# Real-time Red Depth to Point Cloud Visualizer

This script converts red objects from depth images to 3D point clouds and visualizes them in real-time at 10Hz using Intel RealSense cameras and Open3D.

## Features

- **Real-time Processing**: Captures depth and color data from RealSense camera
- **Red Object Detection**: Uses HSV color space for robust red color filtering
- **3D Point Cloud Generation**: Converts depth data to 3D coordinates using camera intrinsics
- **10Hz Visualization**: Updates point cloud visualization at 10Hz for smooth real-time viewing
- **Statistics Tracking**: Monitors point count and performance metrics
- **Robust Error Handling**: Proper cleanup and error handling

## Requirements

- Intel RealSense camera (D405, D415, D435, etc.)
- Python 3.7+
- Required packages:
  ```bash
  pip install pyrealsense2 opencv-python open3d numpy
  ```

## Usage

### Basic Usage

```bash
cd test/point_clouds
python realtime_red_pointcloud.py
```

### Test Script

```bash
cd test/point_clouds
python test_realtime_red_pointcloud.py
```

## How It Works

1. **Camera Setup**: Initializes RealSense pipeline with color (1280x720) and depth (1280x720) streams
2. **Frame Alignment**: Aligns depth frames to color frames for accurate color-depth correspondence
3. **Red Detection**: Uses HSV color space to create masks for red objects:
   - Lower red range: HSV(0,50,50) to HSV(10,255,255)
   - Upper red range: HSV(170,50,50) to HSV(180,255,255)
4. **Point Cloud Generation**: Converts depth pixels to 3D coordinates using camera intrinsics
5. **Real-time Visualization**: Updates Open3D point cloud at 10Hz with color information

## Controls

- **Mouse**: Rotate view in Open3D window
- **Mouse Wheel**: Zoom in/out
- **'q' Key**: Quit the application
- **Ctrl+C**: Force quit from terminal

## Configuration

You can modify the following parameters in the script:

- `target_fps`: Update rate (default: 10Hz)
- `lower_red1/upper_red1`: First red color range in HSV
- `lower_red2/upper_red2`: Second red color range in HSV
- Stream resolution and framerate

## Camera Intrinsics

The script automatically extracts camera intrinsics from the RealSense device:
- Focal length (fx, fy)
- Principal point (ppx, ppy)
- Distortion coefficients

## Performance

- **Update Rate**: 10Hz (configurable)
- **Resolution**: 1280x720 for both color and depth
- **Point Cloud**: Only red pixels are converted to 3D points
- **Memory**: Efficient processing with frame-by-frame updates

## Troubleshooting

### Common Issues

1. **No camera detected**: Ensure RealSense camera is connected and drivers are installed
2. **No red points detected**: Adjust HSV color ranges or check lighting conditions
3. **Poor performance**: Reduce resolution or increase update interval
4. **Open3D window not responding**: Check if Open3D is properly installed

### Debug Information

The script provides real-time statistics:
- Frame count
- Number of red points detected
- Average point count over last 100 frames

## Example Output

```
Real-time Red Point Cloud Visualizer initialized
Camera Matrix:
[[ 920.123   0.     640.456]
 [   0.     920.123  360.789]
 [   0.       0.       1.   ]]
Target FPS: 10
Starting real-time visualization...
Frame 1: 1250 red points (avg: 1250.0)
Frame 2: 1180 red points (avg: 1215.0)
Frame 3: 1320 red points (avg: 1250.0)
...
```

## Integration

This script can be integrated into larger robotics applications by:

1. **Modifying the point cloud processing**: Add filtering, clustering, or object detection
2. **Adding coordinate transformations**: Transform to robot base frame
3. **Implementing callbacks**: Add custom processing functions
4. **Saving data**: Export point clouds for offline analysis

## License

This code is part of the pusht_real project and follows the same licensing terms.

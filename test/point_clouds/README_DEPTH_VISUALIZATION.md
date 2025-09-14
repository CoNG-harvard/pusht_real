# Depth Image Visualization

This document describes how to visualize depth images from the `saved_depth_images` folder using the enhanced `visualize_point_cloud.py` script.

## Overview

The `visualize_point_cloud.py` script has been enhanced to support visualization of depth images stored in the `saved_depth_images` folder. It converts depth images to 3D point clouds and visualizes them using Open3D.

## Features

- **Load depth images**: Automatically loads color, depth, mask, and camera matrix data
- **Point cloud conversion**: Converts depth data to 3D point clouds using camera intrinsics
- **Combined visualization**: Shows both 2D images and 3D point cloud simultaneously
- **Interactive 3D visualization**: Interactive 3D visualization with Open3D
- **Multiple image sets**: Support for visualizing different saved image sets
- **Command-line interface**: Easy-to-use command-line options

## Usage

### Command Line Interface

```bash
# List all available depth image sets
python test/point_clouds/visualize_point_cloud.py --mode depth --list

# Visualize the first depth image set (index 0)
python test/point_clouds/visualize_point_cloud.py --mode depth --image_index 0

# Visualize a specific depth image set
python test/point_clouds/visualize_point_cloud.py --mode depth --image_index 2

# Use a different data directory
python test/point_clouds/visualize_point_cloud.py --mode depth --data_dir /path/to/your/depth/images
```

### Programmatic Usage

```python
from visualize_point_cloud import visualize_depth_images_from_saved, list_available_depth_images

# List available image sets
available_files = list_available_depth_images("saved_depth_images")

# Visualize a specific image set
visualize_depth_images_from_saved("saved_depth_images", image_index=0)
```

## Data Format

The script expects the following files in the `saved_depth_images` directory:

- `color_*.png` - Color images
- `depth_colorized_*.png` - Colorized depth images
- `masked_depth_*.png` - Masked depth images
- `red_mask_*.png` - Red object masks
- `depth_raw_*.npy` - Raw depth data (in mm)
- `masked_depth_raw_*.npy` - Masked raw depth data
- `camera_matrix_*.txt` - Camera intrinsic matrix

## Visualization Controls

The script now displays both 2D images and 3D point cloud simultaneously:

### 2D Images (OpenCV windows):
- **Press any key**: Close all 2D image windows
- Four windows are displayed: Original Color, Depth (Colorized), Masked Depth, and Red Mask

### 3D Point Cloud (Open3D window):
- **Mouse**: Rotate the view
- **Mouse wheel**: Zoom in/out
- **Right mouse + drag**: Pan the view
- **Press 'R'**: Reset view to default
- **Press 'Q' or close window**: Exit 3D visualization

### Workflow:
1. Both 2D images and 3D point cloud open simultaneously
2. You can interact with the 3D point cloud while viewing the 2D images
3. Press any key to close the 2D images (3D visualization continues)
4. Close the 3D window when done viewing

## Output Information

The script provides detailed information about:

- **Image statistics**: Dimensions, number of red pixels, object count
- **Depth analysis**: Min/max/mean depth values for the entire image and red objects
- **Camera parameters**: Focal length and principal point
- **Point cloud statistics**: Number of points, bounding box, dimensions
- **3D coordinates**: Point cloud bounds in millimeters

## Example Output

```
=== VISUALIZING DEPTH IMAGES FROM saved_depth_images ===
Found 5 saved image sets
Visualizing image set: 20250905_180634

=== DEPTH DATA ANALYSIS ===
Red pixels detected: 40911 (4.44% of image)
Number of red objects: 4
Largest red object area: 40365 pixels

Raw depth analysis (actual depth values in mm):
  Raw depth min value: 0 mm
  Raw depth max value: 29433 mm
  Red pixels depth min: 0 mm
  Red pixels depth max: 789 mm
  Red pixels depth mean: 666.3 mm

=== POINT CLOUD CONVERSION ===
Converting 40911 red pixels to 3D points...
Generated 40338 3D points
Point cloud bounds:
  X: [111.3, 359.0] mm
  Y: [-7.1, 207.1] mm
  Z: [588.0, 789.0] mm
```

## Dependencies

- `open3d` - For 3D visualization
- `numpy` - For numerical operations
- `cv2` (OpenCV) - For image processing
- `glob` - For file pattern matching

## Integration with load_depth_images.py

The visualization functionality integrates seamlessly with the existing `load_depth_images.py` module:

- Uses the same data loading functions
- Leverages the existing point cloud conversion logic
- Maintains consistency with the existing codebase

## Troubleshooting

1. **No depth images found**: Ensure the `saved_depth_images` directory exists and contains the required files
2. **Import errors**: Make sure you're running in the correct conda environment (`pusht_real`)
3. **Visualization window doesn't open**: Check that Open3D is properly installed and your display is configured correctly
4. **Empty point cloud**: Verify that the red mask contains valid pixels and the depth data is not all zeros

## Related Files

- `visualize_point_cloud.py` - Main visualization script
- `load_depth_images.py` - Depth image loading utilities
- `example_depth_visualization.py` - Usage example
- `saved_depth_images/` - Directory containing saved depth image data

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os.path as osp
from scipy.spatial.distance import pdist

pkg_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
import sys
sys.path.append(pkg_dir)

# Load the reference point cloud
point_cloud_path = osp.join(osp.dirname(__file__), "t_pcd.ply")
reference_pcd = o3d.io.read_point_cloud(point_cloud_path)

if len(reference_pcd.points) == 0:
    print("Error: Could not load reference point cloud")
    exit(1)

def analyze_point_cloud_units(pcd, expected_size_mm=200):
    """
    Analyze the point cloud to determine its units
    """
    print("=== POINT CLOUD ANALYSIS ===")
    points = np.asarray(pcd.points)
    
    # Calculate bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    dimensions = max_bound - min_bound
    
    print(f"Bounding box min: [{min_bound[0]:.6f}, {min_bound[1]:.6f}, {min_bound[2]:.6f}]")
    print(f"Bounding box max: [{max_bound[0]:.6f}, {max_bound[1]:.6f}, {max_bound[2]:.6f}]")
    print(f"Dimensions: [{dimensions[0]:.6f}, {dimensions[1]:.6f}, {dimensions[2]:.6f}]")
    
    # Calculate the largest dimension
    max_dimension = np.max(dimensions)
    print(f"Largest dimension: {max_dimension:.6f}")
    
    # Estimate units
    if max_dimension > 0.1:  # Likely in meters
        estimated_units = "meters"
        scale_factor = expected_size_mm / 1000.0  # Convert mm to meters
    elif max_dimension > 0.01:  # Likely in centimeters
        estimated_units = "centimeters"
        scale_factor = expected_size_mm / 10.0  # Convert mm to cm
    else:  # Likely in millimeters
        estimated_units = "millimeters"
        scale_factor = expected_size_mm
    
    print(f"Estimated units: {estimated_units}")
    print(f"Expected size: {expected_size_mm}mm")
    print(f"Actual size: {max_dimension * scale_factor:.1f}mm")
    
    # Calculate scale factor to convert to meters
    scale_to_meters = expected_size_mm / (max_dimension * 1000.0)
    print(f"Scale factor to convert to meters: {scale_to_meters:.6f}")
    
    return scale_to_meters, estimated_units

def analyze_camera_units():
    """
    Analyze camera depth data to understand units
    """
    print("\n=== CAMERA ANALYSIS ===")
    
    # Camera setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    
    try:
        profile = pipeline.start(config)
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        
        print(f"Camera intrinsics:")
        print(f"  fx: {intr.fx:.2f}, fy: {intr.fy:.2f}")
        print(f"  ppx: {intr.ppx:.2f}, ppy: {intr.ppy:.2f}")
        print(f"  Width: {intr.width}, Height: {intr.height}")
        
        # Get a few frames to analyze depth values
        print("\nAnalyzing depth values...")
        depth_values = []
        
        for i in range(10):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                valid_depths = depth_image[depth_image > 0]
                if len(valid_depths) > 0:
                    depth_values.extend(valid_depths)
        
        if depth_values:
            depth_values = np.array(depth_values)
            print(f"Depth value statistics:")
            print(f"  Min: {np.min(depth_values)}")
            print(f"  Max: {np.max(depth_values)}")
            print(f"  Mean: {np.mean(depth_values):.1f}")
            print(f"  Median: {np.median(depth_values):.1f}")
            
            # RealSense depth is typically in millimeters
            print(f"  Units: millimeters (RealSense default)")
            print(f"  Typical range: 200-2000mm (20cm-2m)")
            
            # Check if values are in expected range
            if np.mean(depth_values) > 1000:
                print("  ✓ Values appear to be in millimeters")
            else:
                print("  ⚠ Values might be in different units")
        
    except Exception as e:
        print(f"Error analyzing camera: {e}")
    finally:
        pipeline.stop()

def test_point_cloud_scaling(pcd, scale_factor):
    """
    Test scaling the point cloud and show results
    """
    print(f"\n=== SCALING TEST ===")
    
    # Create a scaled version
    scaled_pcd = o3d.geometry.PointCloud(pcd)
    scaled_pcd.scale(scale_factor, center=scaled_pcd.get_center())
    
    # Analyze scaled point cloud
    points = np.asarray(scaled_pcd.points)
    bbox = scaled_pcd.get_axis_aligned_bounding_box()
    dimensions = bbox.max_bound - bbox.min_bound
    max_dimension = np.max(dimensions)
    
    print(f"After scaling by {scale_factor:.6f}:")
    print(f"  Dimensions: [{dimensions[0]:.6f}, {dimensions[1]:.6f}, {dimensions[2]:.6f}]")
    print(f"  Largest dimension: {max_dimension:.6f} meters")
    print(f"  Largest dimension: {max_dimension * 1000:.1f} mm")
    
    return scaled_pcd

def visualize_scale_comparison(original_pcd, scaled_pcd):
    """
    Visualize both original and scaled point clouds
    """
    print("\n=== VISUALIZATION ===")
    print("Opening visualization window...")
    print("Red: Original point cloud")
    print("Green: Scaled point cloud")
    print("Close window to continue...")
    
    # Color the point clouds
    original_pcd.paint_uniform_color([1, 0, 0])  # Red
    scaled_pcd.paint_uniform_color([0, 1, 0])    # Green
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    o3d.visualization.draw_geometries([original_pcd, scaled_pcd, coordinate_frame],
                                     window_name="Unit Comparison - Red: Original, Green: Scaled")

def main():
    print("Point Cloud and Camera Unit Analysis")
    print("=" * 50)
    
    # Analyze point cloud units
    scale_factor, units = analyze_point_cloud_units(reference_pcd, expected_size_mm=200)
    
    # Analyze camera units
    analyze_camera_units()
    
    # Test scaling
    scaled_pcd = test_point_cloud_scaling(reference_pcd, scale_factor)
    
    # Ask user if they want to visualize
    print(f"\n=== RECOMMENDATIONS ===")
    print(f"1. Point cloud appears to be in {units}")
    print(f"2. Camera depth is in millimeters")
    print(f"3. Scale factor to convert point cloud to meters: {scale_factor:.6f}")
    print(f"4. This should make the point cloud compatible with camera data")
    
    # Visualize comparison
    try:
        visualize_scale_comparison(reference_pcd, scaled_pcd)
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print(f"\n=== USAGE IN POSE DETECTION ===")
    print(f"To use the scaled point cloud in pose detection:")
    print(f"1. Apply scale factor {scale_factor:.6f} to the reference point cloud")
    print(f"2. This will ensure units match between camera (mm) and point cloud (m)")

if __name__ == "__main__":
    main()

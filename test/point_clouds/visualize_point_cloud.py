import open3d as o3d
import numpy as np
import os.path as osp
import cv2
import glob
import sys

# Add the parent directory to the path to import load_depth_images
sys.path.append(osp.dirname(osp.dirname(__file__)))
from load_depth_images import load_depth_images, convert_depth_to_point_cloud, analyze_depth_data

pkg_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
point_cloud_path = osp.join(osp.dirname(__file__), "t_pcd.ply")

def visualize_point_cloud():
    """
    Visualize the 3D point cloud from the PLY file
    """
    print(f"Loading point cloud from: {point_cloud_path}")
    
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    
    if len(pcd.points) == 0:
        print("Error: Could not load point cloud or point cloud is empty")
        return
    
    print(f"Point cloud loaded successfully!")
    print(f"Number of points: {len(pcd.points)}")
    print(f"Has colors: {pcd.has_colors()}")
    print(f"Has normals: {pcd.has_normals()}")
    
    # Get point cloud bounds
    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"Bounding box min: {bbox.min_bound}")
    print(f"Bounding box max: {bbox.max_bound}")
    print(f"Bounding box size: {bbox.max_bound - bbox.min_bound}")
    
    # If the point cloud doesn't have colors, add some color based on height (Z coordinate)
    if not pcd.has_colors():
        points = np.asarray(pcd.points)
        z_coords = points[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        
        # Normalize Z coordinates to 0-1 range
        z_normalized = (z_coords - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(z_coords)
        
        # Create colors based on height (blue to red gradient)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = z_normalized  # Red channel
        colors[:, 2] = 1 - z_normalized  # Blue channel
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("Added height-based coloring to point cloud")
    
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Visualize the point cloud
    print("\nVisualization controls:")
    print("- Mouse: Rotate view")
    print("- Mouse wheel: Zoom in/out")
    print("- Right mouse + drag: Pan")
    print("- Press 'R' to reset view")
    print("- Press 'Q' or close window to exit")
    
    o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                     window_name="3D Point Cloud Visualization",
                                     width=1200, height=800)

def analyze_point_cloud():
    """
    Analyze the point cloud and print statistics
    """
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    
    if len(pcd.points) == 0:
        print("Error: Could not load point cloud")
        return
    
    points = np.asarray(pcd.points)
    
    print("\n=== Point Cloud Analysis ===")
    print(f"Total points: {len(points)}")
    print(f"X range: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}]")
    print(f"Y range: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")
    print(f"Z range: [{points[:, 2].min():.4f}, {points[:, 2].max():.4f}]")
    
    # Calculate center
    center = np.mean(points, axis=0)
    print(f"Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    
    # Calculate dimensions
    dimensions = points.max(axis=0) - points.min(axis=0)
    print(f"Dimensions: [{dimensions[0]:.4f}, {dimensions[1]:.4f}, {dimensions[2]:.4f}]")

def visualize_depth_images_from_saved(data_dir="saved_depth_images", image_index=0):
    """
    Load and visualize depth images from the saved_depth_images folder
    """
    print(f"\n=== VISUALIZING DEPTH IMAGES FROM {data_dir} ===")
    
    # Check if directory exists
    if not osp.exists(data_dir):
        print(f"Error: Directory {data_dir} not found!")
        return
    
    # Find all saved image sets
    color_files = glob.glob(f"{data_dir}/color_*.png")
    if not color_files:
        print(f"No saved images found in {data_dir}!")
        return
    
    color_files.sort()
    print(f"Found {len(color_files)} saved image sets")
    
    # Select the image set to visualize
    if image_index >= len(color_files):
        print(f"Error: Image index {image_index} out of range. Available indices: 0-{len(color_files)-1}")
        return
    
    selected_file = color_files[image_index]
    timestamp = osp.basename(selected_file).replace("color_", "").replace(".png", "")
    print(f"Visualizing image set: {timestamp}")
    
    # Load the depth images using a modified version that doesn't show images automatically
    data = load_depth_images_without_display(data_dir)
    if data is None:
        print("Error: Could not load depth images")
        return
    
    # Convert depth to point cloud
    points, colors = convert_depth_to_point_cloud(
        data['raw_depth'], 
        data['mask'], 
        data['camera_matrix'],
        data['color']
    )
    
    if len(points) == 0:
        print("Error: No points generated from depth image")
        return
    
    print(f"Generated {len(points)} 3D points from depth image")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Get point cloud bounds
    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"Point cloud bounds:")
    print(f"  Min: {bbox.min_bound}")
    print(f"  Max: {bbox.max_bound}")
    print(f"  Size: {bbox.max_bound - bbox.min_bound}")
    
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Show both 2D images and 3D point cloud together
    show_images_and_point_cloud(
        data['color'], 
        data['depth'], 
        data['masked'], 
        data['mask'], 
        pcd, 
        coordinate_frame, 
        timestamp
    )

def list_available_depth_images(data_dir="saved_depth_images"):
    """
    List all available depth image sets
    """
    if not osp.exists(data_dir):
        print(f"Directory {data_dir} not found!")
        return []
    
    color_files = glob.glob(f"{data_dir}/color_*.png")
    color_files.sort()
    
    print(f"\nAvailable depth image sets in {data_dir}:")
    for i, file in enumerate(color_files):
        timestamp = osp.basename(file).replace("color_", "").replace(".png", "")
        print(f"  {i}. {timestamp}")
    
    return color_files

def load_depth_images_without_display(data_dir="saved_depth_images"):
    """
    Load depth images without automatically displaying them
    """
    if not osp.exists(data_dir):
        print(f"Directory {data_dir} not found!")
        return None
    
    # Find all saved image sets
    color_files = glob.glob(f"{data_dir}/color_*.png")
    depth_files = glob.glob(f"{data_dir}/depth_colorized_*.png")
    masked_files = glob.glob(f"{data_dir}/masked_depth_*.png")
    mask_files = glob.glob(f"{data_dir}/red_mask_*.png")
    raw_depth_files = glob.glob(f"{data_dir}/depth_raw_*.npy")
    masked_raw_files = glob.glob(f"{data_dir}/masked_depth_raw_*.npy")
    matrix_files = glob.glob(f"{data_dir}/camera_matrix_*.txt")
    
    if not color_files:
        print("No saved images found!")
        return None
    
    # Sort by timestamp
    color_files.sort()
    depth_files.sort()
    masked_files.sort()
    mask_files.sort()
    raw_depth_files.sort()
    masked_raw_files.sort()
    matrix_files.sort()
    
    # Load the first set as example
    print(f"Loading image set: {osp.basename(color_files[0])}")
    
    # Load images
    color_img = cv2.imread(color_files[0])
    depth_img = cv2.imread(depth_files[0])
    masked_img = cv2.imread(masked_files[0])
    mask_img = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
    
    # Load raw depth data
    raw_depth = np.load(raw_depth_files[0])
    masked_raw_depth = np.load(masked_raw_files[0])
    
    # Load camera matrix
    camera_matrix = np.loadtxt(matrix_files[0])
    
    print(f"Color image shape: {color_img.shape}")
    print(f"Depth image shape: {depth_img.shape}")
    print(f"Masked image shape: {masked_img.shape}")
    print(f"Mask image shape: {mask_img.shape}")
    print(f"Raw depth shape: {raw_depth.shape}")
    print(f"Masked raw depth shape: {masked_raw_depth.shape}")
    print(f"Camera matrix shape: {camera_matrix.shape}")
    
    # Analyze the images (but don't show them)
    analyze_depth_data(color_img, depth_img, masked_img, mask_img, camera_matrix, raw_depth)
    
    return {
        'color': color_img,
        'depth': depth_img,
        'masked': masked_img,
        'mask': mask_img,
        'raw_depth': raw_depth,
        'masked_raw_depth': masked_raw_depth,
        'camera_matrix': camera_matrix
    }

def show_images_and_point_cloud(color_img, depth_img, masked_img, mask_img, pcd, coordinate_frame, timestamp):
    """
    Show both 2D images and 3D point cloud together
    """
    print("\nDisplaying 2D images and 3D point cloud together...")
    print("2D Images: Press any key to close image windows")
    print("3D Point Cloud: Use mouse controls, press 'Q' to close")
    
    # Resize images for better display
    display_size = (640, 360)
    
    color_display = cv2.resize(color_img, display_size)
    depth_display = cv2.resize(depth_img, display_size)
    masked_display = cv2.resize(masked_img, display_size)
    mask_display = cv2.resize(mask_img, display_size)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(color_display, "Original Color", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(depth_display, "Depth (Colorized)", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(masked_display, "Masked Depth", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(mask_display, "Red Mask", (10, 30), font, 0.7, (255, 255, 255), 2)
    
    # Show 2D images
    cv2.imshow("Original Color", color_display)
    cv2.imshow("Depth (Colorized)", depth_display)
    cv2.imshow("Masked Depth", masked_display)
    cv2.imshow("Red Mask", mask_display)
    
    # Start 3D visualization in a separate thread or process
    import threading
    
    def show_3d():
        print("\n3D Visualization controls:")
        print("- Mouse: Rotate view")
        print("- Mouse wheel: Zoom in/out")
        print("- Right mouse + drag: Pan")
        print("- Press 'R' to reset view")
        print("- Press 'Q' or close window to exit")
        
        o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                         window_name=f"Depth Image Point Cloud - {timestamp}",
                                         width=1200, height=800)
    
    # Start 3D visualization in a separate thread
    thread_3d = threading.Thread(target=show_3d)
    thread_3d.daemon = True
    thread_3d.start()
    
    # Wait for user to close 2D images
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Wait a bit for 3D visualization to start
    import time
    time.sleep(1)
    
    print("2D images closed. 3D visualization should be running in a separate window.")
    print("Close the 3D window when done viewing.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize point clouds")
    parser.add_argument("--mode", choices=["ply", "depth"], default="depth", 
                       help="Visualization mode: 'ply' for PLY file, 'depth' for depth images")
    parser.add_argument("--data_dir", default="saved_depth_images", 
                       help="Directory containing depth images (for depth mode)")
    parser.add_argument("--image_index", type=int, default=0, 
                       help="Index of the depth image set to visualize (for depth mode)")
    parser.add_argument("--list", action="store_true", 
                       help="List available depth image sets and exit")
    
    args = parser.parse_args()
    
    if args.mode == "ply":
        # Check if the point cloud file exists
        if not osp.exists(point_cloud_path):
            print(f"Error: Point cloud file not found at {point_cloud_path}")
            exit(1)
        
        # Analyze the point cloud first
        analyze_point_cloud()
        
        # Visualize the point cloud
        visualize_point_cloud()
    
    elif args.mode == "depth":
        if args.list:
            # List available depth images
            list_available_depth_images(args.data_dir)
        else:
            # List available images first
            available_files = list_available_depth_images(args.data_dir)
            
            if available_files:
                # Visualize the selected depth image
                visualize_depth_images_from_saved(args.data_dir, args.image_index)
            else:
                print("No depth images found to visualize")

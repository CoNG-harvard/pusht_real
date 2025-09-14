import numpy as np
import cv2
import os
import glob
from datetime import datetime

def load_depth_images(data_dir="saved_depth_images"):
    """
    Load and analyze saved depth images and raw depth data
    """
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found!")
        return
    
    # Find all saved image sets
    color_files = glob.glob(f"{data_dir}/color_*.png")
    depth_files = glob.glob(f"{data_dir}/depth_colorized_*.png")
    masked_files = glob.glob(f"{data_dir}/masked_depth_*.png")
    mask_files = glob.glob(f"{data_dir}/red_mask_*.png")
    raw_depth_files = glob.glob(f"{data_dir}/depth_raw_*.npy")
    masked_raw_files = glob.glob(f"{data_dir}/masked_depth_raw_*.npy")
    matrix_files = glob.glob(f"{data_dir}/camera_matrix_*.txt")
    
    print(f"Found {len(color_files)} image sets in {data_dir}")
    
    if not color_files:
        print("No saved images found!")
        return
    
    # Sort by timestamp
    color_files.sort()
    depth_files.sort()
    masked_files.sort()
    mask_files.sort()
    raw_depth_files.sort()
    masked_raw_files.sort()
    matrix_files.sort()
    
    # Load the first set as example
    print(f"\nLoading first image set: {os.path.basename(color_files[0])}")
    
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
    
    # Analyze the images
    analyze_depth_data(color_img, depth_img, masked_img, mask_img, camera_matrix, raw_depth)
    
    # Show images
    show_images(color_img, depth_img, masked_img, mask_img)
    
    return {
        'color': color_img,
        'depth': depth_img,
        'masked': masked_img,
        'mask': mask_img,
        'raw_depth': raw_depth,
        'masked_raw_depth': masked_raw_depth,
        'camera_matrix': camera_matrix
    }

def analyze_depth_data(color_img, depth_img, masked_img, mask_img, camera_matrix, raw_depth=None):
    """
    Analyze the depth data and provide statistics
    """
    print("\n=== DEPTH DATA ANALYSIS ===")
    
    # Analyze mask
    red_pixels = np.sum(mask_img > 0)
    total_pixels = mask_img.shape[0] * mask_img.shape[1]
    red_percentage = (red_pixels / total_pixels) * 100
    
    print(f"Red pixels detected: {red_pixels} ({red_percentage:.2f}% of image)")
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of red objects: {len(contours)}")
    
    if len(contours) > 0:
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        print(f"Largest red object area: {area:.0f} pixels")
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
        
        # Calculate aspect ratio
        aspect_ratio = w / h
        print(f"Aspect ratio: {aspect_ratio:.2f}")
    
    # Analyze depth distribution
    if raw_depth is not None:
        print(f"\nRaw depth analysis (actual depth values in mm):")
        print(f"  Raw depth min value: {raw_depth.min()} mm")
        print(f"  Raw depth max value: {raw_depth.max()} mm")
        print(f"  Raw depth mean value: {raw_depth.mean():.1f} mm")
        
        # Analyze depth of red pixels only
        red_depth_values = raw_depth[mask_img > 0]
        if len(red_depth_values) > 0:
            print(f"  Red pixels depth min: {red_depth_values.min()} mm")
            print(f"  Red pixels depth max: {red_depth_values.max()} mm")
            print(f"  Red pixels depth mean: {red_depth_values.mean():.1f} mm")
    else:
        # Fallback to colorized depth analysis
        gray_depth = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
        print(f"\nDepth analysis (colorized):")
        print(f"  Depth image min value: {gray_depth.min()}")
        print(f"  Depth image max value: {gray_depth.max()}")
        print(f"  Depth image mean value: {gray_depth.mean():.1f}")
    
    # Camera parameters
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    print(f"\nCamera parameters:")
    print(f"  Focal length: fx={fx:.1f}, fy={fy:.1f}")
    print(f"  Principal point: cx={cx:.1f}, cy={cy:.1f}")

def show_images(color_img, depth_img, masked_img, mask_img):
    """
    Display the loaded images
    """
    print("\nDisplaying images...")
    print("Press any key to close windows")
    
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
    
    # Show images
    cv2.imshow("Original Color", color_display)
    cv2.imshow("Depth (Colorized)", depth_display)
    cv2.imshow("Masked Depth", masked_display)
    cv2.imshow("Red Mask", mask_display)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convert_depth_to_point_cloud(raw_depth, mask_img, camera_matrix, color_img=None):
    """
    Convert raw depth data to 3D point cloud
    """
    print("\n=== POINT CLOUD CONVERSION ===")
    
    height, width = raw_depth.shape
    points = []
    colors = []
    
    # Get red pixels
    red_pixels = np.where(mask_img > 0)
    
    print(f"Converting {len(red_pixels[0])} red pixels to 3D points...")
    
    for y, x in zip(red_pixels[0], red_pixels[1]):
        # Get actual depth value from raw depth data
        depth_value = raw_depth[y, x]
        
        if depth_value > 0:  # Only process valid depth values
            # Convert pixel to 3D coordinates
            z = depth_value  # mm
            x_3d = (x - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
            y_3d = (y - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
            
            points.append([x_3d, y_3d, z])
            
            # Get color if available
            if color_img is not None:
                colors.append(color_img[y, x] / 255.0)
            else:
                colors.append([1.0, 0.0, 0.0])  # Default red color
    
    points = np.array(points)
    colors = np.array(colors)
    
    print(f"Generated {len(points)} 3D points")
    if len(points) > 0:
        print(f"Point cloud bounds:")
        print(f"  X: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}] mm")
        print(f"  Y: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}] mm")
        print(f"  Z: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}] mm")
        
        # Calculate bounding box
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_center = (bbox_min + bbox_max) / 2
        
        print(f"Bounding box center: [{bbox_center[0]:.1f}, {bbox_center[1]:.1f}, {bbox_center[2]:.1f}] mm")
        print(f"Bounding box size: [{bbox_size[0]:.1f}, {bbox_size[1]:.1f}, {bbox_size[2]:.1f}] mm")
    
    return points, colors

def list_all_saved_sets(data_dir="saved_depth_images"):
    """
    List all saved image sets
    """
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found!")
        return
    
    color_files = glob.glob(f"{data_dir}/color_*.png")
    color_files.sort()
    
    print(f"Found {len(color_files)} saved image sets:")
    for i, file in enumerate(color_files):
        timestamp = os.path.basename(file).replace("color_", "").replace(".png", "")
        print(f"  {i+1}. {timestamp}")

def main():
    """
    Main function to load and analyze depth images
    """
    print("=== DEPTH IMAGE LOADER ===")
    
    # List all available sets
    list_all_saved_sets()
    
    # Load the first set
    data = load_depth_images()
    
    if data is not None:
        # Convert to point cloud using raw depth data
        points, colors = convert_depth_to_point_cloud(
            data['raw_depth'], 
            data['mask'], 
            data['camera_matrix'],
            data['color']
        )
        
        print(f"\nPoint cloud conversion completed!")
        print(f"Use these points for offline analysis with your reference model.")
        
        # Save point cloud for further analysis
        if len(points) > 0:
            np.save("camera_point_cloud.npy", points)
            np.save("camera_point_colors.npy", colors)
            print(f"Point cloud saved as 'camera_point_cloud.npy' and 'camera_point_colors.npy'")

if __name__ == "__main__":
    main()

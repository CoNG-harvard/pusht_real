import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os.path as osp
import time
from scipy.spatial.transform import Rotation as R

pkg_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
import sys
sys.path.append(pkg_dir)

# Load the reference point cloud
point_cloud_path = osp.join(osp.dirname(__file__), "t_pcd.ply")
reference_pcd = o3d.io.read_point_cloud(point_cloud_path)

if len(reference_pcd.points) == 0:
    print("Error: Could not load reference point cloud")
    exit(1)

print(f"Reference point cloud loaded: {len(reference_pcd.points)} points")

# Check point cloud bounds and scale to match camera units (millimeters)
print(f"Original point cloud bounds:")
bbox_orig = reference_pcd.get_axis_aligned_bounding_box()
print(f"  Min: {bbox_orig.min_bound}")
print(f"  Max: {bbox_orig.max_bound}")
print(f"  Size: {bbox_orig.max_bound - bbox_orig.min_bound}")

# Scale the point cloud to match camera units (millimeters)
# If the point cloud is in meters, scale by 1000 to convert to millimeters
if np.max(bbox_orig.max_bound - bbox_orig.min_bound) < 1.0:  # Likely in meters
    scale_factor = 1000.0  # Convert meters to millimeters
    reference_pcd.scale(scale_factor, center=reference_pcd.get_center())
    print(f"Scaled point cloud by {scale_factor} to convert from meters to millimeters")
else:
    print("Point cloud appears to already be in millimeters")

print(f"Scaled point cloud bounds:")
bbox_scaled = reference_pcd.get_axis_aligned_bounding_box()
print(f"  Min: {bbox_scaled.min_bound}")
print(f"  Max: {bbox_scaled.max_bound}")
print(f"  Size: {bbox_scaled.max_bound - bbox_scaled.min_bound}")

# Camera setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Create colorizer for depth visualization
colorizer = rs.colorizer()

profile = pipeline.start(config)
color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()

# Camera intrinsic parameters
cameraMatrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
])

distortionCoeffs = np.array(intr.coeffs)

print("Camera Matrix:")
print(cameraMatrix)
print("Distortion Coefficients:")
print(distortionCoeffs)

def depth_to_point_cloud(depth_image, color_image, camera_matrix, red_mask):
    """
    Convert depth image to point cloud, filtering for red objects only
    """
    height, width = depth_image.shape
    points = []
    colors = []
    
    # Get valid depth values (non-zero) and red pixels
    valid_mask = (depth_image > 0) & (red_mask > 0)
    y_coords, x_coords = np.where(valid_mask)
    
    for y, x in zip(y_coords, x_coords):
        depth = depth_image[y, x]
        
        # Convert pixel coordinates to 3D coordinates
        # Keep depth in millimeters to match point cloud units
        z = depth  # Keep in millimeters
        x_3d = (x - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
        y_3d = (y - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
        
        points.append([x_3d, y_3d, z])
        colors.append(color_image[y, x] / 255.0)  # Normalize colors to 0-1
    
    return np.array(points), np.array(colors)

def preprocess_point_cloud(pcd, voxel_size=0.01):
    """
    Preprocess point cloud for better registration
    """
    if len(pcd.points) == 0:
        print("Warning: Empty point cloud in preprocessing")
        return pcd
    
    print(f"Preprocessing: Starting with {len(pcd.points)} points")
    
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"After downsampling: {len(pcd_down.points)} points")
    
    if len(pcd_down.points) == 0:
        print("Warning: No points after downsampling, using original")
        return pcd
    
    # Estimate normals (radius in millimeters)
    try:
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))  # 10mm radius
        print(f"Normals estimated successfully")
    except Exception as e:
        print(f"Warning: Could not estimate normals: {e}")
        # Continue without normals
    
    # Remove outliers (be more conservative)
    if len(pcd_down.points) > 10:  # Only remove outliers if we have enough points
        pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=min(10, len(pcd_down.points)//2), std_ratio=2.0)
        print(f"After outlier removal: {len(pcd_down.points)} points")
    
    if len(pcd_down.points) == 0:
        print("Warning: No points after preprocessing, using downsampled version")
        return pcd.voxel_down_sample(voxel_size)
    
    return pcd_down

def register_point_clouds(source_pcd, target_pcd, initial_transform=None):
    """
    Register source point cloud to target using ICP with partial matching
    """
    if initial_transform is None:
        initial_transform = np.eye(4)
    
    # Check if point clouds have enough points
    if len(source_pcd.points) < 3 or len(target_pcd.points) < 3:
        print(f"Warning: Not enough points for registration - Source: {len(source_pcd.points)}, Target: {len(target_pcd.points)}")
        return initial_transform, 0.0
    
    # Preprocess both point clouds (voxel sizes in millimeters)
    # Use appropriate voxel sizes for millimeter units
    source_processed = preprocess_point_cloud(source_pcd, voxel_size=2.0)  # 2mm voxel for camera data
    target_processed = preprocess_point_cloud(target_pcd, voxel_size=5.0)  # 5mm voxel for reference
    
    # Check if preprocessing resulted in valid point clouds
    if len(source_processed.points) < 3 or len(target_processed.points) < 3:
        print(f"Warning: Not enough points after preprocessing - Source: {len(source_processed.points)}, Target: {len(target_processed.points)}")
        
        # Try with even larger voxel sizes as fallback
        if len(target_processed.points) < 3:
            print("Trying with larger voxel size for target...")
            target_processed = preprocess_point_cloud(target_pcd, voxel_size=10.0)  # 10mm voxel
            print(f"Target after larger voxel: {len(target_processed.points)} points")
        
        if len(source_processed.points) < 3:
            print("Trying with larger voxel size for source...")
            source_processed = preprocess_point_cloud(source_pcd, voxel_size=5.0)  # 5mm voxel
            print(f"Source after larger voxel: {len(source_processed.points)} points")
        
        # Final check
        if len(source_processed.points) < 3 or len(target_processed.points) < 3:
            print("Still not enough points after fallback, skipping registration")
            return initial_transform, 0.0
    
    # For partial matching, we need to be more flexible
    # Try multiple ICP iterations with different thresholds (in millimeters)
    thresholds = [1.0, 2.0, 5.0]  # 1mm, 2mm, 5mm - start tight, get looser
    best_transformation = initial_transform
    best_fitness = 0
    
    for threshold in thresholds:
        try:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_processed, target_processed, threshold, best_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
            
            if reg_p2p.fitness > best_fitness:
                best_fitness = reg_p2p.fitness
                best_transformation = reg_p2p.transformation
                
        except Exception as e:
            print(f"Warning: ICP registration failed with threshold {threshold}: {e}")
            continue
    
    return best_transformation, best_fitness

def pose_to_transformation_matrix(translation, rotation):
    """
    Convert translation and rotation to 4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, 3] = translation
    T[:3, :3] = rotation
    return T

def transformation_matrix_to_pose(T):
    """
    Extract translation and rotation from transformation matrix
    """
    translation = T[:3, 3]
    rotation_matrix = T[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    return translation, rotation

def create_partial_reference_pcd(full_pcd, camera_position, fov_angle=60):
    """
    Create a partial reference point cloud that simulates what the camera would see
    """
    # Convert camera position to numpy array if needed
    if isinstance(camera_position, list):
        camera_position = np.array(camera_position)
    
    # Get all points from the full point cloud
    points = np.asarray(full_pcd.points)
    
    # Calculate vectors from camera to each point
    vectors = points - camera_position
    distances = np.linalg.norm(vectors, axis=1)
    
    # Normalize vectors for angle calculation
    normalized_vectors = vectors / distances[:, np.newaxis]
    
    # Define camera forward direction (assuming camera looks along -Z axis)
    camera_forward = np.array([0, 0, -1])
    
    # Calculate angles between camera forward and point vectors
    cos_angles = np.dot(normalized_vectors, camera_forward)
    angles = np.arccos(np.clip(cos_angles, -1, 1)) * 180 / np.pi
    
    # Filter points within FOV
    fov_mask = angles <= fov_angle / 2
    
    # Create partial point cloud
    partial_pcd = o3d.geometry.PointCloud()
    partial_pcd.points = o3d.utility.Vector3dVector(points[fov_mask])
    
    if full_pcd.has_colors():
        colors = np.asarray(full_pcd.colors)
        partial_pcd.colors = o3d.utility.Vector3dVector(colors[fov_mask])
    
    return partial_pcd

def visualize_registration(source_pcd, target_pcd, transformation):
    """
    Visualize the registration result
    """
    source_pcd_transformed = source_pcd.transform(transformation)
    
    # Color the point clouds differently
    source_pcd_transformed.paint_uniform_color([1, 0, 0])  # Red for source
    target_pcd.paint_uniform_color([0, 1, 0])  # Green for target
    
    o3d.visualization.draw_geometries([source_pcd_transformed, target_pcd],
                                     window_name="Point Cloud Registration")

def visualize_normals_on_image(image, points, normals, camera_matrix, max_points=100):
    """
    Visualize normal vectors on the image
    """
    if len(points) == 0 or len(normals) == 0:
        return image
    
    # Sample points for visualization (don't show too many)
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        normals = normals[indices]
    
    vis_image = image.copy()
    
    for i, (point, normal) in enumerate(zip(points, normals)):
        # Project 3D point to 2D
        x_2d = int(point[0] * camera_matrix[0, 0] / point[2] + camera_matrix[0, 2])
        y_2d = int(point[1] * camera_matrix[1, 1] / point[2] + camera_matrix[1, 2])
        
        if 0 <= x_2d < image.shape[1] and 0 <= y_2d < image.shape[0]:
            # Draw point
            cv2.circle(vis_image, (x_2d, y_2d), 3, (0, 255, 0), -1)
            
            # Draw normal vector (scaled for visibility)
            normal_scale = 20  # pixels
            end_x = int(x_2d + normal[0] * normal_scale)
            end_y = int(y_2d + normal[1] * normal_scale)
            
            if 0 <= end_x < image.shape[1] and 0 <= end_y < image.shape[0]:
                cv2.arrowedLine(vis_image, (x_2d, y_2d), (end_x, end_y), (255, 0, 0), 2)
    
    return vis_image

def process_single_frame():
    """
    Process a single frame with detailed debugging
    """
    print("=== PROCESSING SINGLE FRAME ===")
    
    # Create a partial reference point cloud for better matching
    camera_position = np.array([0, 0, 500])  # 500mm (50cm) in front of object
    print(f"Creating partial reference from camera position: {camera_position}")
    partial_reference = create_partial_reference_pcd(reference_pcd, camera_position, fov_angle=60)
    print(f"Created partial reference with {len(partial_reference.points)} points")
    
    if len(partial_reference.points) == 0:
        print("Warning: Partial reference is empty, using full reference")
        partial_reference = reference_pcd
    
    # Debug: Show reference point cloud statistics
    if len(partial_reference.points) > 0:
        ref_points = np.asarray(partial_reference.points)
        print(f"Reference point cloud: {len(ref_points)} points")
        print(f"  X range: [{ref_points[:, 0].min():.1f}, {ref_points[:, 0].max():.1f}] mm")
        print(f"  Y range: [{ref_points[:, 1].min():.1f}, {ref_points[:, 1].max():.1f}] mm")
        print(f"  Z range: [{ref_points[:, 2].min():.1f}, {ref_points[:, 2].max():.1f}] mm")
    
    # Wait for a single frame
    print("Waiting for camera frame...")
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        print("Error: Could not get camera frames")
        return
    
    # Convert to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    print(f"Camera frame captured: {color_image.shape}")
    print(f"Depth range: {depth_image.min()} - {depth_image.max()} mm")
    
    # Create red mask
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    red_pixels = np.sum(red_mask > 0)
    print(f"Red pixels detected: {red_pixels}")
    
    if red_pixels < 100:
        print("Not enough red pixels detected")
        return
    
    # Convert depth to point cloud
    points, colors = depth_to_point_cloud(depth_image, color_image, cameraMatrix, red_mask)
    
    if len(points) < 100:
        print(f"Not enough points for registration: {len(points)}")
        return
    
    print(f"Camera point cloud: {len(points)} points")
    if len(points) > 0:
        points_array = np.asarray(points)
        print(f"  X range: [{points_array[:, 0].min():.1f}, {points_array[:, 0].max():.1f}] mm")
        print(f"  Y range: [{points_array[:, 1].min():.1f}, {points_array[:, 1].max():.1f}] mm")
        print(f"  Z range: [{points_array[:, 2].min():.1f}, {points_array[:, 2].max():.1f}] mm")
    
    # Create Open3D point cloud
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(points)
    source_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Register with reference point cloud
    print("\n=== REGISTRATION ===")
    transformation, fitness = register_point_clouds(source_pcd, partial_reference)
    
    print(f"Registration fitness: {fitness:.4f}")
    if fitness > 0.1:
        print("✓ Good registration!")
    else:
        print("✗ Poor registration")
    
    # Extract pose
    translation, rotation = transformation_matrix_to_pose(transformation)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    
    print(f"Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}] mm")
    print(f"Rotation (Euler): [{euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}] degrees")
    
    # Create visualizations
    colorized_depth = colorizer.colorize(depth_frame)
    colorized_depth_image = np.asanyarray(colorized_depth.get_data())
    masked_depth = colorized_depth_image.copy()
    masked_depth[red_mask == 0] = [0, 0, 0]
    
    # Add pose information overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(masked_depth, f'Fitness: {fitness:.3f}', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(masked_depth, f'Translation: [{translation[0]:.1f}, {translation[1]:.1f}, {translation[2]:.1f}]', 
               (10, 60), font, 0.5, (255, 255, 255), 1)
    cv2.putText(masked_depth, f'Rotation: [{euler_angles[0]:.0f}, {euler_angles[1]:.0f}, {euler_angles[2]:.0f}]', 
               (10, 90), font, 0.5, (255, 255, 255), 1)
    cv2.putText(masked_depth, f'Points: {len(points)}', (10, 120), font, 0.5, (255, 255, 255), 1)
    
    # Add normal vectors visualization
    if len(points) > 0:
        # Get normals from the processed point cloud
        source_processed = preprocess_point_cloud(source_pcd, voxel_size=2.0)
        if len(source_processed.points) > 0 and source_processed.has_normals():
            normals = np.asarray(source_processed.normals)
            points_for_normals = np.asarray(source_processed.points)
            normal_vis = visualize_normals_on_image(color_image, points_for_normals, normals, cameraMatrix)
            cv2.imshow('Normal Vectors', normal_vis)
    
    # Show images
    cv2.imshow('Pose Detection - Red Objects Only', masked_depth)
    cv2.imshow('Original Color', color_image)
    cv2.imshow('Red Mask', red_mask)
    
    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

try:
    # Process single frame instead of continuous loop
    process_single_frame()

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os.path as osp
import time
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

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
        z = depth  # Keep in millimeters
        x_3d = (x - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
        y_3d = (y - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
        
        points.append([x_3d, y_3d, z])
        colors.append(color_image[y, x] / 255.0)  # Normalize colors to 0-1
    
    return np.array(points), np.array(colors)

def extract_cubic_features(points):
    """
    Extract features suitable for cubic objects:
    1. Principal axes (using PCA)
    2. Bounding box corners
    3. Edge points
    """
    if len(points) < 10:
        return None, None, None
    
    # 1. Principal axes using PCA
    pca = PCA(n_components=3)
    pca.fit(points)
    principal_axes = pca.components_
    eigenvalues = pca.explained_variance_
    
    # 2. Bounding box corners
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min
    
    # Create 8 corners of bounding box
    corners = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                corner = [bbox_min[0] if i == 0 else bbox_max[0],
                         bbox_min[1] if j == 0 else bbox_max[1],
                         bbox_min[2] if k == 0 else bbox_max[2]]
                corners.append(corner)
    corners = np.array(corners)
    
    # 3. Edge points (points on the boundary of the bounding box)
    edge_points = []
    for point in points:
        # Check if point is close to any face of the bounding box
        for axis in range(3):
            if (abs(point[axis] - bbox_min[axis]) < 5 or  # 5mm tolerance
                abs(point[axis] - bbox_max[axis]) < 5):
                edge_points.append(point)
                break
    
    edge_points = np.array(edge_points) if edge_points else points
    
    return {
        'principal_axes': principal_axes,
        'eigenvalues': eigenvalues,
        'bbox_center': bbox_center,
        'bbox_size': bbox_size,
        'corners': corners,
        'edge_points': edge_points
    }

def match_cubic_features(features1, features2):
    """
    Match features between two cubic objects
    """
    if features1 is None or features2 is None:
        return None, 0.0
    
    # 1. Compare bounding box sizes (more lenient for partial views)
    size1 = features1['bbox_size']
    size2 = features2['bbox_size']
    
    # Calculate size ratios for each dimension
    size_ratios = size1 / size2
    print(f"Size ratios: {size_ratios}")
    
    # Use the most similar dimension for comparison
    min_ratio = np.min(size_ratios)
    max_ratio = np.max(size_ratios)
    
    print(f"Size ratio range: {min_ratio:.3f} - {max_ratio:.3f}")
    
    # More lenient size matching for partial views
    if min_ratio < 0.1 or max_ratio > 10.0:  # Very loose size constraints
        print("Warning: Large size difference, but continuing...")
        size_score = 0.1  # Low but not zero
    else:
        # Calculate size similarity score
        size_score = 1.0 - abs(1.0 - np.mean(size_ratios))
        size_score = max(0.0, size_score)
    
    # 2. Compare principal axes alignment
    axes1 = features1['principal_axes']
    axes2 = features2['principal_axes']
    
    # Calculate alignment scores for each axis
    alignment_scores = []
    for i in range(3):
        for j in range(3):
            # Try both directions (axis and -axis)
            score1 = abs(np.dot(axes1[i], axes2[j]))
            score2 = abs(np.dot(axes1[i], -axes2[j]))
            alignment_scores.append(max(score1, score2))
    
    max_alignment = max(alignment_scores)
    print(f"Max axis alignment: {max_alignment:.3f}")
    
    # 3. Compare bounding box centers (more lenient for different scales)
    center1 = features1['bbox_center']
    center2 = features2['bbox_center']
    center_diff = np.linalg.norm(center1 - center2)
    print(f"Center difference: {center_diff:.1f} mm")
    
    # Normalize center score by the larger object size
    max_size = max(np.max(size1), np.max(size2))
    center_score = max(0, 1 - center_diff / (max_size * 2))  # More lenient
    
    print(f"Size score: {size_score:.3f}")
    print(f"Center score: {center_score:.3f}")
    
    # Combined score (more weight on alignment for partial views)
    total_score = 0.5 * max_alignment + 0.3 * size_score + 0.2 * center_score
    
    return {
        'alignment_score': max_alignment,
        'center_score': center_score,
        'total_score': total_score,
        'size_ratio': np.mean(size_ratios)
    }, total_score

def estimate_pose_from_features(features1, features2, match_result):
    """
    Estimate pose transformation from matched features
    """
    if match_result is None:
        return np.eye(4), 0.0
    
    # Simple pose estimation based on bounding box centers and principal axes
    center1 = features1['bbox_center']
    center2 = features2['bbox_center']
    
    # Translation is the difference in centers
    translation = center2 - center1
    
    # Rotation is estimated from principal axes alignment
    axes1 = features1['principal_axes']
    axes2 = features2['principal_axes']
    
    # Find best axis alignment
    best_rotation = np.eye(3)
    for i in range(3):
        for j in range(3):
            # Try both directions
            if abs(np.dot(axes1[i], axes2[j])) > abs(np.dot(axes1[i], -axes2[j])):
                best_rotation[i] = axes2[j]
            else:
                best_rotation[i] = -axes2[j]
    
    # Create transformation matrix
    T = np.eye(4)
    T[:3, 3] = translation
    T[:3, :3] = best_rotation
    
    return T, match_result['total_score']

def visualize_cubic_features(image, features, camera_matrix, max_points=50):
    """
    Visualize cubic features on the image
    """
    if features is None:
        return image
    
    vis_image = image.copy()
    
    # Draw bounding box corners
    corners = features['corners']
    for corner in corners[:min(len(corners), 8)]:  # Show all 8 corners
        x_2d = int(corner[0] * camera_matrix[0, 0] / corner[2] + camera_matrix[0, 2])
        y_2d = int(corner[1] * camera_matrix[1, 1] / corner[2] + camera_matrix[1, 2])
        
        if 0 <= x_2d < image.shape[1] and 0 <= y_2d < image.shape[0]:
            cv2.circle(vis_image, (x_2d, y_2d), 5, (0, 255, 0), -1)  # Green for corners
    
    # Draw principal axes
    center = features['bbox_center']
    axes = features['principal_axes']
    eigenvalues = features['eigenvalues']
    
    for i, (axis, eigenvalue) in enumerate(zip(axes, eigenvalues)):
        # Scale axis by eigenvalue for visibility
        scale = eigenvalue * 20  # Adjust scale as needed
        end_point = center + axis * scale
        
        # Project to 2D
        x1 = int(center[0] * camera_matrix[0, 0] / center[2] + camera_matrix[0, 2])
        y1 = int(center[1] * camera_matrix[1, 1] / center[2] + camera_matrix[1, 2])
        x2 = int(end_point[0] * camera_matrix[0, 0] / end_point[2] + camera_matrix[0, 2])
        y2 = int(end_point[1] * camera_matrix[1, 1] / end_point[2] + camera_matrix[1, 2])
        
        if (0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and
            0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]):
            color = [(255, 0, 0), (0, 0, 255), (0, 255, 255)][i]  # Red, Blue, Cyan
            cv2.arrowedLine(vis_image, (x1, y1), (x2, y2), color, 3)
    
    return vis_image

def draw_bounding_box_3d(image, bbox_center, bbox_size, camera_matrix, color=(0, 255, 0), thickness=2):
    """
    Draw 3D bounding box projected onto 2D image
    """
    # Create 8 corners of the bounding box
    half_size = bbox_size / 2
    corners_3d = []
    
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                corner = bbox_center + np.array([i, j, k]) * half_size
                corners_3d.append(corner)
    
    corners_3d = np.array(corners_3d)
    
    # Project to 2D
    corners_2d = []
    for corner in corners_3d:
        if corner[2] > 0:  # Only project if in front of camera
            x_2d = int(corner[0] * camera_matrix[0, 0] / corner[2] + camera_matrix[0, 2])
            y_2d = int(corner[1] * camera_matrix[1, 1] / corner[2] + camera_matrix[1, 2])
            corners_2d.append((x_2d, y_2d))
        else:
            corners_2d.append(None)
    
    # Draw edges of the bounding box
    # Define the 12 edges of a cube
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    
    for edge in edges:
        pt1 = corners_2d[edge[0]]
        pt2 = corners_2d[edge[1]]
        
        if pt1 is not None and pt2 is not None:
            if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
                cv2.line(image, pt1, pt2, color, thickness)
    
    # Draw corners
    for corner in corners_2d:
        if corner is not None:
            if 0 <= corner[0] < image.shape[1] and 0 <= corner[1] < image.shape[0]:
                cv2.circle(image, corner, 4, color, -1)
    
    return image

def visualize_both_bounding_boxes(image, ref_features, cam_features, camera_matrix):
    """
    Visualize both reference and camera bounding boxes on the same image
    """
    vis_image = image.copy()
    
    # Draw reference bounding box in blue
    if ref_features is not None:
        vis_image = draw_bounding_box_3d(vis_image, ref_features['bbox_center'], 
                                       ref_features['bbox_size'], camera_matrix, 
                                       color=(255, 0, 0), thickness=2)  # Blue
    
    # Draw camera bounding box in green
    if cam_features is not None:
        vis_image = draw_bounding_box_3d(vis_image, cam_features['bbox_center'], 
                                       cam_features['bbox_size'], camera_matrix, 
                                       color=(0, 255, 0), thickness=2)  # Green
    
    # Add legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_image, 'Blue: Reference', (10, 30), font, 0.7, (255, 0, 0), 2)
    cv2.putText(vis_image, 'Green: Camera', (10, 60), font, 0.7, (0, 255, 0), 2)
    
    return vis_image

def process_single_frame():
    """
    Process a single frame with cubic object detection
    """
    print("=== PROCESSING SINGLE FRAME (CUBIC APPROACH) ===")
    
    # Extract features from reference point cloud
    ref_points = np.asarray(reference_pcd.points)
    ref_features = extract_cubic_features(ref_points)
    
    if ref_features is None:
        print("Error: Could not extract reference features")
        return
    
    print(f"Reference features:")
    print(f"  Bounding box center: {ref_features['bbox_center']}")
    print(f"  Bounding box size: {ref_features['bbox_size']}")
    print(f"  Principal axes eigenvalues: {ref_features['eigenvalues']}")
    
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
    
    # Debug: Check if reference bounding box will be visible
    ref_center = ref_features['bbox_center']
    ref_size = ref_features['bbox_size']
    
    # Project reference center to 2D
    if ref_center[2] > 0:
        ref_x_2d = int(ref_center[0] * cameraMatrix[0, 0] / ref_center[2] + cameraMatrix[0, 2])
        ref_y_2d = int(ref_center[1] * cameraMatrix[1, 1] / ref_center[2] + cameraMatrix[1, 2])
        print(f"  Reference center projected to 2D: ({ref_x_2d}, {ref_y_2d})")
        print(f"  Image size: {color_image.shape[1]}x{color_image.shape[0]}")
        
        if 0 <= ref_x_2d < color_image.shape[1] and 0 <= ref_y_2d < color_image.shape[0]:
            print("  ✓ Reference bounding box should be visible in image")
        else:
            print("  ✗ Reference bounding box is outside image bounds")
    else:
        print("  ✗ Reference bounding box is behind camera (negative Z)")
    
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
    
    # Extract features from camera point cloud
    cam_features = extract_cubic_features(points)
    
    if cam_features is None:
        print("Error: Could not extract camera features")
        return
    
    print(f"Camera features:")
    print(f"  Bounding box center: {cam_features['bbox_center']}")
    print(f"  Bounding box size: {cam_features['bbox_size']}")
    print(f"  Principal axes eigenvalues: {cam_features['eigenvalues']}")
    
    # Match features
    print("\n=== FEATURE MATCHING ===")
    match_result, match_score = match_cubic_features(cam_features, ref_features)
    
    # Initialize default values
    translation = np.array([0.0, 0.0, 0.0])
    euler_angles = np.array([0.0, 0.0, 0.0])
    pose_score = 0.0
    
    if match_result is not None:
        print(f"Feature matching score: {match_score:.4f}")
        print(f"  Alignment score: {match_result['alignment_score']:.4f}")
        print(f"  Center score: {match_result['center_score']:.4f}")
        print(f"  Size ratio: {match_result['size_ratio']:.4f}")
        
        if match_score > 0.3:
            print("✓ Reasonable feature match!")
            
            # Estimate pose
            transformation, pose_score = estimate_pose_from_features(cam_features, ref_features, match_result)
            
            # Extract pose information
            translation = transformation[:3, 3]
            rotation_matrix = transformation[:3, :3]
            rotation = R.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler('xyz', degrees=True)
        else:
            print("✗ Poor feature match, using default pose")
    else:
        print("✗ Feature matching failed, using default pose")
    
    print(f"\n=== POSE ESTIMATION ===")
    print(f"Pose score: {pose_score:.4f}")
    print(f"Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}] mm")
    print(f"Rotation (Euler): [{euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}] degrees")
    
    # Create visualizations
    colorized_depth = colorizer.colorize(depth_frame)
    colorized_depth_image = np.asanyarray(colorized_depth.get_data())
    masked_depth = colorized_depth_image.copy()
    masked_depth[red_mask == 0] = [0, 0, 0]
    
    # Add pose information overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(masked_depth, f'Match Score: {match_score:.3f}', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(masked_depth, f'Translation: [{translation[0]:.1f}, {translation[1]:.1f}, {translation[2]:.1f}]', 
               (10, 60), font, 0.5, (255, 255, 255), 1)
    cv2.putText(masked_depth, f'Rotation: [{euler_angles[0]:.0f}, {euler_angles[1]:.0f}, {euler_angles[2]:.0f}]', 
               (10, 90), font, 0.5, (255, 255, 255), 1)
    cv2.putText(masked_depth, f'Points: {len(points)}', (10, 120), font, 0.5, (255, 255, 255), 1)
    
    # Add quality indicator
    if match_score > 0.5:
        quality_color = (0, 255, 0)  # Green
        quality_text = "GOOD"
    elif match_score > 0.3:
        quality_color = (0, 255, 255)  # Yellow
        quality_text = "FAIR"
    else:
        quality_color = (0, 0, 255)  # Red
        quality_text = "POOR"
    
    cv2.putText(masked_depth, f'Quality: {quality_text}', (10, 150), font, 0.6, quality_color, 2)
    
    # Visualize features
    feature_vis = visualize_cubic_features(color_image, cam_features, cameraMatrix)
    
    # Visualize both bounding boxes
    bbox_vis = visualize_both_bounding_boxes(color_image, ref_features, cam_features, cameraMatrix)
    
    # Create separate reference bounding box visualization
    ref_bbox_vis = color_image.copy()
    if ref_features is not None:
        ref_bbox_vis = draw_bounding_box_3d(ref_bbox_vis, ref_features['bbox_center'], 
                                          ref_features['bbox_size'], cameraMatrix, 
                                          color=(255, 0, 0), thickness=3)  # Blue, thicker
        # Add reference info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(ref_bbox_vis, f'Reference BBox Center: [{ref_features["bbox_center"][0]:.1f}, {ref_features["bbox_center"][1]:.1f}, {ref_features["bbox_center"][2]:.1f}]', 
                   (10, 30), font, 0.6, (255, 0, 0), 2)
        cv2.putText(ref_bbox_vis, f'Reference BBox Size: [{ref_features["bbox_size"][0]:.1f}, {ref_features["bbox_size"][1]:.1f}, {ref_features["bbox_size"][2]:.1f}]', 
                   (10, 60), font, 0.6, (255, 0, 0), 2)
    
    # Create separate camera bounding box visualization
    cam_bbox_vis = color_image.copy()
    if cam_features is not None:
        cam_bbox_vis = draw_bounding_box_3d(cam_bbox_vis, cam_features['bbox_center'], 
                                          cam_features['bbox_size'], cameraMatrix, 
                                          color=(0, 255, 0), thickness=3)  # Green, thicker
        # Add camera info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cam_bbox_vis, f'Camera BBox Center: [{cam_features["bbox_center"][0]:.1f}, {cam_features["bbox_center"][1]:.1f}, {cam_features["bbox_center"][2]:.1f}]', 
                   (10, 30), font, 0.6, (0, 255, 0), 2)
        cv2.putText(cam_bbox_vis, f'Camera BBox Size: [{cam_features["bbox_size"][0]:.1f}, {cam_features["bbox_size"][1]:.1f}, {cam_features["bbox_size"][2]:.1f}]', 
                   (10, 60), font, 0.6, (0, 255, 0), 2)
    
    # Show images
    cv2.imshow('Reference Bounding Box Only', ref_bbox_vis)
    cv2.imshow('Camera Bounding Box Only', cam_bbox_vis)
    cv2.imshow('Bounding Boxes Comparison', bbox_vis)
    cv2.imshow('Cubic Features', feature_vis)
    cv2.imshow('Pose Detection - Red Objects Only', masked_depth)
    cv2.imshow('Original Color', color_image)
    cv2.imshow('Red Mask', red_mask)
    
    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

try:
    # Process single frame with cubic approach
    process_single_frame()

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

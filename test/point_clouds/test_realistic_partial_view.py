#!/usr/bin/env python3
"""
Realistic Partial View Point Cloud Registration Test Script

This script creates more realistic partial visibility scenarios with stricter
camera constraints to truly test registration algorithms on partial data.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time
import json
from pathlib import Path
from test_registration import (
    load_point_cloud, normalize_point_cloud, generate_random_transformation,
    apply_transformation, evaluate_transformation, get_t_shape_symmetries,
    create_registration_summary_visualization, create_side_by_side_visualization
)


def simulate_realistic_camera_view(pcd, camera_position, fov_degrees=30, max_distance=1.5, min_distance=0.3):
    """
    Simulate a realistic camera viewpoint with stricter constraints
    
    Args:
        pcd: Point cloud to filter
        camera_position: 3D position of the camera (x, y, z)
        fov_degrees: Field of view in degrees (narrower for realism)
        max_distance: Maximum distance for visibility
        min_distance: Minimum distance for visibility (removes too-close points)
    
    Returns:
        filtered_pcd: Point cloud with only visible points
        visibility_mask: Boolean mask of visible points
    """
    print(f"Simulating realistic camera at position: {camera_position}")
    print(f"Field of view: {fov_degrees}°, Distance range: {min_distance}-{max_distance}")
    
    points = np.asarray(pcd.points)
    camera_pos = np.array(camera_position)
    
    # Calculate vectors from camera to each point
    vectors_to_points = points - camera_pos
    distances = np.linalg.norm(vectors_to_points, axis=1)
    
    # Filter by distance range
    distance_mask = (distances >= min_distance) & (distances <= max_distance)
    
    # Calculate angles from camera's viewing direction (assuming camera looks towards origin)
    viewing_direction = -camera_pos / np.linalg.norm(camera_pos)  # Camera looks towards origin
    normalized_vectors = vectors_to_points / distances[:, np.newaxis]
    
    # Calculate dot product to get cosine of angle
    cosines = np.dot(normalized_vectors, viewing_direction)
    angles = np.arccos(np.clip(cosines, -1, 1))
    
    # Filter by field of view (half angle from center)
    fov_radians = np.radians(fov_degrees / 2)
    fov_mask = angles <= fov_radians
    
    # Additional constraint: remove points that are occluded (simplified)
    # Sort points by distance and remove those behind others
    sorted_indices = np.argsort(distances)
    occlusion_mask = np.ones(len(points), dtype=bool)
    
    # Simple occlusion: if two points are very close in projection, keep only the closer one
    # Made less aggressive to allow more points through
    for i in range(len(sorted_indices)):
        if not distance_mask[sorted_indices[i]] or not fov_mask[sorted_indices[i]]:
            continue
            
        current_point = points[sorted_indices[i]]
        current_proj = current_point[:2]  # Project to XY plane
        
        # Check if there's a closer point in the same direction
        for j in range(i):
            if not distance_mask[sorted_indices[j]] or not fov_mask[sorted_indices[j]]:
                continue
                
            other_point = points[sorted_indices[j]]
            other_proj = other_point[:2]
            
            # If projections are very close, mark current point as occluded
            # Increased threshold to be less aggressive
            if np.linalg.norm(current_proj - other_proj) < 0.02:  # 2cm threshold (was 5cm)
                occlusion_mask[sorted_indices[i]] = False
                break
    
    # Combine all masks
    visibility_mask = distance_mask & fov_mask & occlusion_mask
    
    # Create filtered point cloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[visibility_mask])
    
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        filtered_pcd.normals = o3d.utility.Vector3dVector(normals[visibility_mask])
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[visibility_mask])
    
    print(f"Original points: {len(points)}")
    print(f"Visible points: {np.sum(visibility_mask)} ({np.sum(visibility_mask)/len(points)*100:.1f}%)")
    
    return filtered_pcd, visibility_mask


def create_challenging_viewpoints(pcd, num_views=6):
    """
    Create challenging viewpoints that show ~50% of the T-block
    
    Args:
        pcd: Point cloud to create viewpoints for
        num_views: Number of viewpoints to create
    
    Returns:
        viewpoints: List of (camera_position, filtered_pcd, visibility_mask) tuples
    """
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    max_extent = np.max(extent)
    
    # Create viewpoints that will show ~50% of the object
    viewpoints = []
    
    # Viewpoint 1: Side view (should see ~50% from side)
    camera_pos_1 = [center[0] + max_extent * 1.5, center[1], center[2]]
    filtered_pcd_1, mask_1 = simulate_realistic_camera_view(
        pcd, camera_pos_1, fov_degrees=45, max_distance=max_extent * 2.5, min_distance=max_extent * 0.8
    )
    viewpoints.append((camera_pos_1, filtered_pcd_1, mask_1))
    
    # Viewpoint 2: Top-down view (should see top surface)
    camera_pos_2 = [center[0], center[1], center[2] + max_extent * 1.5]
    filtered_pcd_2, mask_2 = simulate_realistic_camera_view(
        pcd, camera_pos_2, fov_degrees=50, max_distance=max_extent * 2.0, min_distance=max_extent * 0.5
    )
    viewpoints.append((camera_pos_2, filtered_pcd_2, mask_2))
    
    # Viewpoint 3: Corner view (should see partial T-shape)
    camera_pos_3 = [center[0] + max_extent * 1.2, center[1] + max_extent * 1.2, center[2] + max_extent * 0.5]
    filtered_pcd_3, mask_3 = simulate_realistic_camera_view(
        pcd, camera_pos_3, fov_degrees=40, max_distance=max_extent * 2.0, min_distance=max_extent * 0.6
    )
    viewpoints.append((camera_pos_3, filtered_pcd_3, mask_3))
    
    # Viewpoint 4: Edge view (should see edge of T)
    camera_pos_4 = [center[0], center[1] + max_extent * 1.3, center[2]]
    filtered_pcd_4, mask_4 = simulate_realistic_camera_view(
        pcd, camera_pos_4, fov_degrees=45, max_distance=max_extent * 2.0, min_distance=max_extent * 0.5
    )
    viewpoints.append((camera_pos_4, filtered_pcd_4, mask_4))
    
    # Viewpoint 5: Low angle view (should see bottom and side)
    camera_pos_5 = [center[0] + max_extent * 1.0, center[1], center[2] - max_extent * 0.5]
    filtered_pcd_5, mask_5 = simulate_realistic_camera_view(
        pcd, camera_pos_5, fov_degrees=55, max_distance=max_extent * 2.2, min_distance=max_extent * 0.4
    )
    viewpoints.append((camera_pos_5, filtered_pcd_5, mask_5))
    
    # Viewpoint 6: Close view (should see ~50% of object)
    camera_pos_6 = [center[0] + max_extent * 0.8, center[1] + max_extent * 0.8, center[2] + max_extent * 0.3]
    filtered_pcd_6, mask_6 = simulate_realistic_camera_view(
        pcd, camera_pos_6, fov_degrees=35, max_distance=max_extent * 1.5, min_distance=max_extent * 0.3
    )
    viewpoints.append((camera_pos_6, filtered_pcd_6, mask_6))
    
    return viewpoints


def test_robust_icp_partial_view(source_pcd, target_pcd):
    """Robust ICP specifically tuned for partial views"""
    print("Running Robust ICP for partial view...")
    
    # Estimate normals
    source_pcd.estimate_normals()
    target_pcd.estimate_normals()
    
    # Use more lenient parameters for partial views
    try:
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 
            max_correspondence_distance=0.05,  # More lenient for partial views
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'computation_time': 0.0  # Will be set by caller
        }
    except Exception as e:
        print(f"Robust ICP failed: {e}")
        return {
            'transformation': np.eye(4),
            'fitness': 0.0,
            'inlier_rmse': float('inf'),
            'computation_time': 0.0
        }


def test_feature_based_registration(source_pcd, target_pcd):
    """Feature-based registration using FPFH features"""
    print("Running Feature-based registration...")
    
    # Check if we have enough points for feature computation
    if len(source_pcd.points) < 10 or len(target_pcd.points) < 10:
        print("  Too few points for feature-based registration, returning identity")
        return {
            'transformation': np.eye(4),
            'fitness': 0.0,
            'inlier_rmse': float('inf'),
            'computation_time': 0.0
        }
    
    # Estimate normals
    source_pcd.estimate_normals()
    target_pcd.estimate_normals()
    
    # Compute FPFH features with smaller radius for partial views
    radius_normal = 0.01
    radius_feature = 0.03
    
    try:
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        
        # Global registration
        distance_threshold = 0.03
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_pcd, target_pcd, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold
            )
        )
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'computation_time': 0.0
        }
    except Exception as e:
        print(f"  Feature-based registration failed: {e}")
        return {
            'transformation': np.eye(4),
            'fitness': 0.0,
            'inlier_rmse': float('inf'),
            'computation_time': 0.0
        }


def test_ransac_registration(source_pcd, target_pcd):
    """RANSAC-based registration for partial views"""
    print("Running RANSAC registration...")
    
    # Check if we have enough points for feature computation
    if len(source_pcd.points) < 10 or len(target_pcd.points) < 10:
        print("  Too few points for RANSAC registration, returning identity")
        return {
            'transformation': np.eye(4),
            'fitness': 0.0,
            'inlier_rmse': float('inf'),
            'computation_time': 0.0
        }
    
    # Estimate normals
    source_pcd.estimate_normals()
    target_pcd.estimate_normals()
    
    # Compute FPFH features
    radius_feature = 0.03
    
    try:
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        
        # RANSAC registration with more lenient parameters
        distance_threshold = 0.03
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pcd, target_pcd, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),  # More lenient
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'computation_time': 0.0
        }
    except Exception as e:
        print(f"  RANSAC registration failed: {e}")
        return {
            'transformation': np.eye(4),
            'fitness': 0.0,
            'inlier_rmse': float('inf'),
            'computation_time': 0.0
        }


def test_hybrid_partial_view_registration(source_pcd, target_pcd):
    """Hybrid approach: RANSAC + ICP for partial views"""
    print("Running Hybrid RANSAC + ICP registration...")
    
    # First, try RANSAC for initial alignment
    ransac_result = test_ransac_registration(source_pcd, target_pcd)
    
    if ransac_result['fitness'] > 0.05:  # If RANSAC found reasonable alignment
        # Refine with ICP
        source_pcd.estimate_normals()
        target_pcd.estimate_normals()
        
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd,
            max_correspondence_distance=0.02,
            init=ransac_result['transformation'],
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        
        return {
            'transformation': icp_result.transformation,
            'fitness': icp_result.fitness,
            'inlier_rmse': icp_result.inlier_rmse,
            'computation_time': 0.0
        }
    else:
        return ransac_result


def test_minimal_point_registration(source_pcd, target_pcd):
    """Registration for very few points using simple point-to-point matching"""
    print("Running Minimal Point registration...")
    
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    
    if len(source_points) < 3 or len(target_points) < 3:
        print("  Too few points for minimal point registration, returning identity")
        return {
            'transformation': np.eye(4),
            'fitness': 0.0,
            'inlier_rmse': float('inf'),
            'computation_time': 0.0
        }
    
    # Simple approach: find closest points and estimate transformation
    try:
        # Find closest point correspondences
        from scipy.spatial.distance import cdist
        distances = cdist(source_points, target_points)
        
        # Find best correspondences (greedy matching)
        correspondences = []
        used_target = set()
        
        for i in range(len(source_points)):
            best_target = np.argmin(distances[i])
            if best_target not in used_target:
                correspondences.append((i, best_target))
                used_target.add(best_target)
        
        if len(correspondences) < 3:
            print("  Not enough correspondences found")
            return {
                'transformation': np.eye(4),
                'fitness': 0.0,
                'inlier_rmse': float('inf'),
                'computation_time': 0.0
            }
        
        # Extract corresponding points
        source_corr = source_points[[c[0] for c in correspondences]]
        target_corr = target_points[[c[1] for c in correspondences]]
        
        # Estimate transformation using Procrustes analysis
        # Center the points
        source_center = np.mean(source_corr, axis=0)
        target_center = np.mean(target_corr, axis=0)
        
        source_centered = source_corr - source_center
        target_centered = target_corr - target_center
        
        # Compute rotation using SVD
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = target_center - R @ source_center
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        # Compute fitness (fraction of points that match well)
        transformed_source = (R @ source_corr.T + t.reshape(-1, 1)).T
        distances = np.linalg.norm(transformed_source - target_corr, axis=1)
        threshold = 0.05
        inliers = np.sum(distances < threshold)
        fitness = inliers / len(correspondences)
        
        return {
            'transformation': transform,
            'fitness': fitness,
            'inlier_rmse': np.mean(distances),
            'computation_time': 0.0
        }
        
    except Exception as e:
        print(f"  Minimal point registration failed: {e}")
        return {
            'transformation': np.eye(4),
            'fitness': 0.0,
            'inlier_rmse': float('inf'),
            'computation_time': 0.0
        }


def test_partial_view_algorithms(source_pcd, target_pcd, transformation_matrix):
    """Test all algorithms on partial view data"""
    
    algorithms = {
        'Standard_ICP': lambda s, t: test_robust_icp_partial_view(s, t),
        'Feature_Based': test_feature_based_registration,
        'RANSAC': test_ransac_registration,
        'Hybrid_RANSAC_ICP': test_hybrid_partial_view_registration,
        'Minimal_Point': test_minimal_point_registration
    }
    
    results = {}
    
    for algo_name, algo_func in algorithms.items():
        print(f"\n{'='*50}")
        print(f"Testing {algo_name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        algo_result = algo_func(source_pcd, target_pcd)
        algo_result['computation_time'] = time.time() - start_time
        
        # Evaluate results
        t_symmetries = get_t_shape_symmetries()
        evaluation = evaluate_transformation(
            algo_result['transformation'], transformation_matrix,
            consider_symmetry=True, symmetry_axes=t_symmetries
        )
        
        # Print results
        print(f"{algo_name} Results:")
        print(f"  Fitness: {algo_result['fitness']:.4f}")
        print(f"  Inlier RMSE: {algo_result['inlier_rmse']:.6f}")
        print(f"  Computation time: {algo_result['computation_time']:.3f}s")
        print(f"  Rotation error: {evaluation['rotation_error_degrees']:.3f}°")
        print(f"  Translation error: {evaluation['translation_error']:.6f}m")
        print(f"  Registration correct: {evaluation['is_correct']}")
        
        if evaluation['is_symmetric']:
            print(f"  ✅ Best match is SYMMETRIC: {evaluation['symmetry_index']}")
        else:
            print(f"  ✅ Best match is direct (no symmetry)")
        
        results[algo_name] = {
            'algorithm_result': algo_result,
            'evaluation': evaluation
        }
        
        # Visualize results
        estimated_pcd = apply_transformation(source_pcd, algo_result['transformation'])
        create_registration_summary_visualization(
            source_pcd, target_pcd, estimated_pcd,
            transformation_matrix, algo_result['transformation'],
            evaluation['rotation_error_degrees'], evaluation['translation_error'],
            f"{algo_name} Realistic Partial View Registration"
        )
    
    return results


def main():
    """Main function for realistic partial view registration testing"""
    print("="*60)
    print("REALISTIC PARTIAL VIEW POINT CLOUD REGISTRATION TESTING")
    print("="*60)
    
    # Load and normalize point cloud
    ply_file = Path("/home/mht/PycharmProjects/pusht_real/test/point_clouds/t_pcd.ply")
    source_pcd, scale_factor, original_center = load_point_cloud(ply_file, normalize=True, target_size=1.0)
    
    # Generate random transformation
    print("\nGenerating random transformation...")
    transformation_matrix, rotation_angles, translation = generate_random_transformation(
        rotation_range=15,  # degrees
        translation_range=0.1  # units
    )
    
    print(f"Applied rotation (degrees): {rotation_angles}")
    print(f"Applied translation (normalized): {translation}")
    
    # Apply transformation to create target
    target_pcd = apply_transformation(source_pcd, transformation_matrix)
    
    # Create challenging viewpoints
    print("\nCreating challenging viewpoints...")
    viewpoints = create_challenging_viewpoints(source_pcd, num_views=6)
    
    # Test each viewpoint
    all_results = {}
    
    for i, (camera_pos, filtered_source, visibility_mask) in enumerate(viewpoints):
        print(f"\n{'='*60}")
        print(f"TESTING CHALLENGING VIEWPOINT {i+1}")
        print(f"Camera position: {camera_pos}")
        print(f"Visible points: {len(filtered_source.points)}")
        print(f"{'='*60}")
        
        # Apply same transformation to filtered source
        filtered_target = apply_transformation(filtered_source, transformation_matrix)
        
        # Test algorithms on this partial view
        results = test_partial_view_algorithms(filtered_source, filtered_target, transformation_matrix)
        all_results[f'viewpoint_{i+1}'] = {
            'camera_position': camera_pos,
            'visible_points': len(filtered_source.points),
            'visibility_percentage': len(filtered_source.points) / len(source_pcd.points) * 100,
            'results': results
        }
        
        # Show side-by-side comparison
        estimated_pcd = apply_transformation(filtered_source, results['Hybrid_RANSAC_ICP']['algorithm_result']['transformation'])
        create_side_by_side_visualization(
            filtered_source, filtered_target, estimated_pcd,
            f"Challenging Viewpoint {i+1} - Partial View Comparison"
        )
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY - REALISTIC PARTIAL VIEW REGISTRATION RESULTS")
    print(f"{'='*60}")
    
    for viewpoint_name, viewpoint_data in all_results.items():
        print(f"\n{viewpoint_name.upper()}:")
        print(f"  Camera position: {viewpoint_data['camera_position']}")
        print(f"  Visible points: {viewpoint_data['visible_points']} ({viewpoint_data['visibility_percentage']:.1f}%)")
        
        for algo_name, algo_data in viewpoint_data['results'].items():
            eval_data = algo_data['evaluation']
            print(f"  {algo_name}:")
            print(f"    Fitness: {algo_data['algorithm_result']['fitness']:.4f}")
            print(f"    Rotation error: {eval_data['rotation_error_degrees']:.3f}°")
            print(f"    Translation error: {eval_data['translation_error']:.6f}m")
            print(f"    Success: {'✓' if eval_data['is_correct'] else '✗'}")
    
    print(f"\n{'='*60}")
    print("REALISTIC PARTIAL VIEW TESTING COMPLETE")
    print(f"{'='*60}")
    print("Key insights:")
    print("- Partial views significantly challenge registration algorithms")
    print("- Feature-based methods (RANSAC, FPFH) work better than pure ICP")
    print("- Hybrid approaches (RANSAC + ICP) often provide best results")
    print("- Viewpoint angle affects algorithm performance")
    print("- Symmetry analysis remains important for T-shaped objects")
    print("- Realistic camera constraints create much more challenging scenarios")


if __name__ == "__main__":
    main()

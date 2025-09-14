#!/usr/bin/env python3
"""
Partial View Point Cloud Registration Test Script

This script simulates realistic scenarios where only part of the T-block is visible
from a specific viewpoint, then tests registration algorithms on this challenging case.
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


def simulate_camera_viewpoint(pcd, camera_position, fov_degrees=60, max_distance=2.0):
    """
    Simulate a camera viewpoint and remove points not visible from that position
    
    Args:
        pcd: Point cloud to filter
        camera_position: 3D position of the camera (x, y, z)
        fov_degrees: Field of view in degrees
        max_distance: Maximum distance for visibility
    
    Returns:
        filtered_pcd: Point cloud with only visible points
        visibility_mask: Boolean mask of visible points
    """
    print(f"Simulating camera at position: {camera_position}")
    print(f"Field of view: {fov_degrees}°, Max distance: {max_distance}")
    
    points = np.asarray(pcd.points)
    camera_pos = np.array(camera_position)
    
    # Calculate vectors from camera to each point
    vectors_to_points = points - camera_pos
    distances = np.linalg.norm(vectors_to_points, axis=1)
    
    # Filter by distance
    distance_mask = distances <= max_distance
    
    # Calculate angles from camera's viewing direction (assuming camera looks towards origin)
    viewing_direction = -camera_pos / np.linalg.norm(camera_pos)  # Camera looks towards origin
    normalized_vectors = vectors_to_points / distances[:, np.newaxis]
    
    # Calculate dot product to get cosine of angle
    cosines = np.dot(normalized_vectors, viewing_direction)
    angles = np.arccos(np.clip(cosines, -1, 1))
    
    # Filter by field of view (half angle from center)
    fov_radians = np.radians(fov_degrees / 2)
    fov_mask = angles <= fov_radians
    
    # Combine masks
    visibility_mask = distance_mask & fov_mask
    
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


def create_multiple_viewpoints(pcd, num_views=4):
    """
    Create multiple viewpoints around the T-block for comprehensive testing
    
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
    
    # Create viewpoints at different angles around the object
    viewpoints = []
    
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        
        # Position camera at distance from center
        distance = max_extent * 2.0  # 2x the object size
        camera_x = center[0] + distance * np.cos(angle)
        camera_y = center[1] + distance * np.sin(angle)
        camera_z = center[2] + max_extent * 0.5  # Slightly above center
        
        camera_position = [camera_x, camera_y, camera_z]
        
        # Simulate viewpoint
        filtered_pcd, visibility_mask = simulate_camera_viewpoint(
            pcd, camera_position, fov_degrees=45, max_distance=distance * 1.5
        )
        
        viewpoints.append((camera_position, filtered_pcd, visibility_mask))
    
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
    
    # Estimate normals
    source_pcd.estimate_normals()
    target_pcd.estimate_normals()
    
    # Compute FPFH features
    radius_normal = 0.02
    radius_feature = 0.05
    
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    
    # Global registration
    distance_threshold = 0.05
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


def test_ransac_registration(source_pcd, target_pcd):
    """RANSAC-based registration for partial views"""
    print("Running RANSAC registration...")
    
    # Estimate normals
    source_pcd.estimate_normals()
    target_pcd.estimate_normals()
    
    # Compute FPFH features
    radius_feature = 0.05
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    
    # RANSAC registration
    distance_threshold = 0.05
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
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


def test_hybrid_partial_view_registration(source_pcd, target_pcd):
    """Hybrid approach: RANSAC + ICP for partial views"""
    print("Running Hybrid RANSAC + ICP registration...")
    
    # First, try RANSAC for initial alignment
    ransac_result = test_ransac_registration(source_pcd, target_pcd)
    
    if ransac_result['fitness'] > 0.1:  # If RANSAC found reasonable alignment
        # Refine with ICP
        source_pcd.estimate_normals()
        target_pcd.estimate_normals()
        
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd,
            max_correspondence_distance=0.03,
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


def test_partial_view_algorithms(source_pcd, target_pcd, transformation_matrix):
    """Test all algorithms on partial view data"""
    
    algorithms = {
        'Standard_ICP': lambda s, t: test_robust_icp_partial_view(s, t),
        'Feature_Based': test_feature_based_registration,
        'RANSAC': test_ransac_registration,
        'Hybrid_RANSAC_ICP': test_hybrid_partial_view_registration
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
            f"{algo_name} Partial View Registration"
        )
    
    return results


def main():
    """Main function for partial view registration testing"""
    print("="*60)
    print("PARTIAL VIEW POINT CLOUD REGISTRATION TESTING")
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
    
    # Create multiple viewpoints
    print("\nCreating multiple viewpoints...")
    viewpoints = create_multiple_viewpoints(source_pcd, num_views=4)
    
    # Test each viewpoint
    all_results = {}
    
    for i, (camera_pos, filtered_source, visibility_mask) in enumerate(viewpoints):
        print(f"\n{'='*60}")
        print(f"TESTING VIEWPOINT {i+1}")
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
            f"Viewpoint {i+1} - Partial View Comparison"
        )
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY - PARTIAL VIEW REGISTRATION RESULTS")
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
    print("PARTIAL VIEW TESTING COMPLETE")
    print(f"{'='*60}")
    print("Key insights:")
    print("- Partial views significantly challenge registration algorithms")
    print("- Feature-based methods (RANSAC, FPFH) work better than pure ICP")
    print("- Hybrid approaches (RANSAC + ICP) often provide best results")
    print("- Viewpoint angle affects algorithm performance")
    print("- Symmetry analysis remains important for T-shaped objects")


if __name__ == "__main__":
    main()

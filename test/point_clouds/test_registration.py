#!/usr/bin/env python3
"""
Point Cloud Registration Test Script

This script tests registration algorithms by:
1. Loading a point cloud
2. Applying random rotation and translation
3. Using registration to recover the transformation
4. Evaluating the results
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time
import json
from pathlib import Path


def load_point_cloud(file_path, normalize=True, target_size=1.0):
    """Load point cloud from PLY file with optional normalization"""
    print(f"Loading point cloud from: {file_path}")
    pcd = o3d.io.read_point_cloud(str(file_path))
    
    if len(pcd.points) == 0:
        raise ValueError("Failed to load point cloud or point cloud is empty")
    
    print(f"Loaded point cloud with {len(pcd.points)} points")
    
    if normalize:
        print("Normalizing point cloud to unit bounding box...")
        pcd, scale_factor, center = normalize_point_cloud(pcd, target_size)
        print(f"Normalization: scale_factor={scale_factor:.6f}, center={center}")
        return pcd, scale_factor, center
    else:
        return pcd, 1.0, np.zeros(3)


def normalize_point_cloud(pcd, target_size=1.0):
    """Normalize point cloud to unit bounding box and center it at origin"""
    # Get bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    
    # Find the largest dimension
    max_extent = np.max(extent)
    
    # Calculate scale factor to make largest dimension = target_size
    scale_factor = target_size / max_extent
    
    # Step 1: Translate to origin
    translate_to_origin = np.eye(4)
    translate_to_origin[:3, 3] = -center
    
    # Step 2: Scale
    scale_transform = np.eye(4)
    scale_transform[:3, :3] *= scale_factor
    
    # Step 3: Get new center after scaling and translate to center the scaled point cloud
    pcd_temp = o3d.geometry.PointCloud(pcd)
    pcd_temp.transform(translate_to_origin)
    pcd_temp.transform(scale_transform)
    
    # Get new bounding box after scaling
    new_bbox = pcd_temp.get_axis_aligned_bounding_box()
    new_center = new_bbox.get_center()
    
    # Step 4: Translate to center the scaled point cloud at origin
    center_transform = np.eye(4)
    center_transform[:3, 3] = -new_center
    
    # Combine all transformations
    transform = center_transform @ scale_transform @ translate_to_origin
    
    # Apply transformation
    pcd.transform(transform)
    
    return pcd, scale_factor, center


def convert_to_original_scale(translation, scale_factor):
    """Convert translation from normalized scale back to original scale"""
    return translation / scale_factor


def generate_random_transformation(rotation_range=45, translation_range=0.5):
    """
    Generate random rotation and translation
    
    Args:
        rotation_range: Maximum rotation angle in degrees
        translation_range: Maximum translation distance
    
    Returns:
        transformation_matrix: 4x4 homogeneous transformation matrix
        rotation_angles: (rx, ry, rz) in degrees
        translation: (tx, ty, tz)
    """
    # Generate random rotation angles
    rx = np.random.uniform(-rotation_range, rotation_range)
    ry = np.random.uniform(-rotation_range, rotation_range)
    rz = np.random.uniform(-rotation_range, rotation_range)
    
    # Generate random translation
    tx = np.random.uniform(-translation_range, translation_range)
    ty = np.random.uniform(-translation_range, translation_range)
    tz = np.random.uniform(-translation_range, translation_range)
    
    # Create rotation matrix
    rotation = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    rotation_matrix = rotation.as_matrix()
    
    # Create transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]
    
    return transformation_matrix, (rx, ry, rz), (tx, ty, tz)


def apply_transformation(pcd, transformation_matrix):
    """Apply transformation to point cloud"""
    # Create a copy to avoid modifying the original
    
    pcd_copy = o3d.geometry.PointCloud(pcd)
    pcd_copy.transform(transformation_matrix)
    return pcd_copy


def evaluate_transformation(estimated_transform, ground_truth_transform, tolerance=1e-3, 
                           consider_symmetry=True, symmetry_axes=None):
    """
    Evaluate how close the estimated transformation is to ground truth,
    considering potential symmetries of the object
    
    Args:
        estimated_transform: 4x4 estimated transformation matrix
        ground_truth_transform: 4x4 ground truth transformation matrix
        tolerance: tolerance for considering transformations equal
        consider_symmetry: whether to check for symmetric solutions
        symmetry_axes: list of symmetry transformations (rotation matrices)
    
    Returns:
        dict: evaluation metrics
    """
    # Compute the error transformation
    error_transform = np.linalg.inv(ground_truth_transform) @ estimated_transform
    
    # Extract rotation and translation errors
    rotation_error = R.from_matrix(error_transform[:3, :3])
    rotation_error_angle = np.abs(rotation_error.as_rotvec())
    rotation_error_degrees = np.linalg.norm(rotation_error_angle) * 180 / np.pi
    
    translation_error = np.linalg.norm(error_transform[:3, 3])
    
    # Check if transformation is close to identity
    is_correct = (rotation_error_degrees < tolerance * 180 / np.pi and 
                  translation_error < tolerance)
    
    # Check for symmetric solutions if enabled
    symmetric_solutions = []
    if consider_symmetry and symmetry_axes is not None:
        for i, sym_axis in enumerate(symmetry_axes):
            # Apply symmetry transformation to ground truth
            symmetric_gt = ground_truth_transform @ sym_axis
            symmetric_error = np.linalg.inv(symmetric_gt) @ estimated_transform
            
            sym_rotation_error = R.from_matrix(symmetric_error[:3, :3])
            sym_rotation_angle = np.abs(sym_rotation_error.as_rotvec())
            sym_rotation_degrees = np.linalg.norm(sym_rotation_angle) * 180 / np.pi
            sym_translation_error = np.linalg.norm(symmetric_error[:3, 3])
            
            # Check if this symmetric solution is actually close to the estimated transform
            is_close = (sym_rotation_degrees < 5.0 and sym_translation_error < 0.1)  # More lenient threshold
            
            symmetric_solutions.append({
                'symmetry_index': i,
                'rotation_error_degrees': sym_rotation_degrees,
                'translation_error': sym_translation_error,
                'is_correct': (sym_rotation_degrees < tolerance * 180 / np.pi and 
                              sym_translation_error < tolerance),
                'is_close': is_close
            })
    
    # Find the best solution (original or symmetric)
    best_solution = {
        'rotation_error_degrees': rotation_error_degrees,
        'translation_error': translation_error,
        'is_correct': is_correct,
        'is_symmetric': False,
        'symmetry_index': None
    }
    
    if symmetric_solutions:
        # Find the best symmetric solution
        best_symmetric = min(symmetric_solutions, 
                           key=lambda x: x['rotation_error_degrees'] + x['translation_error'])
        
        # Use symmetric solution if it's significantly better
        improvement_threshold = 10.0  # degrees + meters
        current_total_error = best_solution['rotation_error_degrees'] + best_solution['translation_error']
        symmetric_total_error = best_symmetric['rotation_error_degrees'] + best_symmetric['translation_error']
        
        if (current_total_error - symmetric_total_error) > improvement_threshold:
            best_solution = {
                'rotation_error_degrees': best_symmetric['rotation_error_degrees'],
                'translation_error': best_symmetric['translation_error'],
                'is_correct': best_symmetric['is_correct'],
                'is_symmetric': True,
                'symmetry_index': best_symmetric['symmetry_index']
            }
    
    return {
        'rotation_error_degrees': best_solution['rotation_error_degrees'],
        'translation_error': best_solution['translation_error'],
        'is_correct': best_solution['is_correct'],
        'is_symmetric': best_solution['is_symmetric'],
        'symmetry_index': best_solution['symmetry_index'],
        'error_transform': error_transform,
        'symmetric_solutions': symmetric_solutions
    }


def get_t_shape_symmetries():
    """
    Define symmetry transformations for a T-shape object
    
    A T-shape has the following symmetries:
    1. 180° rotation around the vertical axis (Z-axis)
    2. 180° rotation around the horizontal axis (X-axis) 
    3. 180° rotation around the horizontal axis (Y-axis)
    4. Identity (no symmetry)
    
    Returns:
        list: List of 4x4 transformation matrices representing symmetries
    """
    symmetries = []
    
    # Identity (no symmetry)
    symmetries.append(np.eye(4))
    
    # 180° rotation around Z-axis (vertical flip)
    z_rotation = R.from_euler('z', 180, degrees=True)
    z_sym = np.eye(4)
    z_sym[:3, :3] = z_rotation.as_matrix()
    symmetries.append(z_sym)
    
    # 180° rotation around X-axis (horizontal flip)
    x_rotation = R.from_euler('x', 180, degrees=True)
    x_sym = np.eye(4)
    x_sym[:3, :3] = x_rotation.as_matrix()
    symmetries.append(x_sym)
    
    # 180° rotation around Y-axis (horizontal flip)
    y_rotation = R.from_euler('y', 180, degrees=True)
    y_sym = np.eye(4)
    y_sym[:3, :3] = y_rotation.as_matrix()
    symmetries.append(y_sym)
    
    return symmetries


def analyze_symmetry_impact(source_pcd, target_pcd, estimated_transform, ground_truth_transform):
    """
    Analyze how symmetries affect the registration evaluation
    
    Args:
        source_pcd: source point cloud
        target_pcd: target point cloud  
        estimated_transform: estimated transformation
        ground_truth_transform: ground truth transformation
    
    Returns:
        dict: symmetry analysis results
    """
    symmetries = get_t_shape_symmetries()
    symmetry_names = ['Identity', 'Z-180°', 'X-180°', 'Y-180°']
    
    print(f"\n{'='*60}")
    print(f"SYMMETRY ANALYSIS FOR T-SHAPE")
    print(f"{'='*60}")
    
    results = []
    for i, (sym, name) in enumerate(zip(symmetries, symmetry_names)):
        # Apply symmetry to ground truth
        symmetric_gt = ground_truth_transform @ sym
        
        # Evaluate against symmetric ground truth
        error_transform = np.linalg.inv(symmetric_gt) @ estimated_transform
        
        rotation_error = R.from_matrix(error_transform[:3, :3])
        rotation_error_degrees = np.linalg.norm(rotation_error.as_rotvec()) * 180 / np.pi
        translation_error = np.linalg.norm(error_transform[:3, 3])
        
        results.append({
            'symmetry': name,
            'rotation_error': rotation_error_degrees,
            'translation_error': translation_error,
            'total_error': rotation_error_degrees + translation_error
        })
        
        print(f"{name:12}: Rotation {rotation_error_degrees:6.2f}°, Translation {translation_error:8.4f}m")
    
    # Find best symmetry
    best_symmetry = min(results, key=lambda x: x['total_error'])
    print(f"\nBest symmetry match: {best_symmetry['symmetry']}")
    print(f"  Rotation error: {best_symmetry['rotation_error']:.2f}°")
    print(f"  Translation error: {best_symmetry['translation_error']:.4f}m")
    
    return {
        'results': results,
        'best_symmetry': best_symmetry,
        'symmetries': symmetries,
        'symmetry_names': symmetry_names
    }


def create_bounding_box_registration(source_pcd, target_pcd, box_size=1.0):
    """
    Registration based on 3D bounding box alignment
    
    Args:
        source_pcd: source point cloud
        target_pcd: target point cloud
        box_size: size of the cubic bounding box
    
    Returns:
        dict: registration results
    """
    print("Running Bounding Box registration...")
    start_time = time.time()
    
    # Get bounding boxes
    source_bbox = source_pcd.get_axis_aligned_bounding_box()
    target_bbox = target_pcd.get_axis_aligned_bounding_box()
    
    # Get centers
    source_center = source_bbox.get_center()
    target_center = target_bbox.get_center()
    
    # Get extents
    source_extent = source_bbox.get_extent()
    target_extent = target_bbox.get_extent()
    
    print(f"Source center: {source_center}, extent: {source_extent}")
    print(f"Target center: {target_center}, extent: {target_extent}")
    
    # Calculate translation to align centers
    translation = target_center - source_center
    
    # Calculate scaling to normalize to fixed box size
    # Use the maximum extent to determine scaling
    source_scale = box_size / np.max(source_extent)
    target_scale = box_size / np.max(target_extent)
    
    print(f"Source scale: {source_scale}, Target scale: {target_scale}")
    
    # Create transformation matrix
    # First scale, then translate
    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = source_scale
    scale_matrix[1, 1] = source_scale
    scale_matrix[2, 2] = source_scale
    
    # Center the source point cloud
    center_matrix = np.eye(4)
    center_matrix[:3, 3] = -source_center
    
    # Translate to target center
    translate_matrix = np.eye(4)
    translate_matrix[:3, 3] = target_center
    
    # Combine transformations: translate to target center, then scale
    transformation = translate_matrix @ scale_matrix @ center_matrix
    
    end_time = time.time()
    
    # Calculate fitness (how well the bounding boxes align)
    # This is a simplified metric
    center_distance = np.linalg.norm(translation)
    scale_ratio = target_scale / source_scale
    fitness = 1.0 / (1.0 + center_distance + abs(scale_ratio - 1.0))
    
    return {
        'transformation': transformation,
        'fitness': fitness,
        'inlier_rmse': center_distance,
        'correspondence_set': [],
        'computation_time': end_time - start_time,
        'method': 'bounding_box'
    }


def create_pca_bounding_box_registration(source_pcd, target_pcd, box_size=1.0):
    """
    Registration based on PCA-aligned bounding boxes
    
    Args:
        source_pcd: source point cloud
        target_pcd: target point cloud
        box_size: size of the cubic bounding box
    
    Returns:
        dict: registration results
    """
    print("Running PCA Bounding Box registration...")
    start_time = time.time()
    
    # Get point coordinates
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    
    # Center the point clouds
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    
    # Compute PCA
    source_cov = np.cov(source_centered.T)
    target_cov = np.cov(target_centered.T)
    
    source_eigenvals, source_eigenvecs = np.linalg.eigh(source_cov)
    target_eigenvals, target_eigenvecs = np.linalg.eigh(target_cov)
    
    # Sort by eigenvalues (descending)
    source_order = np.argsort(source_eigenvals)[::-1]
    target_order = np.argsort(target_eigenvals)[::-1]
    
    source_eigenvecs = source_eigenvecs[:, source_order]
    target_eigenvecs = target_eigenvecs[:, target_order]
    
    # Ensure right-handed coordinate system
    source_eigenvecs[:, 2] = np.cross(source_eigenvecs[:, 0], source_eigenvecs[:, 1])
    target_eigenvecs[:, 2] = np.cross(target_eigenvecs[:, 0], target_eigenvecs[:, 1])
    
    print(f"Source PCA axes:\n{source_eigenvecs}")
    print(f"Target PCA axes:\n{target_eigenvecs}")
    
    # Calculate rotation to align PCA axes
    rotation_matrix = target_eigenvecs @ source_eigenvecs.T
    
    # Calculate scaling based on eigenvalues
    source_scale = np.sqrt(source_eigenvals[source_order])
    target_scale = np.sqrt(target_eigenvals[target_order])
    
    # Normalize to box_size
    source_scale_factor = box_size / np.max(source_scale)
    target_scale_factor = box_size / np.max(target_scale)
    
    print(f"Source scales: {source_scale * source_scale_factor}")
    print(f"Target scales: {target_scale * target_scale_factor}")
    
    # Create transformation matrix
    # 1. Center source
    center_matrix = np.eye(4)
    center_matrix[:3, 3] = -source_center
    
    # 2. Scale source
    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = source_scale_factor
    scale_matrix[1, 1] = source_scale_factor
    scale_matrix[2, 2] = source_scale_factor
    
    # 3. Rotate to align with target
    rotation_4x4 = np.eye(4)
    rotation_4x4[:3, :3] = rotation_matrix
    
    # 4. Translate to target center
    translate_matrix = np.eye(4)
    translate_matrix[:3, 3] = target_center
    
    # Combine transformations
    transformation = translate_matrix @ rotation_4x4 @ scale_matrix @ center_matrix
    
    end_time = time.time()
    
    # Calculate fitness
    center_distance = np.linalg.norm(target_center - source_center)
    scale_ratio = target_scale_factor / source_scale_factor
    fitness = 1.0 / (1.0 + center_distance + abs(scale_ratio - 1.0))
    
    return {
        'transformation': transformation,
        'fitness': fitness,
        'inlier_rmse': center_distance,
        'correspondence_set': [],
        'computation_time': end_time - start_time,
        'method': 'pca_bounding_box'
    }


def create_hybrid_bounding_box_registration(source_pcd, target_pcd, box_size=1.0):
    """
    Hybrid registration: Bounding box alignment + ICP refinement
    
    Args:
        source_pcd: source point cloud
        target_pcd: target point cloud
        box_size: size of the cubic bounding box
    
    Returns:
        dict: registration results
    """
    print("Running Hybrid Bounding Box + ICP registration...")
    start_time = time.time()
    
    # Step 1: Get initial alignment using PCA bounding box
    pca_result = create_pca_bounding_box_registration(source_pcd, target_pcd, box_size)
    initial_transform = pca_result['transformation']
    
    # Step 2: Apply initial transformation to source
    source_transformed = apply_transformation(source_pcd, initial_transform)
    
    # Step 3: Refine with ICP
    print("Refining with ICP...")
    try:
        # Estimate normals if not present
        if not source_transformed.has_normals():
            source_transformed.estimate_normals()
        if not target_pcd.has_normals():
            target_pcd.estimate_normals()
        
        # Run ICP with the initial transformation
        icp_result = o3d.pipelines.registration.registration_icp(
            source_transformed, target_pcd, 0.05, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
        )
        
        # Combine transformations
        final_transform = icp_result.transformation @ initial_transform
        
        end_time = time.time()
        
        return {
            'transformation': final_transform,
            'fitness': icp_result.fitness,
            'inlier_rmse': icp_result.inlier_rmse,
            'correspondence_set': icp_result.correspondence_set,
            'computation_time': end_time - start_time,
            'method': 'hybrid_bounding_box_icp',
            'initial_transform': initial_transform,
            'icp_transform': icp_result.transformation
        }
        
    except Exception as e:
        print(f"ICP refinement failed: {e}")
        # Return the initial PCA result
        end_time = time.time()
        return {
            'transformation': initial_transform,
            'fitness': pca_result['fitness'],
            'inlier_rmse': pca_result['inlier_rmse'],
            'correspondence_set': [],
            'computation_time': end_time - start_time,
            'method': 'hybrid_bounding_box_icp_fallback'
        }


def test_icp_registration(source_pcd, target_pcd, max_iterations=50, threshold=0.02):
    """
    Test ICP registration algorithm
    
    Args:
        source_pcd: source point cloud
        target_pcd: target point cloud
        max_iterations: maximum ICP iterations
        threshold: distance threshold for ICP
    
    Returns:
        dict: registration results
    """
    print("Running ICP registration...")
    start_time = time.time()
    
    # Estimate normals if not present
    if not source_pcd.has_normals():
        source_pcd.estimate_normals()
    if not target_pcd.has_normals():
        target_pcd.estimate_normals()
    
    # Run ICP with multiple initializations for better results
    best_result = None
    best_fitness = 0
    
    # Try different initial transformations
    initial_transforms = [
        np.eye(4),  # Identity
        # Add some random initializations
        *[generate_random_transformation(rotation_range=10, translation_range=0.1)[0] 
          for _ in range(5)]
    ]
    
    for init_transform in initial_transforms:
        try:
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, threshold, init_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
            )
            
            if result.fitness > best_fitness:
                best_fitness = result.fitness
                best_result = result
        except Exception as e:
            print(f"ICP failed with init transform: {e}")
            continue
    
    # If no result was found, use a simple ICP
    if best_result is None:
        print("All ICP initializations failed, trying simple ICP...")
        best_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
    
    end_time = time.time()
    
    return {
        'transformation': best_result.transformation,
        'fitness': best_result.fitness,
        'inlier_rmse': best_result.inlier_rmse,
        'correspondence_set': best_result.correspondence_set,
        'computation_time': end_time - start_time
    }


def test_robust_icp_registration(source_pcd, target_pcd, max_iterations=50, threshold=0.02):
    """
    Test robust ICP registration using RANSAC for initial alignment
    
    Args:
        source_pcd: source point cloud
        target_pcd: target point cloud
        max_iterations: maximum ICP iterations
        threshold: distance threshold for ICP
    
    Returns:
        dict: registration results
    """
    print("Running Robust ICP registration (RANSAC + ICP)...")
    start_time = time.time()
    
    # Estimate normals if not present
    if not source_pcd.has_normals():
        source_pcd.estimate_normals()
    if not target_pcd.has_normals():
        target_pcd.estimate_normals()
    
    try:
        # First, try RANSAC for initial alignment
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
        )
        
        # RANSAC registration
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pcd, target_pcd, source_fpfh, target_fpfh, True,
            0.05, o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        # Refine with ICP
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, ransac_result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
        
        end_time = time.time()
        
        return {
            'transformation': icp_result.transformation,
            'fitness': icp_result.fitness,
            'inlier_rmse': icp_result.inlier_rmse,
            'correspondence_set': icp_result.correspondence_set,
            'computation_time': end_time - start_time,
            'ransac_transformation': ransac_result.transformation
        }
    
    except Exception as e:
        print(f"RANSAC failed: {e}")
        print("Falling back to standard ICP...")
        
        # Fallback to standard ICP
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
        
        end_time = time.time()
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set': result.correspondence_set,
            'computation_time': end_time - start_time,
            'ransac_transformation': None
        }


def test_global_registration(source_pcd, target_pcd, max_iterations=50, threshold=0.02):
    """
    Test global registration using FPFH features
    
    Args:
        source_pcd: source point cloud
        target_pcd: target point cloud
        max_iterations: maximum ICP iterations
        threshold: distance threshold for ICP
    
    Returns:
        dict: registration results
    """
    print("Running Global registration (FPFH + RANSAC)...")
    start_time = time.time()
    
    # Estimate normals if not present
    if not source_pcd.has_normals():
        source_pcd.estimate_normals()
    if not target_pcd.has_normals():
        target_pcd.estimate_normals()
    
    try:
        # Compute FPFH features
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
        )
        
        # Global registration
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_pcd, target_pcd, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                division_factor=1.5,
                use_absolute_scale=False,
                decrease_mu=False,
                maximum_correspondence_distance=0.05,
                iteration_number=64,
                tuple_scale=0.95,
                maximum_tuple_count=1000
            )
        )
        
        end_time = time.time()
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set': result.correspondence_set,
            'computation_time': end_time - start_time
        }
    
    except Exception as e:
        print(f"Global registration failed: {e}")
        print("Falling back to standard ICP...")
        
        # Fallback to standard ICP
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
        
        end_time = time.time()
        
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set': result.correspondence_set,
            'computation_time': end_time - start_time
        }


def visualize_registration(source_pcd, target_pcd, estimated_pcd, title="Registration Result"):
    """Visualize registration results with labels"""
    # Color the point clouds
    source_pcd.paint_uniform_color([1, 0, 0])  # Red
    target_pcd.paint_uniform_color([0, 1, 0])  # Green
    estimated_pcd.paint_uniform_color([0, 0, 1])  # Blue
    
    # Create coordinate frames for each point cloud
    source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    estimated_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    # Position frames at the center of each point cloud
    source_center = source_pcd.get_center()
    target_center = target_pcd.get_center()
    estimated_center = estimated_pcd.get_center()
    
    source_frame.translate(source_center)
    target_frame.translate(target_center)
    estimated_frame.translate(estimated_center)
    
    # Create text labels (using spheres as placeholders since Open3D doesn't have direct text support)
    def create_label_sphere(center, color, size=5):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        sphere.paint_uniform_color(color)
        sphere.translate(center)
        return sphere
    
    # Create label spheres
    source_label = create_label_sphere(source_center + [0, 0, 30], [1, 0, 0], 8)  # Red sphere for "Source"
    target_label = create_label_sphere(target_center + [0, 0, 30], [0, 1, 0], 8)  # Green sphere for "Target"
    estimated_label = create_label_sphere(estimated_center + [0, 0, 30], [0, 0, 1], 8)  # Blue sphere for "Estimated"
    
    # Create legend
    legend_items = []
    legend_positions = [
        [50, 50, 0],   # Source legend
        [50, 100, 0],  # Target legend
        [50, 150, 0]   # Estimated legend
    ]
    legend_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    legend_names = ["Source (Red)", "Target (Green)", "Estimated (Blue)"]
    
    for i, (pos, color, name) in enumerate(zip(legend_positions, legend_colors, legend_names)):
        legend_sphere = create_label_sphere(pos, color, 5)
        legend_items.append(legend_sphere)
    
    # Create visualization with all elements
    geometries = [
        source_pcd, target_pcd, estimated_pcd,
        source_frame, target_frame, estimated_frame,
        source_label, target_label, estimated_label
    ] + legend_items
    
    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=800)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set up the view
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    # Add text information
    print(f"\nVisualization Legend:")
    print(f"  🔴 Red: Source point cloud (original)")
    print(f"  🟢 Green: Target point cloud (transformed)")
    print(f"  🔵 Blue: Estimated point cloud (registration result)")
    print(f"  📐 Coordinate frames show orientation")
    print(f"  🎯 Large colored spheres mark centers")
    
    vis.run()
    vis.destroy_window()


def create_registration_summary_visualization(source_pcd, target_pcd, estimated_pcd, 
                                            ground_truth_transform, estimated_transform, 
                                            rotation_error, translation_error, title="Registration Summary"):
    """Create a comprehensive visualization showing registration quality"""
    
    # Create copies to avoid modifying originals
    source_vis = o3d.geometry.PointCloud(source_pcd)
    target_vis = o3d.geometry.PointCloud(target_pcd)
    estimated_vis = o3d.geometry.PointCloud(estimated_pcd)
    
    # Color the point clouds with more distinct colors
    source_vis.paint_uniform_color([1, 0, 0])  # Bright Red
    target_vis.paint_uniform_color([0, 1, 0])  # Bright Green
    estimated_vis.paint_uniform_color([0, 0, 1])  # Bright Blue
    
    # Create coordinate frames
    source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    estimated_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    # Position frames at centers
    source_center = source_vis.get_center()
    target_center = target_vis.get_center()
    estimated_center = estimated_vis.get_center()
    
    source_frame.translate(source_center)
    target_frame.translate(target_center)
    estimated_frame.translate(estimated_center)
    
    # Create connection lines to show alignment quality
    def create_line(start, end, color):
        points = np.array([start, end])
        lines = [[0, 1]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color])
        return line_set
    
    # Create lines showing the transformation
    source_to_target = create_line(source_center, target_center, [1, 1, 0])  # Yellow
    source_to_estimated = create_line(source_center, estimated_center, [1, 0, 1])  # Magenta
    
    # Create visualization
    geometries = [
        source_vis, target_vis, estimated_vis,
        source_frame, target_frame, estimated_frame,
        source_to_target, source_to_estimated
    ]
    
    # Print detailed information
    print(f"\n{'='*60}")
    print(f"REGISTRATION QUALITY ASSESSMENT")
    print(f"{'='*60}")
    print(f"Rotation Error: {rotation_error:.3f}°")
    print(f"Translation Error: {translation_error:.6f}m")
    print(f"Registration Quality: {'EXCELLENT' if rotation_error < 5 and translation_error < 0.01 else 'GOOD' if rotation_error < 15 and translation_error < 0.1 else 'POOR'}")
    print(f"\nVisualization Guide:")
    print(f"  🔴 Red: Source (original point cloud)")
    print(f"  🟢 Green: Target (ground truth transformed)")
    print(f"  🔵 Blue: Estimated (registration result)")
    print(f"  🟡 Yellow line: Source → Target (ground truth)")
    print(f"  🟣 Magenta line: Source → Estimated (registration)")
    print(f"  📐 Coordinate frames: Show orientation")
    print(f"\nGood registration: Blue should overlap with Green")
    print(f"Poor registration: Blue will be far from Green")
    
    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1400, height=900)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set up the view
    render_option = vis.get_render_option()
    render_option.point_size = 8.0  # Much larger point size for normalized scale
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # Slightly lighter background
    render_option.line_width = 6.0  # Thicker lines for better visibility
    
    # Add some spacing between point clouds if they're too close
    center_distance = np.linalg.norm(target_center - source_center)
    if center_distance < 0.5:  # If point clouds are very close (adjusted for normalized scale)
        print("⚠️  Point clouds are very close - they may appear to overlap")
        print("   Try rotating the view to see them better")
    
    # Print center positions for debugging
    print(f"Point cloud centers:")
    print(f"  Source (Red): {source_center}")
    print(f"  Target (Green): {target_center}")
    print(f"  Estimated (Blue): {estimated_center}")
    print(f"  Distance Source→Target: {np.linalg.norm(target_center - source_center):.2f}")
    print(f"  Distance Source→Estimated: {np.linalg.norm(estimated_center - source_center):.2f}")
    
    vis.run()
    vis.destroy_window()


def create_side_by_side_visualization(source_pcd, target_pcd, estimated_pcd, title="Side-by-Side Comparison"):
    """Create a side-by-side visualization to better see overlapping point clouds"""
    
    # Create copies and color them
    source_vis = o3d.geometry.PointCloud(source_pcd)
    target_vis = o3d.geometry.PointCloud(target_pcd)
    estimated_vis = o3d.geometry.PointCloud(estimated_pcd)
    
    source_vis.paint_uniform_color([1, 0, 0])  # Red
    target_vis.paint_uniform_color([0, 1, 0])  # Green
    estimated_vis.paint_uniform_color([0, 0, 1])  # Blue
    
    # Get bounding boxes to calculate spacing
    source_bbox = source_vis.get_axis_aligned_bounding_box()
    target_bbox = target_vis.get_axis_aligned_bounding_box()
    estimated_bbox = estimated_vis.get_axis_aligned_bounding_box()
    
    # Calculate spacing based on the largest extent
    max_extent = max(
        np.max(source_bbox.get_extent()),
        np.max(target_bbox.get_extent()),
        np.max(estimated_bbox.get_extent())
    )
    spacing = max_extent * 2.0  # Better separation for normalized scale
    
    # Translate point clouds to be side by side
    target_vis.translate([spacing, 0, 0])
    estimated_vis.translate([spacing * 2, 0, 0])
    
    # Create coordinate frames
    source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=source_vis.get_center())
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=target_vis.get_center())
    estimated_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=estimated_vis.get_center())
    
    # Create labels (spheres)
    def create_label_sphere(center, color, size=0.05):  # Smaller spheres for normalized scale
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        sphere.paint_uniform_color(color)
        sphere.translate(center + [0, 0, 50])
        return sphere
    
    source_label = create_label_sphere(source_vis.get_center(), [1, 0, 0])
    target_label = create_label_sphere(target_vis.get_center(), [0, 1, 0])
    estimated_label = create_label_sphere(estimated_vis.get_center(), [0, 0, 1])
    
    # Create visualization
    geometries = [
        source_vis, target_vis, estimated_vis,
        source_frame, target_frame, estimated_frame,
        source_label, target_label, estimated_label
    ]
    
    print(f"\n{'='*60}")
    print(f"SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")
    print(f"🔴 Left: Source (original)")
    print(f"🟢 Middle: Target (transformed)")
    print(f"🔵 Right: Estimated (registration result)")
    print(f"📐 Coordinate frames show orientation")
    print(f"🎯 Large spheres mark centers")
    
    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1600, height=800)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set up the view
    render_option = vis.get_render_option()
    render_option.point_size = 8.0  # Much larger point size for normalized scale
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    vis.run()
    vis.destroy_window()


def save_results(results, output_file):
    """Save registration results to JSON file"""
    def convert_numpy(obj):
        """Recursively convert numpy arrays and Open3D objects to lists"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__class__') and 'Vector' in obj.__class__.__name__:
            # Handle Open3D Vector types
            return list(obj)
        elif hasattr(obj, '__class__') and 'ndarray' in str(type(obj)):
            # Handle any remaining numpy types
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(convert_numpy(item) for item in obj)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            # Handle other iterable types
            try:
                return [convert_numpy(item) for item in obj]
            except:
                return str(obj)
        else:
            return obj
    
    json_results = convert_numpy(results)
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def main():
    """Main function to run registration test"""
    import sys
    
    # File paths
    ply_file = Path("/home/mht/PycharmProjects/pusht_real/test/point_clouds/t_pcd.ply")
    output_dir = Path("/home/mht/PycharmProjects/pusht_real/test/point_clouds/registration_results")
    output_dir.mkdir(exist_ok=True)
    
    # Check if user wants to test a specific algorithm or force side-by-side view
    test_single_algorithm = len(sys.argv) > 1
    force_side_by_side = '--side-by-side' in sys.argv or '-s' in sys.argv
    
    if test_single_algorithm:
        # Filter out visualization flags
        algorithm_name = [arg for arg in sys.argv[1:] if not arg.startswith('--') and not arg.startswith('-')][0]
        print(f"Testing only: {algorithm_name}")
    
    if force_side_by_side:
        print("Forcing side-by-side visualization for all algorithms")
    
    try:
        # Load point cloud with normalization
        source_pcd, scale_factor, original_center = load_point_cloud(ply_file, normalize=True, target_size=1.0)
        
        # Generate random transformation (adjusted for normalized point cloud)
        print("\nGenerating random transformation...")
        transformation_matrix, rotation_angles, translation = generate_random_transformation(
            rotation_range=15,  # degrees - smaller for better algorithm performance
            translation_range=0.1  # units - smaller for normalized point cloud
        )
        
        print(f"Applied rotation (degrees): {rotation_angles}")
        print(f"Applied translation (normalized): {translation}")
        print(f"Applied translation (original scale): {convert_to_original_scale(translation, scale_factor)}")
        
        # Apply transformation to create target point cloud
        target_pcd = apply_transformation(source_pcd, transformation_matrix)
        
        # Print some debugging information
        print(f"\nPoint cloud statistics:")
        print(f"  Source points: {len(source_pcd.points)}")
        print(f"  Target points: {len(target_pcd.points)}")
        print(f"  Source bounds: {source_pcd.get_axis_aligned_bounding_box()}")
        print(f"  Target bounds: {target_pcd.get_axis_aligned_bounding_box()}")
        
        # Test transformation by checking a few sample points
        source_points = np.asarray(source_pcd.points)
        target_points = np.asarray(target_pcd.points)
        print(f"\nTransformation verification:")
        print(f"  First source point: {source_points[0]}")
        print(f"  First target point: {target_points[0]}")
        print(f"  Expected target point: {transformation_matrix @ np.append(source_points[0], 1)}")
        
        # Check if transformation was applied correctly
        expected_target = transformation_matrix[:3, :3] @ source_points[0] + transformation_matrix[:3, 3]
        print(f"  Manual calculation: {expected_target}")
        print(f"  Difference: {np.linalg.norm(target_points[0] - expected_target)}")
        
        # Test different registration algorithms
        all_algorithms = {
            'Bounding_Box': create_bounding_box_registration,
            'PCA_Bounding_Box': create_pca_bounding_box_registration,
            'Hybrid_Bounding_Box': create_hybrid_bounding_box_registration,
            'ICP': test_icp_registration,
            'Robust_ICP': test_robust_icp_registration,
            'Global_Registration': test_global_registration
        }
        
        # Filter algorithms if testing single one
        if test_single_algorithm:
            if algorithm_name in all_algorithms:
                algorithms = {algorithm_name: all_algorithms[algorithm_name]}
            else:
                print(f"Available algorithms: {list(all_algorithms.keys())}")
                print(f"Unknown algorithm: {algorithm_name}")
                return
        else:
            algorithms = all_algorithms
        
        results = {
            'ground_truth': {
                'transformation': transformation_matrix,
                'rotation_angles': rotation_angles,
                'translation': translation
            },
            'algorithms': {}
        }
        
        for algo_name, algo_func in algorithms.items():
            print(f"\n{'='*50}")
            print(f"Testing {algo_name}")
            print(f"{'='*50}")
            
            # Run registration with appropriate parameters (adjusted for normalized point cloud)
            if algo_name == 'ICP':
                algo_result = algo_func(source_pcd, target_pcd, threshold=0.01)  # Appropriate for normalized scale
            elif algo_name in ['Bounding_Box', 'PCA_Bounding_Box', 'Hybrid_Bounding_Box']:
                algo_result = algo_func(source_pcd, target_pcd, box_size=1.0)
            else:
                algo_result = algo_func(source_pcd, target_pcd)
            
            # Evaluate results with symmetry consideration
            t_symmetries = get_t_shape_symmetries()
            evaluation = evaluate_transformation(
                algo_result['transformation'], 
                transformation_matrix,
                consider_symmetry=True,
                symmetry_axes=t_symmetries
            )
            
            # Store results
            results['algorithms'][algo_name] = {
                'registration_result': algo_result,
                'evaluation': evaluation
            }
            
            # Print results
            print(f"\n{algo_name} Results:")
            print(f"  Fitness: {algo_result['fitness']:.4f}")
            print(f"  Inlier RMSE: {algo_result['inlier_rmse']:.6f}")
            print(f"  Computation time: {algo_result['computation_time']:.3f}s")
            print(f"  Rotation error: {evaluation['rotation_error_degrees']:.3f}°")
            print(f"  Translation error: {evaluation['translation_error']:.6f}m")
            print(f"  Registration correct: {evaluation['is_correct']}")
            
            # Print symmetry information
            if evaluation['is_symmetric']:
                symmetry_names = ['Identity', 'Z-180°', 'X-180°', 'Y-180°']
                sym_name = symmetry_names[evaluation['symmetry_index']]
                print(f"  ⚠️  Best match is SYMMETRIC: {sym_name}")
                print(f"     This means the algorithm found a valid but flipped orientation")
            else:
                print(f"  ✅ Best match is direct (no symmetry)")
            
            # Print transformation matrices for debugging
            print(f"  Ground truth transform:\n{transformation_matrix}")
            print(f"  Estimated transform:\n{algo_result['transformation']}")
            
            # Debug: Check if estimated transform is identity
            estimated_is_identity = np.allclose(algo_result['transformation'], np.eye(4), atol=1e-6)
            if estimated_is_identity:
                print(f"  ⚠️  WARNING: Estimated transform is IDENTITY (no transformation applied)")
                print(f"     This suggests the algorithm failed to find any transformation")
                print(f"     The 'direct match' result is misleading - algorithm didn't work!")
            else:
                # Check if the algorithm actually found a reasonable transformation
                estimated_transform = np.array(algo_result['transformation'])  # Make a copy to avoid read-only issues
                estimated_rotation = R.from_matrix(estimated_transform[:3, :3])
                estimated_angle = np.linalg.norm(estimated_rotation.as_rotvec()) * 180 / np.pi
                estimated_translation = np.linalg.norm(estimated_transform[:3, 3])
                
                print(f"  ✅ Algorithm found a transformation:")
                print(f"     Rotation: {estimated_angle:.2f}°")
                print(f"     Translation: {estimated_translation:.4f}m")
                
                # Check if it's reasonable compared to ground truth
                gt_transform = np.array(transformation_matrix)  # Make a copy to avoid read-only issues
                gt_rotation = R.from_matrix(gt_transform[:3, :3])
                gt_angle = np.linalg.norm(gt_rotation.as_rotvec()) * 180 / np.pi
                gt_translation = np.linalg.norm(gt_transform[:3, 3])
                
                print(f"  Ground truth was:")
                print(f"     Rotation: {gt_angle:.2f}°")
                print(f"     Translation: {gt_translation:.4f}m")
                
                # Calculate relative errors
                rotation_ratio = estimated_angle / gt_angle if gt_angle > 0.1 else 0
                translation_ratio = estimated_translation / gt_translation if gt_translation > 0.01 else 0
                
                print(f"  Relative performance:")
                print(f"     Rotation ratio: {rotation_ratio:.2f} (1.0 = perfect)")
                print(f"     Translation ratio: {translation_ratio:.2f} (1.0 = perfect)")
                
                if 0.5 < rotation_ratio < 2.0 and 0.5 < translation_ratio < 2.0:
                    print(f"  🎯 Algorithm performed reasonably well!")
                elif rotation_ratio < 0.1 or translation_ratio < 0.1:
                    print(f"  ⚠️  Algorithm found much smaller transformation than expected")
                else:
                    print(f"  ❌ Algorithm found very different transformation")
            
            # Visualize results
            estimated_pcd = apply_transformation(source_pcd, algo_result['transformation'])
            
            # Use enhanced visualization for better understanding
            create_registration_summary_visualization(
                source_pcd, target_pcd, estimated_pcd,
                transformation_matrix, algo_result['transformation'],
                evaluation['rotation_error_degrees'], evaluation['translation_error'],
                f"{algo_name} Registration Quality"
            )
            
            # Also show side-by-side comparison if point clouds are close or forced
            center_distance = np.linalg.norm(target_pcd.get_center() - source_pcd.get_center())
            if center_distance < 0.5 or force_side_by_side:  # If point clouds are close or forced (adjusted for normalized scale)
                if center_distance < 0.5:
                    print(f"\nPoint clouds are close (distance: {center_distance:.2f})")
                print("Showing side-by-side comparison for better visibility...")
                create_side_by_side_visualization(
                    source_pcd, target_pcd, estimated_pcd,
                    f"{algo_name} Side-by-Side Comparison"
                )
            
            # Show detailed symmetry analysis for algorithms that might be affected
            if algo_name in ['Robust_ICP', 'Global_Registration', 'PCA_Bounding_Box']:
                print(f"\nDetailed symmetry analysis for {algo_name}:")
                analyze_symmetry_impact(
                    source_pcd, target_pcd, 
                    algo_result['transformation'], 
                    transformation_matrix
                )
                
                # Additional analysis: Check if the algorithm actually worked
                print(f"\nAlgorithm Performance Analysis:")
                if estimated_is_identity:
                    print(f"  ❌ Algorithm FAILED: Returned identity transformation")
                    print(f"     This means no registration was performed")
                    print(f"     The algorithm couldn't find any valid transformation")
                else:
                    # Check if the estimated transform is reasonable
                    estimated_transform = np.array(algo_result['transformation'])  # Make a copy to avoid read-only issues
                    estimated_rotation = R.from_matrix(estimated_transform[:3, :3])
                    estimated_angle = np.linalg.norm(estimated_rotation.as_rotvec()) * 180 / np.pi
                    estimated_translation = np.linalg.norm(estimated_transform[:3, 3])
                    
                    print(f"  Estimated rotation: {estimated_angle:.2f}°")
                    print(f"  Estimated translation: {estimated_translation:.4f}m")
                    
                    if estimated_angle < 1.0 and estimated_translation < 0.01:
                        print(f"  ⚠️  Algorithm returned near-identity (minimal transformation)")
                    else:
                        print(f"  ✅ Algorithm found a significant transformation")
        
        # Save results (skip for now due to serialization issues)
        try:
            results_file = output_dir / f"registration_test_{int(time.time())}.json"
            save_results(results, results_file)
        except Exception as e:
            print(f"Could not save results to JSON: {e}")
            print("Results are displayed above.")
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY WITH SYMMETRY ANALYSIS")
        print(f"{'='*60}")
        for algo_name, algo_data in results['algorithms'].items():
            eval_data = algo_data['evaluation']
            print(f"{algo_name}:")
            print(f"  Success: {'✓' if eval_data['is_correct'] else '✗'}")
            print(f"  Rotation error: {eval_data['rotation_error_degrees']:.3f}°")
            print(f"  Translation error: {eval_data['translation_error']:.6f}m")
            print(f"  Time: {algo_data['registration_result']['computation_time']:.3f}s")
            
            # Show symmetry information
            if eval_data['is_symmetric']:
                symmetry_names = ['Identity', 'Z-180°', 'X-180°', 'Y-180°']
                sym_name = symmetry_names[eval_data['symmetry_index']]
                print(f"  Symmetry: ⚠️  {sym_name} (flipped orientation)")
            else:
                print(f"  Symmetry: ✅ Direct match")
        
        print(f"\n{'='*60}")
        print("SYMMETRY EXPLANATION")
        print(f"{'='*60}")
        print("For T-shaped objects, multiple orientations can look 'correct':")
        print("  • Identity: Direct match (best)")
        print("  • Z-180°: Flipped vertically (still valid for T-shape)")
        print("  • X-180°: Flipped horizontally (still valid for T-shape)")
        print("  • Y-180°: Flipped horizontally (still valid for T-shape)")
        print("\n⚠️  = Algorithm found a valid but flipped orientation")
        print("✅ = Algorithm found the exact orientation")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

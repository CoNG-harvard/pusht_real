#!/usr/bin/env python3
"""
Diagnostic script to understand why registration algorithms are failing
"""

import numpy as np
import open3d as o3d
from pathlib import Path

def diagnose_point_cloud(file_path):
    """Diagnose point cloud characteristics"""
    print("="*60)
    print("POINT CLOUD DIAGNOSTICS")
    print("="*60)
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(file_path))
    
    print(f"Point cloud loaded: {len(pcd.points)} points")
    
    # Basic statistics
    points = np.asarray(pcd.points)
    print(f"Point cloud bounds:")
    print(f"  X: [{np.min(points[:, 0]):.2f}, {np.max(points[:, 0]):.2f}]")
    print(f"  Y: [{np.min(points[:, 1]):.2f}, {np.max(points[:, 1]):.2f}]")
    print(f"  Z: [{np.min(points[:, 2]):.2f}, {np.max(points[:, 2]):.2f}]")
    
    # Calculate extents
    extents = np.max(points, axis=0) - np.min(points, axis=0)
    print(f"Extents: {extents}")
    print(f"Max extent: {np.max(extents):.2f}")
    
    # Check if point cloud has normals
    print(f"Has normals: {pcd.has_normals()}")
    
    # Estimate normals and check
    pcd.estimate_normals()
    print(f"Normals estimated: {pcd.has_normals()}")
    
    # Check point density
    bbox = pcd.get_axis_aligned_bounding_box()
    volume = bbox.volume()
    density = len(pcd.points) / volume if volume > 0 else 0
    print(f"Point density: {density:.2f} points per unit volume")
    
    # Check for duplicate points
    unique_points = np.unique(points, axis=0)
    print(f"Unique points: {len(unique_points)} / {len(points)} ({len(unique_points)/len(points)*100:.1f}%)")
    
    # Check point distribution
    center = pcd.get_center()
    distances = np.linalg.norm(points - center, axis=1)
    print(f"Distance from center:")
    print(f"  Mean: {np.mean(distances):.2f}")
    print(f"  Std: {np.std(distances):.2f}")
    print(f"  Min: {np.min(distances):.2f}")
    print(f"  Max: {np.max(distances):.2f}")
    
    return pcd

def test_simple_registration():
    """Test with a very simple case"""
    print("\n" + "="*60)
    print("SIMPLE REGISTRATION TEST")
    print("="*60)
    
    # Load point cloud
    ply_file = Path("/home/mht/PycharmProjects/pusht_real/test/point_clouds/t_pcd.ply")
    source_pcd = o3d.io.read_point_cloud(str(ply_file))
    
    # Create a simple transformation (just translation)
    simple_transform = np.eye(4)
    simple_transform[0, 3] = 10.0  # Move 10 units in X
    simple_transform[1, 3] = 5.0   # Move 5 units in Y
    
    print(f"Simple transformation (translation only):")
    print(simple_transform)
    
    # Apply transformation
    target_pcd = o3d.geometry.PointCloud(source_pcd)
    target_pcd.transform(simple_transform)
    
    print(f"Source center: {source_pcd.get_center()}")
    print(f"Target center: {target_pcd.get_center()}")
    
    # Try ICP with very lenient parameters
    print("\nTesting ICP with lenient parameters...")
    
    # Estimate normals
    source_pcd.estimate_normals()
    target_pcd.estimate_normals()
    
    # Try different thresholds
    thresholds = [0.1, 0.5, 1.0, 2.0]
    
    for threshold in thresholds:
        print(f"\nICP with threshold {threshold}:")
        try:
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            
            print(f"  Fitness: {result.fitness:.4f}")
            print(f"  Inlier RMSE: {result.inlier_rmse:.6f}")
            print(f"  Transformation:")
            print(f"    {result.transformation}")
            
            # Check if it's close to the expected transformation
            expected = simple_transform
            estimated = result.transformation
            
            # Calculate error
            error_transform = np.linalg.inv(expected) @ estimated
            rotation_error = np.linalg.norm(error_transform[:3, :3] - np.eye(3))
            translation_error = np.linalg.norm(error_transform[:3, 3])
            
            print(f"  Rotation error: {rotation_error:.6f}")
            print(f"  Translation error: {translation_error:.6f}")
            
            if rotation_error < 0.1 and translation_error < 0.1:
                print(f"  ✅ SUCCESS!")
                break
            else:
                print(f"  ❌ Failed")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def main():
    """Main diagnostic function"""
    ply_file = Path("/home/mht/PycharmProjects/pusht_real/test/point_clouds/t_pcd.ply")
    
    if not ply_file.exists():
        print(f"Point cloud file not found: {ply_file}")
        return
    
    # Diagnose point cloud
    pcd = diagnose_point_cloud(ply_file)
    
    # Test simple registration
    test_simple_registration()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("If ICP fails even with simple translation, the issue might be:")
    print("1. Point cloud has too many duplicate points")
    print("2. Point cloud lacks distinctive features")
    print("3. Point cloud is too dense or too sparse")
    print("4. Point cloud has unusual geometry")

if __name__ == "__main__":
    main()

import trimesh
import numpy as np
import open3d as o3d

# Load STL file
mesh = trimesh.load_mesh('t_shape.stl')

# Sample points uniformly on the surface
num_points = 10000  # Adjust based on desired density
points, _ = trimesh.sample.sample_surface(mesh, num_points)

# Optional: convert to Open3D point cloud for visualization or saving
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Visualize
o3d.visualization.draw_geometries([pcd])

# Optional: save to PLY/PCD format
o3d.io.write_point_cloud("t_pcd.ply", pcd)

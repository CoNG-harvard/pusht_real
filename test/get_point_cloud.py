import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Configure depth stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Create Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window("Live Point Cloud from Intel RealSense D405")

# Create a point cloud object
pcd = o3d.geometry.PointCloud()

# Use RealSense's point cloud generator
pc = rs.pointcloud()

# Flag to check if the point cloud has been added to the visualizer
added = False

# Frame counter to limit update rate
frame_count = 0
update_interval = 5  # Update every 5 frames

try:
    while True:
        # Wait for a coherent pair of frames: depth
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # Generate point cloud using RealSense's point cloud generator
        points = pc.calculate(depth_frame)
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # Reshape to Nx3 array

        # Update Open3D point cloud
        pcd.points = o3d.utility.Vector3dVector(verts)

        # Add the point cloud to the visualizer if not already added
        if not added:
            vis.add_geometry(pcd)
            added = True

        # Limit update rate
        frame_count += 1
        if frame_count % update_interval == 0:
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

finally:
    # Stop streaming
    pipeline.stop()
    vis.destroy_window()
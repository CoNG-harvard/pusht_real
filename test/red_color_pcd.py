import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create an Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window("Red Color Point Cloud from Intel RealSense D405")

# Create a point cloud object
pcd = o3d.geometry.PointCloud()

# Use RealSense's point cloud generator
pc = rs.pointcloud()

# Align depth frame to color frame
align_to = rs.stream.color
align = rs.align(align_to)

# Flag to check if the point cloud has been added to the visualizer
added = False

# Define the red color range in RGB
red_lower = np.array([100, 0, 0])  # Lower bound for red (R > 100, G < 50, B < 50)
red_upper = np.array([255, 50, 50])  # Upper bound for red

update_interval = 5
frame_count = 0

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to the color frame
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Generate point cloud
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # Reshape to Nx3 array
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # Texture coordinates

        # Get color image
        color_image = np.asanyarray(color_frame.get_data())

        # Map texture coordinates to color image
        texcoords = (texcoords * [color_image.shape[1], color_image.shape[0]]).astype(int)
        texcoords[texcoords < 0] = 0
        texcoords[:, 0][texcoords[:, 0] >= color_image.shape[1]] = color_image.shape[1] - 1
        texcoords[:, 1][texcoords[:, 1] >= color_image.shape[0]] = color_image.shape[0] - 1

        # Get colors for each point
        colors = color_image[texcoords[:, 1], texcoords[:, 0]] / 255.0  # Normalize to [0, 1]

        # Filter red points
        red_mask = np.all((colors >= red_lower / 255.0) & (colors <= red_upper / 255.0), axis=1)
        red_points = verts[red_mask]
        red_colors = colors[red_mask]

        # Update Open3D point cloud
        pcd.points = o3d.utility.Vector3dVector(red_points)
        pcd.colors = o3d.utility.Vector3dVector(red_colors)

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
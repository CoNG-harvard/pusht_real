import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

# Load the target object point cloud (replace with your file path)
target_point_cloud = o3d.io.read_point_cloud("t_pcd.ply")  # Ensure this is in the correct coordinate system

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create an align object to align depth frames to color frames
align = rs.align(rs.stream.color)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert color image to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Isolate red pixels
        red_mask = cv2.inRange(color_image, (0, 0, 100), (100, 100, 255))
        red_pixels = cv2.bitwise_and(color_image, color_image, mask=red_mask)

        # Create a point cloud from the depth frame
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        vtx = np.array([list(v) for v in vtx])  # Convert to numpy array

        # Filter the point cloud using the red mask
        red_points = vtx[red_mask.reshape(-1) == 255]

        # Remove invalid points (where depth is 0)
        red_points = red_points[red_points[:, 2] != 0]

        if len(red_points) == 0:
            print("No red points found.")
            continue

        # Convert the filtered point cloud to Open3D format
        scene_point_cloud = o3d.geometry.PointCloud()
        scene_point_cloud.points = o3d.utility.Vector3dVector(red_points)

        # Perform ICP alignment
        threshold = 0.02  # Maximum correspondence distance
        trans_init = np.identity(4)  # Initial transformation (identity matrix)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            scene_point_cloud, target_point_cloud, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        # Get the transformation matrix
        transformation_matrix = reg_p2p.transformation
        print("Transformation Matrix (Pose):\n", transformation_matrix)

        # Visualize the red pixels
        cv2.imshow('Red Pixels', red_pixels)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
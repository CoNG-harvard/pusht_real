import pyrealsense2 as rs

# Create a pipeline and start it
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the camera
profile = pipeline.start(config)

# Get stream profile and camera intrinsics
color_stream = profile.get_stream(rs.stream.color)  # Get color stream
intr = color_stream.as_video_stream_profile().get_intrinsics()

# Print camera matrix and distortion coefficients
camera_matrix = [
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
]

distortion_coeffs = intr.coeffs  # [k1, k2, p1, p2, k3]

print("Camera Matrix:")
for row in camera_matrix:
    print(row)

print("\nDistortion Coefficients:")
print(distortion_coeffs)

# Stop the camera
pipeline.stop()

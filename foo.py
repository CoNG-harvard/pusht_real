import pyrealsense2 as rs
import numpy as np
import cv2

ctx = rs.context()
connected_devices = []
for d in ctx.query_devices():
    connected_devices.append(d.get_info(rs.camera_info.serial_number))
    print(d.get_info(rs.camera_info.product_line))
print("Connected devices:", connected_devices)
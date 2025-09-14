import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
from datetime import datetime
import os.path as osp

pkg_dir = osp.dirname(osp.dirname(__file__))
import sys
sys.path.append(pkg_dir)
from utils.marker_util import Marker, MarkerReader, ARUCO_DICT

class MarkerDataSaver:
    def __init__(self, output_dir="marker_data"):
        self.output_dir = output_dir
        self.ensure_output_dir()
        
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def save_marker_data(self, markers_data, image, timestamp=None):
        """Save marker data and image to files"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        # Save JSON data
        json_filename = f"marker_data_{timestamp}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(markers_data, f, indent=2)
        
        # Save image
        image_filename = f"marker_image_{timestamp}.png"
        image_path = os.path.join(self.output_dir, image_filename)
        cv2.imwrite(image_path, image)
        
        print(f"Saved marker data to: {json_path}")
        print(f"Saved image to: {image_path}")
        
        return json_path, image_path

def detect_all_markers(image, cameraMatrix, distortionCoeffs, markerSize=38):
    """Detect all visible markers in the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    markerDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_4X4_50"])
    parameters = cv2.aruco.DetectorParameters_create()
    
    # Detect all markers
    (allCorners, ids, rejected) = cv2.aruco.detectMarkers(gray, markerDict, parameters=parameters)
    
    markers_data = {
        "timestamp": datetime.now().isoformat(),
        "camera_matrix": cameraMatrix.tolist(),
        "distortion_coefficients": distortionCoeffs.tolist(),
        "marker_size": markerSize,
        "num_markers_detected": len(allCorners) if len(allCorners) > 0 else 0,
        "markers": []
    }
    
    annotated_image = image.copy()
    
    if len(allCorners) > 0:
        # Estimate pose for all detected markers
        rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(
            allCorners, markerSize, cameraMatrix, distortionCoeffs)
        
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(annotated_image, allCorners, ids)
        
        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            rvec = rvecs[i][0]  # Shape: (3,)
            tvec = tvecs[i][0]  # Shape: (3,)
            
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            
            # Calculate distance from camera
            distance = np.linalg.norm(tvec)
            
            # Create marker data entry
            marker_data = {
                "id": marker_id,
                "translation": {
                    "x": float(tvec[0]),
                    "y": float(tvec[1]), 
                    "z": float(tvec[2])
                },
                "rotation_vector": {
                    "x": float(rvec[0]),
                    "y": float(rvec[1]),
                    "z": float(rvec[2])
                },
                "rotation_matrix": R.tolist(),
                "distance_from_camera": float(distance),
                "corners": allCorners[i].reshape(4, 2).tolist()
            }
            
            markers_data["markers"].append(marker_data)
            
            # Draw coordinate axes for this marker
            cv2.drawFrameAxes(annotated_image, cameraMatrix, distortionCoeffs, 
                            rvec, tvec, 40)
            
            # Add text label with marker ID and distance
            corner = allCorners[i][0][0]  # Top-left corner
            label = f"ID:{marker_id} d:{distance:.1f}mm"
            cv2.putText(annotated_image, label, 
                       (int(corner[0]), int(corner[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return markers_data, annotated_image

def main():
    # Create pipeline
    pipeline = rs.pipeline()
    
    # Configure the pipeline
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Get stream profile and camera intrinsics
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    
    # Create camera matrix and distortion coefficients
    cameraMatrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ])
    
    distortionCoeffs = np.array(intr.coeffs)
    
    # Initialize marker data saver
    saver = MarkerDataSaver()
    
    print("Marker Detection and Data Saving")
    print("Press 's' to save current frame data")
    print("Press 'q' to quit")
    print("Press 'c' to save continuously (every 2 seconds)")
    
    continuous_save = False
    last_save_time = 0
    
    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            # Detect all markers
            markers_data, annotated_image = detect_all_markers(
                color_image, cameraMatrix, distortionCoeffs)
            
            # Show the annotated image
            cv2.imshow('Marker Detection', annotated_image)
            
            # Handle continuous saving
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if continuous_save and (current_time - last_save_time) > 2.0:  # Save every 2 seconds
                saver.save_marker_data(markers_data, annotated_image)
                last_save_time = current_time
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                saver.save_marker_data(markers_data, annotated_image)
            elif key == ord('c'):
                # Toggle continuous saving
                continuous_save = not continuous_save
                print(f"Continuous saving: {'ON' if continuous_save else 'OFF'}")
                if continuous_save:
                    last_save_time = current_time
                    
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

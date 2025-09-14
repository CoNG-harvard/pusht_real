#!/usr/bin/env python3
"""
Test script to verify the marker data JSON format without requiring a camera
"""

import json
import numpy as np
from datetime import datetime

def create_sample_marker_data():
    """Create sample marker data to test the JSON format"""
    
    # Sample camera matrix and distortion coefficients
    cameraMatrix = np.array([
        [640.0, 0, 320.0],
        [0, 640.0, 240.0],
        [0, 0, 1]
    ])
    
    distortionCoeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0])
    
    # Sample marker data
    markers_data = {
        "timestamp": datetime.now().isoformat(),
        "camera_matrix": cameraMatrix.tolist(),
        "distortion_coefficients": distortionCoeffs.tolist(),
        "marker_size": 38,
        "num_markers_detected": 2,
        "markers": [
            {
                "id": 1,
                "translation": {
                    "x": 10.5,
                    "y": -5.2,
                    "z": 100.0
                },
                "rotation_vector": {
                    "x": 0.1,
                    "y": 0.2,
                    "z": 0.3
                },
                "rotation_matrix": [
                    [0.936, -0.275, 0.218],
                    [0.289, 0.956, -0.047],
                    [-0.198, 0.108, 0.975]
                ],
                "distance_from_camera": 100.8,
                "corners": [
                    [100.0, 50.0],
                    [150.0, 50.0],
                    [150.0, 100.0],
                    [100.0, 100.0]
                ]
            },
            {
                "id": 2,
                "translation": {
                    "x": -15.3,
                    "y": 8.7,
                    "z": 120.5
                },
                "rotation_vector": {
                    "x": -0.1,
                    "y": 0.4,
                    "z": -0.2
                },
                "rotation_matrix": [
                    [0.921, 0.388, -0.025],
                    [-0.390, 0.920, -0.040],
                    [0.007, 0.045, 0.999]
                ],
                "distance_from_camera": 122.1,
                "corners": [
                    [200.0, 100.0],
                    [250.0, 100.0],
                    [250.0, 150.0],
                    [200.0, 150.0]
                ]
            }
        ]
    }
    
    return markers_data

def test_json_serialization():
    """Test that the marker data can be properly serialized to JSON"""
    try:
        markers_data = create_sample_marker_data()
        
        # Test JSON serialization
        json_str = json.dumps(markers_data, indent=2)
        print("✓ JSON serialization successful")
        
        # Test JSON deserialization
        loaded_data = json.loads(json_str)
        print("✓ JSON deserialization successful")
        
        # Verify data integrity
        assert loaded_data["num_markers_detected"] == 2
        assert len(loaded_data["markers"]) == 2
        assert loaded_data["markers"][0]["id"] == 1
        assert loaded_data["markers"][1]["id"] == 2
        print("✓ Data integrity verified")
        
        # Save sample file
        with open("sample_marker_data.json", "w") as f:
            json.dump(markers_data, f, indent=2)
        print("✓ Sample file saved as 'sample_marker_data.json'")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing marker data JSON format...")
    success = test_json_serialization()
    
    if success:
        print("\n✓ All tests passed! The marker data format is working correctly.")
    else:
        print("\n✗ Tests failed. Please check the implementation.")

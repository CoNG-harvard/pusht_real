#!/usr/bin/env python3
"""
Symmetry Demo for T-Shape Point Cloud Registration

This script demonstrates how symmetry affects registration evaluation
for T-shaped objects.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def demonstrate_t_shape_symmetry():
    """Demonstrate T-shape symmetries"""
    
    print("="*60)
    print("T-SHAPE SYMMETRY DEMONSTRATION")
    print("="*60)
    print()
    print("A T-shape has multiple orientations that look identical:")
    print()
    
    # Define the symmetries
    symmetries = [
        ("Identity", np.eye(4)),
        ("Z-180° (Vertical flip)", R.from_euler('z', 180, degrees=True).as_matrix()),
        ("X-180° (Horizontal flip)", R.from_euler('x', 180, degrees=True).as_matrix()),
        ("Y-180° (Horizontal flip)", R.from_euler('y', 180, degrees=True).as_matrix())
    ]
    
    # Create a simple T-shape transformation
    original_transform = np.array([
        [0.9, 0.1, 0.0, 1.0],
        [-0.1, 0.9, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    print("Original transformation:")
    print(original_transform)
    print()
    
    print("Symmetric transformations (all look identical for T-shape):")
    print("-" * 60)
    
    for name, sym_matrix in symmetries:
        if name == "Identity":
            symmetric_transform = original_transform
        else:
            # Create 4x4 matrix
            sym_4x4 = np.eye(4)
            sym_4x4[:3, :3] = sym_matrix
            symmetric_transform = original_transform @ sym_4x4
        
        print(f"\n{name}:")
        print(symmetric_transform)
        
        # Calculate rotation angle
        rotation_matrix = symmetric_transform[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        angle = np.linalg.norm(rotation.as_rotvec()) * 180 / np.pi
        print(f"Rotation angle: {angle:.1f}°")
    
    print("\n" + "="*60)
    print("WHY THIS MATTERS FOR REGISTRATION")
    print("="*60)
    print()
    print("1. Registration algorithms may find any of these orientations")
    print("2. All orientations are 'correct' for a T-shape")
    print("3. Without symmetry analysis, we might think registration failed")
    print("4. With symmetry analysis, we can identify valid but flipped solutions")
    print()
    print("Example scenarios:")
    print("  • Algorithm finds Z-180° flip → Still valid for T-shape")
    print("  • Algorithm finds X-180° flip → Still valid for T-shape") 
    print("  • Algorithm finds Y-180° flip → Still valid for T-shape")
    print("  • Algorithm finds Identity → Perfect match")
    print()
    print("This is why Robust_ICP and Global_Registration might appear")
    print("to work well even with high rotation errors - they're finding")
    print("valid symmetric orientations!")

if __name__ == "__main__":
    demonstrate_t_shape_symmetry()

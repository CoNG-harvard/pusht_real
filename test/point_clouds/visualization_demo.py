#!/usr/bin/env python3
"""
Point Cloud Registration Visualization Demo

This script demonstrates the different visualization options available
for the registration test.
"""

import sys
from pathlib import Path

def print_usage():
    """Print usage instructions"""
    print("="*60)
    print("POINT CLOUD REGISTRATION VISUALIZATION DEMO")
    print("="*60)
    print()
    print("Usage:")
    print("  python test_registration.py                    # Test all algorithms")
    print("  python test_registration.py ICP                # Test only ICP")
    print("  python test_registration.py Robust_ICP         # Test only Robust ICP")
    print("  python test_registration.py Bounding_Box       # Test only Bounding Box")
    print("  python test_registration.py PCA_Bounding_Box   # Test only PCA Bounding Box")
    print("  python test_registration.py Hybrid_Bounding_Box # Test only Hybrid")
    print("  python test_registration.py Global_Registration # Test only Global")
    print()
    print("Visualization Options:")
    print("  python test_registration.py ICP --side-by-side # Force side-by-side view")
    print("  python test_registration.py ICP -s             # Short form for side-by-side")
    print()
    print("Available Algorithms:")
    print("  • ICP: Iterative Closest Point (good for small transformations)")
    print("  • Robust_ICP: RANSAC + ICP (good for larger transformations)")
    print("  • Bounding_Box: Simple center-based alignment (very fast)")
    print("  • PCA_Bounding_Box: PCA-based alignment (handles rotations)")
    print("  • Hybrid_Bounding_Box: PCA + ICP refinement (balanced)")
    print("  • Global_Registration: Feature-based matching (most robust)")
    print()
    print("Visualization Features:")
    print("  🔴 Red: Source point cloud (original)")
    print("  🟢 Green: Target point cloud (transformed)")
    print("  🔵 Blue: Estimated point cloud (registration result)")
    print("  📐 Coordinate frames: Show orientation")
    print("  🟡 Yellow line: Source → Target (ground truth)")
    print("  🟣 Magenta line: Source → Estimated (registration)")
    print()
    print("Quality Assessment:")
    print("  • EXCELLENT: < 5° rotation, < 0.01m translation")
    print("  • GOOD: < 15° rotation, < 0.1m translation")
    print("  • POOR: > 15° rotation or > 0.1m translation")
    print()
    print("Tips:")
    print("  • Use single algorithm mode for better visualization")
    print("  • ICP works best for small transformations")
    print("  • Robust_ICP is good for larger transformations")
    print("  • Bounding box methods are very fast but less accurate")
    print("  • Use --side-by-side if point clouds appear to overlap")
    print("  • Close the 3D viewer window to continue to next algorithm")
    print()
    print("Troubleshooting:")
    print("  • If red point cloud doesn't show: Use --side-by-side flag")
    print("  • If point clouds overlap: Try rotating the view or use side-by-side")
    print("  • For better visibility: Use single algorithm mode")
    print()
    print("Symmetry Analysis:")
    print("  • T-shapes have multiple valid orientations (symmetries)")
    print("  • Algorithms may find 'correct' but flipped orientations")
    print("  • ✅ = Direct match (exact orientation)")
    print("  • ⚠️  = Symmetric match (valid but flipped orientation)")
    print("  • Symmetry analysis shows all possible orientations")
    print("  • This explains why some algorithms appear to work well")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
    else:
        print("Run 'python visualization_demo.py help' for usage instructions")
        print("Or run 'python test_registration.py ICP' to test with visualization")

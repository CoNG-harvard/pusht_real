#!/usr/bin/env python3
"""
Demo script showing different configurations for the real-time red point cloud visualizer
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from realtime_red_pointcloud import RealTimeRedPointCloud

def demo_high_fps():
    """
    Demo with higher frame rate (20Hz)
    """
    print("=== High FPS Demo (20Hz) ===")
    print("This demo runs at 20Hz for smoother visualization")
    print("Press 'q' in the Open3D window to quit")
    print()
    
    try:
        visualizer = RealTimeRedPointCloud(target_fps=20)
        visualizer.run()
    except Exception as e:
        print(f"Error: {e}")

def demo_low_fps():
    """
    Demo with lower frame rate (5Hz) for better performance
    """
    print("=== Low FPS Demo (5Hz) ===")
    print("This demo runs at 5Hz for better performance on slower systems")
    print("Press 'q' in the Open3D window to quit")
    print()
    
    try:
        visualizer = RealTimeRedPointCloud(target_fps=5)
        visualizer.run()
    except Exception as e:
        print(f"Error: {e}")

def demo_standard():
    """
    Standard demo at 10Hz
    """
    print("=== Standard Demo (10Hz) ===")
    print("This is the standard configuration at 10Hz")
    print("Press 'q' in the Open3D window to quit")
    print()
    
    try:
        visualizer = RealTimeRedPointCloud(target_fps=10)
        visualizer.run()
    except Exception as e:
        print(f"Error: {e}")

def main():
    """
    Main demo function
    """
    print("Real-time Red Point Cloud Visualizer - Demo")
    print("=" * 50)
    print()
    print("Choose a demo:")
    print("1. Standard (10Hz)")
    print("2. High FPS (20Hz)")
    print("3. Low FPS (5Hz)")
    print("4. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                demo_standard()
                break
            elif choice == "2":
                demo_high_fps()
                break
            elif choice == "3":
                demo_low_fps()
                break
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()

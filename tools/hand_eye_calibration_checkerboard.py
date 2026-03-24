#!/usr/bin/env python3
"""
Hand-Eye Calibration using checkerboard patterns

This script performs hand-eye calibration using standard checkerboard patterns.
Supports both eye-in-hand and eye-to-hand calibration modes.

Usage:
    python hand_eye_calibration_checkerboard.py --mode eye_in_hand
    python hand_eye_calibration_checkerboard.py --mode eye_to_hand

Arguments:
    --mode: Calibration mode ('eye_in_hand' or 'eye_to_hand')
    --squares_x: Number of inner corners in X direction (default: 9)
    --squares_y: Number of inner corners in Y direction (default: 6)
    --square_size: Size of each square in meters (default: 0.025)
    --robot_ip: IP address of UR robot (default: 192.168.0.191)
    --min_poses: Minimum number of poses for calibration (default: 10)
    --camera_width: Camera resolution width (default: 1280)
    --camera_height: Camera resolution height (default: 720)
    --fps: Camera FPS (default: 30)
"""

import cv2
import numpy as np
import argparse
import sys
import tty
import termios
import os.path as osp
import os
import pyrealsense2 as rs
import json
from datetime import datetime

# Add parent directory to path
pkg_dir = osp.dirname(osp.dirname(__file__))
sys.path.append(pkg_dir)

# Robot control imports
try:
    import rtde_receive
    import rtde_control
    ROBOT_AVAILABLE = True
except ImportError:
    print("Warning: rtde libraries not available. Running in simulation mode.")
    ROBOT_AVAILABLE = False


class CheckerboardHandEyeCalibrator:
    def __init__(self, mode, squares_x=9, squares_y=6, square_size=0.025,
                 robot_ip="192.168.0.191", min_poses=10,
                 camera_width=1280, camera_height=720, fps=30, run_name="calibration"):
        """
        Initialize the Checkerboard Hand-Eye Calibrator

        Args:
            mode: 'eye_in_hand' or 'eye_to_hand'
            squares_x: Number of inner corners in X direction
            squares_y: Number of inner corners in Y direction
            square_size: Size of each square in meters
            robot_ip: IP address of UR robot
            min_poses: Minimum number of poses for calibration
            camera_width: Camera resolution width
            camera_height: Camera resolution height
            fps: Camera FPS
            run_name: Name for this calibration run
        """
        self.mode = mode
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_size = square_size
        self.robot_ip = robot_ip
        self.min_poses = min_poses
        self.run_name = run_name

        # Validate mode
        if mode not in ['eye_in_hand', 'eye_to_hand']:
            raise ValueError("Mode must be 'eye_in_hand' or 'eye_to_hand'")

        # Create checkerboard object points
        self.objp = np.zeros((squares_x * squares_y, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:squares_x, 0:squares_y].T.reshape(-1, 2)
        self.objp *= square_size

        # Initialize camera
        self.pipeline, self.camera_matrix, self.dist_coeffs = self._init_camera(
            camera_width, camera_height, fps
        )

        # Initialize robot connection
        self.rtde_c = None
        self.rtde_r = None
        if ROBOT_AVAILABLE:
            self._init_robot()

        # Data storage for calibration
        self.robot_poses = []  # Robot poses (base to gripper/camera)
        self.board_poses = []  # Checkerboard poses relative to camera
        self.recorded_images = []  # Store recorded images
        self.recorded_corners = []  # Store detected corners
        self.translation_matrices = []  # Store translation matrices for each recording

        print(f"Initialized {mode} calibration with checkerboard pattern")
        print(f"Pattern: {squares_x}x{squares_y} inner corners, Square size: {square_size}m")

    def _init_camera(self, width, height, fps):
        """Initialize RealSense camera"""
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        profile = pipeline.start(config)

        # Get camera intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()

        camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ])

        dist_coeffs = np.array(intr.coeffs)

        print(f"Camera initialized: {width}x{height}@{fps}fps")
        return pipeline, camera_matrix, dist_coeffs

    def _init_robot(self):
        """Initialize robot connection"""
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)

            # Set TCP (Tool Center Point) - adjust as needed
            self.rtde_c.setTcp([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            print(f"Robot connected at {self.robot_ip}")
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            self.rtde_c = None
            self.rtde_r = None

    def detect_checkerboard(self, image):
        """
        Detect checkerboard in image and estimate pose

        Returns:
            success: bool indicating if board was detected
            rvec: rotation vector of board relative to camera
            tvec: translation vector of board relative to camera
            annotated_image: image with detected board drawn
            corners: detected corners (None if not detected)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, (self.squares_x, self.squares_y), None
        )

        annotated_image = image.copy()

        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw corners
            cv2.drawChessboardCorners(annotated_image, (self.squares_x, self.squares_y), corners, ret)

            # Estimate pose
            success, rvec, tvec = cv2.solvePnP(
                self.objp, corners, self.camera_matrix, self.dist_coeffs
            )

            if success:
                # Draw coordinate axes
                axis_length = self.square_size * 3
                try:
                    # Try newer OpenCV API first
                    cv2.drawFrameAxes(annotated_image, self.camera_matrix, self.dist_coeffs,
                                    rvec, tvec, axis_length)
                except AttributeError:
                    # Fall back to older API - draw axes manually
                    axis_points = np.float32([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,-axis_length]]).reshape(-1,3)
                    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                    imgpts = np.int32(imgpts).reshape(-1,2)

                    origin = tuple(imgpts[0].ravel())
                    x_axis = tuple(imgpts[1].ravel())
                    y_axis = tuple(imgpts[2].ravel())
                    z_axis = tuple(imgpts[3].ravel())

                    # Draw X axis (red)
                    cv2.line(annotated_image, origin, x_axis, (0,0,255), 5)
                    # Draw Y axis (green)
                    cv2.line(annotated_image, origin, y_axis, (0,255,0), 5)
                    # Draw Z axis (blue)
                    cv2.line(annotated_image, origin, z_axis, (255,0,0), 5)

                return True, rvec, tvec, annotated_image, corners

        return False, None, None, annotated_image, None

    def get_robot_pose(self):
        """Get current robot pose"""
        if not ROBOT_AVAILABLE or self.rtde_r is None:
            # Return dummy pose for simulation
            raise ValueError("Robot not available")

        return np.array(self.rtde_r.getActualTCPPose())

    def collect_calibration_data(self):
        """
        Collect calibration data interactively

        Instructions:
        - Press 'r' to record current pose
        - Press 'q' to finish data collection and run calibration
        - Press 'c' to clear all collected data
        """
        print("\n" + "="*60)
        print("Hand-Eye Calibration Data Collection (Checkerboard)")
        print("="*60)
        print(f"Mode: {self.mode}")
        print(f"Minimum poses required: {self.min_poses}")
        print(f"Checkerboard: {self.squares_x}x{self.squares_y} inner corners")
        print("\nControls:")
        print("  'r' - Record current pose")
        print("  'c' - Clear all data")
        print("  'q' - Finish and run calibration")
        print("="*60)

        # Set up terminal for non-blocking input
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)

            while True:
                # Get camera frame
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())

                # Detect checkerboard
                success, rvec, tvec, annotated_image, corners = self.detect_checkerboard(color_image)

                # Add status text to image
                status_text = f"Poses: {len(self.robot_poses)}/{self.min_poses} | "
                if success:
                    status_text += "Checkerboard: DETECTED"
                    cv2.putText(annotated_image, status_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    status_text += "Checkerboard: NOT DETECTED"
                    cv2.putText(annotated_image, status_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('Hand-Eye Calibration (Checkerboard)', annotated_image)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('r'):
                    if success:
                        # Record pose
                        robot_pose = self.get_robot_pose()

                        # Create transformation matrix from board pose
                        board_R, _ = cv2.Rodrigues(rvec)
                        T_board2cam = np.eye(4)
                        T_board2cam[:3, :3] = board_R
                        T_board2cam[:3, 3] = tvec.ravel()

                        # Store data
                        self.robot_poses.append(robot_pose)
                        self.board_poses.append((rvec.copy(), tvec.copy()))
                        self.recorded_images.append(color_image.copy())
                        self.recorded_corners.append(corners.copy())
                        self.translation_matrices.append(T_board2cam.copy())

                        print(f"\n--- Recorded pose {len(self.robot_poses)} ---")
                        print(f"Robot pose: {robot_pose}")
                        print(f"Board translation: {tvec.ravel()}")
                        print(f"Board rotation (rodrigues): {rvec.ravel()}")
                        print(f"Translation Matrix (Board to Camera):")
                        print(T_board2cam)
                        print("-" * 40)
                        # Auto-run calibration after each new sample once we have >= 4 samples
                        if len(self.robot_poses) >= 4:
                            try:
                                T_cam2gripper, _, _ = self.perform_calibration(min_required=4, save=True, run_suffix=f"live_{len(self.robot_poses):03d}")
                                self.validate_calibration(T_cam2gripper)
                            except Exception as e:
                                print(f"Live calibration failed: {e}")
                    else:
                        print("Cannot record: Checkerboard not detected!")

                elif key == ord('c'):
                    self.robot_poses.clear()
                    self.board_poses.clear()
                    self.recorded_images.clear()
                    self.recorded_corners.clear()
                    self.translation_matrices.clear()
                    print("Cleared all calibration data")

                elif key == ord('q'):
                    if len(self.robot_poses) >= self.min_poses:
                        break
                    else:
                        print(f"Need at least {self.min_poses} poses. Currently have {len(self.robot_poses)}")

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            cv2.destroyAllWindows()

    def perform_calibration(self, min_required=None, save=True, run_suffix=None):
        """Perform hand-eye calibration using collected data

        Args:
            min_required: Override minimum required poses for this calibration run.
            save: If True, write artifacts to disk; otherwise only print results.
            run_suffix: Optional suffix to append to run folder name.
        """
        required_poses = self.min_poses if min_required is None else min_required
        if len(self.robot_poses) < required_poses:
            raise ValueError(f"Need at least {required_poses} poses for calibration")

        print(f"\nPerforming {self.mode} calibration with {len(self.robot_poses)} poses...")

        # Prepare data for OpenCV calibration
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for i, (robot_pose, (board_rvec, board_tvec)) in enumerate(zip(self.robot_poses, self.board_poses)):
            # Robot pose (base to gripper/camera)
            robot_tvec = robot_pose[:3]
            robot_rvec = robot_pose[3:]
            robot_R, _ = cv2.Rodrigues(robot_rvec)
            if self.mode == 'eye_in_hand':
                R_gripper2base.append(robot_R)
                t_gripper2base.append(robot_tvec.reshape(3, 1))
            else:
                # if eye-to-hand, we need to reverse the rotation and translation for gripper to base
                R_gripper2base.append(robot_R.T)
                t_gripper2base.append(-robot_R.T @ robot_tvec.reshape(3, 1))
                

            # Board pose (camera to board)
            board_R, _ = cv2.Rodrigues(board_rvec)
            R_target2cam.append(board_R)
            t_target2cam.append(board_tvec)

        # Choose calibration method based on mode
        method = cv2.CALIB_HAND_EYE_TSAI
        # if self.mode == 'eye_in_hand':
        #     # Camera attached to robot gripper
        #     method = cv2.CALIB_HAND_EYE_TSAI
        #     print("Using Tsai method for eye-in-hand calibration")
        # else:
        #     # Camera fixed in environment
        #     method = cv2.CALIB_HAND_EYE_PARK
        #     print("Using Park method for eye-to-hand calibration")

        # Perform calibration
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=method
        )

        # Convert to transformation matrix
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.ravel()

        # Convert rotation to axis-angle representation
        rvec_result, _ = cv2.Rodrigues(R_cam2gripper)

        # Save results (optional)
        results_dir = None
        images_dir = None
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        if save:
            run_folder = f"{self.run_name}_{timestamp}"
            if run_suffix:
                run_folder = f"{run_folder}_{run_suffix}"
            results_dir = osp.join(pkg_dir, 'data', run_folder)

            # Create directories
            os.makedirs(results_dir, exist_ok=True)
            images_dir = osp.join(results_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)

            # Save recorded images, corners, translation matrices, and robot poses
            for i, (image, corners, trans_matrix, robot_pose) in enumerate(zip(self.recorded_images, self.recorded_corners, self.translation_matrices, self.robot_poses)):
                # Save image
                cv2.imwrite(osp.join(images_dir, f"pose_{i:03d}.jpg"), image)

                # Save corners
                np.save(osp.join(images_dir, f"corners_{i:03d}.npy"), corners)

                # Save translation matrix
                np.save(osp.join(images_dir, f"translation_matrix_{i:03d}.npy"), trans_matrix)

                # Save robot pose
                np.save(osp.join(images_dir, f"robot_pose_{i:03d}.npy"), robot_pose)

            # Save calibration matrices as both numpy and JSON
            np.save(osp.join(results_dir, "hand_eye_rvec.npy"), rvec_result)
            np.save(osp.join(results_dir, "hand_eye_tvec.npy"), t_cam2gripper)
            np.save(osp.join(results_dir, "hand_eye_matrix.npy"), T_cam2gripper)

            # Save as JSON for easier reading
            calibration_data = {
                "mode": self.mode,
                "run_name": self.run_name,
                "timestamp": timestamp,
                "num_poses": len(self.robot_poses),
                "checkerboard_pattern": {
                    "squares_x": self.squares_x,
                    "squares_y": self.squares_y,
                    "square_size": self.square_size
                },
                "camera_matrix": self.camera_matrix.tolist(),
                "dist_coeffs": self.dist_coeffs.tolist(),
                "hand_eye_calibration": {
                    "rotation_vector": rvec_result.ravel().tolist(),
                    "translation_vector": t_cam2gripper.ravel().tolist(),
                    "transformation_matrix": T_cam2gripper.tolist()
                },
                "robot_poses": [pose.tolist() for pose in self.robot_poses],
                "board_poses": [(rvec.ravel().tolist(), tvec.ravel().tolist()) for rvec, tvec in self.board_poses],
                "translation_matrices": [matrix.tolist() for matrix in self.translation_matrices]
            }

            with open(osp.join(results_dir, "calibration_results.json"), 'w') as f:
                json.dump(calibration_data, f, indent=2)

        # Print results
        print("\n" + "="*60)
        print("CALIBRATION RESULTS (Checkerboard)")
        print("="*60)
        print(f"Mode: {self.mode}")
        print(f"Number of poses used: {len(self.robot_poses)}")
        print(f"Calibration method: {method}")
        print(f"Checkerboard pattern: {self.squares_x}x{self.squares_y} corners, {self.square_size}m squares")
        print("\nTransformation (Camera to Gripper/Base):")
        print("Rotation vector (rad):", rvec_result.ravel())
        print("Translation vector (m):", t_cam2gripper.ravel())
        print("\nTransformation Matrix:")
        print(T_cam2gripper)
        if save:
            print(f"\nResults saved to: {results_dir}")
            print(f"Images saved to: {images_dir}")
            print(f"JSON results: {osp.join(results_dir, 'calibration_results.json')}")
        else:
            print("\nLive calibration run complete (no files saved)")
        print("="*60)

        return T_cam2gripper, rvec_result, t_cam2gripper

    def validate_calibration(self, T_cam2gripper):
        """
        Validate calibration by using T_cam2gripper and the kinematic chain to
        predict checkerboard poses, then compute reprojection error against
        recorded image corners.

        The method:
        1) Build T_gripper2base from each recorded robot pose.
        2) Compute a reference T_target2base from the first sample as:
           T_gripper2base @ T_cam2gripper @ T_target2cam.
        3) For each sample i, predict T_target2cam_pred = inv(T_cam2gripper) @
           inv(T_gripper2base_i) @ T_target2base_ref.
        4) Project checkerboard points with the predicted pose and compute RMS
           reprojection error versus recorded corners.
        """
        print("\nValidating calibration...")

        if len(self.robot_poses) == 0 or len(self.board_poses) == 0:
            print("No data to validate.")
            return float('nan')

        # Helper: build SE3 from rvec/tvec
        def rt_to_T(rvec, tvec):
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.ravel()
            return T

        # Precompute inverse of T_cam2gripper
        T_gripper2cam = np.linalg.inv(T_cam2gripper)

        # Construct gripper->base transforms for each sample consistent with perform_calibration
        T_list_gripper2base = []
        T_list_target2cam = []
        for robot_pose, (board_rvec, board_tvec) in zip(self.robot_poses, self.board_poses):
            robot_tvec = robot_pose[:3]
            robot_rvec = robot_pose[3:]
            R, _ = cv2.Rodrigues(robot_rvec)
            if self.mode == 'eye_in_hand':
                R_g2b = R
                t_g2b = robot_tvec.reshape(3, 1)
            else:
                # eye-to-hand branch mirrored from perform_calibration
                R_g2b = R.T
                t_g2b = -R.T @ robot_tvec.reshape(3, 1)

            T_g2b = np.eye(4)
            T_g2b[:3, :3] = R_g2b
            T_g2b[:3, 3] = t_g2b.ravel()
            T_list_gripper2base.append(T_g2b)

            # Measured checkerboard pose in camera frame
            T_t2c = rt_to_T(board_rvec, board_tvec)
            T_list_target2cam.append(T_t2c)

        # Reference target pose in base frame from the first sample
        T_target2base_ref = T_list_gripper2base[0] @ T_cam2gripper @ T_list_target2cam[0]

        # Compute RMS reprojection error over all samples
        total_error = 0.0
        total_points = 0
        for i, (T_g2b, corners_meas) in enumerate(zip(T_list_gripper2base, self.recorded_corners)):
            # Predict target pose in camera for this sample
            T_t2c_pred = T_gripper2cam @ np.linalg.inv(T_g2b) @ T_target2base_ref

            # Convert predicted pose to rvec/tvec
            R_pred = T_t2c_pred[:3, :3]
            t_pred = T_t2c_pred[:3, 3].reshape(3, 1)
            rvec_pred, _ = cv2.Rodrigues(R_pred)

            # Project 3D checkerboard points
            proj_pred, _ = cv2.projectPoints(self.objp, rvec_pred, t_pred, self.camera_matrix, self.dist_coeffs)
            proj_pred = proj_pred.reshape(-1, 2)

            # Ensure corners_meas shape is Nx1x2 or Nx2
            corners_2d = corners_meas.reshape(-1, 2)

            # RMS error for this sample
            diffs = proj_pred - corners_2d
            errs = np.linalg.norm(diffs, axis=1)
            total_error += np.sum(errs ** 2)
            total_points += errs.size

        rms_error = np.sqrt(total_error / max(total_points, 1))
        print(f"RMS reprojection error (pixels): {rms_error:.4f}")
        print("Validation complete")

        return rms_error

    def cleanup(self):
        """Clean up resources"""
        if self.pipeline:
            self.pipeline.stop()

        if self.rtde_c:
            self.rtde_c.disconnect()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Hand-Eye Calibration using checkerboard patterns')

    parser.add_argument('--mode', required=True, choices=['eye_in_hand', 'eye_to_hand'],
                       help='Calibration mode: eye_in_hand (camera on robot) or eye_to_hand (fixed camera)')
    parser.add_argument('--squares_x', type=int, default=9,
                       help='Number of inner corners in X direction (default: 9)')
    parser.add_argument('--squares_y', type=int, default=6,
                       help='Number of inner corners in Y direction (default: 6)')
    parser.add_argument('--square_size', type=float, default=0.022,
                       help='Size of each square in meters (default: 0.022)')
    parser.add_argument('--robot_ip', default='192.168.0.191',
                       help='IP address of UR robot (default: 192.168.0.191)')
    parser.add_argument('--min_poses', type=int, default=10,
                       help='Minimum number of poses for calibration (default: 10)')
    parser.add_argument('--camera_width', type=int, default=1280,
                       help='Camera resolution width (default: 1280)')
    parser.add_argument('--camera_height', type=int, default=720,
                       help='Camera resolution height (default: 720)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Camera FPS (default: 30)')
    parser.add_argument('--run_name', default='calibration',
                       help='Name for this calibration run (default: calibration)')

    args = parser.parse_args()

    calibrator = None
    try:
        # Initialize calibrator
        calibrator = CheckerboardHandEyeCalibrator(
            mode=args.mode,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_size=args.square_size,
            robot_ip=args.robot_ip,
            min_poses=args.min_poses,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            fps=args.fps,
            run_name=args.run_name
        )

        # Collect calibration data
        calibrator.collect_calibration_data()

        # Perform calibration
        T_cam2gripper, _, _ = calibrator.perform_calibration()

        # Optional: validate calibration
        calibrator.validate_calibration(T_cam2gripper)

    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
    except Exception as e:
        print(f"Error during calibration: {e}")
    finally:
        if calibrator:
            calibrator.cleanup()


if __name__ == '__main__':
    main()
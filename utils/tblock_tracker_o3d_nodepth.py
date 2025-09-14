import cv2
import open3d as o3d
import numpy as np

from pathlib import Path

from shapely.strtree import STRtree
from shapely.geometry import LineString, box
import numpy as np
from itertools import combinations

from skimage.draw import line as skimage_line
from scipy.ndimage import binary_dilation

from scipy.spatial.transform import Rotation
import trimesh

# TODO: Add corrector

def get_points_table(obj_only, T_camera_to_world, camera_matrix, Zworld=0.0):
    a, b, c, d = T_camera_to_world[-2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    H, W, _ = obj_only.shape
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    num = Zworld - d
    denom = a*(u-cx)/fx + b*(v-cy)/fy + c
    Z_cam_img = num / denom
    Y_cam_img = (v-cy)/fy * Z_cam_img
    X_cam_img = (u-cx)/fx * Z_cam_img
    coord_cam_img = np.stack([X_cam_img, Y_cam_img, Z_cam_img, np.ones_like(Z_cam_img)], axis=-1)
    coord_world_img = T_camera_to_world[None, None] @ coord_cam_img[..., None]
    coord_world_img = coord_world_img[..., :3, 0]
    points_table = coord_world_img[np.any(obj_only, axis=-1)]
    return points_table


class TBlockTracker:
    # hardcoded color limits..
    color_u = np.array([250, 120, 80])
    color_l = np.array([90, 0, 0])
    MESH_FILE = 'data/t_shape2.stl'
    N_MESH_PTS = 2000
    MAX_LOSS = 0.95

    def __init__(self, camera_matrix, fixed_camera_pose, mode='bgr'):
        self.mode = mode
        self.camera_matrix = camera_matrix
        self.fixed_camera_pose = fixed_camera_pose
        
        METHOD = 0
        
        if METHOD == 0:
            self.mesh = trimesh.load(self.MESH_FILE)
            self.mesh.apply_translation(-np.array([0, 0.025, 0.04]))
            self.model_pts, _ = trimesh.sample.sample_surface(
                self.mesh, self.N_MESH_PTS)
            
            self.model_pcd = o3d.geometry.PointCloud()
            self.model_pcd.points = o3d.utility.Vector3dVector(self.model_pts)
        elif METHOD == 1:
            ply_file = Path("/home/mht/PycharmProjects/pusht_real/test/point_clouds/t_pcd.ply")
            self.model_pcd = o3d.io.read_point_cloud(str(ply_file))

    def get_obj(self, img, min_cnt_area=1000):
        img = img.copy()
        if self.mode == 'bgr':
            color_l = self.color_l[::-1]
            color_u = self.color_u[::-1]
        else:
            color_l = self.color_l
            color_u = self.color_u
        mask = np.all((color_l <= img) & (
            img <= color_u), axis=-1).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = tuple(
            contour for contour in contours if cv2.contourArea(contour) > min_cnt_area)
        contours = tuple(cv2.approxPolyDP(contour, 1, True)
                         for contour in contours)
        if len(contours) == 0:
            print('no contours')
            return None

        mask = cv2.drawContours(
            np.zeros(img.shape[:-1]), contours, -1, color=255, thickness=-1)
        
        obj_only = img.copy()
        obj_only[mask == 0] = 0
        # cv2.imshow("Masked", obj_only)
        return obj_only
    
    def get_wf_obj_pts(self, obj_mask, depth, max_z=0.065):
        cf_coords = depth_to_camera_coords(depth / 1000, self.camera_matrix)
        T_camera_to_world = rodrigues_to_matrix(self.fixed_camera_pose[3:], self.fixed_camera_pose[:3])
        cf_coords_homo = np.concatenate(
            (cf_coords, np.ones((*cf_coords.shape[:-1], 1))), axis=-1)
        wf_coords_homo = T_camera_to_world[None, None] @ cf_coords_homo[..., None]
        wf_coords = wf_coords_homo[..., :3, 0]

        wf_mask = wf_coords.copy()
        wf_mask[~obj_mask] = 0
        wf_mask[wf_mask[..., -1] > max_z] = 0

        points_table = wf_mask[wf_mask[..., -1] != 0]
        return points_table

    def estimate(self, img, depth, n_angles=24, max_iterations=50, threshold=0.025):
        obj = self.get_obj(img)
        if obj is None:
            return None
        obj_mask = np.any(obj, axis=-1)
        points_table = self.get_wf_obj_pts(obj_mask, depth)
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(points_table)
        
        # Run ICP with multiple initializations for better results
        best_result = None
        best_fitness = -np.inf
        
        # Try different initial transformations
        for yaw in np.linspace(0, 2*np.pi * ((n_angles-1) / n_angles), n_angles):
            tvec = points_table.mean(axis=0)
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            
            try:
                result = o3d.pipelines.registration.registration_icp(
                    self.model_pcd, target_pcd, threshold, T,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
                )
                
                if result.fitness > best_fitness:
                    best_fitness = result.fitness
                    best_result = result
            except Exception as e:
                print(f"ICP failed with init transform: {e}")
                continue
        
        loss = best_fitness
        pose = transformation_matrix_to_pose(best_result.transformation.copy())
        
        self.last_transform = best_result.transformation.copy()
        self.last_pose = pose
        print('initial est done')
        print(loss, pose)
        return loss, pose

    def track(self, img, depth, last_pose=None, safe=True, max_iterations=50, threshold=0.025):
        assert self.last_pose is not None or last_pose is not None, 'No last pose provided'
        last_pose = last_pose if last_pose is not None else self.last_pose
        obj = self.get_obj(img)
        if obj is None:
            return None
        obj_mask = np.any(obj, axis=-1)
        points_table = self.get_wf_obj_pts(obj_mask, depth)
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(points_table)
        
        icp_result = o3d.pipelines.registration.registration_icp(
            self.model_pcd, target_pcd, threshold, self.last_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
        loss = icp_result.fitness
        pose = transformation_matrix_to_pose(icp_result.transformation.copy())
        
        self.last_pose = pose
        print(loss, pose)
        return loss, pose


def depth_to_camera_coords(depth, camera_matrix):
    """
    Converts a depth image to 3D camera coordinates.

    Args:
        depth: (H, W) depth image (in meters, typically)
        fx, fy: focal lengths
        cx, cy: principal point offsets

    Returns:
        (H, W, 3) array of (X, Y, Z) coordinates
    """
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    coords = np.stack((X, Y, Z), axis=-1)  # shape (H, W, 3)
    return coords


def transformation_matrix_to_pose(T):
    """
    Extract translation and rotation from transformation matrix
    """
    translation = T[:3, 3]
    rotation_matrix = T[:3, :3]
    rotation = Rotation.from_matrix(rotation_matrix)
    return (translation[0], translation[1], rotation.as_euler('zyx')[0])

def rodrigues_to_matrix(rvec, tvec):
    """Convert Rodrigues rotation vector to 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def generate_random_transformation(rotation_range=45, translation_range=0.5):
    """
    Generate random rotation and translation
    
    Args:
        rotation_range: Maximum rotation angle in degrees
        translation_range: Maximum translation distance
    
    Returns:
        transformation_matrix: 4x4 homogeneous transformation matrix
        rotation_angles: (rx, ry, rz) in degrees
        translation: (tx, ty, tz)
    """
    # Generate random rotation angles
    rx = np.random.uniform(-rotation_range, rotation_range)
    ry = np.random.uniform(-rotation_range, rotation_range)
    rz = np.random.uniform(-rotation_range, rotation_range)
    
    # Generate random translation
    tx = np.random.uniform(-translation_range, translation_range)
    ty = np.random.uniform(-translation_range, translation_range)
    tz = np.random.uniform(-translation_range, translation_range)
    
    # Create rotation matrix
    rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    rotation_matrix = rotation.as_matrix()
    
    # Create transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]
    
    return transformation_matrix, (rx, ry, rz), (tx, ty, tz)
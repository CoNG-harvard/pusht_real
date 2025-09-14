import cv2
import numpy as np

from shapely.strtree import STRtree
from shapely.geometry import LineString, box
import numpy as np
from itertools import combinations

from skimage.draw import line as skimage_line
from scipy.ndimage import binary_dilation

from scipy.spatial import KDTree
from scipy.optimize import minimize
from scipy.optimize import least_squares
import trimesh

# TODO: Add corrector


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
        self.mesh = trimesh.load(self.MESH_FILE)
        self.mesh.apply_translation(-np.array([0, 0.025, 0.04]))
        self.model_pts, _ = trimesh.sample.sample_surface(
            self.mesh, self.N_MESH_PTS)

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

    def estimate(self, img, depth, n_angles=12):
        obj = self.get_obj(img)
        if obj is None:
            return None
        obj_mask = np.any(obj, axis=-1)
        points_table = self.get_wf_obj_pts(obj_mask, depth)
        loss, pose = np.inf, None
        for yaw in np.linspace(0, 2*np.pi * ((n_angles-1) / n_angles), n_angles):
            init_guess = np.array([*points_table.mean(axis=0)[:2], yaw])
            _loss, _pose = self.optimize(points_table, init_guess=init_guess)
            if _loss < loss:
                loss = _loss
                pose = _pose
        self.last_pose = pose
        print('initial est done')
        print(loss, pose)
        return loss, pose

    def track(self, img, depth, last_pose=None, safe=True):
        assert self.last_pose is not None or last_pose is not None, 'No last pose provided'
        last_pose = last_pose if last_pose is not None else self.last_pose
        obj = self.get_obj(img)
        if obj is None:
            return None
        obj_mask = np.any(obj, axis=-1)
        points_table = self.get_wf_obj_pts(obj_mask, depth)
        loss, pose = self.optimize(points_table, init_guess=last_pose)
        if safe and loss > self.MAX_LOSS:
            print('lost track')
            return None
        self.last_pose = pose
        print(loss, pose)
        return loss, pose

    def optimize(self, points_table, init_guess=None, z=0.04):
        def transform_model(model_pts, x, y, yaw):
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]])
            t = np.array([x, y, z])
            return (R @ model_pts.T).T + t

        def loss_fn(pose):
            x, y, yaw = pose
            transformed_model = transform_model(self.model_pts, x, y, yaw)

            dists, _ = tree.query(transformed_model)
            return dists
        
        tree = KDTree(points_table)
        result = least_squares(loss_fn, init_guess, method='lm')
        pose = result.x
        pose[-1] = - pose[-1] # NOTE: invert yaw to be consistent
        return np.sum(result.fun**2), pose


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


def rodrigues_to_matrix(rvec, tvec):
    """Convert Rodrigues rotation vector to 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T
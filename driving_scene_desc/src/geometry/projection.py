from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from pyquaternion import Quaternion

from src.common.config import camera_names

@dataclass
class ProjectionResult:
    camera_name: str
    valid_indices: np.ndarray
    pixel_coords: np.ndarray
    depth_values: np.ndarray
    valid_mask: np.ndarray
    points_cam: np.ndarray


def rotate(points: np.ndarray, rotation_quat) -> np.ndarray:
    r = Quaternion(rotation_quat).rotation_matrix
    return points @ r.T


def translate(points: np.ndarray, translation) -> np.ndarray:
    return points + np.asarray(translation, dtype=np.float32).reshape(1, 3)


def inverse_rotate(points: np.ndarray, rotation_quat) -> np.ndarray:
    r = Quaternion(rotation_quat).rotation_matrix
    return points @ r


def inverse_translate(points: np.ndarray, translation) -> np.ndarray:
    return points - np.asarray(translation, dtype=np.float32).reshape(1, 3)


def transform_lidar_to_camera(points_lidar: np.ndarray, nusc, lidar_sd: dict, cam_sd: dict) -> np.ndarray:
    """nuScenes 标准链路：lidar -> ego(lidar) -> global -> ego(camera) -> camera。"""
    pts = points_lidar[:, :3].astype(np.float32).copy()
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])

    pts = rotate(pts, lidar_cs['rotation'])
    pts = translate(pts, lidar_cs['translation'])
    pts = rotate(pts, lidar_pose['rotation'])
    pts = translate(pts, lidar_pose['translation'])

    pts = inverse_translate(pts, cam_pose['translation'])
    pts = inverse_rotate(pts, cam_pose['rotation'])
    pts = inverse_translate(pts, cam_cs['translation'])
    pts = inverse_rotate(pts, cam_cs['rotation'])
    return pts


def project_to_image(points_lidar: np.ndarray, nusc, lidar_sd: dict, cam_sd: dict, image_size) -> ProjectionResult:
    pts_cam = transform_lidar_to_camera(points_lidar, nusc, lidar_sd, cam_sd)
    cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    K = np.asarray(cam_cs['camera_intrinsic'], dtype=np.float32)
    depth = pts_cam[:, 2]
    eps = 1e-6
    uvw = pts_cam @ K.T
    u = uvw[:, 0] / np.maximum(uvw[:, 2], eps)
    v = uvw[:, 1] / np.maximum(uvw[:, 2], eps)
    w, h = image_size
    valid = (depth > 1.0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    valid_indices = np.nonzero(valid)[0]
    pixels = np.stack([u[valid], v[valid]], axis=1).astype(np.float32)
    return ProjectionResult(
        camera_name=cam_sd.get('channel', ''),
        valid_indices=valid_indices,
        pixel_coords=pixels,
        depth_values=depth[valid].astype(np.float32),
        valid_mask=valid,
        points_cam=pts_cam[valid].astype(np.float32),
    )


def project_frame(frame, nusc) -> Dict[str, ProjectionResult]:
    results = {}
    for cam in camera_names():
        img = frame.images[cam]
        results[cam] = project_to_image(frame.lidar_points, nusc, frame.lidar_sd, frame.camera_sds[cam], img.size)
        results[cam].camera_name = cam
    return results

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from PIL import Image

from src.common.config import camera_names
from src.datasets.nuscenes_reader import TARGET_CLASSES

# 统计量来自该方向相机中的有效投影点。
LIDAR_FEATURE_NAMES = [
    'lidar_num_points', 'lidar_mean_x', 'lidar_mean_y', 'lidar_mean_z', 'lidar_std_x', 'lidar_std_y', 'lidar_std_z',
    'lidar_mean_intensity', 'lidar_std_intensity', 'lidar_mean_range', 'lidar_std_range',
    'lidar_near_ratio', 'lidar_front_ratio', 'lidar_left_ratio', 'lidar_right_ratio', 'lidar_height_pos_ratio',
]
IMAGE_FEATURE_NAMES = [
    'img_mean_r', 'img_mean_g', 'img_mean_b', 'img_std_r', 'img_std_g', 'img_std_b',
    'img_brightness', 'img_contrast', 'img_edge_strength', 'img_dark_ratio', 'img_green_ratio', 'img_red_ratio',
]
CLIP_CLASSES = ['vehicle', 'pedestrian', 'obstacle', 'road', 'building', 'vegetation', 'traffic_light', 'traffic_sign', 'barrier', 'traffic_cone']
CLIP_FEATURE_NAMES = [f'clip_{c}' for c in CLIP_CLASSES]
EVIDENCE_FEATURE_NAMES = [f'evidence_{c}' for c in TARGET_CLASSES]

# 六相机方向 one-hot 与方向角特征。
CAMERA_ONEHOT_NAME = {
    'CAM_FRONT': 'cam_front',
    'CAM_FRONT_LEFT': 'cam_front_left',
    'CAM_FRONT_RIGHT': 'cam_front_right',
    'CAM_BACK': 'cam_back',
    'CAM_BACK_LEFT': 'cam_back_left',
    'CAM_BACK_RIGHT': 'cam_back_right',
}

# 近似方向角：前方为0度，左侧为正，右侧为负。sin/cos 比单一角度数值更适合神经网络学习方向关系。
CAMERA_ANGLE_DEG = {
    'CAM_FRONT': 0.0,
    'CAM_FRONT_LEFT': 45.0,
    'CAM_FRONT_RIGHT': -45.0,
    'CAM_BACK': 180.0,
    'CAM_BACK_LEFT': 135.0,
    'CAM_BACK_RIGHT': -135.0,
}

DIRECTION_FEATURE_NAMES = list(CAMERA_ONEHOT_NAME.values()) + ['camera_angle_sin', 'camera_angle_cos']

CONTEXT_FEATURE_NAMES = [
    'scene_idx_norm', 'is_day_sunny', 'is_night_rainy',
    'projection_valid_ratio', 'front_valid_ratio', 'mean_depth_norm',
] + DIRECTION_FEATURE_NAMES
ALL_FEATURE_NAMES = LIDAR_FEATURE_NAMES + IMAGE_FEATURE_NAMES + CLIP_FEATURE_NAMES + EVIDENCE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES


def safe_stats(arr: np.ndarray, dim: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    if arr is None or arr.size == 0:
        return np.zeros(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32)
    return arr.mean(axis=0).astype(np.float32), arr.std(axis=0).astype(np.float32)


def _empty_lidar_stats() -> Dict[str, float]:
    return {k: 0.0 for k in LIDAR_FEATURE_NAMES}


def _empty_image_stats() -> Dict[str, float]:
    return {k: 0.0 for k in IMAGE_FEATURE_NAMES}


def add_camera_direction_features(camera_name: str) -> Dict[str, float]:
    """返回当前相机方向的 one-hot 与方向角特征。"""
    row = {}
    for cam, col in CAMERA_ONEHOT_NAME.items():
        row[col] = 1.0 if camera_name == cam else 0.0

    angle_deg = CAMERA_ANGLE_DEG.get(camera_name, 0.0)
    angle_rad = np.deg2rad(angle_deg)
    row['camera_angle_sin'] = float(np.sin(angle_rad))
    row['camera_angle_cos'] = float(np.cos(angle_rad))
    return row


def _projected_lidar_points_for_camera(frame, projections, camera_name: str):
    p = projections.get(camera_name)
    if p is None or len(p.valid_indices) == 0:
        dim = frame.lidar_points.shape[1] if frame.lidar_points is not None and frame.lidar_points.ndim == 2 else 5
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    idx = np.asarray(p.valid_indices, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(frame.lidar_points))]
    if len(idx) == 0:
        return np.zeros((0, frame.lidar_points.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    n = min(len(idx), len(p.depth_values))
    return frame.lidar_points[idx[:n]].astype(np.float32), np.asarray(p.depth_values[:n], dtype=np.float32)


def _collect_projected_lidar_points(frame, projections):
    pts_list, depth_list = [], []
    for cam in camera_names():
        pts, dep = _projected_lidar_points_for_camera(frame, projections, cam)
        if len(pts) > 0:
            pts_list.append(pts); depth_list.append(dep)
    if not pts_list:
        dim = frame.lidar_points.shape[1] if frame.lidar_points is not None and frame.lidar_points.ndim == 2 else 5
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(pts_list, axis=0).astype(np.float32), np.concatenate(depth_list, axis=0).astype(np.float32)


def extract_lidar_stats(points: np.ndarray) -> Dict[str, float]:
    if points is None or len(points) == 0:
        return _empty_lidar_stats()
    xyz = points[:, :3].astype(np.float32)
    intensity = points[:, 3].astype(np.float32) if points.shape[1] > 3 else np.zeros(len(points), dtype=np.float32)
    planar = np.linalg.norm(xyz[:, :2], axis=1)
    mean_xyz, std_xyz = safe_stats(xyz, dim=3)
    return {
        'lidar_num_points': float(min(len(points), 50000) / 50000.0),
        'lidar_mean_x': float(mean_xyz[0] / 50.0),
        'lidar_mean_y': float(mean_xyz[1] / 50.0),
        'lidar_mean_z': float(mean_xyz[2] / 5.0),
        'lidar_std_x': float(std_xyz[0] / 50.0),
        'lidar_std_y': float(std_xyz[1] / 50.0),
        'lidar_std_z': float(std_xyz[2] / 5.0),
        'lidar_mean_intensity': float(np.nan_to_num(np.mean(intensity), nan=0.0)),
        'lidar_std_intensity': float(np.nan_to_num(np.std(intensity), nan=0.0)),
        'lidar_mean_range': float(np.nan_to_num(np.mean(planar) / 60.0, nan=0.0)),
        'lidar_std_range': float(np.nan_to_num(np.std(planar) / 30.0, nan=0.0)),
        'lidar_near_ratio': float(np.mean(planar < 20.0)),
        'lidar_front_ratio': float(np.mean(xyz[:, 0] > 0)),
        'lidar_left_ratio': float(np.mean(xyz[:, 1] > 0)),
        'lidar_right_ratio': float(np.mean(xyz[:, 1] < 0)),
        'lidar_height_pos_ratio': float(np.mean(xyz[:, 2] > 0.5)),
    }


def _sample_projected_pixels(img: Image.Image, pixel_coords: np.ndarray) -> Dict[str, np.ndarray]:
    if pixel_coords is None or len(pixel_coords) == 0:
        return {}
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    uv = np.rint(pixel_coords).astype(np.int64)
    u = np.clip(uv[:, 0], 0, w - 1)
    v = np.clip(uv[:, 1], 0, h - 1)
    rgb = arr[v, u, :3]
    gray_img = arr.mean(axis=2)
    gray = gray_img[v, u]

    pad = np.pad(gray_img, ((1, 1), (1, 1)), mode='edge')
    vp, up = v + 1, u + 1
    neigh = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neigh.append(pad[vp + dy, up + dx])
    local_contrast = np.stack(neigh, axis=1).std(axis=1)
    gy, gx = np.gradient(gray_img)
    grad = np.sqrt(gx * gx + gy * gy)[v, u]
    return {'rgb': rgb.astype(np.float32), 'gray': gray.astype(np.float32), 'local_contrast': local_contrast.astype(np.float32), 'edge_strength': grad.astype(np.float32)}


def extract_projected_image_stats_for_camera(frame, projections, camera_name: str) -> Dict[str, float]:
    p = projections.get(camera_name)
    if p is None or len(p.pixel_coords) == 0:
        return _empty_image_stats()
    sampled = _sample_projected_pixels(frame.images[camera_name], p.pixel_coords)
    if not sampled:
        return _empty_image_stats()
    rgb = sampled['rgb']
    gray = sampled['gray']
    local_contrast = sampled['local_contrast']
    edge = sampled['edge_strength']
    mean, std = rgb.mean(axis=0), rgb.std(axis=0)
    green = (rgb[:, 1] > rgb[:, 0] * 1.08) & (rgb[:, 1] > rgb[:, 2] * 1.05)
    red = (rgb[:, 0] > rgb[:, 1] * 1.15) & (rgb[:, 0] > rgb[:, 2] * 1.15)
    return {
        'img_mean_r': float(mean[0]), 'img_mean_g': float(mean[1]), 'img_mean_b': float(mean[2]),
        'img_std_r': float(std[0]), 'img_std_g': float(std[1]), 'img_std_b': float(std[2]),
        'img_brightness': float(gray.mean()),
        'img_contrast': float(local_contrast.mean()),
        'img_edge_strength': float(edge.mean()),
        'img_dark_ratio': float(np.mean(gray < 0.18)),
        'img_green_ratio': float(np.mean(green)),
        'img_red_ratio': float(np.mean(red)),
    }


def extract_projected_image_stats(frame, projections) -> Dict[str, float]:
    vals = []
    for cam in camera_names():
        vals.append(extract_projected_image_stats_for_camera(frame, projections, cam))
    return {k: float(np.mean([v[k] for v in vals])) for k in IMAGE_FEATURE_NAMES}


def extract_projection_context_for_camera(frame, projections, camera_name: str) -> Dict[str, float]:
    npts = max(len(frame.lidar_points), 1)
    p = projections.get(camera_name)
    valid_ratio = len(p.valid_indices) / npts if p is not None else 0.0
    depth = p.depth_values if (p is not None and len(p.depth_values) > 0) else np.array([0.0], dtype=np.float32)
    row = {
        'scene_idx_norm': float(frame.scene_idx / 10.0),
        'is_day_sunny': float(frame.scene_condition == 'day_sunny'),
        'is_night_rainy': float(frame.scene_condition == 'night_rainy'),
        'projection_valid_ratio': float(valid_ratio),
        'front_valid_ratio': float(valid_ratio if camera_name == 'CAM_FRONT' else 0.0),
        'mean_depth_norm': float(np.mean(depth) / 60.0),
    }
    row.update(add_camera_direction_features(camera_name))
    return row


def extract_projection_context(frame, projections) -> Dict[str, float]:
    vals = [extract_projection_context_for_camera(frame, projections, cam) for cam in camera_names()]
    return {k: float(np.mean([v[k] for v in vals])) for k in CONTEXT_FEATURE_NAMES}


def compute_geometry_evidence_for_camera(frame, projections, camera_name: str) -> Dict[str, float]:
    pts, _ = _projected_lidar_points_for_camera(frame, projections, camera_name)
    if len(pts) == 0:
        return {k: 0.0 for k in EVIDENCE_FEATURE_NAMES}
    xyz = pts[:, :3]
    rng = np.linalg.norm(xyz[:, :2], axis=1)
    near = np.mean(rng < 25.0)
    front = np.mean((xyz[:, 0] > 0) & (rng < 35.0))
    side = np.mean((np.abs(xyz[:, 1]) > 5) & (rng < 30.0))
    vertical = np.std(xyz[:, 2]) / 3.0
    cone_like = np.mean((rng < 25) & (xyz[:, 2] > -1.5) & (xyz[:, 2] < 1.0))
    return {
        'evidence_vehicle': float(np.clip(front * 2.0 + near * 0.3, 0, 1)),
        'evidence_pedestrian': float(np.clip(side * 1.5 + vertical * 0.2, 0, 1)),
        'evidence_obstacle': float(np.clip(side * 1.2 + cone_like * 1.4 + near * 0.25, 0, 1)),
    }


def compute_geometry_evidence(frame, projections) -> Dict[str, float]:
    vals = [compute_geometry_evidence_for_camera(frame, projections, cam) for cam in camera_names()]
    return {k: float(np.mean([v[k] for v in vals])) for k in EVIDENCE_FEATURE_NAMES}


def clip_score_dict_to_features(clip_scores: Dict[str, float] | None) -> Dict[str, float]:
    clip_scores = clip_scores or {}
    return {f'clip_{c}': float(clip_scores.get(c, 0.0)) for c in CLIP_CLASSES}


def make_direction_feature_row(frame, projections, clip_scores: Dict[str, float] | None, camera_name: str) -> Dict[str, float]:
    projected_pts, _ = _projected_lidar_points_for_camera(frame, projections, camera_name)
    row = {}
    row.update(extract_lidar_stats(projected_pts))
    row.update(extract_projected_image_stats_for_camera(frame, projections, camera_name))
    row.update(clip_score_dict_to_features(clip_scores))
    row.update(compute_geometry_evidence_for_camera(frame, projections, camera_name))
    row.update(extract_projection_context_for_camera(frame, projections, camera_name))
    for name in ALL_FEATURE_NAMES:
        row.setdefault(name, 0.0)
        if not np.isfinite(row[name]):
            row[name] = 0.0
    return row


def make_feature_row(frame, projections, clip_scores: Dict[str, float] | None) -> Dict[str, float]:
    projected_pts, _ = _collect_projected_lidar_points(frame, projections)
    row = {}
    row.update(extract_lidar_stats(projected_pts))
    row.update(extract_projected_image_stats(frame, projections))
    row.update(clip_score_dict_to_features(clip_scores))
    row.update(compute_geometry_evidence(frame, projections))
    row.update(extract_projection_context(frame, projections))
    for name in ALL_FEATURE_NAMES:
        row.setdefault(name, 0.0)
        if not np.isfinite(row[name]):
            row[name] = 0.0
    return row


def vector_from_row(row: Dict[str, float], names) -> np.ndarray:
    return np.array([float(row.get(n, 0.0)) for n in names], dtype=np.float32)

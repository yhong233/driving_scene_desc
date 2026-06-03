from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import math
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from src.common.config import camera_names

try:
    from nuscenes.nuscenes import NuScenes
except Exception:
    NuScenes = None

TARGET_CLASSES = ['vehicle', 'pedestrian', 'obstacle']

# 六相机近似方位角
CAMERA_ANGLE_DEG = {
    'CAM_FRONT': 0.0,
    'CAM_FRONT_LEFT': 60.0,
    'CAM_BACK_LEFT': 120.0,
    'CAM_BACK': 180.0,
    'CAM_BACK_RIGHT': -120.0,
    'CAM_FRONT_RIGHT': -60.0,
}


def map_nuscenes_category(category_name: str) -> str | None:
    """把 nuScenes 原始类别映射为本文三类方向级评价目标。"""
    if not category_name:
        return None
    if category_name.startswith('vehicle.'):
        return 'vehicle'
    if category_name.startswith('human.pedestrian'):
        return 'pedestrian'
    if category_name in {
        'movable_object.barrier',
        'movable_object.trafficcone',
        'movable_object.debris',
        'movable_object.pushable_pullable',
    }:
        return 'obstacle'
    return None


def _angle_distance_deg(a: float, b: float) -> float:
    """计算两个角度的最小夹角，单位为度。"""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def camera_from_ego_xy(x: float, y: float) -> str:
    """根据目标中心在 ego 坐标中的方位，分配到最接近的相机方向。只用于训练/评价标签构建，不用于推理特征输入。"""
    angle = math.degrees(math.atan2(float(y), float(x)))
    return min(CAMERA_ANGLE_DEG.keys(), key=lambda cam: _angle_distance_deg(angle, CAMERA_ANGLE_DEG[cam]))


@dataclass
class FrameData:
    frame_idx: int
    sample_token: str
    scene_token: str
    scene_name: str
    scene_idx: int
    scene_condition: str
    sample: Dict[str, Any]
    lidar_sd: Dict[str, Any]
    camera_sds: Dict[str, Dict[str, Any]]
    lidar_path: Path
    image_paths: Dict[str, Path]
    lidar_points: np.ndarray
    images: Dict[str, Image.Image]
    annotations: List[Dict[str, Any]]
    target_labels: Dict[str, int]


class MiniNuScenesReader:
    def __init__(self, dataroot: str | Path, version: str = 'v1.0-mini', verbose: bool = False):
        if NuScenes is None:
            raise ImportError('未安装 nuscenes-devkit，请先 pip install nuscenes-devkit')
        self.dataroot = Path(str(dataroot).replace('\\', '/'))
        self.version = version
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=verbose)
        self.samples = list(self.nusc.sample)
        self.scene_token_to_index = {scene['token']: i + 1 for i, scene in enumerate(self.nusc.scene)}
        self.scene_token_to_name = {scene['token']: scene.get('name', f'scene-{i+1}') for i, scene in enumerate(self.nusc.scene)}

    def __len__(self) -> int:
        return len(self.samples)

    def scene_condition(self, scene_idx: int) -> str:
        return 'day_sunny' if scene_idx <= 7 else 'night_rainy'

    def get_sample_scene_token(self, sample: Dict[str, Any]) -> str:
        if 'scene_token' in sample:
            return sample['scene_token']
        for scene in self.nusc.scene:
            token = scene['first_sample_token']
            while token:
                if token == sample['token']:
                    return scene['token']
                s = self.nusc.get('sample', token)
                token = s['next']
        raise KeyError('无法找到 sample 对应 scene_token')

    def read_lidar_bin(self, lidar_path: Path) -> np.ndarray:
        pts = np.fromfile(str(lidar_path), dtype=np.float32)
        if pts.size % 5 != 0:
            pts = pts[:pts.size // 5 * 5]
        return pts.reshape(-1, 5)

    def get_frame(self, frame_idx: int, load_images: bool = True, max_points: int | None = None) -> FrameData:
        sample = self.samples[frame_idx]
        scene_token = self.get_sample_scene_token(sample)
        scene_idx = self.scene_token_to_index[scene_token]
        scene_name = self.scene_token_to_name[scene_token]

        lidar_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path = self.dataroot / lidar_sd['filename']
        if not lidar_path.exists():
            raise FileNotFoundError(f'LiDAR 文件不存在: {lidar_path}')
        lidar_points = self.read_lidar_bin(lidar_path)
        if max_points and lidar_points.shape[0] > max_points:
            idx = np.linspace(0, lidar_points.shape[0] - 1, max_points).astype(int)
            lidar_points = lidar_points[idx]

        camera_sds, image_paths, images = {}, {}, {}
        for cam in camera_names():
            sd = self.nusc.get('sample_data', sample['data'][cam])
            path = self.dataroot / sd['filename']
            if not path.exists():
                raise FileNotFoundError(f'图像文件不存在: {path}')
            camera_sds[cam] = sd
            image_paths[cam] = path
            if load_images:
                images[cam] = Image.open(path).convert('RGB')

        anns = [self.nusc.get('sample_annotation', tok) for tok in sample.get('anns', [])]
        labels = {c: 0 for c in TARGET_CLASSES}
        for ann in anns:
            if ann.get('num_lidar_pts', 1) <= 0:
                continue
            c = map_nuscenes_category(ann.get('category_name', ''))
            if c in labels:
                labels[c] = 1

        return FrameData(
            frame_idx=frame_idx,
            sample_token=sample['token'],
            scene_token=scene_token,
            scene_name=scene_name,
            scene_idx=scene_idx,
            scene_condition=self.scene_condition(scene_idx),
            sample=sample,
            lidar_sd=lidar_sd,
            camera_sds=camera_sds,
            lidar_path=lidar_path,
            image_paths=image_paths,
            lidar_points=lidar_points,
            images=images,
            annotations=anns,
            target_labels=labels,
        )

    def ann_center_in_lidar_ego(self, frame: FrameData, ann: Dict[str, Any]) -> np.ndarray:
        """将 sample_annotation 的全局中心点转换到当前 LiDAR sample_data 对应的 ego 坐标。"""
        pose = self.nusc.get('ego_pose', frame.lidar_sd['ego_pose_token'])
        center_global = np.asarray(ann.get('translation', [0.0, 0.0, 0.0]), dtype=np.float32).reshape(1, 3)
        trans = np.asarray(pose['translation'], dtype=np.float32).reshape(1, 3)
        rot = Quaternion(pose['rotation']).rotation_matrix
        center_ego = (center_global - trans) @ rot
        return center_ego.reshape(3)

    def get_direction_labels(self, frame: FrameData) -> Dict[str, Dict[str, int]]:
        """构建六相机方向级标签"""
        labels = {cam: {c: 0 for c in TARGET_CLASSES} for cam in camera_names()}
        for ann in frame.annotations:
            if ann.get('num_lidar_pts', 1) <= 0:
                continue
            cls = map_nuscenes_category(ann.get('category_name', ''))
            if cls not in TARGET_CLASSES:
                continue
            center = self.ann_center_in_lidar_ego(frame, ann)
            cam = camera_from_ego_xy(center[0], center[1])
            labels[cam][cls] = 1
        return labels

    def make_index_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for i in range(len(self)):
            sample = self.samples[i]
            scene_token = self.get_sample_scene_token(sample)
            scene_idx = self.scene_token_to_index[scene_token]
            rows.append({
                'frame_idx': i,
                'sample_token': sample['token'],
                'scene_idx': scene_idx,
                'scene_name': self.scene_token_to_name[scene_token],
                'condition': self.scene_condition(scene_idx),
                'lidar_token': sample['data']['LIDAR_TOP'],
                **{cam: sample['data'][cam] for cam in camera_names()},
            })
        return rows

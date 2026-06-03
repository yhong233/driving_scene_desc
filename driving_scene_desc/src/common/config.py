from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_path(value: str | Path, project_root: Path | None = None) -> Path:
    p = Path(str(value).replace('\\', '/'))
    if p.is_absolute():
        return p
    return (project_root or PROJECT_ROOT) / p

@dataclass
class AppConfig:
    project_root: Path
    nuscenes_root: Path
    nuscenes_version: str
    clip_ckpt: Path
    fusion_ckpt: Path
    output_root: Path
    cache_root: Path
    class_cfg: Dict[str, Any]
    scene_cfg: Dict[str, Any]
    exp_cfg: Dict[str, Any]

    @property
    def csv_dir(self) -> Path:
        return self.output_root / 'csv'

    @property
    def json_dir(self) -> Path:
        return self.output_root / 'json'

    @property
    def vis_dir(self) -> Path:
        return self.output_root / 'vis'

    @property
    def chart_dir(self) -> Path:
        return self.output_root / 'charts'


def load_app_config(config_dir: str | Path | None = None) -> AppConfig:
    cfg_dir = Path(config_dir) if config_dir else PROJECT_ROOT / 'configs'
    paths = read_yaml(cfg_dir / 'paths.yaml')
    project_root = resolve_path(paths.get('project_root', PROJECT_ROOT), PROJECT_ROOT)
    app = AppConfig(
        project_root=project_root,
        nuscenes_root=resolve_path(paths['nuscenes_root'], project_root),
        nuscenes_version=str(paths.get('nuscenes_version', 'v1.0-mini')),
        clip_ckpt=resolve_path(paths['clip_ckpt'], project_root),
        fusion_ckpt=resolve_path(paths.get('fusion_ckpt', project_root / 'models/fusion/fusion_mlp.pt'), project_root),
        output_root=resolve_path(paths.get('output_root', project_root / 'outputs'), project_root),
        cache_root=resolve_path(paths.get('cache_root', project_root / 'outputs/cache'), project_root),
        class_cfg=read_yaml(cfg_dir / 'classes.yaml'),
        scene_cfg=read_yaml(cfg_dir / 'scene_split.yaml'),
        exp_cfg=read_yaml(cfg_dir / 'experiments.yaml') if (cfg_dir / 'experiments.yaml').exists() else {},
    )
    for d in [app.csv_dir, app.json_dir, app.vis_dir, app.chart_dir, app.cache_root, app.fusion_ckpt.parent]:
        d.mkdir(parents=True, exist_ok=True)
    return app


def camera_names():
    return ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def camera_to_zh_direction(cam: str) -> str:
    return {
        'CAM_FRONT': '前方',
        'CAM_FRONT_LEFT': '左前方',
        'CAM_FRONT_RIGHT': '右前方',
        'CAM_BACK': '后方',
        'CAM_BACK_LEFT': '左后方',
        'CAM_BACK_RIGHT': '右后方',
    }.get(cam, cam)

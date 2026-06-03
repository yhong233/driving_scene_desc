from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

from src.common.config import load_app_config, camera_names, camera_to_zh_direction
from src.common.timer import RuntimeMeter
from src.common.io_utils import save_json
from src.datasets.nuscenes_reader import MiniNuScenesReader, TARGET_CLASSES
from src.geometry.projection import project_frame
from src.features.feature_extract import make_direction_feature_row
from src.semantics.clip_adapter import ClipAdapter
from src.models.train_utils import load_fusion_model
from src.baselines.traditional_rule import predict_traditional_rule
from src.nlg.generator import build_structured_semantics
from src.nlg.temporal import apply_temporal_context
from src.eval.metrics import frame_prf, score_description_details
from src.visualization.vis import save_six_camera_projection, save_bev, save_six_camera_projected_with_stats, compute_fused_bev_strength


# 消融实验方法。
DEFAULT_ABLATION_METHODS = ['image_only', 'lidar_only', 'normal_fusion']
ABLATION_METHODS_WITH_THRESHOLDS = ['clip_only'] + DEFAULT_ABLATION_METHODS
ABLATION_FIXED_THRESHOLD = 0.38
ABLATION_FALLBACK_THRESHOLD = 0.22
DEFAULT_ABLATION_THRESHOLDS = {
    method: {c: ABLATION_FIXED_THRESHOLD for c in TARGET_CLASSES}
    for method in ABLATION_METHODS_WITH_THRESHOLDS
}


def _safe_float(row: Dict[str, float], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def score_ablation_method(row: Dict[str, float], method: str) -> Dict[str, float]:
    """
    计算消融实验方法的三类目标分数。
    """
    method = str(method).strip()
    clip = {c: _safe_float(row, f'clip_{c}', 0.0) for c in TARGET_CLASSES}
    evi = {c: _safe_float(row, f'evidence_{c}', 0.0) for c in TARGET_CLASSES}
    img_signal = float(
        _safe_float(row, 'img_contrast', 0.0)
        + _safe_float(row, 'img_edge_strength', 0.0)
        + _safe_float(row, 'img_red_ratio', 0.0)
    )

    if method == 'clip_only':
        probs = clip
    elif method == 'lidar_only':
        probs = evi
    elif method == 'image_only':
        probs = {
            c: float(np.clip(0.75 * clip[c] + 0.25 * img_signal, 0.0, 1.0))
            for c in TARGET_CLASSES
        }
    elif method == 'normal_fusion':
        # 普通融合消融
        probs = {
            c: float(np.clip(0.45 * clip[c] + 0.40 * evi[c] + 0.15 * img_signal, 0.0, 1.0))
            for c in TARGET_CLASSES
        }
    else:
        raise ValueError(f'未知消融方法: {method}')

    return {c: float(probs.get(c, 0.0)) for c in TARGET_CLASSES}


class DrivingScenePipeline:
    def __init__(
        self,
        config_dir=None,
        device='auto',
        use_clip=True,
        load_model=True,
        nuscenes_root=None,
        fusion_ckpt=None,
        clip_ckpt=None,
    ):
        """驾驶场景语义描述系统主流程。"""
        self.cfg = load_app_config(config_dir)

        if nuscenes_root is not None and str(nuscenes_root).strip() != "":
            self.cfg.nuscenes_root = Path(nuscenes_root)

        if fusion_ckpt is not None and str(fusion_ckpt).strip() != "":
            self.cfg.fusion_ckpt = Path(fusion_ckpt)

        if clip_ckpt is not None and str(clip_ckpt).strip() != "":
            self.cfg.clip_ckpt = Path(clip_ckpt)

        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)

        self.reader = MiniNuScenesReader(
            self.cfg.nuscenes_root,
            self.cfg.nuscenes_version,
            verbose=False
        )

        self.clip = ClipAdapter(
            self.cfg.clip_ckpt,
            self.cfg.class_cfg,
            device=self.device,
            enabled=use_clip
        )

        if load_model:
            self.model, self.thresholds, self.ckpt = load_fusion_model(
                self.cfg.fusion_ckpt,
                device=self.device
            )
        else:
            self.model = None
            self.thresholds = {c: 0.5 for c in TARGET_CLASSES}
            self.ckpt = None

        # 消融实验固定阈值。以便突出不同特征输入方式本身对 F1-score 的影响。
        self.ablation_thresholds = {
            method: vals.copy()
            for method, vals in DEFAULT_ABLATION_THRESHOLDS.items()
        }

    def set_ablation_thresholds(self, thresholds_by_method: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """使用固定阈值"""
        cleaned = {}
        for method in ABLATION_METHODS_WITH_THRESHOLDS:
            base = DEFAULT_ABLATION_THRESHOLDS.get(method, {c: ABLATION_FIXED_THRESHOLD for c in TARGET_CLASSES}).copy()
            incoming = thresholds_by_method.get(method, {}) if thresholds_by_method else {}
            for c in TARGET_CLASSES:
                if c in incoming:
                    base[c] = float(incoming[c])
            cleaned[method] = base
        self.ablation_thresholds = cleaned
        return self.ablation_thresholds

    def method_predict(self, row: Dict[str, float], method: str) -> Tuple[List[str], Dict[str, float]]:
        method = method.strip()
        if method in ['ours', 'no_clip', 'no_image_proxy'] and self.model is not None:
            df = pd.DataFrame([row])
            from src.models.train_utils import predict_model_on_df
            probs = predict_model_on_df(self.model, df, device=self.device, zero_clip=(method=='no_clip'), zero_image=(method=='no_image_proxy'))[0]
            probs_dict = {c: float(probs[i]) for i, c in enumerate(TARGET_CLASSES)}
            pred = [c for c in TARGET_CLASSES if probs_dict[c] >= float(self.thresholds.get(c, 0.5))]
            return pred, probs_dict

        # 对比/消融方法。
        if method == 'traditional_rule':
            pred, probs = predict_traditional_rule(row)
            probs = {c: float(probs.get(c, 0.0)) for c in TARGET_CLASSES}
            return pred, probs

        if method in ABLATION_METHODS_WITH_THRESHOLDS:
            probs = score_ablation_method(row, method)
        else:
            method = 'normal_fusion'
            probs = score_ablation_method(row, method)

        thresholds = self.ablation_thresholds.get(method, DEFAULT_ABLATION_THRESHOLDS.get(method, {}))
        pred = [
            c for c in TARGET_CLASSES
            if probs.get(c, 0.0) >= float(thresholds.get(c, ABLATION_FIXED_THRESHOLD))
        ]
        if not pred:
            best_cls = max(TARGET_CLASSES, key=lambda c: probs.get(c, 0.0))
            if probs.get(best_cls, 0.0) > ABLATION_FALLBACK_THRESHOLD:
                pred = [best_cls]
        return pred, probs

    @staticmethod
    def _union_classes(direction_results: List[Dict]) -> List[str]:
        out = []
        for item in direction_results:
            for c in item.get('pred_classes', []):
                if c not in out:
                    out.append(c)
        return out

    @staticmethod
    def _max_probs(direction_results: List[Dict]) -> Dict[str, float]:
        probs = {c: 0.0 for c in TARGET_CLASSES}
        for item in direction_results:
            for c, v in item.get('confidence_scores', {}).items():
                if c in probs:
                    probs[c] = max(probs[c], float(v))
        return probs

    def process_frame(self, frame_idx: int, method='ours', save_outputs=True, save_vis=True, save_result_json=True, max_points=None, prev_structured: Dict | None = None) -> Dict:
        meter = RuntimeMeter()
        with meter.track('read_data'):
            frame = self.reader.get_frame(frame_idx, load_images=True, max_points=max_points)
        with meter.track('projection'):
            projections = project_frame(frame, self.reader.nusc)
        with meter.track('clip_align'):
            camera_scores = self.clip.score_images(frame.images)
        with meter.track('feature_fusion'):
            direction_labels = self.reader.get_direction_labels(frame)
            direction_rows = {}
            for cam in camera_names():
                direction_rows[cam] = make_direction_feature_row(frame, projections, camera_scores.get(cam, {}), cam)
        with meter.track('fusion_mlp'):
            direction_results = []
            for cam in camera_names():
                pred, probs = self.method_predict(direction_rows[cam], method)
                true_classes = [c for c, v in direction_labels[cam].items() if v]
                dp, dr, df1 = frame_prf(true_classes, pred)
                direction_results.append({
                    'camera_name': cam,
                    'direction': camera_to_zh_direction(cam),
                    'pred_classes': pred,
                    'true_classes': true_classes,
                    'confidence_scores': probs,
                    'metrics': {'precision': dp, 'recall': dr, 'f1': df1},
                })
            pred_classes = self._union_classes(direction_results)
            probs = self._max_probs(direction_results)
        with meter.track('description'):
            structured = build_structured_semantics(frame, pred_classes, probs, camera_scores, method=method, direction_predictions=direction_results)
            structured = apply_temporal_context(prev_structured, structured)


            structured['frame_idx'] = int(frame_idx)
            structured['sample_token'] = getattr(frame, 'sample_token', structured.get('sample_token', ''))
            structured['scene_idx'] = int(getattr(frame, 'scene_idx', structured.get('scene_idx', -1)))
            structured['scene_name'] = getattr(frame, 'scene_name', structured.get('scene_name', ''))
            structured['scene_condition'] = getattr(frame, 'scene_condition', structured.get('scene_condition', 'unknown'))
            structured['method'] = method

            desc = structured['description_text']

        true_frame_classes = [c for c, v in frame.target_labels.items() if v]
        p, r, f1 = frame_prf(true_frame_classes, pred_classes)

        direction_precisions = []
        direction_recalls = []
        direction_f1s = []
        for item in direction_results:
            m = item.get('metrics', {})
            direction_precisions.append(float(m.get('precision', 0.0)))
            direction_recalls.append(float(m.get('recall', 0.0)))
            direction_f1s.append(float(m.get('f1', 0.0)))

        camera_mean_precision = float(np.mean(direction_precisions)) if direction_precisions else 0.0
        camera_mean_recall = float(np.mean(direction_recalls)) if direction_recalls else 0.0
        camera_mean_f1 = float(np.mean(direction_f1s)) if direction_f1s else 0.0

        # 描述连贯性：结构化语义的五项加权评分。
        coherence_detail = score_description_details(
            description=desc,
            predicted_classes=pred_classes,
            direction_targets=structured.get('direction_targets', {}),
            scene_condition=structured.get('scene_condition'),
            scene_context=structured.get('scene_context', {}),
            dynamic_events=structured.get('dynamic_events', []),
        )
        spatial = float(coherence_detail.get('semantic_consistency_score', 0.0))
        coherence = float(coherence_detail.get('coherence_score', 0.0))

        structured['reference_classes'] = true_frame_classes
        structured['metrics'] = {

            'frame_precision': p,
            'frame_recall': r,
            'frame_f1': f1,
            # 六相机方向平均指标
            'precision': camera_mean_precision,
            'recall': camera_mean_recall,
            'f1': camera_mean_f1,
            'camera_mean_precision': camera_mean_precision,
            'camera_mean_recall': camera_mean_recall,
            'camera_mean_f1': camera_mean_f1,
            # 描述连贯性相关指标。
            'spatial_score': spatial,
            'coherence_score': coherence,
            'coherence_detail': coherence_detail,
        }
        structured['runtime_ms'] = {**meter.times, 'core_runtime_ms': meter.core_ms()}
        structured['feature_rows_by_camera'] = direction_rows

        if save_outputs:
            stem = f'{method}_frame_{frame_idx:04d}'

            if save_vis:
                try:
                    save_six_camera_projection(frame, projections, self.cfg.vis_dir / f'{stem}_sixcam.jpg')
                    projected_path, projection_stats = save_six_camera_projected_with_stats(
                        frame,
                        projections,
                        self.cfg.vis_dir / f'{stem}_sixcam_projected.jpg',
                        self.cfg.vis_dir / f'{stem}_projection_stats.csv',
                    )
                    structured['projection_overlay_path'] = str(projected_path)
                    structured['projection_stats'] = projection_stats
                    # 原始点云颜色表示距离，融合图颜色表示点级融合特征强度。
                    save_bev(
                        frame.lidar_points,
                        self.cfg.vis_dir / f'{stem}_bev_raw.jpg',
                        title='Raw LiDAR top-down view',
                    )
                    strength = compute_fused_bev_strength(frame, projections, camera_scores)
                    save_bev(
                        frame.lidar_points,
                        self.cfg.vis_dir / f'{stem}_bev_fused.jpg',
                        title='Fused LiDAR feature strength',
                        strength=strength,
                    )
                except Exception as e:
                    structured['visualization_error'] = str(e)

            if save_result_json:
                save_json(self.cfg.json_dir / f'{stem}.json', structured)

        return structured

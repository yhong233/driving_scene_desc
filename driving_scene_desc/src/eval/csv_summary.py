from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd

from src.datasets.nuscenes_reader import TARGET_CLASSES
from src.eval.run_experiments import (
    evaluate_prediction_df,
    evaluate_by_camera,
    evaluate_camera_mean,
    evaluate_by_condition_camera_mean,
)


def check_prediction_columns(df: pd.DataFrame) -> list[str]:
    required = [
        'method', 'frame_idx', 'camera_name',
        *[f'true_{c}' for c in TARGET_CLASSES],
        *[f'pred_{c}' for c in TARGET_CLASSES],
    ]
    return [c for c in required if c not in df.columns]


def build_single_method_summaries(df: pd.DataFrame, method: str) -> dict[str, pd.DataFrame]:
    """由方向级 predictions DataFrame 构建各种评价汇总。"""
    summary, per_class = evaluate_prediction_df(df, method)

    condition_rows = []
    if 'condition' in df.columns:
        for cond, g in df.groupby('condition'):
            s, _ = evaluate_prediction_df(g, method)
            condition_rows.append({'condition': cond, **s})

    scene_rows = []
    if 'scene_idx' in df.columns:
        for scene, g in df.groupby('scene_idx'):
            s, _ = evaluate_prediction_df(g, method)
            condition = g['condition'].iloc[0] if 'condition' in g.columns and len(g) else ''
            scene_rows.append({'scene_idx': scene, 'condition': condition, **s})

    camera_rows, camera_class_rows = evaluate_by_camera(df, method)
    cam_mean, class_cam_mean = evaluate_camera_mean(camera_rows, camera_class_rows, method)
    cond_cam_mean = evaluate_by_condition_camera_mean(df, method)

    return {
        'summary_by_method': pd.DataFrame([summary]),
        'summary_per_class': pd.DataFrame(per_class),
        'summary_by_condition': pd.DataFrame(condition_rows),
        'summary_by_scene': pd.DataFrame(scene_rows),
        'summary_by_camera': pd.DataFrame(camera_rows),
        'summary_by_camera_class': pd.DataFrame(camera_class_rows),
        'summary_by_method_camera_mean': pd.DataFrame([cam_mean]),
        'summary_per_class_camera_mean': pd.DataFrame(class_cam_mean),
        'summary_by_condition_camera_mean': pd.DataFrame(cond_cam_mean),
    }


def build_multi_method_summaries(prediction_dfs: Iterable[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """多个方法的方向级 predictions DataFrame 构建统一汇总。"""
    summaries = []
    per_classes = []
    condition_rows = []
    scene_rows = []
    camera_rows_all = []
    camera_class_rows_all = []
    camera_mean_rows = []
    per_class_camera_mean_rows = []
    condition_camera_mean_rows = []

    for df in prediction_dfs:
        if df is None or len(df) == 0:
            continue
        method = str(df['method'].iloc[0]) if 'method' in df.columns and len(df) else 'unknown'
        pack = build_single_method_summaries(df, method)
        summaries.append(pack['summary_by_method'])
        per_classes.append(pack['summary_per_class'])
        condition_rows.append(pack['summary_by_condition'])
        scene_rows.append(pack['summary_by_scene'])
        camera_rows_all.append(pack['summary_by_camera'])
        camera_class_rows_all.append(pack['summary_by_camera_class'])
        camera_mean_rows.append(pack['summary_by_method_camera_mean'])
        per_class_camera_mean_rows.append(pack['summary_per_class_camera_mean'])
        condition_camera_mean_rows.append(pack['summary_by_condition_camera_mean'])

    def concat(items):
        items = [x for x in items if x is not None and len(x) > 0]
        return pd.concat(items, ignore_index=True) if items else pd.DataFrame()

    return {
        'summary_by_method': concat(summaries),
        'summary_per_class': concat(per_classes),
        'summary_by_condition': concat(condition_rows),
        'summary_by_scene': concat(scene_rows),
        'summary_by_camera': concat(camera_rows_all),
        'summary_by_camera_class': concat(camera_class_rows_all),
        'summary_by_method_camera_mean': concat(camera_mean_rows),
        'summary_per_class_camera_mean': concat(per_class_camera_mean_rows),
        'summary_by_condition_camera_mean': concat(condition_camera_mean_rows),
    }


def save_summary_pack(pack: dict[str, pd.DataFrame], out_dir: str | Path, prefix: str = '') -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in pack.items():
        filename = f'{prefix}{name}.csv' if prefix else f'{name}.csv'
        df.to_csv(out_dir / filename, index=False, encoding='utf-8-sig')


def slim_summary_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    """图表"""
    keep = [c for c in ['method', 'precision', 'recall', 'f1', 'support', 'pred_positive', 'num_frames'] if c in df.columns]
    return df[keep].copy()

def slim_ablation_f1_summary_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    """消融实验表"""
    keep = [c for c in ['method', 'f1', 'support', 'pred_positive', 'num_frames'] if c in df.columns]
    return df[keep].copy()


def slim_ablation_per_class_f1_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    """消融实验类别级 F1 表。"""
    keep = [c for c in ['method', 'class', 'f1', 'support', 'pred_positive'] if c in df.columns]
    return df[keep].copy()

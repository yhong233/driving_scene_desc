from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import math
import pandas as pd


def split_strategy(scene_cfg: dict) -> str:
    """返回数据划分策略。"""
    return str(scene_cfg.get('split_strategy', scene_cfg.get('mode', 'frame_stratified'))).strip()


def _frame_split_params(scene_cfg: dict) -> dict:
    params = scene_cfg.get('frame_split', {}) or {}
    train_ratio = float(params.get('train_ratio', scene_cfg.get('train_ratio', 0.60)))
    val_ratio = float(params.get('val_ratio', scene_cfg.get('val_ratio', 0.20)))
    test_ratio = float(params.get('test_ratio', scene_cfg.get('test_ratio', 0.20)))
    mode = str(params.get('mode', scene_cfg.get('frame_split_mode', 'block'))).strip()
    gap = int(params.get('gap', scene_cfg.get('split_gap', 0)))
    return {
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'mode': mode,
        'gap': max(gap, 0),
    }


def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    s = train_ratio + val_ratio + test_ratio
    if s <= 0:
        raise ValueError('train/val/test ratio 之和必须大于 0')
    return train_ratio / s, val_ratio / s, test_ratio / s


def make_frame_split_from_features(df: pd.DataFrame, scene_cfg: dict) -> pd.DataFrame:
    """
    按每个 scene 内的 frame_idx 划分 train/val/test。
    """
    required = {'frame_idx', 'scene_idx'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'构建 frame_split 需要列: {sorted(missing)}')

    params = _frame_split_params(scene_cfg)
    tr, vr, ter = _normalize_ratios(params['train_ratio'], params['val_ratio'], params['test_ratio'])
    mode = params['mode']
    gap = params['gap']

    frame_df = df[['frame_idx', 'scene_idx']].drop_duplicates().copy()
    if 'condition' in df.columns:
        cond = df[['frame_idx', 'condition']].drop_duplicates('frame_idx')
        frame_df = frame_df.merge(cond, on='frame_idx', how='left')
    else:
        frame_df['condition'] = frame_df['scene_idx'].apply(lambda x: 'day_sunny' if int(x) <= 7 else 'night_rainy')
    if 'scene_name' in df.columns:
        names = df[['frame_idx', 'scene_name']].drop_duplicates('frame_idx')
        frame_df = frame_df.merge(names, on='frame_idx', how='left')
    else:
        frame_df['scene_name'] = frame_df['scene_idx'].apply(lambda x: f'scene-{int(x):04d}')

    rows = []
    for scene_idx, g in frame_df.sort_values(['scene_idx', 'frame_idx']).groupby('scene_idx'):
        frames = g.sort_values('frame_idx').to_dict('records')
        n = len(frames)
        if n < 5:
            n_train = max(1, int(round(n * tr)))
            n_val = max(1, int(round(n * vr))) if n >= 3 else 0
        else:
            n_train = max(1, int(math.floor(n * tr)))
            n_val = max(1, int(math.floor(n * vr)))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_val > 0:
                n_val -= 1
            else:
                n_train = max(1, n_train - 1)

        if mode == 'interleaved':
            # 每个 scene 都均匀出现在三个集合里。
            for k, rec in enumerate(frames):
                r = k % 5
                if r in (0, 1, 2):
                    split = 'train'
                elif r == 3:
                    split = 'val'
                else:
                    split = 'test'
                rows.append({**rec, 'split': split})
        else:
            #降低随机带来的相邻帧泄漏。
            train_end = n_train
            val_start = min(n, train_end + gap)
            val_end = min(n, val_start + n_val)
            test_start = min(n, val_end + gap)
            for k, rec in enumerate(frames):
                if k < train_end:
                    split = 'train'
                elif val_start <= k < val_end:
                    split = 'val'
                elif k >= test_start:
                    split = 'test'
                else:
                    split = 'unused'
                rows.append({**rec, 'split': split})

    out = pd.DataFrame(rows).sort_values('frame_idx').reset_index(drop=True)
    return out


def save_frame_split(frame_split: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_split.to_csv(path, index=False, encoding='utf-8-sig')
    return path


def load_or_create_frame_split(features_df: pd.DataFrame, scene_cfg: dict, csv_dir: str | Path) -> pd.DataFrame:
    path = Path(csv_dir) / 'frame_split.csv'
    if path.exists():
        split = pd.read_csv(path)
        expected = set(int(x) for x in features_df['frame_idx'].dropna().unique())
        got = set(int(x) for x in split['frame_idx'].dropna().unique()) if 'frame_idx' in split else set()
        if expected == got and 'split' in split.columns:
            return split
    split = make_frame_split_from_features(features_df, scene_cfg)
    save_frame_split(split, path)
    return split


def apply_split_to_features(df: pd.DataFrame, scene_cfg: dict, csv_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """根据配置把方向级样本切分为 train/val/test。"""
    strategy = split_strategy(scene_cfg)
    if strategy == 'scene_level':
        train_scenes = set(scene_cfg.get('train_scenes', [1, 5, 6, 7, 10]))
        val_scenes = set(scene_cfg.get('val_scenes', [2, 8]))
        test_scenes = set(scene_cfg.get('test_scenes', [3, 4, 9]))
        if train_scenes & val_scenes or train_scenes & test_scenes or val_scenes & test_scenes:
            raise ValueError('scene_level 划分中 train/val/test scene 不能重叠')
        train_df = df[df['scene_idx'].isin(train_scenes)].reset_index(drop=True)
        val_df = df[df['scene_idx'].isin(val_scenes)].reset_index(drop=True)
        test_df = df[df['scene_idx'].isin(test_scenes)].reset_index(drop=True)
        return train_df, val_df, test_df, None

    if strategy != 'frame_stratified':
        raise ValueError(f'未知 split_strategy: {strategy}')

    split = load_or_create_frame_split(df, scene_cfg, csv_dir)
    use_cols = ['frame_idx', 'split']
    merged = df.merge(split[use_cols], on='frame_idx', how='left')
    if merged['split'].isna().any():
        raise ValueError('部分 frame_idx 未能匹配 frame_split.csv')
    train_df = merged[merged['split'] == 'train'].reset_index(drop=True)
    val_df = merged[merged['split'] == 'val'].reset_index(drop=True)
    test_df = merged[merged['split'] == 'test'].reset_index(drop=True)
    return train_df, val_df, test_df, split


def frame_indices_for_evaluation(cfg, reader, eval_all: bool = False, split_name: str = 'test') -> list[int]:
    """返回待评价 frame_idx。"""
    if eval_all:
        return list(range(len(reader)))
    strategy = split_strategy(cfg.scene_cfg)
    if strategy == 'frame_stratified':
        path = cfg.csv_dir / 'frame_split.csv'
        if not path.exists():
            features_path = cfg.csv_dir / 'train_features.csv'
            if not features_path.exists():
                raise FileNotFoundError('未找到 frame_split.csv 或 train_features.csv，请先运行 02_build_train_data.py 和 03_train_fusion_mlp.py')
            df = pd.read_csv(features_path)
            split = load_or_create_frame_split(df, cfg.scene_cfg, cfg.csv_dir)
        else:
            split = pd.read_csv(path)
        indices = sorted(int(x) for x in split.loc[split['split'] == split_name, 'frame_idx'].unique())
        return indices

    test_scenes = cfg.scene_cfg.get('test_scenes', [3, 4, 9])
    indices = []
    for i, s in enumerate(reader.samples):
        scene_idx = reader.scene_token_to_index[reader.get_sample_scene_token(s)]
        if scene_idx in test_scenes:
            indices.append(i)
    return indices


def print_split_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, label_cols: Iterable[str], split: pd.DataFrame | None = None):
    label_cols = list(label_cols)
    print('数据集划分：')
    if split is not None:
        print('  split_strategy = frame_stratified（每个 scene 内按 frame_idx 划分；同一帧六方向不拆开）')
        frame_counts = split[split['split'].isin(['train', 'val', 'test'])].groupby('split')['frame_idx'].nunique().to_dict()
        print(f'  frame counts   = {frame_counts}')
        scene_cover = split[split['split'].isin(['train','val','test'])].groupby('split')['scene_idx'].nunique().to_dict()
        print(f'  scene coverage = {scene_cover}')
    else:
        print('  split_strategy = scene_level（按 scene 整体划分）')
    print(f'  train: 样本数={len(train_df)}, 帧数={train_df["frame_idx"].nunique() if "frame_idx" in train_df else "未知"}')
    print(f'  val:   样本数={len(val_df)}, 帧数={val_df["frame_idx"].nunique() if "frame_idx" in val_df else "未知"}')
    print(f'  test:  样本数={len(test_df)}, 帧数={test_df["frame_idx"].nunique() if "frame_idx" in test_df else "未知"}，训练阶段不使用')
    print('\n训练集标签均值：')
    print(train_df[label_cols].mean())
    print('\n验证集标签均值：')
    print(val_df[label_cols].mean())
    if len(test_df) > 0:
        print('\n测试集标签均值，仅检查，不参与训练：')
        print(test_df[label_cols].mean())

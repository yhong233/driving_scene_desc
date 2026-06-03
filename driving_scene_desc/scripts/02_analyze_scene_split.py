from pathlib import Path
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common.config import load_app_config
from src.datasets.nuscenes_reader import TARGET_CLASSES
from src.datasets.split_utils import (
    save_frame_split,
    apply_split_to_features,
)

LABEL_COLS = [f'label_{c}' for c in TARGET_CLASSES]


def mean_for(df, scenes):
    return df[df['scene_idx'].isin(scenes)][LABEL_COLS].mean()


def count_for(df, scenes):
    return df[df['scene_idx'].isin(scenes)][LABEL_COLS].sum().astype(int)


def night_count(scenes):
    return sum(int(s) >= 8 for s in scenes)


def score_scene_split(df, train, val, test):
    """scene 级补充划分评分：越小代表 train/val/test 标签分布越接近整体分布。"""
    overall = df[LABEL_COLS].mean().values
    means = np.vstack([
        mean_for(df, train).values,
        mean_for(df, val).values,
        mean_for(df, test).values,
    ])
    return (
        np.abs(means[0] - overall).mean()
        + np.abs(means[1] - overall).mean()
        + np.abs(means[2] - overall).mean()
        + 2.0 * np.std(means, axis=0).mean()
    )


def print_split_stats(name, df):
    print(f'\n{name}: 样本数={len(df)}, 帧数={df["frame_idx"].nunique() if "frame_idx" in df else "未知"}')
    if len(df):
        print('  标签均值:', df[LABEL_COLS].mean().round(3).to_dict())
        print('  正样本数:', df[LABEL_COLS].sum().astype(int).to_dict())
        if 'scene_idx' in df:
            print('  覆盖 scene:', sorted(int(x) for x in df['scene_idx'].unique()))
        if 'condition' in df:
            frame_level = df.drop_duplicates('frame_idx')
            print('  场景条件帧数:', frame_level.groupby('condition').size().to_dict())


def build_scene_suggestions(df, topk=10):
    scenes = sorted(int(x) for x in df['scene_idx'].unique())
    best = []
    for val in itertools.combinations(scenes, 2):
        remaining = [s for s in scenes if s not in val]
        for test in itertools.combinations(remaining, 3):
            train = [s for s in remaining if s not in test]
            if len(train) != 5:
                continue
            # 保证 train / val / test 至少各包含一个夜雨场景，避免天气条件完全缺失。
            if night_count(train) < 1 or night_count(val) < 1 or night_count(test) < 1:
                continue
            # 保证三类目标都有一定正样本，避免某类在某个集合中完全缺失。
            if (count_for(df, train) < 10).any():
                continue
            if (count_for(df, val) < 10).any():
                continue
            if (count_for(df, test) < 10).any():
                continue
            score = score_scene_split(df, train, list(val), list(test))
            best.append((score, train, list(val), list(test)))
    return sorted(best, key=lambda x: x[0])[:topk]


def print_scene_suggestions(best, df):
    print('\n================ scene 级划分建议 Top 10（只用于补充泛化实验，不是当前主流程划分） ================')
    print('说明：Rank 是候选 scene_level 划分的分布均衡评分；score 越小越均衡。')
    print('当前主流程默认使用 frame_stratified 时，这些 Rank 不会被 03/04 自动采用。')
    for i, (score, train, val, test) in enumerate(best, 1):
        print(f'\nRank {i}: score={score:.4f}')
        print(f'  train_scenes: {train}')
        print(f'  val_scenes:   {val}')
        print(f'  test_scenes:  {test}')
        print('  train mean:', mean_for(df, train).round(3).to_dict())
        print('  val mean:  ', mean_for(df, val).round(3).to_dict())
        print('  test mean: ', mean_for(df, test).round(3).to_dict())
        print('  train count:', count_for(df, train).to_dict())
        print('  val count:  ', count_for(df, val).to_dict())
        print('  test count: ', count_for(df, test).to_dict())


def save_best_scene_suggestion(best, cfg):
    if not best:
        return None
    _, train, val, test = best[0]
    out_path = cfg.csv_dir / 'scene_split_suggestion.yaml'
    data = {
        'conditions': {
            'day_sunny': [1, 2, 3, 4, 5, 6, 7],
            'night_rainy': [8, 9, 10],
        },
        'split_strategy': 'scene_level',
        'train_scenes': [int(x) for x in train],
        'val_scenes': [int(x) for x in val],
        'test_scenes': [int(x) for x in test],
        'note': '仅供 scene_level 补充泛化实验使用；主流程 frame_stratified 不会自动采用本文件。',
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-scene-suggestions', action='store_true', help='额外打印 scene 级划分 Rank，用于补充泛化实验')
    parser.add_argument('--save-scene-suggestion', action='store_true', help='保存 Rank 1 scene 级推荐划分到 scene_split_suggestion.yaml')
    args = parser.parse_args()

    cfg = load_app_config()
    csv_path = cfg.csv_dir / 'train_features.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'未找到 {csv_path}，请先运行 scripts/02_build_train_data.py --use-clip')

    df = pd.read_csv(csv_path)
    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'train_features.csv 缺少三类标签列: {missing}。请删除旧 train_features.csv 后重新运行 02_build_train_data.py --use-clip')

    print('整体标签均值：')
    print(df[LABEL_COLS].mean())
    print('\n每个 scene 的方向级标签均值：')
    print(df.groupby('scene_idx')[LABEL_COLS].mean())
    print('\n每个 scene 样本数：')
    print(df.groupby('scene_idx').size())

    print('\n================ 当前主流程划分检查 ================')
    train_df, val_df, test_df, frame_split = apply_split_to_features(df, cfg.scene_cfg, cfg.csv_dir)
    print_split_stats('train', train_df)
    print_split_stats('val', val_df)
    print_split_stats('test', test_df)
    if frame_split is not None:
        path = save_frame_split(frame_split, cfg.csv_dir / 'frame_split.csv')
        print(f'\n已保存/更新场景内按帧划分文件: {path}')
        print('frame_split 统计（每个 split 在各 scene 中的帧数）：')
        print(frame_split.groupby(['split', 'scene_idx']).size().unstack(fill_value=0))
        print('\n提示：当前主流程采用 frame_stratified。只要 configs/scene_split.yaml 中 split_strategy 不改为 scene_level，')
        print('      03_train_fusion_mlp.py 和 04_run_inference.py 都会使用 frame_split.csv，而不是下面的 scene Rank。')

    if args.show_scene_suggestions or args.save_scene_suggestion:
        best = build_scene_suggestions(df, topk=10)
        if args.show_scene_suggestions:
            print_scene_suggestions(best, df)
        if args.save_scene_suggestion:
            out_path = save_best_scene_suggestion(best, cfg)
            if out_path:
                print(f'\n已保存 scene 级推荐划分到: {out_path}')
    else:
        print('\n如需查看 scene 级补充泛化实验划分 Rank，请运行：')
        print('python scripts/02_analyze_scene_split.py --show-scene-suggestions')


if __name__ == '__main__':
    main()

from pathlib import Path
import sys
import argparse
import json
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline import DrivingScenePipeline
from src.common.config import load_app_config
from src.eval.run_experiments import flatten_direction_results
from src.eval.csv_summary import build_single_method_summaries, save_summary_pack, slim_summary_for_paper
from src.datasets.split_utils import frame_indices_for_evaluation, apply_split_to_features
from src.baselines.traditional_rule import (
    tune_traditional_thresholds,
    set_traditional_thresholds,
    get_traditional_thresholds,
)


def _csv_matches_indices(csv_path: Path, indices: list[int]) -> bool:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    if 'frame_idx' not in df.columns:
        return False
    return set(df['frame_idx'].dropna().astype(int).unique()) == set(int(x) for x in indices)


def _tune_thresholds_from_train_features(cfg, tune_split: str, threshold_min: float, threshold_max: float, num_steps: int):
    """用 train/val split 搜索传统规则阈值，严禁用 test split 调阈值。"""
    features_csv = cfg.csv_dir / 'train_features.csv'
    if not features_csv.exists():
        raise FileNotFoundError(
            '未找到 train_features.csv，无法为传统规则方法搜索阈值。\n'
            '请先运行: python scripts/02_build_train_data.py'
        )

    df = pd.read_csv(features_csv)
    train_df, val_df, test_df, _ = apply_split_to_features(df, cfg.scene_cfg, cfg.csv_dir)

    tune_split = tune_split.strip().lower()
    if tune_split == 'val':
        tune_df = val_df
    elif tune_split == 'train':
        tune_df = train_df
    else:
        raise ValueError('传统规则阈值只能用 train 或 val split 确定，不能使用 test。')

    if len(tune_df) == 0:
        raise ValueError(f'{tune_split} split 为空，无法搜索传统规则阈值。')

    thresholds, detail = tune_traditional_thresholds(
        tune_df,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        num_steps=num_steps,
    )
    set_traditional_thresholds(thresholds)

    # 保存阈值和搜索明细。
    json_out = cfg.csv_dir / 'traditional_rule_thresholds.json'
    csv_out = cfg.csv_dir / 'traditional_rule_thresholds.csv'
    meta = {
        'tune_split': tune_split,
        'threshold_min': float(threshold_min),
        'threshold_max': float(threshold_max),
        'num_steps': int(num_steps),
        'thresholds': thresholds,
        'note': 'Thresholds are tuned on train/val split only. Test split is not used for threshold tuning.',
    }
    json_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    detail.insert(0, 'tune_split', tune_split)
    detail.to_csv(csv_out, index=False, encoding='utf-8-sig')

    print('\n传统规则方法阈值搜索完成：')
    print(f'  tune_split = {tune_split}')
    for c, t in thresholds.items():
        row = detail[detail['class'] == c].iloc[0]
        print(
            f'  {c:12s}: threshold={t:.3f}, '
            f'tune_p={row["precision"]:.3f}, tune_r={row["recall"]:.3f}, tune_f1={row["f1"]:.3f}'
        )
    print('  阈值文件:', csv_out)
    return thresholds


def main():
    parser = argparse.ArgumentParser(
        description='06：运行传统规则方法。该方法只使用投影点统计与几何规则，不使用 CLIP、FusionMLP 和真实标签参与预测。'
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument('--eval-all', action='store_true', help='评估全部 mini-nuScenes 帧；默认只评估 test split')
    parser.add_argument('--limit', type=int, default=None, help='只评估前 N 帧，调试用；若要与 04 对比，04 也必须使用相同 limit')
    parser.add_argument('--skip-existing', action='store_true', help='如果 predictions_traditional_rule.csv 已存在且测试帧范围一致，则直接汇总不重新运行')
    parser.add_argument('--tune-split', default='val', choices=['train', 'val'], help='传统规则阈值搜索使用 train 或 val；默认 val，不能使用 test')
    parser.add_argument('--threshold-min', type=float, default=0.45, help='传统规则阈值搜索下限')
    parser.add_argument('--threshold-max', type=float, default=0.80, help='传统规则阈值搜索上限')
    parser.add_argument('--threshold-steps', type=int, default=36, help='传统规则阈值搜索步数')
    args = parser.parse_args()

    method = 'traditional_rule'

    cfg = load_app_config()

    # 传统规则阈值用 train/val split 确定，不使用 test split。
    _tune_thresholds_from_train_features(
        cfg,
        tune_split=args.tune_split,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        num_steps=args.threshold_steps,
    )

    # 传统规则方法不使用 CLIP，也不加载 FusionMLP。
    pipe = DrivingScenePipeline(device=args.device, use_clip=False, load_model=False)
    indices = frame_indices_for_evaluation(pipe.cfg, pipe.reader, eval_all=args.eval_all, split_name='test')
    if args.limit:
        indices = indices[:args.limit]

    out = pipe.cfg.csv_dir / 'predictions_traditional_rule.csv'
    if args.skip_existing and out.exists() and _csv_matches_indices(out, indices):
        df = pd.read_csv(out)
        df['method'] = method
        print('[SKIP] 已存在传统规则方法预测结果，且测试帧范围一致，直接汇总:', out)
        print('[WARN] 注意：如果刚刚重新搜索了阈值，建议不要使用 --skip-existing，应重新运行传统方法预测。')
    else:
        if args.skip_existing and out.exists():
            print('[WARN] 已有传统方法 CSV，但测试帧范围与当前参数不一致，将重新运行。')
        rows = []
        prev_scene_idx = None
        prev_structured = None

        for idx in tqdm(indices, desc='run traditional_rule'):
            scene_idx = pipe.reader.scene_token_to_index[
                pipe.reader.get_sample_scene_token(pipe.reader.samples[idx])
            ]
            if prev_scene_idx is not None and scene_idx != prev_scene_idx:
                prev_structured = None

            res = pipe.process_frame(
                idx,
                method=method,
                save_outputs=False,
                save_vis=False,
                save_result_json=False,
                prev_structured=prev_structured,
            )
            rows.extend(flatten_direction_results(res))
            prev_structured = res
            prev_scene_idx = scene_idx

        df = pd.DataFrame(rows)
        df['method'] = method

        # 记录本次传统规则预测所用阈值
        used_thresholds = get_traditional_thresholds()
        for c, t in used_thresholds.items():
            df[f'trad_threshold_{c}'] = float(t)

        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding='utf-8-sig')
        print('saved:', out)

    pack = build_single_method_summaries(df, method)
    # 传统方法单独加 traditional_ 前缀
    save_summary_pack(pack, pipe.cfg.csv_dir, prefix='traditional_')

    # 传统方法自身汇总
    paper_summary = slim_summary_for_paper(pack['summary_by_method_camera_mean'])
    keep = [c for c in ['method', 'precision', 'recall', 'f1', 'support', 'pred_positive', 'num_frames'] if c in paper_summary.columns]
    paper_summary = paper_summary[keep]
    paper_out = pipe.cfg.csv_dir / 'traditional_rule_paper_summary.csv'
    paper_summary.to_csv(paper_out, index=False, encoding='utf-8-sig')

    print('传统规则方法完成。')
    print('预测结果:', out)
    print('论文精简汇总:', paper_out)
    print('说明：06 不进行流程耗时对比；07 只比较 Precision / Recall / F1-score，不比较描述连贯性。')
    print('下一步运行: python scripts/07_make_traditional_comparison_charts.py')


if __name__ == '__main__':
    main()

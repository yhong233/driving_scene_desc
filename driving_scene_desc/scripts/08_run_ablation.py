from pathlib import Path
import sys
import argparse
import json
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline import (
    DrivingScenePipeline,
    DEFAULT_ABLATION_METHODS,
    ABLATION_FIXED_THRESHOLD,
    ABLATION_FALLBACK_THRESHOLD,
)
from src.datasets.nuscenes_reader import TARGET_CLASSES
from src.datasets.split_utils import frame_indices_for_evaluation
from src.eval.run_experiments import flatten_direction_results
from src.eval.csv_summary import (
    build_multi_method_summaries,
    save_summary_pack,
    slim_ablation_f1_summary_for_paper,
    slim_ablation_per_class_f1_for_paper,
)


OURS_METHOD = 'ours'


def _unique_frame_indices_from_csv(csv_path: Path) -> set[int]:
    df = pd.read_csv(csv_path)
    if 'frame_idx' not in df.columns:
        raise ValueError(f'{csv_path} 中缺少 frame_idx 列，无法确认测试集是否一致。')
    return set(df['frame_idx'].dropna().astype(int).unique())


def _csv_matches_indices(csv_path: Path, indices: list[int]) -> bool:
    try:
        got = _unique_frame_indices_from_csv(csv_path)
    except Exception:
        return False
    return got == set(int(x) for x in indices)


def _assert_csv_matches_indices(csv_path: Path, indices: list[int], hint: str = '') -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f'未找到 {csv_path}。{hint}')
    df = pd.read_csv(csv_path)
    if 'frame_idx' not in df.columns:
        raise ValueError(f'{csv_path} 中缺少 frame_idx 列，无法确认测试集是否一致。')
    expected = set(int(x) for x in indices)
    got = set(df['frame_idx'].dropna().astype(int).unique())
    if got != expected:
        missing = sorted(expected - got)[:10]
        extra = sorted(got - expected)[:10]
        raise ValueError(
            f'{csv_path.name} 的测试帧范围与当前 08 消融实验不一致。\n'
            f'  当前应有帧数: {len(expected)}，CSV中帧数: {len(got)}。\n'
            f'  缺少示例: {missing}\n'
            f'  多出示例: {extra}\n'
            f'请使用同样参数重新运行 04_run_inference.py，例如：python scripts/04_run_inference.py --method ours'
        )
    return df


def _normalize_method_df(df: pd.DataFrame, method: str) -> pd.DataFrame:
    df = df.copy()
    df['method'] = method
    return df


def _remove_old_val_threshold_files(csv_dir: Path) -> None:
    for name in ['ablation_thresholds.csv', 'ablation_thresholds.json']:
        path = csv_dir / name
        if path.exists():
            path.unlink()
            print('[CLEAN] 已删除旧消融阈值搜索文件:', path)


def save_fixed_threshold_note(csv_dir: Path) -> None:
    """保存固定阈值"""
    rows = []
    for method in DEFAULT_ABLATION_METHODS:
        for cls_name in TARGET_CLASSES:
            rows.append({
                'method': method,
                'class': cls_name,
                'threshold': float(ABLATION_FIXED_THRESHOLD),
                'fallback_threshold': float(ABLATION_FALLBACK_THRESHOLD),
                'note': 'fixed threshold for non-training ablation methods',
            })
    out_csv = csv_dir / 'ablation_fixed_thresholds.csv'
    out_json = csv_dir / 'ablation_fixed_thresholds.json'
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
    with out_json.open('w', encoding='utf-8') as f:
        json.dump({
            'methods': DEFAULT_ABLATION_METHODS,
            'target_classes': TARGET_CLASSES,
            'fixed_threshold': float(ABLATION_FIXED_THRESHOLD),
            'fallback_threshold': float(ABLATION_FALLBACK_THRESHOLD),
            'rule': (
                'For image_only/lidar_only/normal_fusion, output all classes with score >= fixed_threshold. '
                'If no class reaches fixed_threshold and the best score > fallback_threshold, output the best class.'
            ),
            'ours': 'reused from outputs/csv/predictions_ours.csv; thresholds are those saved in FusionMLP checkpoint',
        }, f, ensure_ascii=False, indent=2)
    print('消融实验固定阈值说明已保存:')
    print('  -', out_csv)
    print('  -', out_json)


def _f1_only_pack_for_ablation(pack: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    消融实验汇总文件
    """
    keep_map = {
        'summary_by_method': ['method', 'f1', 'support', 'pred_positive', 'num_frames'],
        'summary_per_class': ['method', 'class', 'f1', 'support', 'pred_positive'],
        'summary_by_condition': ['method', 'condition', 'f1', 'support', 'pred_positive', 'num_frames'],
        'summary_by_scene': ['method', 'scene_idx', 'condition', 'f1', 'support', 'pred_positive', 'num_frames'],
        'summary_by_camera': ['method', 'camera_name', 'f1', 'support', 'pred_positive', 'num_frames'],
        'summary_by_camera_class': ['method', 'camera_name', 'class', 'f1', 'support', 'pred_positive'],
        'summary_by_method_camera_mean': ['method', 'f1', 'support', 'pred_positive', 'num_frames'],
        'summary_per_class_camera_mean': ['method', 'class', 'f1', 'support', 'pred_positive'],
        'summary_by_condition_camera_mean': ['method', 'condition', 'f1', 'support', 'pred_positive', 'num_frames'],
    }
    out = {}
    for name, df in pack.items():
        if df is None or len(df) == 0:
            out[name] = df
            continue
        keep = [c for c in keep_map.get(name, ['method', 'f1']) if c in df.columns]
        out[name] = df[keep].copy() if keep else df.copy()
    return out


def run_one_method(pipe, method: str, indices: list[int], skip_existing: bool = False) -> pd.DataFrame:
    out = pipe.cfg.csv_dir / f'predictions_{method}.csv'
    if skip_existing and out.exists() and _csv_matches_indices(out, indices):
        print(f'[SKIP] {method}: 已存在并覆盖当前测试帧范围，直接读取 {out}')
        return _normalize_method_df(pd.read_csv(out), method)

    if method == OURS_METHOD:
        return _normalize_method_df(
            _assert_csv_matches_indices(
                pipe.cfg.csv_dir / 'predictions_ours.csv',
                indices,
                hint='请先运行：python scripts/04_run_inference.py --method ours'
            ),
            OURS_METHOD,
        )

    rows = []
    prev_scene_idx = None
    prev_structured = None

    for idx in tqdm(indices, desc=f'run ablation {method}'):
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
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print('saved:', out)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='08：运行特征融合消融实验。image_only/lidar_only/normal_fusion 使用统一固定阈值，ours 直接复用 04 的结果。'
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument('--eval-all', action='store_true', help='评估全部 mini-nuScenes 帧；默认只评估 test split')
    parser.add_argument('--limit', type=int, default=None, help='只评估前 N 帧，调试用；必须与 04 的 ours 结果一致')
    parser.add_argument('--skip-existing', action='store_true', help='已有 predictions_METHOD.csv 且帧范围匹配时跳过重跑')
    parser.add_argument('--methods', nargs='+', default=DEFAULT_ABLATION_METHODS,
                        help='需要重新运行的消融方法。默认 image_only lidar_only normal_fusion；ours 永远从 04 的 CSV 复用。')
    args = parser.parse_args()

    # 消融方法
    pipe = DrivingScenePipeline(device=args.device, use_clip=True, load_model=False)
    indices = frame_indices_for_evaluation(pipe.cfg, pipe.reader, eval_all=args.eval_all, split_name='test')
    if args.limit:
        indices = indices[:args.limit]

    methods_to_run = []
    for m in args.methods:
        m = str(m).strip()
        if not m or m == OURS_METHOD:
            continue
        if m not in DEFAULT_ABLATION_METHODS:
            raise ValueError(f'不支持的消融方法: {m}。可选: {DEFAULT_ABLATION_METHODS}')
        if m not in methods_to_run:
            methods_to_run.append(m)

    _remove_old_val_threshold_files(pipe.cfg.csv_dir)
    save_fixed_threshold_note(pipe.cfg.csv_dir)

    print('消融实验设置：')
    print('  rerun methods      =', methods_to_run)
    print('  reuse method       = ours  # 来自 outputs/csv/predictions_ours.csv')
    print('  frames             =', len(indices))
    print('  fixed threshold    =', ABLATION_FIXED_THRESHOLD)
    print('  fallback threshold =', ABLATION_FALLBACK_THRESHOLD)
    print('  说明：image_only/lidar_only/normal_fusion 统一阈值；08 不重新运行 ours。')
    print('  若 predictions_ours.csv 与当前测试帧不一致会直接报错。')

    dfs = []
    for method in methods_to_run:
        dfs.append(run_one_method(pipe, method, indices, skip_existing=args.skip_existing))

    # 加入04本文方法结果
    ours_df = run_one_method(pipe, OURS_METHOD, indices, skip_existing=True)
    dfs.append(ours_df)

    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    all_out = pipe.cfg.csv_dir / 'predictions_ablation_all.csv'
    all_df.to_csv(all_out, index=False, encoding='utf-8-sig')

    pack = build_multi_method_summaries(dfs)
    pack_for_save = _f1_only_pack_for_ablation(pack)
    save_summary_pack(pack_for_save, pipe.cfg.csv_dir, prefix='ablation_')

    # 消融实验记录
    paper_summary = slim_ablation_f1_summary_for_paper(pack['summary_by_method_camera_mean'])
    paper_out = pipe.cfg.csv_dir / 'ablation_paper_summary.csv'
    paper_summary.to_csv(paper_out, index=False, encoding='utf-8-sig')

    paper_per_class = slim_ablation_per_class_f1_for_paper(pack['summary_per_class_camera_mean'])
    paper_per_class_out = pipe.cfg.csv_dir / 'ablation_paper_per_class_f1.csv'
    paper_per_class.to_csv(paper_per_class_out, index=False, encoding='utf-8-sig')

    print('消融实验完成。')
    print('总预测结果:', all_out)
    print('F1精简汇总:', paper_out)
    print('类别级F1汇总:', paper_per_class_out)
    print('下一步运行: python scripts/09_make_fusion_ablation_charts.py')


if __name__ == '__main__':
    main()

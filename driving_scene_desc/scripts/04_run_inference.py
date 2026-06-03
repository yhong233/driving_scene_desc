from pathlib import Path
import sys
import argparse
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline import DrivingScenePipeline
from src.eval.run_experiments import flatten_direction_results, evaluate_prediction_df
from src.datasets.split_utils import frame_indices_for_evaluation


def main():
    parser = argparse.ArgumentParser(description='运行单个方法推理；默认 ours 会保存可视化与 JSON。')
    parser.add_argument('--method', default='ours')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--eval-all', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--no-clip', action='store_true')
    parser.add_argument('--no-vis', action='store_true', help='不保存可视化图片，只生成 CSV；调试速度更快')
    parser.add_argument('--no-json', action='store_true', help='不保存逐帧 JSON')
    parser.add_argument('--warmup', type=int, default=1, help='正式记录前预热帧数，默认1，用于排除CLIP/PyTorch首次前向初始化开销')
    parser.add_argument('--timing-only', action='store_true', help='只统计核心推理耗时，不保存可视化和逐帧JSON；用于生成更稳定的耗时统计')
    args = parser.parse_args()

    if args.timing_only:
        args.no_vis = True
        args.no_json = True

    pipe = DrivingScenePipeline(device=args.device, use_clip=not args.no_clip, load_model=True)

    indices = frame_indices_for_evaluation(pipe.cfg, pipe.reader, eval_all=args.eval_all, split_name='test')

    if args.limit:
        indices = indices[:args.limit]

    # 预热：不保存、不记录。用于排除 CLIP 首次前向、CUDA/PyTorch 初始化、文件缓存建立等开销。
    if args.warmup and len(indices) > 0:
        warm_indices = indices[:max(0, min(args.warmup, len(indices)))]
        warm_prev = None
        for warm_idx in tqdm(warm_indices, desc='warmup', leave=False):
            try:
                warm_prev = pipe.process_frame(
                    warm_idx,
                    method=args.method,
                    save_outputs=False,
                    save_vis=False,
                    save_result_json=False,
                    prev_structured=warm_prev,
                )
            except Exception as e:
                print(f'[WARN] warmup frame {warm_idx} failed: {e}')

    rows = []
    prev_structured = None
    prev_scene_idx = None

    for idx in tqdm(indices, desc=f'inference {args.method}'):
        scene_idx = pipe.reader.scene_token_to_index[
            pipe.reader.get_sample_scene_token(pipe.reader.samples[idx])
        ]
        if prev_scene_idx is not None and scene_idx != prev_scene_idx:
            prev_structured = None

        # 04 用于展示。默认只对 ours 保存图和 JSON；其它方法主要用于快速调试 CSV。
        save_outputs = (args.method == 'ours') and (not args.no_json or not args.no_vis)
        res = pipe.process_frame(
            idx,
            method=args.method,
            save_outputs=save_outputs,
            save_vis=not args.no_vis,
            save_result_json=not args.no_json,
            prev_structured=prev_structured,
        )
        rows.extend(flatten_direction_results(res))
        prev_structured = res
        prev_scene_idx = scene_idx

    df = pd.DataFrame(rows)
    out = pipe.cfg.csv_dir / f'predictions_{args.method}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8-sig')

    summary, _ = evaluate_prediction_df(df, args.method)
    print('saved:', out)
    print('direction-level summary:')
    print(summary)
    if 'core_runtime_ms' in df.columns and len(df):
        rt = df.drop_duplicates('frame_idx')['core_runtime_ms']
        if len(rt) > 1:
            stable_rt = rt.iloc[1:]
        else:
            stable_rt = rt
        print(f'core runtime mean(all)    : {rt.mean():.1f} ms')
        print(f'core runtime mean(stable) : {stable_rt.mean():.1f} ms  # skipped first recorded frame')


if __name__ == '__main__':
    main()

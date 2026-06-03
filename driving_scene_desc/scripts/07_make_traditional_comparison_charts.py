from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common.config import load_app_config
from src.eval.csv_summary import build_multi_method_summaries, save_summary_pack, slim_summary_for_paper
from src.eval.plotting import plot_traditional_vs_ours_metrics, plot_traditional_vs_ours_f1


def _frame_set(df: pd.DataFrame, name: str) -> set[int]:
    if 'frame_idx' not in df.columns:
        raise ValueError(f'{name} 缺少 frame_idx 列，无法判断是否使用同一测试集。')
    return set(df['frame_idx'].dropna().astype(int).unique())


def _assert_same_frames(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> None:
    a = _frame_set(left, left_name)
    b = _frame_set(right, right_name)
    if a != b:
        raise ValueError(
            f'{left_name} 与 {right_name} 使用的测试帧不一致，不能直接生成对比图。\n'
            f'  {left_name} 帧数: {len(a)}，{right_name} 帧数: {len(b)}\n'
            f'  {left_name} 缺少示例: {sorted(b - a)[:10]}\n'
            f'  {right_name} 缺少示例: {sorted(a - b)[:10]}\n'
            f'请用相同的 --eval-all / --limit 参数重新运行 04 和 06。'
        )


def main():
    cfg = load_app_config()
    ours_path = cfg.csv_dir / 'predictions_ours.csv'
    trad_path = cfg.csv_dir / 'predictions_traditional_rule.csv'

    if not ours_path.exists():
        raise FileNotFoundError('未找到 predictions_ours.csv，请先运行: python scripts/04_run_inference.py --method ours')
    if not trad_path.exists():
        raise FileNotFoundError('未找到 predictions_traditional_rule.csv，请先运行: python scripts/06_run_traditional.py')

    ours_df = pd.read_csv(ours_path)
    trad_df = pd.read_csv(trad_path)
    ours_df['method'] = 'ours'
    trad_df['method'] = 'traditional_rule'

    _assert_same_frames(ours_df, trad_df, 'predictions_ours.csv', 'predictions_traditional_rule.csv')

    pack = build_multi_method_summaries([trad_df, ours_df])
    # 保存传统 vs 本文
    save_summary_pack(pack, cfg.csv_dir, prefix='traditional_vs_ours_')

    summary_path = cfg.csv_dir / 'traditional_vs_ours_summary_by_method_camera_mean.csv'
    if not summary_path.exists() or len(pd.read_csv(summary_path)) == 0:
        summary_path = cfg.csv_dir / 'traditional_vs_ours_summary_by_method.csv'

    paper_summary = slim_summary_for_paper(pd.read_csv(summary_path))
    keep = [c for c in ['method', 'precision', 'recall', 'f1', 'support', 'pred_positive', 'num_frames'] if c in paper_summary.columns]
    paper_summary = paper_summary[keep]
    paper_out = cfg.csv_dir / 'traditional_vs_ours_paper_summary.csv'
    paper_summary.to_csv(paper_out, index=False, encoding='utf-8-sig')

    plot_traditional_vs_ours_metrics(summary_path, cfg.chart_dir, filename='07_traditional_vs_ours_metrics.png')
    plot_traditional_vs_ours_f1(summary_path, cfg.chart_dir, filename='07_traditional_vs_ours_f1.png')

    print('传统规则方法与本文方法对比图已保存:', cfg.chart_dir)
    print('生成图表:')
    print('  - 07_traditional_vs_ours_metrics.png')
    print('  - 07_traditional_vs_ours_f1.png')
    print('论文精简汇总:', paper_out)
    print('说明：07 只比较 Precision / Recall / F1-score，不比较描述连贯性和流程耗时。')


if __name__ == '__main__':
    main()

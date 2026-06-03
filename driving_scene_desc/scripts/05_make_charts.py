from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common.config import load_app_config
from src.eval.csv_summary import check_prediction_columns, build_single_method_summaries, save_summary_pack
from src.eval.plotting import (
    plot_ours_metric_summary,
    plot_ours_condition,
    plot_ours_per_class,
    plot_camera_compare,
    plot_camera_class_heatmap,
    plot_runtime,
    plot_fusion_train_loss,
)


def build_ours_summaries_from_predictions(cfg):
    """05：由 04_run_inference.py 生成的 predictions_ours.csv 构建本文方法统计 CSV。"""
    pred_path = cfg.csv_dir / 'predictions_ours.csv'
    if not pred_path.exists():
        print('[WARN] 未找到 predictions_ours.csv。请先运行: python scripts/04_run_inference.py --method ours')
        return False

    df = pd.read_csv(pred_path)
    if len(df) == 0:
        print('[WARN] predictions_ours.csv 为空，无法生成性能图。')
        return False

    missing = check_prediction_columns(df)
    if missing:
        print(f'[WARN] predictions_ours.csv 缺少必要列: {missing}')
        return False

    pack = build_single_method_summaries(df, 'ours')
    # 本实验主结果仍保存为通用 summary 文件，方便 GUI/论文图表读取。
    save_summary_pack(pack, cfg.csv_dir)
    print('[OK] 已基于 predictions_ours.csv 生成本文方法性能统计 CSV。')
    return True


def main():
    cfg = load_app_config()

    # 05 只绘制本文方法的实验结果图。
    if not (cfg.csv_dir / 'summary_by_method_camera_mean.csv').exists():
        build_ours_summaries_from_predictions(cfg)

    generated = []

    main_summary = cfg.csv_dir / 'summary_by_method_camera_mean.csv'
    if main_summary.exists():
        plot_ours_metric_summary(main_summary, cfg.chart_dir, filename='05_ours_main_metrics_camera_mean.png')
        generated.append('05_ours_main_metrics_camera_mean.png')
    elif (cfg.csv_dir / 'summary_by_method.csv').exists():
        plot_ours_metric_summary(cfg.csv_dir / 'summary_by_method.csv', cfg.chart_dir, filename='05_ours_main_metrics_direction_overall.png')
        generated.append('05_ours_main_metrics_direction_overall.png')

    condition_summary = cfg.csv_dir / 'summary_by_condition_camera_mean.csv'
    if condition_summary.exists():
        plot_ours_condition(condition_summary, cfg.chart_dir, filename='05_ours_condition_f1_camera_mean.png')
        generated.append('05_ours_condition_f1_camera_mean.png')
    elif (cfg.csv_dir / 'summary_by_condition.csv').exists():
        plot_ours_condition(cfg.csv_dir / 'summary_by_condition.csv', cfg.chart_dir, filename='05_ours_condition_f1_direction_overall.png')
        generated.append('05_ours_condition_f1_direction_overall.png')

    per_class_summary = cfg.csv_dir / 'summary_per_class_camera_mean.csv'
    if per_class_summary.exists():
        plot_ours_per_class(per_class_summary, cfg.chart_dir, filename='05_ours_per_class_f1_camera_mean.png')
        generated.append('05_ours_per_class_f1_camera_mean.png')
    elif (cfg.csv_dir / 'summary_per_class.csv').exists():
        plot_ours_per_class(cfg.csv_dir / 'summary_per_class.csv', cfg.chart_dir, filename='05_ours_per_class_f1_direction_overall.png')
        generated.append('05_ours_per_class_f1_direction_overall.png')

    if (cfg.csv_dir / 'summary_by_camera.csv').exists():
        plot_camera_compare(cfg.csv_dir / 'summary_by_camera.csv', cfg.chart_dir, filename='05_ours_camera_f1.png')
        generated.append('05_ours_camera_f1.png')

    if (cfg.csv_dir / 'summary_by_camera_class.csv').exists():
        plot_camera_class_heatmap(cfg.csv_dir / 'summary_by_camera_class.csv', cfg.chart_dir, filename='05_ours_camera_class_heatmap.png')
        generated.append('05_ours_camera_class_heatmap.png')


    if cfg.fusion_ckpt.exists():
        plot_fusion_train_loss(cfg.fusion_ckpt, cfg.chart_dir, filename='05_fusion_mlp_train_loss.png')
        generated.append('05_fusion_mlp_train_loss.png')

    # 实时性分析图
    if (cfg.csv_dir / 'predictions_ours.csv').exists():
        plot_runtime(cfg.csv_dir / 'predictions_ours.csv', cfg.chart_dir, filename='05_runtime_core_10frames.png')
        generated.append('05_runtime_core_10frames.png')

    print('本文方法实验图已保存:', cfg.chart_dir)
    print('生成图表:')
    for name in generated:
        print('  -', name)
    print('传统规则方法请运行: python scripts/06_run_traditional.py')
    print('传统方法对比图请运行: python scripts/07_make_traditional_comparison_charts.py')


if __name__ == '__main__':
    main()

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common.config import load_app_config
from src.eval.plotting import plot_ablation_f1, plot_per_class_compare


def main():
    cfg = load_app_config()
    summary = cfg.csv_dir / 'ablation_summary_by_method_camera_mean.csv'
    per_class = cfg.csv_dir / 'ablation_summary_per_class_camera_mean.csv'

    if not summary.exists():
        raise FileNotFoundError('未找到 ablation_summary_by_method_camera_mean.csv，请先运行: python scripts/08_run_ablation.py')

    # 消融实验对比 F1-score。
    plot_ablation_f1(summary, cfg.chart_dir, filename='09_fusion_ablation_f1.png')

    if per_class.exists():
        plot_per_class_compare(
            per_class,
            cfg.chart_dir,
            title='不同融合策略三类目标 F1-score 对比',
            filename='09_fusion_ablation_per_class_f1.png',
            methods=['image_only', 'lidar_only', 'normal_fusion', 'ours'],
        )

    print('特征融合效果对比图已保存:', cfg.chart_dir)
    print('生成图表:')
    print('  - 09_fusion_ablation_f1.png')
    if per_class.exists():
        print('  - 09_fusion_ablation_per_class_f1.png')
    print('说明：消融实验只比较 F1-score，不比较描述连贯性和流程耗时。')


if __name__ == '__main__':
    main()

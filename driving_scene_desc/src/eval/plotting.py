from __future__ import annotations

from pathlib import Path
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch

DEFAULT_COMPARE_METHODS = ['traditional_rule', 'ours']
METHOD_ORDER = DEFAULT_COMPARE_METHODS
CAMERA_ORDER = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
METHOD_NAME = {
    'traditional_rule': '传统规则',
    'clip_only': '仅CLIP',
    'lidar_only': '仅LiDAR',
    'normal_fusion': '普通融合',
    'ours': '本文方法',
    'image_only': '仅图像',
    'no_clip': '无CLIP',
    'no_image_proxy': '无图像代理',
}
CLASS_NAME = {
    'vehicle': '车辆',
    'pedestrian': '行人',
    'obstacle': '障碍物',
}
CONDITION_NAME = {
    'day_sunny': '白天晴天',
    'night_rainy': '夜间雨天',
}
CAMERA_NAME_ZH = {
    'CAM_FRONT': '前方',
    'CAM_FRONT_LEFT': '左前方',
    'CAM_FRONT_RIGHT': '右前方',
    'CAM_BACK': '后方',
    'CAM_BACK_LEFT': '左后方',
    'CAM_BACK_RIGHT': '右后方',
}


def setup_chinese_font():
    candidates = [
        'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi',
        'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Micro Hei',
        'Arial Unicode MS', 'DejaVu Sans'
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in installed:
            chosen = name
            break
    if chosen:
        plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore', message='Glyph .* missing from font.*')


setup_chinese_font()


def _ensure_out(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def add_value_labels(ax, fmt='{:.3f}', fontsize=8):
    for p in ax.patches:
        h = p.get_height()
        if pd.notna(h):
            ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h), ha='center', va='bottom', fontsize=fontsize)


def set_xlabels_horizontal(ax):
    ax.tick_params(axis='x', labelrotation=0)
    for label in ax.get_xticklabels():
        label.set_ha('center')


def _filter_methods(df, methods=None):
    if 'method' not in df:
        return df
    methods = methods or DEFAULT_COMPARE_METHODS
    df = df[df['method'].isin(methods)].copy()
    df['method'] = pd.Categorical(df['method'], categories=methods, ordered=True)
    df = df.sort_values('method')
    df['method_label'] = df['method'].astype(str).map(METHOD_NAME).fillna(df['method'].astype(str))
    return df


def _only_ours(df):
    if 'method' in df.columns:
        sub = df[df['method'] == 'ours'].copy()
        if len(sub) > 0:
            return sub
    return df.copy()


def plot_ours_metric_summary(summary_csv, out_dir, filename='06_ours_main_metrics.png'):
    """本文方法总体性能指标图。"""
    out_dir = _ensure_out(out_dir)
    df = pd.read_csv(summary_csv)
    row_df = _only_ours(df)
    if len(row_df) == 0:
        return
    row = row_df.iloc[0]
    metrics = []
    labels = []
    for col, label in [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1-score'),
                       ('coherence_score', '描述连贯性')]:
        if col in row:
            metrics.append(float(row[col]))
            labels.append(label)
    if not metrics:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, metrics)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('score')
    ax.set_title('本文方法性能指标')
    add_value_labels(ax)
    set_xlabels_horizontal(ax)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def plot_ours_condition(condition_csv, out_dir, filename='06_ours_condition_f1.png'):
    """本文方法在不同场景条件下的 F1。"""
    out_dir = _ensure_out(out_dir)
    df = pd.read_csv(condition_csv)
    sub = _only_ours(df)
    if len(sub) == 0 or 'condition' not in sub.columns:
        return
    sub = sub.copy()
    sub['condition_label'] = sub['condition'].map(CONDITION_NAME).fillna(sub['condition'])
    ax = sub.plot(x='condition_label', y='f1', kind='bar', legend=False, figsize=(7, 4))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1-score')
    ax.set_xlabel('场景条件')
    ax.set_title('本文方法不同场景条件 F1-score')
    add_value_labels(ax)
    set_xlabels_horizontal(ax)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def plot_ours_per_class(per_class_csv, out_dir, filename='06_ours_per_class_f1.png'):
    """本文方法三类目标 F1。"""
    out_dir = _ensure_out(out_dir)
    df = pd.read_csv(per_class_csv)
    sub = _only_ours(df)
    if len(sub) == 0 or 'class' not in sub.columns:
        return
    sub = sub.copy()
    sub['class_label'] = sub['class'].map(CLASS_NAME).fillna(sub['class'])
    ax = sub.plot(x='class_label', y='f1', kind='bar', legend=False, figsize=(7, 4))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1-score')
    ax.set_xlabel('类别')
    ax.set_title('本文方法三类目标 F1-score')
    add_value_labels(ax)
    set_xlabels_horizontal(ax)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def plot_camera_compare(camera_csv, out_dir, filename='06_ours_camera_f1.png'):
    """本文方法六相机方向 F1 对比。"""
    out_dir = _ensure_out(out_dir)
    df = pd.read_csv(camera_csv)
    if len(df) == 0:
        return
    sub = _only_ours(df)
    if len(sub) == 0:
        return
    sub = sub.copy()
    sub['camera_name'] = pd.Categorical(sub['camera_name'], categories=CAMERA_ORDER, ordered=True)
    sub = sub.sort_values('camera_name')
    sub['camera_label'] = sub['camera_name'].astype(str).map(CAMERA_NAME_ZH).fillna(sub['camera_name'].astype(str))
    ax = sub.plot(x='camera_label', y='f1', kind='bar', legend=False, figsize=(8, 4))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1-score')
    ax.set_xlabel('相机方向')
    ax.set_title('本文方法六相机方向 F1-score')
    add_value_labels(ax)
    set_xlabels_horizontal(ax)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def plot_camera_class_heatmap(camera_class_csv, out_dir, filename='06_ours_camera_class_heatmap.png'):
    """本文方法六相机×类别 F1 热力图。"""
    out_dir = _ensure_out(out_dir)
    df = pd.read_csv(camera_class_csv)
    if len(df) == 0:
        return
    sub = _only_ours(df)
    if len(sub) == 0:
        return
    pivot = sub.pivot(index='camera_name', columns='class', values='f1').reindex(CAMERA_ORDER)
    pivot = pivot.rename(index=CAMERA_NAME_ZH, columns=CLASS_NAME)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0, ha='center')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title('本文方法六相机方向×类别 F1 矩阵')
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, label='F1-score')
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def plot_runtime(predictions_csv, out_dir, filename='06_runtime_core_10frames.png', skip_warmup=1, num_frames=10):
    """绘制稳定阶段核心流程耗时。"""
    out_dir = _ensure_out(out_dir)
    df = pd.read_csv(predictions_csv)
    if len(df) == 0 or 'core_runtime_ms' not in df.columns:
        return

    df = df.drop_duplicates('frame_idx').sort_values('frame_idx').copy()
    df['core_runtime_ms'] = pd.to_numeric(df['core_runtime_ms'], errors='coerce')
    df = df.dropna(subset=['core_runtime_ms'])
    if len(df) == 0:
        return

    # 去掉第一个正式记录帧，减少 warm-up 对图的影响。
    if skip_warmup > 0 and len(df) > skip_warmup:
        plot_df = df.iloc[skip_warmup:].head(num_frames).copy()
    else:
        plot_df = df.head(num_frames).copy()

    if len(plot_df) == 0:
        plot_df = df.head(num_frames).copy()

    ax = plot_df.plot(x='frame_idx', y='core_runtime_ms', kind='bar', legend=False, figsize=(9, 4))
    ax.set_ylabel('ms')
    ax.set_xlabel('帧序号')
    ax.set_title('单帧耗时图')
    add_value_labels(ax, fmt='{:.1f}')
    set_xlabels_horizontal(ax)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()

    # 同步保存绘图所用的耗时明细
    plot_df[['frame_idx', 'core_runtime_ms']].to_csv(
        out_dir / filename.replace('.png', '.csv'),
        index=False,
        encoding='utf-8-sig'
    )


def plot_fusion_train_loss(ckpt_path, out_dir, filename='05_fusion_mlp_train_loss.png'):
    """绘制 FusionMLP 训练损失变化曲线。"""
    out_dir = _ensure_out(out_dir)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return
    ckpt = torch.load(ckpt_path, map_location='cpu')
    history = ckpt.get('history', [])
    if not history:
        return
    hist_df = pd.DataFrame(history)
    if 'epoch' not in hist_df.columns or 'loss' not in hist_df.columns:
        return
    cols = [c for c in ['loss', 'cls_loss', 'distill_loss'] if c in hist_df.columns]
    ax = hist_df.plot(x='epoch', y=cols, kind='line', marker='o', figsize=(8, 4))
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('loss')
    ax.set_title('FusionMLP 训练损失变化曲线')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()

    save_cols = ['epoch'] + cols + [c for c in ['val_f1', 'val_precision', 'val_recall'] if c in hist_df.columns]
    hist_df[save_cols].to_csv(out_dir / filename.replace('.png', '.csv'), index=False, encoding='utf-8-sig')


def plot_method_compare(summary_csv, out_dir, title='不同方法 F1-score 对比', filename='07_method_f1_compare.png', methods=None):
    """不同方法对比图。"""
    out_dir = _ensure_out(out_dir)
    df = _filter_methods(pd.read_csv(summary_csv), methods=methods)
    if len(df) == 0:
        return
    ax = df.plot(x='method_label', y='f1', kind='bar', legend=False, figsize=(8, 4))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1-score')
    ax.set_xlabel('方法')
    ax.set_title(title)
    add_value_labels(ax)
    set_xlabels_horizontal(ax)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()

    cols = [c for c in ['f1', 'coherence_score'] if c in df.columns]
    if len(cols) >= 2:
        plot_df = df[['method_label'] + cols].copy()
        ax = plot_df.plot(x='method_label', y=cols, kind='bar', figsize=(9, 4))
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('score')
        ax.set_xlabel('方法')
        ax.set_title(title.replace('F1-score', '综合指标'))
        set_xlabels_horizontal(ax)
        plt.tight_layout()
        plt.savefig(out_dir / filename.replace('f1', 'quality'), dpi=200)
        plt.close()


def plot_condition_compare(condition_csv, out_dir, title='不同场景条件下 F1-score 对比', filename='07_condition_f1_compare.png', methods=None):
    """不同方法在不同场景条件下的 F1 对比。"""
    out_dir = _ensure_out(out_dir)
    df = _filter_methods(pd.read_csv(condition_csv), methods=methods)
    if len(df) == 0 or 'condition' not in df.columns:
        return
    df = df.copy()
    df['condition_label'] = df['condition'].map(CONDITION_NAME).fillna(df['condition'])
    pivot = df.pivot(index='condition_label', columns='method_label', values='f1')
    ax = pivot.plot(kind='bar', figsize=(9, 4))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1-score')
    ax.set_xlabel('场景条件')
    ax.set_title(title)
    set_xlabels_horizontal(ax)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def plot_per_class_compare(per_class_csv, out_dir, title='不同方法三类目标 F1 对比', filename='07_per_class_f1_compare.png', methods=None):
    """不同方法在三个目标类别上的 F1 对比。"""
    out_dir = _ensure_out(out_dir)
    methods = methods or ['image_only', 'lidar_only', 'normal_fusion', 'ours']
    df = _filter_methods(pd.read_csv(per_class_csv), methods=methods)
    if len(df) == 0 or 'class' not in df.columns:
        return
    df = df.copy()
    df['class_label'] = df['class'].map(CLASS_NAME).fillna(df['class'])

    # 类别顺序固定，避免不同 CSV 行顺序导致论文图顺序变化。
    class_order = [CLASS_NAME[c] for c in ['pedestrian', 'vehicle', 'obstacle']]
    method_order = [METHOD_NAME.get(m, m) for m in methods]

    pivot = df.pivot(index='class_label', columns='method_label', values='f1')
    pivot = pivot.reindex(index=class_order)
    pivot = pivot.reindex(columns=method_order)

    ax = pivot.plot(kind='bar', figsize=(9, 4))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1-score')
    ax.set_xlabel('类别')
    ax.set_title(title)
    add_value_labels(ax, fmt='{:.3f}', fontsize=8)
    set_xlabels_horizontal(ax)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def plot_traditional_gap(summary_csv, out_dir, filename='07_ours_vs_traditional_gap.png'):
    """本文方法与传统规则方法性能差异图。"""
    out_dir = _ensure_out(out_dir)
    df = pd.read_csv(summary_csv)
    if not {'method', 'f1'}.issubset(df.columns):
        return
    sub = df[df['method'].isin(['traditional_rule', 'ours'])].copy()
    if len(sub) < 2:
        return
    sub['method_label'] = sub['method'].map(METHOD_NAME)
    ax = sub.plot(x='method_label', y='f1', kind='bar', legend=False, figsize=(6, 4))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1-score')
    ax.set_xlabel('方法')
    ax.set_title('本文方法与传统规则方法 F1 对比')
    add_value_labels(ax)
    set_xlabels_horizontal(ax)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()



def plot_traditional_vs_ours_metrics(summary_csv, out_dir, filename='07_traditional_vs_ours_metrics.png'):
    """传统规则方法与本文方法的 Precision / Recall / F1 对比。"""
    out_dir = _ensure_out(out_dir)
    df = pd.read_csv(summary_csv)
    df = _filter_methods(df, methods=['traditional_rule', 'ours'])
    if len(df) == 0:
        return
    metrics = [c for c in ['precision', 'recall', 'f1'] if c in df.columns]
    if not metrics:
        return
    metric_name = {
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-score',
    }
    plot_df = df[['method_label'] + metrics].rename(columns=metric_name)
    ax = plot_df.plot(x='method_label', y=[metric_name[m] for m in metrics], kind='bar', figsize=(8, 4))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('score')
    ax.set_xlabel('方法')
    ax.set_title('传统规则方法与本文方法识别性能对比')
    add_value_labels(ax)
    set_xlabels_horizontal(ax)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()

def plot_traditional_vs_ours_f1(summary_csv, out_dir, filename='07_traditional_vs_ours_f1.png'):
    """传统规则方法与本文方法 F1-score 单项对比。"""
    out_dir = _ensure_out(out_dir)
    df = pd.read_csv(summary_csv)
    df = _filter_methods(df, methods=['traditional_rule', 'ours'])
    if len(df) == 0 or 'f1' not in df.columns:
        return
    ax = df.plot(x='method_label', y='f1', kind='bar', legend=False, figsize=(6, 4))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1-score')
    ax.set_xlabel('方法')
    ax.set_title('传统规则方法与本文方法 F1-score 对比')
    add_value_labels(ax)
    set_xlabels_horizontal(ax)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def plot_ablation_metrics(summary_csv, out_dir, filename='09_fusion_ablation_metrics.png'):

    return plot_ablation_f1(summary_csv, out_dir, filename=filename)


def plot_ablation_f1(summary_csv, out_dir, filename='09_fusion_ablation_f1.png'):
    """特征融合消融实验 F1-score 对比。"""
    out_dir = _ensure_out(out_dir)
    methods = ['image_only', 'lidar_only', 'normal_fusion', 'ours']
    df = _filter_methods(pd.read_csv(summary_csv), methods=methods)
    if len(df) == 0 or 'f1' not in df.columns:
        return
    ax = df.plot(x='method_label', y='f1', kind='bar', legend=False, figsize=(7, 4))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1-score')
    ax.set_xlabel('融合策略')
    ax.set_title('不同融合策略 F1-score 对比')
    add_value_labels(ax)
    set_xlabels_horizontal(ax)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()

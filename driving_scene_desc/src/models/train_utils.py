from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.nuscenes_reader import TARGET_CLASSES
from src.features.feature_extract import (
    LIDAR_FEATURE_NAMES,
    IMAGE_FEATURE_NAMES,
    CLIP_FEATURE_NAMES,
    EVIDENCE_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
)
from src.models.fusion_mlp import FusionMLP, fusion_loss, make_tensor_batch
from src.eval.metrics import multilabel_counts, metrics_from_counts
from src.datasets.split_utils import apply_split_to_features, print_split_report, split_strategy


FEATURE_COLS = (
    LIDAR_FEATURE_NAMES
    + IMAGE_FEATURE_NAMES
    + CLIP_FEATURE_NAMES
    + EVIDENCE_FEATURE_NAMES
    + CONTEXT_FEATURE_NAMES
)
LABEL_COLS = [f'label_{c}' for c in TARGET_CLASSES]


def _resolve_device(device: str):
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def _check_scene_split(scene_cfg: dict):
    train_scenes = set(scene_cfg.get('train_scenes', [1, 2, 6, 7, 9]))
    val_scenes = set(scene_cfg.get('val_scenes', [3, 4]))
    test_scenes = set(scene_cfg.get('test_scenes', [5, 8, 10]))

    if train_scenes & val_scenes:
        raise ValueError(f'train_scenes 与 val_scenes 有重叠: {sorted(train_scenes & val_scenes)}')
    if train_scenes & test_scenes:
        raise ValueError(f'train_scenes 与 test_scenes 有重叠: {sorted(train_scenes & test_scenes)}')
    if val_scenes & test_scenes:
        raise ValueError(f'val_scenes 与 test_scenes 有重叠: {sorted(val_scenes & test_scenes)}')

    return train_scenes, val_scenes, test_scenes


def compute_pos_weight(train_df: pd.DataFrame, label_cols=LABEL_COLS, max_weight: float = 6.0):
    """根据训练集标签分布计算 BCEWithLogitsLoss 的 pos_weight。"""
    y = train_df[label_cols].values.astype(np.float32)
    pos = y.sum(axis=0)
    neg = len(train_df) - pos
    pos_weight = neg / np.maximum(pos, 1.0)
    pos_weight = np.clip(pos_weight, 1.0, max_weight)
    return pos_weight.astype(np.float32)


def train_fusion_model(
    features_csv: str | Path,
    ckpt_path: str | Path,
    scene_cfg: dict,
    epochs=60,
    batch_size=64,
    lr=1e-3,
    device='auto',
    early_stop_patience: int = 15,
    max_pos_weight: float = 6.0,
    threshold_min: float = 0.20,
    distill_weight: float = 0.02,
):
    device = _resolve_device(device)
    df = pd.read_csv(features_csv)

    required = ['frame_idx', 'scene_idx'] + FEATURE_COLS + LABEL_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'train_features.csv 缺少必要列: {missing}')

    # 支持两种划分：
    # 1) scene_level：按 scene 整体划分，泛化评价更严格；
    # 2) frame_stratified：每个 scene 内按 frame_idx 划分，主流程验证更稳定。
    train_df, val_df, test_df, frame_split = apply_split_to_features(
        df,
        scene_cfg,
        Path(features_csv).parent,
    )

    if len(train_df) == 0:
        raise ValueError('训练集为空，请检查 scene_split.yaml、frame_split.csv 和 train_features.csv')
    if len(val_df) == 0:
        raise ValueError('验证集为空，请检查 scene_split.yaml、frame_split.csv 和 train_features.csv')
    if len(test_df) == 0:
        print('警告：测试集为空。训练仍可进行，但后续最终评价无有效样本。')

    print_split_report(train_df, val_df, test_df, LABEL_COLS, frame_split)

    pos_weight_np = compute_pos_weight(train_df, max_weight=max_pos_weight)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    print('\npos_weight，用于处理类别不平衡：')
    for cls_name, w in zip(TARGET_CLASSES, pos_weight_np):
        print(f'  {cls_name:12s}: {w:.3f}')

    print('\n训练超参数：')
    print(f'  distill_weight      = {distill_weight}')
    print(f'  threshold_min       = {threshold_min}')
    print(f'  early_stop_patience = {early_stop_patience}')
    print(f'  max_pos_weight      = {max_pos_weight}')

    model = FusionMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    y_train = torch.tensor(train_df[LABEL_COLS].values, dtype=torch.float32, device=device)
    xs = make_tensor_batch(train_df, device=device)
    ds = TensorDataset(*xs, y_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    best_thresholds = {c: 0.5 for c in TARGET_CLASSES}
    history = []
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        cls_losses = []
        distill_losses = []

        for batch in dl:
            *xb, yb = batch
            opt.zero_grad(set_to_none=True)
            logits, aux = model(*xb)
            loss, loss_dict = fusion_loss(
                logits,
                yb,
                aux,
                pos_weight=pos_weight,
                distill_weight=distill_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))
            cls_losses.append(loss_dict['cls_loss'])
            distill_losses.append(loss_dict['distill_loss'])

        val_metrics, thresholds = evaluate_model_on_df(
            model,
            val_df,
            device=device,
            tune_thresholds=True,
            threshold_min=threshold_min,
        )
        # 训练模型选择以“六相机方向平均 F1”为主，
        val_camera = val_metrics.get('camera_mean', val_metrics['overall'])
        record = {
            'epoch': ep,
            'loss': float(np.mean(losses)),
            'cls_loss': float(np.mean(cls_losses)),
            'distill_loss': float(np.mean(distill_losses)),
            'val_f1': val_camera['f1'],
            'val_precision': val_camera['precision'],
            'val_recall': val_camera['recall'],
            'val_overall_f1': val_metrics['overall']['f1'],
            'val_overall_precision': val_metrics['overall']['precision'],
            'val_overall_recall': val_metrics['overall']['recall'],
        }
        history.append(record)

        improved = record['val_f1'] > best_val_f1 + 1e-6
        if improved:
            best_val_f1 = record['val_f1']
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thresholds = thresholds.copy()
            no_improve = 0
        else:
            no_improve += 1

        if ep == 1 or ep % 10 == 0 or ep == epochs or improved:
            print(
                f"epoch={ep:03d} "
                f"loss={record['loss']:.4f} "
                f"cls={record['cls_loss']:.4f} "
                f"distill={record['distill_loss']:.4f} "
                f"val_cam_p={record['val_precision']:.4f} "
                f"val_cam_r={record['val_recall']:.4f} "
                f"val_cam_f1={record['val_f1']:.4f} "
                f"val_micro_f1={record['val_overall_f1']:.4f} "
                f"best_cam={best_val_f1:.4f}@{best_epoch}"
            )

        if early_stop_patience and no_improve >= early_stop_patience:
            print(f'早停触发：连续 {early_stop_patience} 轮验证集 F1 未提升，停止训练。')
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = history[-1]['epoch'] if history else epochs
        best_val_f1 = history[-1]['val_f1'] if history else 0.0

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state': best_state,
            'thresholds': best_thresholds,
            'target_classes': TARGET_CLASSES,
            'feature_cols': FEATURE_COLS,
            'label_cols': LABEL_COLS,
            'history': history,
            'scene_cfg': scene_cfg,
            'split_strategy': split_strategy(scene_cfg),
            'pos_weight': {c: float(w) for c, w in zip(TARGET_CLASSES, pos_weight_np)},
            'best_epoch': int(best_epoch),
            'best_val_f1': float(best_val_f1),
            'train_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'early_stop_patience': early_stop_patience,
                'max_pos_weight': max_pos_weight,
                'threshold_min': threshold_min,
                'distill_weight': distill_weight,
                'selection_metric': 'val_camera_mean_f1',
            },
            'note': 'FusionMLP trained on direction-level samples. Gate is not used. Best model is selected by validation camera-mean F1. Test split is not used during training or threshold tuning.',
        },
        ckpt_path,
    )
    print(f'已保存最佳模型: {ckpt_path}')
    print(f'最佳验证集 epoch = {best_epoch}, best_val_camera_mean_f1 = {best_val_f1:.4f}')
    print('最佳验证集阈值：')
    for c in TARGET_CLASSES:
        print(f'  {c:12s}: {best_thresholds[c]:.3f}')
    return ckpt_path


def load_fusion_model(ckpt_path: str | Path, device='auto'):
    device = _resolve_device(device)
    model = FusionMLP().to(device)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return model, {c: 0.5 for c in TARGET_CLASSES}, None
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt.get('thresholds', {c: 0.5 for c in TARGET_CLASSES}), ckpt


def predict_model_on_df(model, df: pd.DataFrame, device='cpu', zero_clip=False, zero_image=False) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xs = make_tensor_batch(df, device=device, zero_clip=zero_clip, zero_image=zero_image)
        logits, _ = model(*xs)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs


def camera_mean_metrics_from_arrays(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray):
    """按相机方向分别计算 micro-F1，再取六方向平均。"""
    if 'camera_name' not in df.columns:
        counts = multilabel_counts(y_true, y_pred, TARGET_CLASSES)
        return metrics_from_counts(counts)['overall']

    camera_metrics = []
    for _, idx in df.groupby('camera_name').groups.items():
        idx = list(idx)
        if len(idx) == 0:
            continue
        counts = multilabel_counts(y_true[idx], y_pred[idx], TARGET_CLASSES)
        camera_metrics.append(metrics_from_counts(counts)['overall'])

    if not camera_metrics:
        counts = multilabel_counts(y_true, y_pred, TARGET_CLASSES)
        return metrics_from_counts(counts)['overall']

    return {
        'precision': float(np.mean([m['precision'] for m in camera_metrics])),
        'recall': float(np.mean([m['recall'] for m in camera_metrics])),
        'f1': float(np.mean([m['f1'] for m in camera_metrics])),
        'support': int(sum(m['support'] for m in camera_metrics)),
        'pred_positive': int(sum(m['pred_positive'] for m in camera_metrics)),
        'num_cameras': int(len(camera_metrics)),
    }


def evaluate_model_on_df(model, df: pd.DataFrame, device='cpu', tune_thresholds=False, threshold_min: float = 0.20):
    if len(df) == 0:
        raise ValueError('evaluate_model_on_df 收到空 DataFrame')
    y_true = df[LABEL_COLS].values.astype(int)
    probs = predict_model_on_df(model, df, device=device)
    if tune_thresholds:
        thresholds = tune_per_class_thresholds(y_true, probs, threshold_min=threshold_min)
    else:
        thresholds = {c: 0.5 for c in TARGET_CLASSES}
    y_pred = np.zeros_like(y_true)
    for i, c in enumerate(TARGET_CLASSES):
        y_pred[:, i] = probs[:, i] >= thresholds[c]
    counts = multilabel_counts(y_true, y_pred, TARGET_CLASSES)
    metrics = metrics_from_counts(counts)
    metrics['camera_mean'] = camera_mean_metrics_from_arrays(df, y_true, y_pred)
    return metrics, thresholds


def tune_per_class_thresholds(y_true, probs, threshold_min: float = 0.20):
    """在验证集上搜索阈值。"""
    thresholds = {}
    threshold_min = float(threshold_min)
    if threshold_min < 0.05 or threshold_min >= 0.9:
        raise ValueError('threshold_min 应位于 [0.05, 0.9) 范围内')
    for i, c in enumerate(TARGET_CLASSES):
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(threshold_min, 0.9, 29):
            pred = (probs[:, i] >= t).astype(int)
            tp = int(((pred == 1) & (y_true[:, i] == 1)).sum())
            fp = int(((pred == 1) & (y_true[:, i] == 0)).sum())
            fn = int(((pred == 0) & (y_true[:, i] == 1)).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[c] = best_t
    return thresholds

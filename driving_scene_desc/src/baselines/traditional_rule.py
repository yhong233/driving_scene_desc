from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.datasets.nuscenes_reader import TARGET_CLASSES


# 传统规则基线
DEFAULT_TRADITIONAL_RULE_THRESHOLDS = {
    "vehicle": 0.50,
    "pedestrian": 0.55,
    "obstacle": 0.52,
}

TRADITIONAL_RULE_THRESHOLDS = DEFAULT_TRADITIONAL_RULE_THRESHOLDS.copy()

def _clip01(value: float) -> float:
    """把规则分数限制在 [0, 1]，避免异常特征影响预测。"""
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except Exception:
        return 0.0


def _get(row: Dict[str, float], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def set_traditional_thresholds(thresholds: Dict[str, float]) -> Dict[str, float]:
    """设置传统规则方法阈值。"""
    global TRADITIONAL_RULE_THRESHOLDS
    cleaned = {}
    for c in TARGET_CLASSES:
        cleaned[c] = float(thresholds.get(c, DEFAULT_TRADITIONAL_RULE_THRESHOLDS.get(c, 0.5)))
    TRADITIONAL_RULE_THRESHOLDS = cleaned
    return TRADITIONAL_RULE_THRESHOLDS.copy()


def get_traditional_thresholds() -> Dict[str, float]:
    return TRADITIONAL_RULE_THRESHOLDS.copy()


def score_traditional_rule(row: Dict[str, float]) -> Dict[str, float]:
    """计算传统规则方法的三类规则分数。"""

    projection_valid = _clip01(_get(row, "projection_valid_ratio") * 8.0)
    point_density = _clip01(_get(row, "lidar_num_points") * 30.0)
    near_ratio = _clip01(_get(row, "lidar_near_ratio"))
    front_ratio = _clip01(_get(row, "lidar_front_ratio"))
    left_ratio = _clip01(_get(row, "lidar_left_ratio"))
    right_ratio = _clip01(_get(row, "lidar_right_ratio"))
    side_ratio = max(left_ratio, right_ratio)
    height_ratio = _clip01(_get(row, "lidar_height_pos_ratio") * 1.5)

    edge = _clip01(_get(row, "img_edge_strength") * 8.0)
    contrast = _clip01(_get(row, "img_contrast") * 6.0)
    red_ratio = _clip01(_get(row, "img_red_ratio") * 4.0)

    evidence_vehicle = _clip01(_get(row, "evidence_vehicle"))
    evidence_pedestrian = _clip01(_get(row, "evidence_pedestrian"))
    evidence_obstacle = _clip01(_get(row, "evidence_obstacle"))

    # vehicle
    vehicle_score = (
        0.50 * evidence_vehicle
        + 0.16 * near_ratio
        + 0.12 * front_ratio
        + 0.12 * projection_valid
        + 0.10 * point_density
    )

    # pedestrian
    pedestrian_score = (
        0.38 * evidence_pedestrian
        + 0.20 * height_ratio
        + 0.17 * side_ratio
        + 0.15 * edge
        + 0.10 * contrast
    )

    # obstacle
    obstacle_score = (
        0.44 * evidence_obstacle
        + 0.18 * side_ratio
        + 0.14 * near_ratio
        + 0.12 * edge
        + 0.12 * red_ratio
    )

    return {
        "vehicle": _clip01(vehicle_score),
        "pedestrian": _clip01(pedestrian_score),
        "obstacle": _clip01(obstacle_score),
    }


def predict_traditional_rule(
    row: Dict[str, float],
    thresholds: Dict[str, float] | None = None,
) -> Tuple[List[str], Dict[str, float]]:
    """基于投影点统计与几何规则的传统基线。"""
    thresholds = thresholds or TRADITIONAL_RULE_THRESHOLDS
    probs = score_traditional_rule(row)

    candidates = [
        (c, probs.get(c, 0.0))
        for c in TARGET_CLASSES
        if probs.get(c, 0.0) >= float(thresholds.get(c, DEFAULT_TRADITIONAL_RULE_THRESHOLDS.get(c, 0.5)))
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    pred = [c for c, _ in candidates]

    return pred, probs


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return float(precision), float(recall), float(f1)


def tune_traditional_thresholds(
    df: pd.DataFrame,
    threshold_min: float = 0.45,
    threshold_max: float = 0.80,
    num_steps: int = 36,
    prefer_precision_when_tie: bool = True,
) -> tuple[Dict[str, float], pd.DataFrame]:
    """在 train/val split 上搜索传统规则阈值。"""
    if len(df) == 0:
        raise ValueError("传统规则阈值搜索收到空 DataFrame")

    thresholds = {}
    rows = []
    grid = np.linspace(float(threshold_min), float(threshold_max), int(num_steps))

    score_rows = [score_traditional_rule(rec) for rec in df.to_dict("records")]
    score_df = pd.DataFrame(score_rows)

    for c in TARGET_CLASSES:
        label_col = f"label_{c}"
        if label_col not in df.columns:
            raise ValueError(f"阈值搜索缺少标签列: {label_col}")

        y_true = df[label_col].astype(int).values
        scores = score_df[c].astype(float).values

        best = None
        for t in grid:
            y_pred = (scores >= float(t)).astype(int)
            p, r, f1 = _binary_f1(y_true, y_pred)
            item = {
                "class": c,
                "threshold": float(t),
                "precision": p,
                "recall": r,
                "f1": f1,
                "support": int(y_true.sum()),
                "pred_positive": int(y_pred.sum()),
            }
            if best is None:
                best = item
            else:
                # 主目标最大化 F1；并列时优先更高 precision，再优先更高阈值，减少过度召回。
                if item["f1"] > best["f1"] + 1e-12:
                    best = item
                elif abs(item["f1"] - best["f1"]) <= 1e-12 and prefer_precision_when_tie:
                    if (item["precision"], item["threshold"]) > (best["precision"], best["threshold"]):
                        best = item

        thresholds[c] = float(best["threshold"])
        rows.append(best)

    return thresholds, pd.DataFrame(rows)

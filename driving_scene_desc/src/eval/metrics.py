from __future__ import annotations

import re
from typing import Dict, List, Tuple, Any, Set

import numpy as np
import pandas as pd


TARGET_CLASSES = ["vehicle", "pedestrian", "obstacle"]

CLASS_ZH = {
    "vehicle": "车辆",
    "pedestrian": "行人",
    "obstacle": "障碍物",
}

CLASS_KEYWORDS = {
    "vehicle": [
        "车辆", "前车", "后车", "跟车", "汽车", "公交车", "货车",
        "相邻车道车辆", "后方车辆", "车距", "跟随交通"
    ],
    "pedestrian": [
        "行人", "人群", "人员", "行人活动", "靠近车道", "与车道的相对位置"
    ],
    "obstacle": [
        "障碍物", "路障", "交通锥", "锥桶", "隔离墩", "道路障碍", "通行空间"
    ],
}

DIRECTION_ORDER = ["前方", "左前方", "右前方", "后方", "左后方", "右后方"]
FRONT_DIRS = ["前方", "左前方", "右前方"]
SIDE_FRONT_DIRS = ["左前方", "右前方"]
REAR_DIRS = ["后方", "左后方", "右后方"]

DIRECTION_KEYWORDS = {
    "前方": ["前方", "前侧", "前向", "当前行驶路径", "前车距离", "前向安全距离"],
    "左前方": ["左前方", "左侧", "左右前方", "两侧", "前侧"],
    "右前方": ["右前方", "右侧", "左右前方", "两侧", "前侧"],
    "后方": ["后方", "后车", "跟随交通", "跟车", "减速", "变道"],
    "左后方": ["左后方", "侧后方", "后方及侧后方", "变道"],
    "右后方": ["右后方", "侧后方", "后方及侧后方", "变道"],
}

DYNAMIC_KEYWORDS = [
    "相邻帧", "上一帧", "变化", "连续出现", "新检测到", "新出现",
    "移动", "靠近", "由", "转向", "进入"
]


def _safe_set(items) -> Set[str]:
    if not items:
        return set()
    return set(str(x) for x in items if str(x).strip())


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return float(p), float(r), float(f1)


def frame_prf(true_set, pred_set) -> Tuple[float, float, float]:
    """
    多标签集合 precision / recall / f1。
    true_set:
        当前方向或当前帧的参考类别集合。
    pred_set:
        系统预测类别集合。
    1. 如果 true 和 pred 都为空，认为该方向没有目标且系统也没有误检，F1=1；
    2. 如果 true 为空但 pred 不为空，表示误检，F1=0；
    3. 如果 true 不为空但 pred 为空，表示漏检，F1=0。
    """
    true_set = _safe_set(true_set)
    pred_set = _safe_set(pred_set)

    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0

    if not pred_set:
        return 0.0, 0.0, 0.0

    if not true_set:
        return 0.0, 0.0, 0.0

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    return prf(tp, fp, fn)


def multilabel_counts(y_true, y_pred, classes):
    counts = {}
    for i, c in enumerate(classes):
        yt = y_true[:, i].astype(bool)
        yp = y_pred[:, i].astype(bool)
        counts[c] = {
            'tp': int(np.logical_and(yt, yp).sum()),
            'fp': int(np.logical_and(~yt, yp).sum()),
            'fn': int(np.logical_and(yt, ~yp).sum()),
            'tn': int(np.logical_and(~yt, ~yp).sum()),
        }
    return counts


def metrics_from_counts(counts):
    out = {}
    total_tp = total_fp = total_fn = 0
    for c, d in counts.items():
        p, r, f1 = prf(d['tp'], d['fp'], d['fn'])
        support = int(d['tp'] + d['fn'])
        pred_positive = int(d['tp'] + d['fp'])
        out[c] = {
            **d,
            'support': support,
            'pred_positive': pred_positive,
            'precision': p,
            'recall': r,
            'f1': f1,
        }
        total_tp += d['tp']
        total_fp += d['fp']
        total_fn += d['fn']
    p, r, f1 = prf(total_tp, total_fp, total_fn)
    out['overall'] = {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'support': int(total_tp + total_fn),
        'pred_positive': int(total_tp + total_fp),
        'precision': p,
        'recall': r,
        'f1': f1,
    }
    return out


def safe_mean(values):
    values = [float(v) for v in values if pd.notna(v)]
    return float(np.mean(values)) if values else 0.0


def _normalize_text(text: str) -> str:
    text = text or ""
    text = str(text)
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    return text


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def _class_mentioned(text: str, cls: str) -> bool:
    return _contains_any(text, CLASS_KEYWORDS.get(cls, []))


def _direction_mentioned(text: str, direction: str) -> bool:
    return _contains_any(text, DIRECTION_KEYWORDS.get(direction, [direction]))


def _class_exists_in_targets(
    cls: str,
    direction_targets: Dict[str, List[str]],
    predicted_classes: List[str] | None = None
) -> bool:
    if predicted_classes and cls in predicted_classes:
        return True

    for _, classes in (direction_targets or {}).items():
        if cls in classes:
            return True

    return False


def _flatten_targets(direction_targets: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    pairs = []
    for d in DIRECTION_ORDER:
        classes = direction_targets.get(d, []) if direction_targets else []
        for c in classes:
            if c in TARGET_CLASSES:
                pairs.append((d, c))
    return pairs


def _extract_key_semantics(direction_targets: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """提取用于描述连贯性评价的关键驾驶语义。"""
    direction_targets = direction_targets or {}
    key_items: List[Tuple[str, str]] = []

    for c in direction_targets.get("前方", []):
        if c in TARGET_CLASSES:
            key_items.append(("前方", c))

    for d in SIDE_FRONT_DIRS:
        classes = direction_targets.get(d, [])
        for c in ["pedestrian", "obstacle"]:
            if c in classes:
                key_items.append((d, c))

    has_front_vehicle = "vehicle" in direction_targets.get("前方", [])
    if not has_front_vehicle:
        for d in SIDE_FRONT_DIRS:
            if "vehicle" in direction_targets.get(d, []):
                key_items.append((d, "vehicle"))

    for d in REAR_DIRS:
        if "vehicle" in direction_targets.get(d, []):
            key_items.append((d, "vehicle"))

    if not key_items:
        key_items = _flatten_targets(direction_targets)[:4]

    out = []
    for item in key_items:
        if item not in out:
            out.append(item)
    return out


def _direction_class_covered(text: str, direction: str, cls: str) -> bool:
    """判断描述是否覆盖某个方向-类别关键语义。"""
    if not _class_mentioned(text, cls):
        return False

    if _direction_mentioned(text, direction):
        return True

    if direction == "前方" and cls == "vehicle":
        if "前车" in text or "前向安全距离" in text or "车距" in text:
            return True

    if direction == "前方" and cls == "pedestrian":
        if "行人与车道" in text or "行人与当前车道" in text:
            return True

    if direction in SIDE_FRONT_DIRS:
        if any(k in text for k in ["左右前方", "两侧", "侧道路空间", "侧通行空间", "左侧", "右侧"]):
            return True

    if direction in REAR_DIRS and cls == "vehicle":
        if any(k in text for k in ["后方及侧后方", "后方跟随交通", "后方车辆", "跟车", "变道"]):
            return True

    return False


def _score_scene(description: str, scene_condition: str | None = None) -> float:
    """场景状态完整性：判断描述是否包含天气、光照和道路类型。"""
    text = _normalize_text(description)
    score = 0.0

    if any(k in text for k in ["城市道路", "道路", "车道", "路口", "驾驶场景"]):
        score += 0.4

    if scene_condition:
        if "night" in scene_condition:
            if any(k in text for k in ["夜间", "夜晚", "光照较弱", "可见度"]):
                score += 0.3
        elif "day" in scene_condition:
            if any(k in text for k in ["白天", "日间", "晴天"]):
                score += 0.3
        else:
            if any(k in text for k in ["白天", "夜间", "日间", "夜晚"]):
                score += 0.3
    else:
        if any(k in text for k in ["白天", "夜间", "日间", "夜晚", "光照"]):
            score += 0.3

    if scene_condition:
        if "rain" in scene_condition:
            if any(k in text for k in ["雨天", "雨夜", "路面条件", "可见度", "湿滑"]):
                score += 0.3
        elif "sunny" in scene_condition:
            if any(k in text for k in ["晴天", "光照", "白天"]):
                score += 0.3
        else:
            if any(k in text for k in ["晴天", "雨天", "可见度", "天气"]):
                score += 0.3
    else:
        if any(k in text for k in ["晴天", "雨天", "可见度", "天气", "光照"]):
            score += 0.3

    return float(min(score, 1.0))


def _score_key_semantics(description: str, direction_targets: Dict[str, List[str]]) -> float:
    """关键驾驶语义覆盖度。"""
    text = _normalize_text(description)
    key_items = _extract_key_semantics(direction_targets)
    if not key_items:
        return 1.0

    covered = 0
    for direction, cls in key_items:
        if _direction_class_covered(text, direction, cls):
            covered += 1

    return float(covered / max(len(key_items), 1))


def _find_direction_class_mentions(text: str) -> List[Tuple[str, str]]:
    """从描述中粗略抽取显式方向-类别提法，用于一致性扣分。"""
    text = _normalize_text(text)
    mentions: List[Tuple[str, str]] = []

    direction_patterns = {
        "前方": ["前方"],
        "左前方": ["左前方"],
        "右前方": ["右前方"],
        "后方": ["后方"],
        "左后方": ["左后方"],
        "右后方": ["右后方"],
        "左右前方": ["左右前方"],
        "后方及侧后方": ["后方及侧后方", "侧后方"],
    }

    for pattern_name, patterns in direction_patterns.items():
        for pat in patterns:
            for m in re.finditer(re.escape(pat), text):
                start = max(0, m.start() - 8)
                end = min(len(text), m.end() + 35)
                window = text[start:end]

                for cls in TARGET_CLASSES:
                    if _class_mentioned(window, cls):
                        if pattern_name == "左右前方":
                            mentions.append(("左前方", cls))
                            mentions.append(("右前方", cls))
                        elif pattern_name == "后方及侧后方":
                            mentions.append(("后方", cls))
                            mentions.append(("左后方", cls))
                            mentions.append(("右后方", cls))
                        else:
                            mentions.append((pattern_name, cls))

    out = []
    for item in mentions:
        if item not in out:
            out.append(item)
    return out


def _score_consistency(
    description: str,
    predicted_classes: List[str] | None,
    direction_targets: Dict[str, List[str]],
    scene_context: Dict[str, float] | None = None
) -> float:
    """结构化语义一致性：判断描述是否乱写不存在的类别或方向级目标。"""
    text = _normalize_text(description)
    direction_targets = direction_targets or {}
    predicted_classes = predicted_classes or []
    scene_context = scene_context or {}

    score = 1.0

    for cls in TARGET_CLASSES:
        if _class_mentioned(text, cls):
            if not _class_exists_in_targets(cls, direction_targets, predicted_classes):
                score -= 0.20

    mentions = _find_direction_class_mentions(text)
    for direction, cls in mentions:
        true_classes = direction_targets.get(direction, [])

        if direction in REAR_DIRS and cls == "vehicle":
            if any("vehicle" in direction_targets.get(d, []) for d in REAR_DIRS):
                continue

        if direction in SIDE_FRONT_DIRS:
            if cls in direction_targets.get("左前方", []) or cls in direction_targets.get("右前方", []):
                continue

        if cls not in true_classes:
            score -= 0.15

    has_facility = any(k in text for k in ["交通信号灯", "交通灯", "交通标志"])
    if has_facility:
        facility_score = max(
            float(scene_context.get("traffic_light", 0.0)),
            float(scene_context.get("traffic_sign", 0.0)),
            0.0,
        )
        if scene_context and facility_score < 0.10:
            score -= 0.10

        # 交通信号灯/交通标志当前是帧级背景语义
        if re.search(r"(前方|左前方|右前方|后方|左后方|右后方).{0,8}(交通信号灯|交通灯|交通标志)", text):
            score -= 0.15

    return float(max(0.0, min(score, 1.0)))


def _score_dynamic(description: str, dynamic_events: List[Dict] | None = None) -> float:
    """动态表达有效性：有动态事件时检查是否表达；无动态事件时检查是否避免乱写动态。"""
    text = _normalize_text(description)
    dynamic_events = dynamic_events or []
    has_dynamic_words = any(k in text for k in DYNAMIC_KEYWORDS)

    if not dynamic_events:
        return 0.60 if has_dynamic_words else 1.0

    scores = []
    for ev in dynamic_events:
        cls = ev.get("class")
        from_dir = ev.get("from", "")
        to_dir = ev.get("to", "")

        cls_ok = _class_mentioned(text, cls) if cls in TARGET_CLASSES else True
        dynamic_ok = has_dynamic_words

        dir_ok = True
        if from_dir:
            dir_ok = dir_ok and (from_dir in text)
        if to_dir:
            dir_ok = dir_ok and (to_dir in text)

        if cls_ok and dynamic_ok and dir_ok:
            scores.append(1.0)
        elif cls_ok and dynamic_ok:
            scores.append(0.75)
        elif dynamic_ok:
            scores.append(0.50)
        else:
            scores.append(0.30)

    return float(sum(scores) / max(len(scores), 1))


def _chinese_char_count(text: str) -> int:
    return len(_normalize_text(text))


def _sentence_count(text: str) -> int:
    text = str(text or "")
    parts = re.split(r"[。！？!?]", text)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


def _score_language(description: str) -> float:
    """语言组织质量：从长度、重复度和句子数量三个角度评价。"""
    text = _normalize_text(description)
    n_chars = _chinese_char_count(text)

    if 60 <= n_chars <= 180:
        length_score = 1.0
    elif 40 <= n_chars < 60 or 180 < n_chars <= 230:
        length_score = 0.80
    else:
        length_score = 0.60

    repeat_score = 1.0
    repeated_checks = {
        "检测到": 4,
        "需关注": 4,
        "前方": 6,
        "车辆": 8,
        "行人": 6,
        "障碍物": 5,
    }

    for word, max_count in repeated_checks.items():
        count = text.count(word)
        if count > max_count:
            repeat_score -= min(0.25, 0.05 * (count - max_count))

    semicolon_count = text.count("；") + text.count(";")
    if semicolon_count > 3:
        repeat_score -= min(0.20, 0.05 * (semicolon_count - 3))

    direction_count = sum(text.count(d) for d in DIRECTION_ORDER)
    if direction_count > 7:
        repeat_score -= min(0.20, 0.04 * (direction_count - 7))

    repeat_score = max(0.0, min(repeat_score, 1.0))

    n_sent = _sentence_count(description)
    if 3 <= n_sent <= 5:
        sent_score = 1.0
    elif n_sent in [2, 6]:
        sent_score = 0.85
    else:
        sent_score = 0.70

    language_score = 0.40 * length_score + 0.40 * repeat_score + 0.20 * sent_score
    return float(max(0.0, min(language_score, 1.0)))


def score_description_details(
    description: str,
    predicted_classes: List[str] | None = None,
    direction_targets: Dict[str, List[str]] | None = None,
    scene_condition: str | None = None,
    scene_context: Dict[str, float] | None = None,
    dynamic_events: List[Dict] | None = None,
) -> Dict[str, float]:
    """
    结构化语义约束的描述连贯性评分。
    S_coh = 0.20*S_scene
          + 0.30*S_key
          + 0.25*S_consistency
          + 0.10*S_dynamic
          + 0.15*S_language
    """
    direction_targets = direction_targets or {}
    predicted_classes = predicted_classes or []
    scene_context = scene_context or {}
    dynamic_events = dynamic_events or []

    scene_score = _score_scene(description, scene_condition)
    key_semantic_score = _score_key_semantics(description, direction_targets)
    consistency_score = _score_consistency(
        description,
        predicted_classes,
        direction_targets,
        scene_context=scene_context,
    )
    dynamic_score = _score_dynamic(description, dynamic_events)
    language_score = _score_language(description)

    coherence_score = (
        0.20 * scene_score
        + 0.30 * key_semantic_score
        + 0.25 * consistency_score
        + 0.10 * dynamic_score
        + 0.15 * language_score
    )

    return {
        "scene_score": float(scene_score),
        "key_semantic_score": float(key_semantic_score),
        "semantic_consistency_score": float(consistency_score),
        "dynamic_score": float(dynamic_score),
        "language_score": float(language_score),
        "coherence_score": float(max(0.0, min(coherence_score, 1.0))),
    }


def score_description(
    description: str,
    predicted_classes: List[str] | None = None,
    direction_targets: Dict[str, List[str]] | None = None,
    scene_condition: str | None = None,
    scene_context: Dict[str, float] | None = None,
    dynamic_events: List[Dict] | None = None,
) -> Tuple[float, float]:

    detail = score_description_details(
        description=description,
        predicted_classes=predicted_classes,
        direction_targets=direction_targets,
        scene_condition=scene_condition,
        scene_context=scene_context,
        dynamic_events=dynamic_events,
    )
    return float(detail["semantic_consistency_score"]), float(detail["coherence_score"])

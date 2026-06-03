from __future__ import annotations

from typing import Dict, List, Any, Tuple


TARGET_DESCRIPTION_CLASSES = ["vehicle", "pedestrian", "obstacle"]

CLASS_ZH = {
    "vehicle": "车辆",
    "pedestrian": "行人",
    "obstacle": "障碍物",
}

BACKGROUND_ZH = {
    "road": "道路",
    "building": "建筑物",
    "vegetation": "植被",
    "traffic_light": "交通信号灯",
    "traffic_sign": "交通标志",
}

DIRECTION_ORDER = ["前方", "左前方", "右前方", "后方", "左后方", "右后方"]
FRONT_DIRS = ["前方", "左前方", "右前方"]
SIDE_FRONT_DIRS = ["左前方", "右前方"]
REAR_DIRS = ["后方", "左后方", "右后方"]


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _unique_keep_order(items: List[str]) -> List[str]:
    out = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def _get_frame_attr(frame: Any, name: str, default=None):
    if hasattr(frame, name):
        return getattr(frame, name)
    if isinstance(frame, dict):
        return frame.get(name, default)
    return default


def _scene_condition_zh(scene_condition: str) -> str:
    if scene_condition == "day_sunny":
        return "白天晴天城市道路"
    if scene_condition == "night_rainy":
        return "夜间雨天城市道路"
    if scene_condition == "day_rainy":
        return "白天雨天城市道路"
    if scene_condition == "night":
        return "夜间城市道路"
    return "城市道路"


def _build_direction_targets_from_predictions(direction_predictions: List[Dict]) -> Dict[str, List[str]]:
    """根据六相机方向预测结果生成完整方向语义。"""
    direction_targets: Dict[str, List[str]] = {}

    if not direction_predictions:
        return direction_targets

    for item in direction_predictions:
        direction = item.get("direction", item.get("camera_name", "未知方向"))
        pred_classes = item.get("pred_classes", []) or []

        keep = []
        for c in pred_classes:
            if c in TARGET_DESCRIPTION_CLASSES and c not in keep:
                keep.append(c)

        if keep:
            direction_targets[direction] = keep

    return {d: direction_targets[d] for d in DIRECTION_ORDER if d in direction_targets}


def _build_direction_targets_from_frame_level(predicted_classes: List[str]) -> Dict[str, List[str]]:
    keep = []
    for c in predicted_classes:
        if c in TARGET_DESCRIPTION_CLASSES and c not in keep:
            keep.append(c)
    return {"前方": keep} if keep else {}


def _aggregate_scene_context(camera_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    根据六相机 CLIP 分数生成帧级背景语义。
    traffic_light / traffic_sign 只是帧级背景语义，不作为方向级目标。
    """
    context_classes = ["road", "building", "vegetation", "traffic_light", "traffic_sign"]
    out: Dict[str, float] = {}

    for cls in context_classes:
        vals = []
        for _, score_dict in (camera_scores or {}).items():
            if isinstance(score_dict, dict) and cls in score_dict:
                vals.append(_safe_float(score_dict.get(cls, 0.0)))

        if vals:
            max_score = max(vals)
            mean_score = sum(vals) / len(vals)
            out[cls] = 0.60 * max_score + 0.40 * mean_score

    return out


def _select_scene_context_names(scene_context: Dict[str, float]) -> Tuple[List[str], List[str]]:
    """返回普通背景类别和道路设施类别。"""
    if not scene_context:
        return [], []

    normal_candidates = ["road", "building", "vegetation"]
    facility_candidates = ["traffic_light", "traffic_sign"]

    normal = [c for c in normal_candidates if scene_context.get(c, 0.0) >= 0.18]
    facility = [c for c in facility_candidates if scene_context.get(c, 0.0) >= 0.20]

    if not normal:
        sorted_normal = sorted(
            [(c, scene_context.get(c, 0.0)) for c in normal_candidates],
            key=lambda x: x[1],
            reverse=True,
        )
        normal = [c for c, s in sorted_normal[:2] if s > 0.05]

    return normal[:3], facility[:2]


def _count_targets(direction_targets: Dict[str, List[str]]) -> int:
    return sum(len(v) for v in direction_targets.values())


def _scene_overview_sentence(scene_condition: str, direction_targets: Dict[str, List[str]]) -> str:
    scene_zh = _scene_condition_zh(scene_condition)

    total = _count_targets(direction_targets)
    front_count = sum(len(direction_targets.get(d, [])) for d in FRONT_DIRS)
    front_vehicle_dirs = [d for d in FRONT_DIRS if "vehicle" in direction_targets.get(d, [])]
    rear_vehicle = any("vehicle" in direction_targets.get(d, []) for d in REAR_DIRS)
    has_front_ped = any("pedestrian" in direction_targets.get(d, []) for d in FRONT_DIRS)
    has_front_obs = any("obstacle" in direction_targets.get(d, []) for d in FRONT_DIRS)

    if scene_condition == "night_rainy":
        if total >= 6:
            return f"当前场景为{scene_zh}，可见度和路面条件相对复杂，车辆周边交通目标较多。"
        return f"当前场景为{scene_zh}，环境光照较弱，需结合前方目标和道路空间进行判断。"

    if total >= 8:
        return f"当前场景为{scene_zh}，车辆周边交通目标较多。"

    if len(front_vehicle_dirs) >= 2 and not has_front_ped and not has_front_obs:
        return f"当前场景为{scene_zh}，车辆主要分布在前方及道路两侧。"

    if front_count >= 4:
        return f"当前场景为{scene_zh}，前侧道路区域存在多类交通目标。"

    if rear_vehicle and front_count <= 1:
        return f"当前场景为{scene_zh}，前方通行空间相对较清晰，后方存在跟随交通。"

    return f"当前场景为{scene_zh}。"


def _front_sentence(direction_targets: Dict[str, List[str]]) -> str:
    front = direction_targets.get("前方", [])

    if not front:
        return "前方未检测到明显影响当前行驶路径的目标。"

    has_vehicle = "vehicle" in front
    has_ped = "pedestrian" in front
    has_obs = "obstacle" in front

    if has_vehicle and has_ped and has_obs:
        return "前方同时检测到车辆、行人和障碍物，需关注前车距离、行人与车道的相对位置及前方通行空间。"

    if has_vehicle and has_ped:
        return "前方检测到车辆和行人，需同时关注前车距离及行人与车道的相对位置。"

    if has_vehicle and has_obs:
        return "前方检测到车辆和障碍物，需保持前向安全距离并关注当前通行空间。"

    if has_ped and has_obs:
        return "前方检测到行人和障碍物，可能影响当前行驶路径，需重点关注前方通行情况。"

    if has_ped:
        return "前方检测到行人，应重点关注其与当前车道的相对位置。"

    if has_obs:
        return "前方存在障碍物，可能影响当前行驶路径。"

    if has_vehicle:
        return "前方检测到车辆，需注意与前车保持安全距离。"

    return ""


def _side_front_sentence(direction_targets: Dict[str, List[str]]) -> str:
    left = direction_targets.get("左前方", [])
    right = direction_targets.get("右前方", [])

    if not left and not right:
        return ""

    union = _unique_keep_order(left + right)

    left_has = bool(left)
    right_has = bool(right)

    has_vehicle = "vehicle" in union
    has_ped = "pedestrian" in union
    has_obs = "obstacle" in union

    if left_has and right_has:
        if has_ped and has_obs and has_vehicle:
            return "左右前方检测到车辆、行人和障碍物，需关注两侧通行空间及行人靠近车道的可能。"

        if has_ped and has_obs:
            return "左右前方检测到行人和障碍物，需关注两侧道路空间变化及行人靠近车道的可能。"

        if has_ped and has_vehicle:
            return "左右前方检测到车辆和行人，需关注相邻车道车辆状态及行人与车道的相对位置。"

        if has_obs and has_vehicle:
            return "左右前方检测到车辆和障碍物，需注意相邻车道车辆状态及两侧通行空间。"

        if has_ped:
            return "左右前方检测到行人，需关注其靠近车道或横向移动的可能。"

        if has_obs:
            return "左右前方存在障碍物，可能影响两侧通行空间。"

        if has_vehicle:
            return "左右前方均检测到车辆，应关注相邻车道或路侧车辆状态。"

    direction = "左前方" if left_has else "右前方"
    cls = left if left_has else right

    has_vehicle = "vehicle" in cls
    has_ped = "pedestrian" in cls
    has_obs = "obstacle" in cls

    if has_ped and has_obs:
        return f"{direction}检测到行人和障碍物，需关注该侧道路空间及行人靠近车道的可能。"

    if has_ped and has_vehicle:
        return f"{direction}检测到车辆和行人，需关注相邻车道车辆状态及行人与车道的相对位置。"

    if has_obs and has_vehicle:
        return f"{direction}检测到车辆和障碍物，需关注该侧通行空间。"

    if has_ped:
        return f"{direction}检测到行人，需关注其靠近车道或横向移动的可能。"

    if has_obs:
        return f"{direction}存在障碍物，可能影响该侧通行空间。"

    if has_vehicle:
        return f"{direction}检测到车辆，应关注相邻车道或路侧车辆状态。"

    return ""


def _rear_sentence(direction_targets: Dict[str, List[str]]) -> str:
    rear_sets = [direction_targets.get(d, []) for d in REAR_DIRS]
    rear_union = _unique_keep_order([c for s in rear_sets for c in s])

    if not rear_union:
        return ""

    rear_vehicle_dirs = [d for d in REAR_DIRS if "vehicle" in direction_targets.get(d, [])]
    has_vehicle = bool(rear_vehicle_dirs)

    if has_vehicle:
        if len(rear_vehicle_dirs) >= 2:
            return "后方及侧后方检测到车辆，变道或减速时需关注后方跟随交通。"
        if "后方" in rear_vehicle_dirs:
            return "后方检测到车辆，说明当前存在跟车情况，减速时需关注后方车辆状态。"
        return "侧后方检测到车辆，变道时需关注侧后方交通情况。"

    return "后方及侧后方检测到少量道路目标，对当前前向行驶影响相对较小。"


def _context_sentence(scene_context: Dict[str, float]) -> str:
    normal, facility = _select_scene_context_names(scene_context)

    if normal:
        names = [BACKGROUND_ZH[c] for c in normal if c in BACKGROUND_ZH]
        if len(names) == 1:
            return f"周围环境中可见{names[0]}等背景信息。"
        if len(names) == 2:
            return f"周围环境中可见{names[0]}和{names[1]}等背景信息。"
        return f"周围环境中可见{'、'.join(names[:-1])}和{names[-1]}等背景信息。"

    if facility:
        names = [BACKGROUND_ZH[c] for c in facility if c in BACKGROUND_ZH]
        if names:
            return f"场景中可见{'、'.join(names)}等道路设施，说明当前区域可能包含路口或交通引导信息。"

    return ""


def _focus_regions(direction_targets: Dict[str, List[str]]) -> List[str]:
    focus = []

    front = direction_targets.get("前方", [])
    if "pedestrian" in front:
        focus.append("前方行人")
    if "obstacle" in front:
        focus.append("前方障碍物")
    if "vehicle" in front:
        focus.append("前方车辆")

    for d in SIDE_FRONT_DIRS:
        cls = direction_targets.get(d, [])
        if "pedestrian" in cls:
            focus.append(f"{d}行人")
        if "obstacle" in cls:
            focus.append(f"{d}障碍物")

    if any("vehicle" in direction_targets.get(d, []) for d in REAR_DIRS):
        focus.append("后方车辆")

    return _unique_keep_order(focus)[:4]


def _focus_sentence(focus_regions: List[str]) -> str:
    """只在存在行人或障碍物时补充关注建议。"""
    important = [x for x in (focus_regions or []) if ("行人" in x or "障碍物" in x)]

    if len(important) >= 3:
        return f"建议重点关注{important[0]}、{important[1]}和{important[2]}。"
    if len(important) == 2:
        return f"建议重点关注{important[0]}和{important[1]}。"
    if len(important) == 1:
        return f"建议重点关注{important[0]}。"
    return ""


def _post_process_description(sentences: List[str]) -> str:
    clean = []
    for s in sentences:
        if not s:
            continue
        s = str(s).strip()
        if not s:
            continue
        if not s.endswith(("。", "！", "？")):
            s += "。"
        if s not in clean:
            clean.append(s)

    # 避免GUI太长。
    clean = clean[:5]

    return "".join(clean)


def _join_zh(items: List[str]) -> str:
    items = [x for x in items if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]}和{items[1]}"
    return "、".join(items[:-1]) + f"和{items[-1]}"


def _build_traditional_description(structured: Dict[str, Any]) -> str:
    """传统规则方法的保守模板"""
    scene_condition = structured.get("scene_condition", "unknown")
    direction_targets = structured.get("direction_targets", {}) or {}
    scene_zh = _scene_condition_zh(scene_condition)

    parts: List[str] = []
    for direction in DIRECTION_ORDER:
        classes = direction_targets.get(direction, []) or []
        names = []
        if "vehicle" in classes:
            names.append("疑似车辆区域")
        if "pedestrian" in classes:
            names.append("疑似行人区域")
        if "obstacle" in classes:
            names.append("疑似障碍物区域")
        if names:
            parts.append(f"{direction}存在{_join_zh(names)}")

    if not parts:
        return f"当前场景为{scene_zh}。根据投影点统计结果，未检测到明显的车辆、行人或障碍物区域，建议继续关注周围道路环境。"

    # 避免传统方法文本过长，最多展示4个方向级结果。
    shown = parts[:4]
    target_sentence = "根据投影点统计结果，" + "，".join(shown) + "。"

    focus_dirs = []
    for direction in DIRECTION_ORDER:
        if direction in direction_targets and direction not in focus_dirs:
            focus_dirs.append(direction)
    focus_dirs = focus_dirs[:4]
    focus_sentence = f"建议关注{_join_zh(focus_dirs)}目标。" if focus_dirs else "建议继续关注周围道路环境。"

    return f"当前场景为{scene_zh}。{target_sentence}{focus_sentence}"


def build_description(structured: Dict[str, Any]) -> str:
    if structured.get("method") == "traditional_rule":
        return _build_traditional_description(structured)

    scene_condition = structured.get("scene_condition", "unknown")
    direction_targets = structured.get("direction_targets", {}) or {}
    scene_context = structured.get("scene_context", {}) or {}
    focus_regions = structured.get("focus_regions", []) or []

    sentences = [
        _scene_overview_sentence(scene_condition, direction_targets),
        _front_sentence(direction_targets),
        _side_front_sentence(direction_targets),
        _rear_sentence(direction_targets),
    ]

    context_sent = _context_sentence(scene_context)
    if context_sent:
        sentences.append(context_sent)

    focus_sent = _focus_sentence(focus_regions)
    if focus_sent:
        sentences.append(focus_sent)

    return _post_process_description(sentences)


def build_structured_semantics(
    frame: Any,
    predicted_classes: List[str],
    probs: Dict[str, float],
    camera_scores: Dict[str, Dict[str, float]],
    method: str = "ours",
    direction_predictions: List[Dict] | None = None,
) -> Dict[str, Any]:
    scene_idx = _get_frame_attr(frame, "scene_idx", None)
    scene_condition = _get_frame_attr(frame, "scene_condition", "unknown")
    sample_token = _get_frame_attr(frame, "sample_token", None)

    if direction_predictions:
        direction_targets = _build_direction_targets_from_predictions(direction_predictions)
    else:
        direction_targets = _build_direction_targets_from_frame_level(predicted_classes)

    scene_context = _aggregate_scene_context(camera_scores)
    focus_regions = _focus_regions(direction_targets)

    structured: Dict[str, Any] = {
        "sample_token": sample_token,
        "scene_idx": scene_idx,
        "scene_condition": scene_condition,
        "method": method,
        "direction_targets": direction_targets,
        "scene_context": scene_context,
        "predicted_classes": predicted_classes,
        "confidence_scores": probs,
        "direction_results": direction_predictions or [],
        "focus_regions": focus_regions,
        "dynamic_events": [],
    }

    structured["description_text"] = build_description(structured)

    return structured

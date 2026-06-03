from __future__ import annotations

from typing import Dict, List, Any


DIRECTION_ORDER = ["前方", "左前方", "右前方", "后方", "左后方", "右后方"]
FRONT_DIRS = ["前方", "左前方", "右前方"]
SIDE_FRONT_DIRS = ["左前方", "右前方"]
REAR_DIRS = ["后方", "左后方", "右后方"]

CLASS_ZH = {
    "vehicle": "车辆",
    "pedestrian": "行人",
    "obstacle": "障碍物",
}


def _reverse_direction_map(direction_targets: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out = {
        "vehicle": [],
        "pedestrian": [],
        "obstacle": [],
    }

    for direction, classes in (direction_targets or {}).items():
        for c in classes:
            if c in out and direction not in out[c]:
                out[c].append(direction)

    for c in out:
        out[c] = [d for d in DIRECTION_ORDER if d in out[c]]

    return out


def _first_front_direction(directions: List[str]) -> str | None:
    for d in FRONT_DIRS:
        if d in directions:
            return d
    return directions[0] if directions else None


def _append_sentence(desc: str, sentence: str) -> str:
    if not sentence:
        return desc

    sentence = sentence.strip()
    if not sentence:
        return desc

    if not sentence.endswith(("。", "！", "？")):
        sentence += "。"

    if sentence in desc:
        return desc

    if desc and not desc.endswith(("。", "！", "？")):
        desc += "。"

    return desc + sentence


def _build_dynamic_event(prev: Dict[str, Any], curr: Dict[str, Any]) -> tuple[List[Dict], str]:
    """生成轻量动态事件。"""
    prev_targets = prev.get("direction_targets", {}) if prev else {}
    curr_targets = curr.get("direction_targets", {}) if curr else {}

    if not prev_targets or not curr_targets:
        return [], ""

    prev_map = _reverse_direction_map(prev_targets)
    curr_map = _reverse_direction_map(curr_targets)

    events: List[Dict] = []

    # 1. 行人方向变化
    prev_ped = prev_map.get("pedestrian", [])
    curr_ped = curr_map.get("pedestrian", [])

    if prev_ped and curr_ped:
        prev_main = _first_front_direction(prev_ped)
        curr_main = _first_front_direction(curr_ped)

        if prev_main and curr_main and prev_main != curr_main:
            events.append({
                "class": "pedestrian",
                "type": "direction_shift",
                "from": prev_main,
                "to": curr_main,
            })
            sentence = (
                f"相邻帧变化显示，行人由{prev_main}向{curr_main}区域变化，"
                f"可能靠近当前行驶路径，需持续关注。"
            )
            return events, sentence

    # 2. 前侧新出现行人
    if not prev_ped and curr_ped:
        curr_main = _first_front_direction(curr_ped)
        if curr_main in FRONT_DIRS:
            events.append({
                "class": "pedestrian",
                "type": "new_appeared",
                "to": curr_main,
            })
            return events, f"与上一帧相比，{curr_main}新检测到行人，需关注其与当前车道的相对位置。"

    # 3. 前侧新出现障碍物
    prev_obs = prev_map.get("obstacle", [])
    curr_obs = curr_map.get("obstacle", [])

    if not prev_obs and curr_obs:
        curr_main = _first_front_direction(curr_obs)
        if curr_main in FRONT_DIRS:
            events.append({
                "class": "obstacle",
                "type": "new_appeared",
                "to": curr_main,
            })
            return events, f"与上一帧相比，{curr_main}新检测到障碍物，可能影响通行空间，需要持续观察。"

    # 4. 车辆由侧前方进入前方时动态
    prev_vehicle = prev_map.get("vehicle", [])
    curr_vehicle = curr_map.get("vehicle", [])

    if prev_vehicle and curr_vehicle:
        prev_has_side_front = any(d in SIDE_FRONT_DIRS for d in prev_vehicle)
        curr_has_front = "前方" in curr_vehicle
        prev_has_front = "前方" in prev_vehicle

        if prev_has_side_front and curr_has_front and not prev_has_front:
            events.append({
                "class": "vehicle",
                "type": "direction_shift_to_front",
                "from": "侧前方",
                "to": "前方",
            })
            return events, "相邻帧变化显示，车辆由侧前方向前方区域变化，需关注前向车距和行驶状态。"

    return events, ""


def apply_temporal_context(prev_structured: Dict | None, curr_structured: Dict) -> Dict:
    """将上一帧与当前帧的方向级结构化语义进行比较，"""
    if not curr_structured:
        return curr_structured

    # 传统规则方法只做单帧投影点统计，不表达动态语义。
    if curr_structured.get("method") == "traditional_rule":
        curr_structured["dynamic_events"] = []
        return curr_structured

    if not prev_structured:
        curr_structured["dynamic_events"] = []
        return curr_structured

    events, sentence = _build_dynamic_event(prev_structured, curr_structured)
    curr_structured["dynamic_events"] = events

    if sentence:
        curr_structured["description_text"] = _append_sentence(
            curr_structured.get("description_text", ""),
            sentence,
        )

    return curr_structured

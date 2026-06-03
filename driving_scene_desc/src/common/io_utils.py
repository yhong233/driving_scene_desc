from __future__ import annotations
from pathlib import Path
import json
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def save_json(path: str | Path, data):
    p = Path(path)
    ensure_dir(p.parent)
    with p.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)


def load_json(path: str | Path):
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)

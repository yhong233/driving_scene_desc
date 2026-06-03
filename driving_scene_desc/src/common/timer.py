from __future__ import annotations
import time
from contextlib import contextmanager

class RuntimeMeter:
    def __init__(self):
        self.times = {}

    @contextmanager
    def track(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0

    def core_ms(self) -> float:
        keys = ['read_data', 'projection', 'feature_fusion', 'clip_align', 'fusion_mlp', 'description']
        return sum(self.times.get(k, 0.0) for k in keys)

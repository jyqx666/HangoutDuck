"""固定频率循环，带周期统计（验收 A5：50 Hz 抖动 p99 < 5 ms）。"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np


class RateLoop:
    def __init__(self, hz: float, spin_s: float = 0.0005) -> None:
        self.period = 1.0 / hz
        self.spin_s = spin_s  # 最后这一小段用忙等，sleep 的粒度在 Nano 上不够细
        self._next: Optional[float] = None
        self._last: Optional[float] = None
        self.periods: List[float] = []
        self.overruns = 0

    def wait(self) -> float:
        """等到下一个周期起点，返回上一周期实际时长（第一次调用返回 0）。"""
        now = time.perf_counter()
        if self._next is None:
            self._next = now + self.period
            self._last = now
            return 0.0
        remaining = self._next - now
        if remaining > 0:
            if remaining > self.spin_s:
                time.sleep(remaining - self.spin_s)
            while time.perf_counter() < self._next:
                pass
            self._next += self.period
        else:
            self.overruns += 1
            # 落后超过一个周期就不追了，从现在重新计时
            self._next = now + self.period if -remaining > self.period else self._next + self.period
        t = time.perf_counter()
        dt = t - self._last
        self._last = t
        self.periods.append(dt)
        return dt

    def stats(self) -> Dict[str, float]:
        if not self.periods:
            return {"cycles": 0}
        arr = np.asarray(self.periods)
        jitter = np.abs(arr - self.period)
        return {
            "cycles": int(arr.size),
            "mean_ms": float(arr.mean() * 1e3),
            "max_ms": float(arr.max() * 1e3),
            "jitter_p99_ms": float(np.percentile(jitter, 99) * 1e3),
            "overruns": self.overruns,
        }

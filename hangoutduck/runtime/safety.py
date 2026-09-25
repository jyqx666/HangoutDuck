"""安全监控：任何一项越界就返回故障原因，由运行时立即卸力。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..robot import JointState


@dataclass
class SafetyLimits:
    max_temperature_c: float = 65.0
    min_voltage_v: float = 6.5
    max_voltage_v: float = 8.6
    # 连续多少个周期有舵机没应答就停（50 Hz 下 5 个周期 = 0.1 s）
    max_missed_cycles: int = 5
    # 机身倾角超过多少度认为摔倒（需要 IMU）；None 表示不检查
    max_tilt_deg: Optional[float] = 60.0


class SafetyMonitor:
    def __init__(self, limits: SafetyLimits, names: List[str]) -> None:
        self.limits = limits
        self.names = names
        self._missed = 0

    def check(self, state: JointState, tilt_deg: Optional[float] = None) -> Optional[str]:
        lim = self.limits
        if state.all_valid:
            self._missed = 0
        else:
            self._missed += 1
            if self._missed >= lim.max_missed_cycles:
                lost = [n for n, ok in zip(self.names, state.valid) if not ok]
                return f"no reply for {self._missed} cycles: {', '.join(lost)}"
        seen = state.valid
        if seen.any():
            hot = np.where(seen & (state.temperature >= lim.max_temperature_c))[0]
            if hot.size:
                k = int(hot[0])
                return f"{self.names[k]} temperature {state.temperature[k]:.0f} C"
            v = state.voltage[seen]
            if v.min() < lim.min_voltage_v:
                return f"supply voltage low: {v.min():.1f} V"
            if v.max() > lim.max_voltage_v:
                return f"supply voltage high: {v.max():.1f} V"
        if lim.max_tilt_deg is not None and tilt_deg is not None and tilt_deg > lim.max_tilt_deg:
            return f"tilt {tilt_deg:.0f} deg, robot fell"
        return None

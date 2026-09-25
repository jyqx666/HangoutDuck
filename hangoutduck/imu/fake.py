from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base import G, Imu, ImuSample


class FakeImu(Imu):
    """静止平放的 IMU（机体系与 IMU 系重合），带一点噪声。"""

    def __init__(self, noise: float = 0.0, seed: int = 0) -> None:
        self.noise = noise
        self._rng = np.random.default_rng(seed)

    def latest(self) -> Optional[ImuSample]:
        n = self._rng.normal(0.0, self.noise, 6) if self.noise else np.zeros(6)
        return ImuSample(
            t=time.perf_counter(),
            gyro=n[:3],
            accel=np.array([0.0, 0.0, G]) + n[3:],
        )

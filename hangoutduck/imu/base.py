from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

G = 9.80665


@dataclass
class ImuSample:
    t: float  # time.perf_counter() 时刻
    gyro: np.ndarray  # rad/s，IMU 自身坐标系
    accel: np.ndarray  # m/s²（比力：静止平放时 z 约 +9.8），IMU 自身坐标系


class Imu:
    """IMU 驱动接口。latest() 不阻塞，返回最近一次完整样本；还没有数据时返回 None。"""

    def start(self) -> None:
        pass

    def latest(self) -> Optional[ImuSample]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class GravityEstimator:
    """互补滤波：用陀螺积分预测机体系下的重力方向，再按加速度计缓慢修正。

    输出 projected gravity（单位向量，指向地面，平放静止为 [0, 0, -1]），
    与仿真里策略观测的 projected_gravity 同一约定。
    """

    def __init__(self, rotation: np.ndarray, tau_s: float = 0.3, accel_trust_band: float = 0.2) -> None:
        self.rotation = np.asarray(rotation, dtype=float)
        self.tau_s = tau_s
        self.accel_trust_band = accel_trust_band
        self.gravity: Optional[np.ndarray] = None
        self.gyro = np.zeros(3)
        self._t: Optional[float] = None

    def reset(self) -> None:
        self.gravity = None
        self._t = None

    def update(self, sample: ImuSample) -> np.ndarray:
        gyro = self.rotation @ sample.gyro
        accel = self.rotation @ sample.accel
        self.gyro = gyro
        norm = float(np.linalg.norm(accel))
        measured = -accel / norm if norm > 1e-6 else None
        dt = None if self._t is None else sample.t - self._t
        if self.gravity is not None and dt is not None and dt <= 0:
            return self.gravity  # IMU 比控制循环慢时会拿到同一个样本，保持原估计
        self._t = sample.t
        if self.gravity is None or dt is None or dt > 0.1:
            self.gravity = measured if measured is not None else np.array([0.0, 0.0, -1.0])
            return self.gravity
        # 世界系固定的向量在转动的机体系里：dg/dt = -ω × g
        g = self.gravity - np.cross(gyro, self.gravity) * dt
        g /= np.linalg.norm(g)
        if measured is not None:
            # 加速度模长偏离 1 g 越多（在加速/冲击），越不信任加速度计
            trust = max(0.0, 1.0 - abs(norm / G - 1.0) / self.accel_trust_band)
            gain = (1.0 - math.exp(-dt / self.tau_s)) * trust
            g = (1.0 - gain) * g + gain * measured
            g /= np.linalg.norm(g)
        self.gravity = g
        return g

    def tilt_deg(self) -> float:
        """机体 z 轴与竖直方向的夹角。"""
        if self.gravity is None:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, -float(self.gravity[2])))))

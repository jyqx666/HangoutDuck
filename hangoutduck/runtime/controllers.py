"""控制器接口与几个基础控制器。

控制器每个周期拿到关节状态和 IMU 观测，返回 10 个关节的目标角（rad，仿真约定）。
保底步态、RL 策略都实现同一个接口，运行时不需要改。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from ..config import RobotConfig
from ..robot import JointState


@dataclass
class ImuObs:
    gyro: np.ndarray  # 机体系角速度 rad/s
    gravity: np.ndarray  # 机体系重力方向单位向量，平放为 [0, 0, -1]


class Controller:
    name = "base"

    def __init__(self, cfg: RobotConfig) -> None:
        self.cfg = cfg

    def reset(self, state: JointState) -> None:
        """上力前调用一次，state 为当前实际姿态。"""

    def step(self, t: float, state: JointState, imu: Optional[ImuObs]) -> np.ndarray:
        raise NotImplementedError

    def done(self, t: float) -> bool:
        return False


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class MoveTo(Controller):
    """从当前姿态平滑插值到目标姿态，之后保持。"""

    name = "move_to"

    def __init__(self, cfg: RobotConfig, target: np.ndarray, duration_s: float = 2.0) -> None:
        super().__init__(cfg)
        self.target = np.asarray(target, dtype=float)
        self.duration_s = duration_s
        self.start = self.target.copy()

    def reset(self, state: JointState) -> None:
        self.start = state.q.copy()

    def step(self, t, state, imu):
        a = smoothstep(t / self.duration_s) if self.duration_s > 0 else 1.0
        return self.start + a * (self.target - self.start)


class Stand(MoveTo):
    """平滑站到 joints.yaml 里的 home 姿态并保持。"""

    name = "stand"

    def __init__(self, cfg: RobotConfig, duration_s: float = 2.0) -> None:
        super().__init__(cfg, cfg.home_rad, duration_s)


class Sweep(Controller):
    """先站到 home，再让一个关节绕 home 做正弦摆动，用来核对方向和零位。"""

    name = "sweep"

    def __init__(
        self,
        cfg: RobotConfig,
        joint: str,
        amplitude_deg: float = 10.0,
        period_s: float = 2.0,
        cycles: int = 3,
        settle_s: float = 2.0,
    ) -> None:
        super().__init__(cfg)
        self.k = cfg.index(joint)
        self.amplitude = math.radians(amplitude_deg)
        self.period_s = period_s
        self.cycles = cycles
        self.settle_s = settle_s
        self._stand = Stand(cfg, settle_s)

    def reset(self, state):
        self._stand.reset(state)

    def step(self, t, state, imu):
        q = self._stand.step(t, state, imu)
        ts = t - self.settle_s
        if 0 <= ts <= self.period_s * self.cycles:
            q = q.copy()
            q[self.k] += self.amplitude * math.sin(2 * math.pi * ts / self.period_s)
        return q

    def done(self, t):
        return t > self.settle_s * 2 + self.period_s * self.cycles


# 名字 -> 工厂函数，`hduck run --controller <名字>` 用。
# 新控制器（保底步态、RL 策略）在这里注册即可。
CONTROLLERS: Dict[str, Callable[..., Controller]] = {
    "stand": Stand,
    "sweep": Sweep,
}

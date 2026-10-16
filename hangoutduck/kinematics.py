"""腿的正 / 逆运动学（躯干坐标系：x 前、y 左、z 上，单位 m / rad，关节角为仿真约定）。

每条腿 5 个关节：髋 yaw → 髋 roll → 髋 pitch → 膝 → 踝（pitch）。几何取自仿真模型
（xgoduck MJCF）在关节全为 0 时各关节轴的位置和方向，用指数积公式计算正运动学。
rl 分支的 scripts/check_gait_kinematics.py 会拿 MuJoCo 核对这里的结果。

逆运动学：给定脚（脚底参考点）在躯干系的位置，髋 yaw 取给定值，解髋 roll、髋 pitch、膝；
踝关节让脚底保持和站立时一样的俯仰（三个 pitch 轴平行，所以只要它们的转角之和不变）。
没有踝 roll，髋 roll 转多少，脚底就跟着侧倾多少。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

# 关节全为 0 时各关节轴经过的点（mm）和方向，以及脚底参考点（xgoduck MJCF 的 left_foot / right_foot site）
_LEFT_ANCHORS_MM = [(20.0, 17.5, -26.0), (34.5, 17.5, -40.5), (18.0, 40.0, -40.5), (-17.75, 40.0, -62.5),
                    (-17.75, 64.0, -104.5)]
_LEFT_AXES = [(0, 0, -1), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 1, 0)]
_LEFT_FOOT_MM = (-12.38, 30.67, -127.49)

SIDES = ("left", "right")
HIP_YAW, HIP_ROLL, HIP_PITCH, KNEE, ANKLE = range(5)


def _mirror(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.array(points, dtype=float)
    arr[..., 1] *= -1
    return arr


GEOMETRY: Dict[str, Dict[str, np.ndarray]] = {
    "left": {
        "anchors": np.array(_LEFT_ANCHORS_MM) / 1000.0,
        "axes": np.array(_LEFT_AXES, dtype=float),
        "foot": np.array(_LEFT_FOOT_MM) / 1000.0,
    },
    "right": {
        # 右腿是左腿的镜像：点的 y 取反；pitch 类轴（y 方向）也取反，yaw / roll 轴不变
        "anchors": _mirror(_LEFT_ANCHORS_MM) / 1000.0,
        "axes": np.array([(0, 0, -1), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, -1, 0)], dtype=float),
        "foot": _mirror([_LEFT_FOOT_MM])[0] / 1000.0,
    },
}


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


class Leg:
    def __init__(self, side: str) -> None:
        if side not in GEOMETRY:
            raise ValueError(f"side must be one of {SIDES}")
        g = GEOMETRY[side]
        self.side = side
        self.anchors = g["anchors"]
        self.axes = g["axes"]
        self.foot0 = g["foot"]
        # 三个 pitch 类关节绕同一方向（±y）转，脚底俯仰 = 它们带符号的转角之和
        self._pitch_sign = self.axes[[HIP_PITCH, KNEE, ANKLE], 1]

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """5 个关节角 → 脚底参考点在躯干系的位置。"""
        v = self.foot0.copy()
        for k in (ANKLE, KNEE, HIP_PITCH, HIP_ROLL, HIP_YAW):
            c = self.anchors[k]
            v = _rotation(self.axes[k], float(q[k])) @ (v - c) + c
        return v

    def foot_pitch(self, q: Sequence[float]) -> float:
        """脚底绕 pitch 轴相对零位转过的角度。"""
        return float(self._pitch_sign @ np.asarray([q[HIP_PITCH], q[KNEE], q[ANKLE]], dtype=float))

    def ankle_for(self, q: Sequence[float], foot_pitch: float) -> float:
        """在给定髋 pitch、膝的情况下，让脚底俯仰等于 foot_pitch 的踝角。"""
        sp, sk, sa = self._pitch_sign
        return (foot_pitch - sp * q[HIP_PITCH] - sk * q[KNEE]) / sa

    def ik(
        self,
        foot: Sequence[float],
        foot_pitch: float,
        hip_yaw: float = 0.0,
        q_init: Optional[Sequence[float]] = None,
        tol: float = 1e-7,
        max_iter: int = 20,
    ) -> np.ndarray:
        """脚底参考点位置 + 脚底俯仰 → 5 个关节角（牛顿迭代，q_init 用上一周期的解收敛最快）。

        到不了的位置返回最接近的解，不抛异常；调用方用 fk 检查误差。
        """
        target = np.asarray(foot, dtype=float)
        q = np.zeros(5) if q_init is None else np.array(q_init, dtype=float)
        q[HIP_YAW] = hip_yaw
        free = [HIP_ROLL, HIP_PITCH, KNEE]

        def residual(qq: np.ndarray) -> np.ndarray:
            qq[ANKLE] = self.ankle_for(qq, foot_pitch)
            return self.fk(qq) - target

        r = residual(q)
        eps = 1e-6
        for _ in range(max_iter):
            if float(r @ r) < tol * tol:
                break
            jac = np.empty((3, 3))
            for col, k in enumerate(free):
                dq = q.copy()
                dq[k] += eps
                jac[:, col] = (residual(dq) - r) / eps
            # 阻尼最小二乘，接近奇异（腿伸直）时也稳定
            step = np.linalg.solve(jac.T @ jac + 1e-6 * np.eye(3), jac.T @ r)
            q[free] -= step
            r = residual(q)
        return q

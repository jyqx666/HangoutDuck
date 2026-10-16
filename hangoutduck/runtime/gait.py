"""保底步态：手写的开环行走，不依赖 RL。

每个步态周期左右脚各迈一步，一步分三段：
  1. 双脚着地，躯干向支撑脚一侧晃过去（两只脚相对躯干同时往另一侧移）；
  2. 摆动脚抬起、向前迈 step_length，支撑脚相对躯干向后移同样的距离（躯干因此前进）；
  3. 摆动脚落地，进入下一段双脚支撑。
脚的轨迹在躯干坐标系里规划，再用 kinematics.Leg 的逆运动学换成关节角；脚底保持站立时的俯仰。
起步和停步时所有幅度在 ramp_cycles 个周期内渐变，所以从 home 站姿开始、回到 home 站姿结束。

参数在 sim2sim 里调（rl 分支 scripts/sim2sim.py）；真机上先在吊架上看动作，再落地。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Dict, Optional

import numpy as np

from ..config import RobotConfig
from ..kinematics import HIP_ROLL, HIP_YAW, SIDES, Leg
from ..robot import JointState
from .controllers import Controller, ImuObs, MoveTo, smoothstep


@dataclass
class GaitParams:
    period_s: float = 0.5  # 一个完整周期（左右各一步）
    step_length_m: float = 0.015  # 每步摆动脚相对躯干从后 L/2 迈到前 L/2
    lift_m: float = 0.012  # 抬脚高度
    sway_m: float = 0.012  # 重心左右晃动幅度（两只脚相对躯干的侧移）
    sway_lead: float = 0.0  # 晃动相对迈步提前的相位（周期的比例）
    double_support: float = 0.2  # 每半个周期里双脚着地的比例
    crouch_m: float = 0.004  # 走路时躯干比站立低多少，给膝关节留伸展余量
    narrow_m: float = 0.0  # 走路时两只脚各向中线收多少（脚在髋正下方时重心要横移的距离更短）
    turn_rad: float = 0.0  # 每步转向角（正 = 向左转），用髋 yaw 实现
    ramp_cycles: float = 2.0  # 起步 / 停步渐变的周期数
    settle_s: float = 1.5  # 起步前先平滑站到 home 的时间
    imu_roll_gain: float = 0.0  # 机身侧倾时用髋 roll 回正（rad / rad），0 = 纯开环
    imu_pitch_gain: float = 0.0  # 机身前后倾时用踝回正（rad / rad）

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> "GaitParams":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown gait parameters: {sorted(unknown)}; known: {sorted(known)}")
        return cls(**{k: float(v) for k, v in values.items()})


class Walk(Controller):
    """手写步态。steps = 走多少步（左右各算一步）后停下；None 表示一直走，直到外部停止。"""

    name = "walk"

    def __init__(self, cfg: RobotConfig, steps: Optional[int] = 20, params: Optional[GaitParams] = None,
                 **overrides: float) -> None:
        super().__init__(cfg)
        base = params or GaitParams()
        self.p = GaitParams.from_dict({**{f.name: getattr(base, f.name) for f in fields(GaitParams)}, **overrides})
        self.steps = steps
        self.legs = {s: Leg(s) for s in SIDES}
        self._cols = {s: [cfg.index(f"{s}_{j}") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
                      for s in SIDES}
        home = cfg.home_rad
        self.home = home.copy()
        self.home_foot = {s: self.legs[s].fk(home[self._cols[s]]) for s in SIDES}
        self.home_pitch = {s: self.legs[s].foot_pitch(home[self._cols[s]]) for s in SIDES}
        self._q = {s: home[self._cols[s]].copy() for s in SIDES}
        self._stand = MoveTo(cfg, home, self.p.settle_s)
        self.max_ik_error_m = 0.0

    # ---- 时间轴 ------------------------------------------------------------

    @property
    def walk_cycles(self) -> Optional[float]:
        return None if self.steps is None else self.steps / 2.0

    def _amplitude(self, tw: float) -> float:
        """起步、停步时幅度在 ramp_cycles 个周期内渐变；停步后为 0。"""
        p = self.p
        ramp = max(p.ramp_cycles * p.period_s, 1e-6)
        a = smoothstep(tw / ramp)
        if self.walk_cycles is not None:
            end = (self.walk_cycles + 2 * p.ramp_cycles) * p.period_s
            a = min(a, smoothstep((end - tw) / ramp))
        return a

    def done(self, t: float) -> bool:
        if self.walk_cycles is None:
            return False
        p = self.p
        return t > p.settle_s + (self.walk_cycles + 2 * p.ramp_cycles + 1) * p.period_s

    # ---- 脚的轨迹 ----------------------------------------------------------

    def foot_offsets(self, phase: float, a: float) -> Dict[str, np.ndarray]:
        """相位 phase ∈ [0, 1)、幅度 a ∈ [0, 1] 时两只脚相对 home 位置的偏移（躯干系，m）。

        phase ∈ [0, 0.5) 左脚摆动，[0.5, 1) 右脚摆动。
        """
        p = self.p
        d = 0.5 * p.double_support / 2.0  # 每段摆动前后各留的双支撑相位
        swing_len = 0.5 - 2 * d
        out = {}
        # 左脚摆动时躯干要在右脚上方：两只脚相对躯干往 +y（左）移
        sway = p.sway_m * a * math.sin(2 * math.pi * (phase + p.sway_lead))
        for side, start in (("left", d), ("right", 0.5 + d)):
            u = (phase - start) % 1.0  # 本脚摆动开始后经过的相位
            if u < swing_len:
                s = u / swing_len
                x = -0.5 + smoothstep(s)
                z = math.sin(math.pi * s)
            else:
                s = (u - swing_len) / (1.0 - swing_len)
                x = 0.5 - s
                z = 0.0
            inward = -p.narrow_m * a if side == "left" else p.narrow_m * a
            out[side] = np.array([p.step_length_m * a * x, sway + inward, p.lift_m * a * z + p.crouch_m * a])
        return out

    # ---- 控制器接口 --------------------------------------------------------

    def reset(self, state: JointState) -> None:
        self._stand.reset(state)
        self._q = {s: self.home[self._cols[s]].copy() for s in SIDES}
        self.max_ik_error_m = 0.0

    def step(self, t: float, state: JointState, imu: Optional[ImuObs]) -> np.ndarray:
        p = self.p
        if t < p.settle_s:
            return self._stand.step(t, state, imu)
        tw = t - p.settle_s
        a = self._amplitude(tw)
        if a <= 0.0 and tw > 0.0:
            return self.home.copy()
        phase = (tw / p.period_s) % 1.0
        offsets = self.foot_offsets(phase, a)

        # 机身倾斜说明舵机被压弯了：脚底平贴地面时，机身倾角 = −(腿相对机身的转角)，
        # 所以按倾斜方向多给一点同号的角度，舵机会把机身推回去
        roll_fix = pitch_fix = 0.0
        if imu is not None and (p.imu_roll_gain or p.imu_pitch_gain):
            g = imu.gravity  # 平放为 [0, 0, -1]
            roll_fix = p.imu_roll_gain * math.atan2(-g[1], -g[2])
            pitch_fix = p.imu_pitch_gain * math.atan2(g[0], -g[2])

        q = self.home.copy()
        for side in SIDES:
            leg, cols = self.legs[side], self._cols[side]
            target = self.home_foot[side] + offsets[side]
            # 转向：摆动时两条腿的髋 yaw 反向，落地后保持，下一步回来
            yaw = 0.0
            if p.turn_rad:
                yaw = 0.5 * p.turn_rad * a * math.sin(2 * math.pi * phase) * (1 if side == "left" else -1)
            sol = leg.ik(target, self.home_pitch[side] + pitch_fix, hip_yaw=self.home[cols[HIP_YAW]] + yaw,
                         q_init=self._q[side])
            err = float(np.linalg.norm(leg.fk(sol) - target))
            self.max_ik_error_m = max(self.max_ik_error_m, err)
            self._q[side] = sol
            q[cols] = sol
            q[cols[HIP_ROLL]] += roll_fix
        return q


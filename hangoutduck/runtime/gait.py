"""保底步态：手写的脚步轨迹，不依赖 RL。

快步频（默认约 3 步/秒）左右交替：摆动脚抬起、从后 L/2 迈到前 L/2，同时支撑脚相对躯干向后移，躯干因此前进；
两段摆动之间留一小段双脚支撑。重心不主动横移，靠快步频下的动态摇摆换脚（这条腿没有踝 roll，
慢慢横移重心反而抬不起脚，见 GaitParams 的说明）。
脚的轨迹在躯干坐标系里规划，再用 kinematics.Leg 的逆运动学换成关节角；脚底保持站立时的俯仰。
接了 IMU 时加两个小修正：机身前后倾时用踝回正、侧倒时摆动脚往倒的一侧迈；不接 IMU 也能走。
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
    """默认值是在 sim2sim 里挑出来的（rl 分支 scripts/sim2sim.py）。

    这条腿没有踝 roll：髋 roll 一转，脚底就跟着侧倾，所以「先把重心平移到支撑脚上再抬脚」的慢步态
    抬不起脚（机器人只会在脚掌边上来回摇）。可行的是快步频（约 3 步/秒）、不主动晃重心、靠动态摇摆
    换脚，再用 IMU 做两个小修正（前后倾时用踝回正；侧倒时摆动脚往倒的一侧迈）。不接 IMU 也能走，只是更晃。
    在 15 种硬件偏差下（舵机 P 5–10、延迟 35 ms、整机轻 20% / 重 15%、地面摩擦 0.6、各关节零位 ±2°、
    供电 7.2 V、IMU 噪声且装歪 3°、躯干质心前后偏 5 mm、不接 IMU）走 20 步都不摔，前进 6–21 cm；
    方向会偏（最多约 40°），走直线需要在真机上用 turn_rad 微调（仿真里 turn_rad=0.3 约每 20 步转 45–55°，
    0.1 以下几乎被脚底打滑吃掉）。
    """

    period_s: float = 0.35  # 一个完整周期（左右各一步）
    step_length_m: float = 0.015  # 每步摆动脚相对躯干从后 L/2 迈到前 L/2
    lift_m: float = 0.015  # 抬脚高度
    sway_m: float = 0.0  # 主动左右晃重心的幅度（两只脚相对躯干的侧移）；快步频下靠动态摇摆就够，见上面的说明
    sway_lead: float = 0.0  # 晃动相对迈步提前的相位（周期的比例）
    double_support: float = 0.1  # 每半个周期里双脚着地的比例
    crouch_m: float = 0.004  # 走路时躯干比站立低多少，给膝关节留伸展余量
    narrow_m: float = 0.0  # 走路时两只脚各向中线收多少（脚在髋正下方时重心要横移的距离更短）
    pitch_trim_rad: float = 0.0  # 走路时机身后仰多少（脚底平贴地面时，脚相对机身的俯仰反过来就是机身的俯仰）
    turn_rad: float = 0.0  # 每个周期的转向角（正 = 向左转），用髋 yaw 实现
    ramp_cycles: float = 2.0  # 起步 / 停步渐变的周期数
    settle_s: float = 1.5  # 起步前先平滑站到 home 的时间
    step_pitch_gain: float = 0.0  # 落脚点反馈：机身前倾 1 rad，摆动脚多往前迈多少米（向后倾则往后）
    step_roll_gain: float = 0.03  # 落脚点反馈：机身向一侧倒 1 rad，摆动脚往那一侧多迈多少米
    imu_roll_gain: float = 0.0  # 机身侧倾时用髋 roll 回正（rad / rad），0 = 纯开环
    imu_pitch_gain: float = 0.2  # 机身前后倾时用踝回正（rad / rad）

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

    def _shape(self, phase: float, side: str):
        """本脚在周期里的进度：前后位置 x ∈ [−½, ½]（摆动中从后到前、支撑中匀速从前到后）、
        抬脚高度 z ∈ [0, 1]、落脚点修正权重 w ∈ [0, 1]（摆动中加上、支撑中收回）。"""
        d = 0.5 * self.p.double_support / 2.0  # 每段摆动前后各留的双支撑相位
        swing_len = 0.5 - 2 * d
        start = d if side == "left" else 0.5 + d
        u = (phase - start) % 1.0  # 本脚摆动开始后经过的相位
        if u < swing_len:
            s = u / swing_len
            return -0.5 + smoothstep(s), math.sin(math.pi * s), smoothstep(s)
        s = (u - swing_len) / (1.0 - swing_len)
        return 0.5 - s, 0.0, 1.0 - s

    def foot_offsets(self, phase: float, a: float, pitch: float = 0.0, roll: float = 0.0) -> Dict[str, np.ndarray]:
        """相位 phase ∈ [0, 1)、幅度 a ∈ [0, 1] 时两只脚相对 home 位置的偏移（躯干系，m）。

        phase ∈ [0, 0.5) 左脚摆动，[0.5, 1) 右脚摆动。pitch / roll 是机身倾角（前倾、左侧抬高为正），
        用于落脚点反馈：修正量在摆动中逐渐加上，落地后在支撑期里逐渐收回，轨迹保持连续。
        """
        p = self.p
        out = {}
        # 左脚摆动时躯干要在右脚上方：两只脚相对躯干往 +y（左）移
        sway = p.sway_m * a * math.sin(2 * math.pi * (phase + p.sway_lead))
        for side in SIDES:
            x, z, w = self._shape(phase, side)
            inward = -p.narrow_m * a if side == "left" else p.narrow_m * a
            dx = p.step_pitch_gain * pitch * w
            dy = -p.step_roll_gain * roll * w  # 左侧抬高（roll > 0）说明在往右倒，脚往右（−y）迈
            out[side] = np.array([p.step_length_m * a * x + dx, sway + inward + dy, p.lift_m * a * z + p.crouch_m * a])
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
        pitch = roll = 0.0
        if imu is not None:
            g = imu.gravity  # 平放为 [0, 0, -1]
            pitch = math.atan2(g[0], -g[2])  # 前倾为正
            roll = math.atan2(-g[1], -g[2])  # 左侧抬高为正
        offsets = self.foot_offsets(phase, a, pitch * a, roll * a)

        # 机身倾斜说明舵机被压弯了：脚底平贴地面时，机身倾角 = −(腿相对机身的转角)，
        # 所以按倾斜方向多给一点同号的角度，舵机会把机身推回去
        roll_fix = p.imu_roll_gain * roll
        pitch_fix = p.imu_pitch_gain * pitch

        q = self.home.copy()
        for side in SIDES:
            leg, cols = self.legs[side], self._cols[side]
            target = self.home_foot[side] + offsets[side]
            # 转向：腿相对躯干绕 +z 的转角和前后迈步同一个节奏——支撑中从 +½ 匀速转到 −½（脚踩在地上，
            # 躯干因此转过 turn_rad），摆动中在空中转回来。髋 yaw 轴朝 −z，所以关节角取负
            yaw = -p.turn_rad * a * self._shape(phase, side)[0] if p.turn_rad else 0.0
            sol = leg.ik(target, self.home_pitch[side] + p.pitch_trim_rad * a + pitch_fix,
                         hip_yaw=self.home[cols[HIP_YAW]] + yaw,
                         q_init=self._q[side])
            err = float(np.linalg.norm(leg.fk(sol) - target))
            self.max_ik_error_m = max(self.max_ik_error_m, err)
            self._q[side] = sol
            q[cols] = sol
            q[cols[HIP_ROLL]] += roll_fix
        return q


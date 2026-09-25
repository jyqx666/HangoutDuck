"""整机抽象：按关节表把「关节角（弧度，仿真约定）」和「舵机刻度」互相换算，
一次同步读拿全部关节状态，一次同步写下发全部目标。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import RobotConfig, rad_per_tick
from .feetech import protocol as p
from .feetech import registers as reg
from .feetech.bus import FeetechBus


@dataclass
class JointState:
    t: float  # time.perf_counter() 时刻
    q: np.ndarray  # 关节角 rad
    dq: np.ndarray  # 关节角速度 rad/s
    load: np.ndarray  # 负载，0.1 % 为单位，带方向
    voltage: np.ndarray  # V
    temperature: np.ndarray  # °C
    valid: np.ndarray  # 本次是否收到了该舵机的应答

    @property
    def all_valid(self) -> bool:
        return bool(self.valid.all())


class Robot:
    def __init__(self, bus: FeetechBus, cfg: RobotConfig) -> None:
        self.bus = bus
        self.cfg = cfg
        self.ids: List[int] = cfg.ids
        n = len(self.ids)
        self._zero = np.array([j.zero for j in cfg.joints], dtype=float)
        self._sign = np.array([j.sign for j in cfg.joints], dtype=float)
        self._rpt = rad_per_tick(cfg)
        self.lower, self.upper = cfg.soft_limits_rad()
        self._last = JointState(
            t=0.0,
            q=np.zeros(n),
            dq=np.zeros(n),
            load=np.zeros(n),
            voltage=np.zeros(n),
            temperature=np.zeros(n),
            valid=np.zeros(n, dtype=bool),
        )
        self.last_target: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return len(self.ids)

    # ---- 换算 -------------------------------------------------------------

    def ticks_to_rad(self, ticks) -> np.ndarray:
        return (np.asarray(ticks, dtype=float) - self._zero) * self._rpt * self._sign

    def rad_to_ticks(self, q) -> np.ndarray:
        ticks = self._zero + np.asarray(q, dtype=float) / self._rpt * self._sign
        return np.clip(np.rint(ticks), 0, self.cfg.ticks_per_rev - 1).astype(int)

    def clamp(self, q) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.lower, self.upper)

    # ---- 读写 -------------------------------------------------------------

    def ping_all(self) -> List[int]:
        """返回没有应答的 ID。"""
        return [i for i in self.ids if not self.bus.ping(i)]

    def read_state(self) -> JointState:
        raw = self.bus.sync_read(reg.PRESENT_POSITION, reg.STATE_BLOCK_LEN, self.ids)
        prev = self._last
        st = JointState(
            t=time.perf_counter(),
            q=prev.q.copy(),
            dq=prev.dq.copy(),
            load=prev.load.copy(),
            voltage=prev.voltage.copy(),
            temperature=prev.temperature.copy(),
            valid=np.zeros(self.n, dtype=bool),
        )
        for k, sid in enumerate(self.ids):
            data = raw.get(sid)
            if data is None:
                continue
            ticks = p.parse_u16_le(data, 0)
            speed = p.decode_sign_magnitude(p.parse_u16_le(data, 2), 15)
            st.q[k] = (ticks - self._zero[k]) * self._rpt * self._sign[k]
            st.dq[k] = speed * reg.SPEED_UNIT_TICKS_PER_S * self._rpt * self._sign[k]
            st.load[k] = p.decode_sign_magnitude(p.parse_u16_le(data, 4), 10)
            st.voltage[k] = data[6] / 10.0
            st.temperature[k] = data[7]
            st.valid[k] = True
        self._last = st
        return st

    def command(self, q_target) -> np.ndarray:
        """下发目标关节角（rad），先裁剪到软限位。返回实际下发的角度。"""
        q = self.clamp(q_target)
        ticks = self.rad_to_ticks(q)
        # goal_position, goal_time=0, goal_speed=0（0 表示不限速，由舵机 PD 决定）
        block = {sid: p.u16_le(int(t)) + b"\x00\x00\x00\x00" for sid, t in zip(self.ids, ticks)}
        self.bus.sync_write(reg.GOAL_POSITION, reg.GOAL_BLOCK_LEN, block)
        self.last_target = q
        return q

    def set_torque(self, enable: bool) -> None:
        """上力前先把目标位置设为当前位置，避免上力瞬间跳变。"""
        if enable:
            raw = self.bus.sync_read(reg.PRESENT_POSITION, 2, self.ids)
            missing = [i for i in self.ids if i not in raw]
            if missing:
                raise RuntimeError(f"cannot enable torque, no reply from ids {missing}")
            block = {sid: raw[sid] + b"\x00\x00\x00\x00" for sid in self.ids}
            self.bus.sync_write(reg.GOAL_POSITION, reg.GOAL_BLOCK_LEN, block)
        self.bus.sync_write(reg.TORQUE_ENABLE, 1, {sid: bytes([1 if enable else 0]) for sid in self.ids})

    def torque_off(self) -> None:
        """急停用：广播卸力，发两次防止丢包；不等应答。"""
        for _ in range(2):
            self.bus.write_u8(p.BROADCAST_ID, reg.TORQUE_ENABLE, 0)

    def apply_run_pd(self) -> None:
        if self.cfg.run_pd is None:
            return
        kp, kd = self.cfg.run_pd
        self.bus.sync_write(reg.TEMP_P, 2, {sid: bytes([kp, kd]) for sid in self.ids})

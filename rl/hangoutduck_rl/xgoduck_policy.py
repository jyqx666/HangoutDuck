"""把 xgoduck 的预训练行走策略（14 个关节、61 维观测）当作 hangoutduck 控制器，跑在 10 关节的 HangoutDuck 上。

用途：sim2sim 自检。这个策略在训练仿真里能走，如果它通过真机程序的整条链路在 sim2sim 里也能走，
说明 sim2sim 搭得对。头颈是固定的，正好停在 xgoduck 的 home 角度，所以头颈的关节角、角速度
观测都是 0，头部 / 身体姿态指令也给 0，策略输出的头颈动作丢掉。

xgoduck 的 ONNX 元数据里站姿写的是 ±20°，但它自己的运行时和训练环境都用 ±24°，这里也用 ±24°（joints.yaml 的 home）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import onnxruntime as ort

from hangoutduck.config import RobotConfig
from hangoutduck.robot import JointState
from hangoutduck.runtime.controllers import Controller, ImuObs

OBS_DIM = 61
LEG_IN_14 = list(range(0, 5)) + list(range(9, 14))  # 14 个关节里的腿：左 5、（头颈 4）、右 5


class XgoduckWalk(Controller):
    name = "xgoduck_walk"

    def __init__(self, cfg: RobotConfig, onnx_path: str, command: Sequence[float] = (0.2, 0.0, 0.0)) -> None:
        super().__init__(cfg)
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        names = meta["joint_names"].split(",")
        legs = [names[i] for i in LEG_IN_14]
        if legs != list(cfg.names):
            raise ValueError(f"policy leg joints {legs} != joints.yaml {cfg.names}")
        self.home = cfg.home_rad.copy()
        self.command = np.asarray(command, dtype=np.float32)
        self.prev = np.zeros(14, dtype=np.float32)

    def reset(self, state: JointState) -> None:
        self.prev[:] = 0.0

    def step(self, t: float, state: JointState, imu: Optional[ImuObs]) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        if imu is not None:
            obs[0:3] = imu.gyro
            obs[3:6] = imu.gravity
        else:
            obs[5] = -1.0
        obs[6:20][LEG_IN_14] = state.q - self.home
        obs[20:34][LEG_IN_14] = state.dq
        obs[34:48] = self.prev
        obs[48:51] = self.command
        action = self.sess.run(None, {"obs": obs[None]})[0][0].astype(np.float32)
        self.prev = action
        return self.home + action[LEG_IN_14]

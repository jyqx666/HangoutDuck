"""在真机上运行 rl 分支导出的 ONNX 行走策略。

观测、动作约定与 rl/hangoutduck_rl/tasks.py 完全一致（39 维观测、10 维动作）：
  [0:3]   机体系角速度 rad/s        [3:6]   projected gravity
  [6:16]  关节角 - home，rad        [16:26] 关节角速度 rad/s
  [26:36] 上一步动作（网络原始输出） [36:39] 速度指令 [vx, vy, wz]
目标关节角 = home + action。
"""

from __future__ import annotations

import json
from typing import Callable, Optional, Sequence

import numpy as np

from ..config import RobotConfig
from ..robot import JointState
from .controllers import Controller, ImuObs

OBS_DIM = 39
ACTION_DIM = 10
OBS_TERMS = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command"]


def load_onnx(path: str) -> "tuple[Callable[[np.ndarray], np.ndarray], dict]":
    """返回 (推理函数, 元数据)。推理函数输入 (39,) float32，输出 (10,)。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    if inp.shape[-1] not in (OBS_DIM, None, "obs_dim"):
        raise ValueError(f"{path}: policy expects {inp.shape[-1]} observations, runtime builds {OBS_DIM}")
    meta = dict(sess.get_modelmeta().custom_metadata_map)

    def infer(obs: np.ndarray) -> np.ndarray:
        out = sess.run(None, {inp.name: obs.reshape(1, -1).astype(np.float32)})[0]
        return np.asarray(out, dtype=float).reshape(-1)

    return infer, meta


def _meta_list(meta: dict, key: str) -> Optional[list]:
    """mjlab 导出的元数据是逗号分隔字符串或 JSON；取不到时返回 None。"""
    raw = meta.get(key)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
        return list(value) if isinstance(value, (list, tuple)) else None
    except (TypeError, ValueError):
        return [x.strip() for x in str(raw).split(",")]


class PolicyController(Controller):
    name = "policy"

    def __init__(
        self,
        cfg: RobotConfig,
        onnx_path: Optional[str] = None,
        command: Sequence[float] = (0.0, 0.0, 0.0),
        infer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        meta: Optional[dict] = None,
        action_alpha: float = 0.0,
    ) -> None:
        super().__init__(cfg)
        if infer is None:
            if onnx_path is None:
                raise ValueError("need onnx_path or infer")
            infer, meta = load_onnx(onnx_path)
        self.infer = infer
        self.meta = meta or {}
        self.home = cfg.home_rad
        self.command = np.asarray(command, dtype=float)
        # 动作低通：0 表示不滤波（与训练一致）；真机抖动大时可试 0.3–0.5
        self.action_alpha = action_alpha
        self.prev_action = np.zeros(ACTION_DIM)
        self._smoothed = np.zeros(ACTION_DIM)
        self._obs = np.zeros(OBS_DIM, dtype=np.float32)
        self._check_meta()

    def _check_meta(self) -> None:
        terms = _meta_list(self.meta, "observation_names")
        if terms and list(terms) != OBS_TERMS:
            raise ValueError(f"policy observation terms {terms} != runtime layout {OBS_TERMS}")
        names = _meta_list(self.meta, "joint_names")
        if names and list(names) != self.cfg.names:
            raise ValueError(f"policy joint order {names} != joints.yaml {self.cfg.names}")
        defaults = _meta_list(self.meta, "default_joint_pos")
        if defaults and len(defaults) == ACTION_DIM:
            diff = np.abs(np.asarray(defaults, dtype=float) - self.home)
            if diff.max() > np.radians(1.0):
                raise ValueError(f"policy home pose differs from joints.yaml home by {np.degrees(diff.max()):.1f} deg")

    def reset(self, state: JointState) -> None:
        self.prev_action[:] = 0.0
        self._smoothed[:] = 0.0

    def build_obs(self, state: JointState, imu: Optional[ImuObs]) -> np.ndarray:
        obs = self._obs
        if imu is None:
            obs[0:3] = 0.0
            obs[3:6] = (0.0, 0.0, -1.0)
        else:
            obs[0:3] = imu.gyro
            obs[3:6] = imu.gravity
        obs[6:16] = state.q - self.home
        obs[16:26] = state.dq
        obs[26:36] = self.prev_action
        obs[36:39] = self.command
        return obs

    def step(self, t, state, imu):
        action = self.infer(self.build_obs(state, imu))
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"policy returned shape {action.shape}, expected ({ACTION_DIM},)")
        self.prev_action = action.copy()
        if self.action_alpha > 0:
            self._smoothed = self.action_alpha * self._smoothed + (1 - self.action_alpha) * action
            action = self._smoothed
        return self.home + action

"""HangoutDuck 下半身模型。

在 xgoduck 的 MJCF 上删掉整个颈/头子树（neck → neck_pitch → yaw_roll_motion → jaw_soft）
和对应的 4 个执行器，腿、HLS1910 的 BAM 执行器模型、碰撞设置全部沿用 xgoduck。
"""

from __future__ import annotations

import math
from typing import Optional

import mujoco
import numpy as np
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_microduck.robot import xgoduck_constants as xgo

NECK_ROOT_BODY = "neck"
REMOVED_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")

# 与 main 分支 hardware/joints.yaml 的顺序一致，也是策略动作/观测里的关节顺序
LEG_JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)

# ---- 需要按实物修改的参数 ----------------------------------------------------
# 躯干（trunk_base）质量，kg：称「躯干打印件 + 躯干上的一切（IMU、线材、螺丝），不含腿」。
# None 表示沿用 xgoduck 的值（0.238 kg，是它按整机 0.8 kg 缩放出来的）。
# 我们身上没有主控和电池，躯干很可能更轻，务必实称后填写。
# 改完用 `python rl/scripts/check_model.py` 看整机质量和质心。
TRUNK_MASS_KG: Optional[float] = None


def strip_head(spec: mujoco.MjSpec, trunk_mass_kg: Optional[float] = None) -> mujoco.MjSpec:
    """原地修改 spec：删掉颈/头，按需改躯干质量（惯量按同比例缩放）。"""
    for actuator in list(spec.actuators):
        if actuator.target in REMOVED_JOINTS:
            spec.delete(actuator)
    spec.delete(spec.body(NECK_ROOT_BODY))
    if trunk_mass_kg is not None:
        trunk = spec.body("trunk_base")
        scale = trunk_mass_kg / trunk.mass
        trunk.mass = trunk_mass_kg
        trunk.fullinertia = np.asarray(trunk.fullinertia) * scale
    return spec


def get_walk_spec() -> mujoco.MjSpec:
    return strip_head(xgo.get_walk_spec(), TRUNK_MASS_KG)


def get_standup_spec() -> mujoco.MjSpec:
    return strip_head(xgo.get_standup_spec(), TRUNK_MASS_KG)


_DEG24 = math.radians(24)

# 与 xgoduck 相同的站姿（hip_pitch / ankle ±24°，hip_roll ±5°），去掉颈/头
HOME_JOINT_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -_DEG24,
    "left_knee": -0.0049,
    "left_ankle": _DEG24,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": _DEG24,
    "right_knee": 0.0049,
    "right_ankle": -_DEG24,
}

# 腿的几何没变，站立高度沿用 xgoduck
STAND_Z = xgo.XGODUCK_STAND_Z

HOME_FRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, STAND_Z),
    joint_pos=dict(HOME_JOINT_POS),
    joint_vel={".*": 0.0},
)

HANGOUTDUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(xgo.XGODUCK_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(xgo.actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

HANGOUTDUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_standup_spec,
    init_state=HOME_FRAME,
    collisions=(xgo.XGODUCK_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(xgo.actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

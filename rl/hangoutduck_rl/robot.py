"""HangoutDuck 仿真模型：10 个腿部关节可动，头颈保留为固定件。

真机头颈不装舵机和传感器，但仿真里保留整个头颈子树
（neck → neck_pitch → yaw_roll_motion → jaw_soft）的几何和质量：
删掉 4 个颈/头关节和执行器，把头颈按 xgoduck 的 home 角度焊死在躯干上。
腿、HLS1910 的 BAM 执行器模型、碰撞设置全部沿用 xgoduck。
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import mujoco
import numpy as np
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_microduck.robot import xgoduck_constants as xgo

NECK_ROOT_BODY = "neck"
HEAD_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")

# 头颈关节焊死的角度（rad），即 xgoduck 的 home 姿态
HEAD_FIXED_POS: Dict[str, float] = {
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
}

# 与 main 分支 hardware/joints.yaml 的顺序一致，也是策略动作/观测里的关节顺序
LEG_JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)

# ---- 需要按实物修改的参数 ----------------------------------------------------
# 改完用 `python rl/scripts/check_model.py` 看整机质量和质心。
#
# 躯干（trunk_base）质量，kg：称「躯干打印件 + 躯干上的一切（IMU、线材、螺丝），不含腿和头」。
# None 表示沿用 xgoduck 的值（0.238 kg，是它按整机 0.8 kg 缩放出来的）。
TRUNK_MASS_KG: Optional[float] = None
# 头颈总质量，kg（neck、neck_pitch、yaw_roll_motion、jaw_soft 四个 body 合计）。
# None 表示沿用 xgoduck 的 0.251 kg，其中包含 4 个颈/头舵机；
# 我们的头不装舵机，实物会更轻，装好后称重填写（各 body 按比例缩放）。
HEAD_MASS_KG: Optional[float] = None


def _scale_body_mass(body, scale: float) -> None:
    body.mass = body.mass * scale
    body.fullinertia = np.asarray(body.fullinertia) * scale


def _rotate_about_local_z(quat, angle: float) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(quat, dtype=float), np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]))
    return out


def freeze_head(
    spec: mujoco.MjSpec,
    trunk_mass_kg: Optional[float] = None,
    head_mass_kg: Optional[float] = None,
) -> mujoco.MjSpec:
    """原地修改 spec：删掉颈/头的关节和执行器，头颈按 HEAD_FIXED_POS 焊在躯干上。

    xgoduck 的颈/头关节都在各自 body 的原点、绕本地 z 轴转动，
    所以把 body 的姿态绕本地 z 轴转过对应角度，就等于把关节固定在这个角度。
    """
    for actuator in list(spec.actuators):
        if actuator.target in HEAD_FIXED_POS:
            spec.delete(actuator)
    for joint in list(spec.joints):
        if joint.name in HEAD_FIXED_POS:
            body = joint.parent
            if np.any(np.asarray(joint.pos) != 0) or list(joint.axis) != [0, 0, 1]:
                raise ValueError(f"{joint.name}: expected a z-axis hinge at the body origin")
            body.quat = _rotate_about_local_z(body.quat, HEAD_FIXED_POS[joint.name])
            spec.delete(joint)
    if trunk_mass_kg is not None:
        trunk = spec.body("trunk_base")
        _scale_body_mass(trunk, trunk_mass_kg / trunk.mass)
    if head_mass_kg is not None:
        bodies = [spec.body(n) for n in HEAD_BODIES]
        scale = head_mass_kg / sum(b.mass for b in bodies)
        for b in bodies:
            _scale_body_mass(b, scale)
    return spec


def remove_head(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """只用于对比：完全去掉头颈（之前的方案）。训练不用。"""
    for actuator in list(spec.actuators):
        if actuator.target in HEAD_FIXED_POS:
            spec.delete(actuator)
    spec.delete(spec.body(NECK_ROOT_BODY))
    return spec


def get_walk_spec() -> mujoco.MjSpec:
    return freeze_head(xgo.get_walk_spec(), TRUNK_MASS_KG, HEAD_MASS_KG)


def get_standup_spec() -> mujoco.MjSpec:
    return freeze_head(xgo.get_standup_spec(), TRUNK_MASS_KG, HEAD_MASS_KG)


_DEG24 = math.radians(24)

# 与 xgoduck 相同的站姿（hip_pitch / ankle ±24°，hip_roll ±5°）；头颈是固定件，不在这里
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

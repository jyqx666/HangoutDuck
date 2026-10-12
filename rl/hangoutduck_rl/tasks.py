"""HangoutDuck 行走任务：10 个腿部关节可动，头颈是固定件。

直接复用 xgoduck 的 velocity 配方（奖励、域随机化、课程都不变），只做两件事：
1. 机器人换成 HangoutDuck 模型（头颈焊死在 home 角度，只剩 10 个关节）；
2. 去掉头颈关节相关的项：头部/身体姿态指令、对应的观测、奖励和课程。
   头颈的质心随机化保留（头还在，只是不能动），用来覆盖实物头部重量的不确定。

策略观测因此是 39 维：
  [0:3]   base_ang_vel      机体系角速度 rad/s
  [3:6]   projected_gravity 机体系重力方向
  [6:16]  joint_pos         关节角 - home，rad（LEG_JOINTS 顺序）
  [16:26] joint_vel         rad/s
  [26:36] actions           上一步动作
  [36:39] command           速度指令 [vx, vy, wz]
动作 10 维：目标关节角 = home + action（rad，scale 1.0）。
"""

from __future__ import annotations

import os
from dataclasses import replace

from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.registry import register_mjlab_task

# 导入 mjlab_microduck.tasks 会注册上游全部任务，并提供 xgoduck 的 velocity 配置
from mjlab_microduck.tasks import MicroduckOnPolicyRunner, XgoduckRlCfg, _xgoduck_velocity_cfg

from .robot import HANGOUTDUCK_WALK_ROBOT_CFG, PARAMS

OBS_DIM = 39
ACTION_DIM = 10

_HEAD_COMMANDS = ("head_pose", "body_pose")
_HEAD_OBS_TERMS = ("head_command", "body_command")
_HEAD_REWARDS = ("head_pose_tracking", "head_pose_bias", "body_pose_tracking")
_HEAD_CURRICULA = ("head_pose_range", "body_pose_range", "head_pose_bias_weight")


def strip_head_joint_terms(cfg):
    for name in _HEAD_COMMANDS:
        cfg.commands.pop(name, None)
    for group in cfg.observations.values():
        for name in _HEAD_OBS_TERMS:
            group.terms.pop(name, None)
    for name in _HEAD_REWARDS:
        cfg.rewards.pop(name, None)
    for name in _HEAD_CURRICULA:
        cfg.curriculum.pop(name, None)
    return cfg


def apply_sim_params(cfg, params=PARAMS):
    """把 sim_params.yaml 里的地面摩擦和速度指令范围写进任务配置（质量、舵机等在 robot.py 里）。"""
    cfg.events["foot_friction"].params["ranges"] = params.foot_friction
    ranges = cfg.commands["twist"].ranges
    ranges.lin_vel_x = params.cmd_lin_vel_x
    ranges.lin_vel_y = params.cmd_lin_vel_y
    ranges.ang_vel_z = params.cmd_ang_vel_z
    return cfg


def make_hangoutduck_velocity_env_cfg(play: bool = False, rough: bool = False):
    cfg = _xgoduck_velocity_cfg(play=play, rough=rough)
    cfg.scene.entities = {"robot": HANGOUTDUCK_WALK_ROBOT_CFG}
    return apply_sim_params(strip_head_joint_terms(cfg))


class HangoutduckOnPolicyRunner(MicroduckOnPolicyRunner):
    """导出 ONNX 时多写几项元数据：用的哪套仿真参数，以及舵机 P（真机 run_pd 必须与之相同）。"""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        _, _, onnx_path = self._get_export_paths(path)
        if os.path.exists(onnx_path):
            attach_metadata_to_onnx(str(onnx_path), {
                "sim_params_version": PARAMS.version,
                "sim_params_unconfirmed": PARAMS.unconfirmed(),
                "servo_kp": PARAMS.servo_kp,
            })


# 日志在 logs/rsl_rl/hangoutduck_velocity/，目录名带上仿真参数的 version
HangoutduckRlCfg = replace(
    XgoduckRlCfg,
    experiment_name="hangoutduck_velocity",
    run_name=f"hangoutduck-{PARAMS.version}",
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-HangoutDuck",
    env_cfg=make_hangoutduck_velocity_env_cfg(),
    play_env_cfg=make_hangoutduck_velocity_env_cfg(play=True),
    rl_cfg=HangoutduckRlCfg,
    runner_cls=HangoutduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-HangoutDuck",
    env_cfg=make_hangoutduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_hangoutduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=HangoutduckRlCfg,
    runner_cls=HangoutduckOnPolicyRunner,
)

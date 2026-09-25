"""HangoutDuck 下半身行走任务。

直接复用 xgoduck 的 velocity 配方（奖励、域随机化、课程都不变），只做两件事：
1. 机器人换成下半身模型（10 个关节）；
2. 去掉所有和头相关的项：头部/身体姿态指令、对应的观测、奖励、事件和课程。

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

from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task

# 导入 mjlab_microduck.tasks 会注册上游全部任务，并提供 xgoduck 的 velocity 配置
from mjlab_microduck.tasks import MicroduckOnPolicyRunner, XgoduckRlCfg, _xgoduck_velocity_cfg

from .robot import HANGOUTDUCK_WALK_ROBOT_CFG

OBS_DIM = 39
ACTION_DIM = 10

_HEAD_COMMANDS = ("head_pose", "body_pose")
_HEAD_OBS_TERMS = ("head_command", "body_command")
_HEAD_REWARDS = ("head_pose_tracking", "head_pose_bias", "body_pose_tracking")
_HEAD_EVENTS = ("randomize_head_com",)
_HEAD_CURRICULA = ("head_pose_range", "body_pose_range", "head_com_range", "head_pose_bias_weight")


def strip_head_terms(cfg):
    for name in _HEAD_COMMANDS:
        cfg.commands.pop(name, None)
    for group in cfg.observations.values():
        for name in _HEAD_OBS_TERMS:
            group.terms.pop(name, None)
    for name in _HEAD_REWARDS:
        cfg.rewards.pop(name, None)
    for name in _HEAD_EVENTS:
        cfg.events.pop(name, None)
    for name in _HEAD_CURRICULA:
        cfg.curriculum.pop(name, None)
    return cfg


def make_hangoutduck_velocity_env_cfg(play: bool = False, rough: bool = False):
    cfg = _xgoduck_velocity_cfg(play=play, rough=rough)
    cfg.scene.entities = {"robot": HANGOUTDUCK_WALK_ROBOT_CFG}
    return strip_head_terms(cfg)


# 日志在 logs/rsl_rl/hangoutduck_velocity/
HangoutduckRlCfg = replace(
    XgoduckRlCfg,
    experiment_name="hangoutduck_velocity",
    run_name="hangoutduck",
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-HangoutDuck",
    env_cfg=make_hangoutduck_velocity_env_cfg(),
    play_env_cfg=make_hangoutduck_velocity_env_cfg(play=True),
    rl_cfg=HangoutduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-HangoutDuck",
    env_cfg=make_hangoutduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_hangoutduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=HangoutduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

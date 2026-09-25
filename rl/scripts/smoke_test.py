"""不训练，只把 HangoutDuck 任务建出来跑几步：确认观测 39 维、动作 10 维、观测各段顺序。

CPU 上也能跑（慢），用来在装好环境后、开训之前快速自检：
    cd rl/third_party/xgoduck_rl && uv run python ../../scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import sys

import mjlab  # noqa: F401  加载任务插件
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from hangoutduck_rl.robot import LEG_JOINTS
from hangoutduck_rl.tasks import ACTION_DIM, OBS_DIM

EXPECTED_ACTOR_TERMS = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-HangoutDuck")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    obs, _ = env.reset()

    om = env.observation_manager
    terms = om.active_terms["actor"]
    dims = om.group_obs_term_dim["actor"]
    layout, start = [], 0
    for name, shape in zip(terms, dims):
        width = int(shape[-1])
        layout.append((name, start, start + width))
        start += width
    print("actor observation layout:")
    for name, a, b in layout:
        print(f"  [{a:2d}:{b:2d}] {name}")

    robot = env.scene["robot"]
    action_dim = env.action_manager.total_action_dim
    actor_dim = obs["actor"].shape[-1]
    print(f"actor obs dim {actor_dim}, critic obs dim {obs['critic'].shape[-1]}, action dim {action_dim}")
    print(f"robot joints: {list(robot.joint_names)}")

    total = torch.zeros(args.num_envs, device=args.device)
    for _ in range(args.steps):
        action = torch.zeros(args.num_envs, action_dim, device=args.device)  # 保持 home 姿态
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
    print(f"{args.steps} steps holding home: mean reward {total.mean().item():.3f}, "
          f"terminated {int(terminated.sum())}/{args.num_envs}, obs finite: {bool(torch.isfinite(obs['actor']).all())}")
    env.close()

    problems = []
    if actor_dim != OBS_DIM:
        problems.append(f"actor obs dim {actor_dim} != {OBS_DIM}")
    if action_dim != ACTION_DIM:
        problems.append(f"action dim {action_dim} != {ACTION_DIM}")
    if [t for t in terms] != EXPECTED_ACTOR_TERMS:
        problems.append(f"actor terms {terms} != {EXPECTED_ACTOR_TERMS}")
    if tuple(robot.joint_names) != LEG_JOINTS:
        problems.append(f"joint order {robot.joint_names} != {LEG_JOINTS}")
    for p in problems:
        print("ERROR:", p)
    print("smoke test:", "OK" if not problems else "FAILED")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

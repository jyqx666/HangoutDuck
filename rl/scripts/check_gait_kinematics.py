"""核对 main 分支 hangoutduck/kinematics.py（保底步态用的腿部运动学）与仿真模型是否一致。

kinematics.py 里的关节轴位置和方向是从 xgoduck MJCF 抄出来的常数；模型改了（换打印件、改尺寸）
就要重新核对。随机取关节角，比较 kinematics.Leg.fk 与 MuJoCo 里 left_foot / right_foot site 的位置。

    cd rl/third_party/xgoduck_rl && uv run python ../../scripts/check_gait_kinematics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import mjlab  # noqa: E402,F401
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from hangoutduck.kinematics import SIDES, Leg  # noqa: E402
from hangoutduck_rl import robot as hd  # noqa: E402

JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")


def main() -> int:
    model = hd.get_walk_spec().compile()
    data = mujoco.MjData(model)
    trunk = model.body("trunk_base").id
    legs = {s: Leg(s) for s in SIDES}
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        q = {s: rng.uniform(-0.6, 0.6, 5) for s in SIDES}
        mujoco.mj_resetData(model, data)
        data.qpos[3:7] = [1, 0, 0, 0]
        for s in SIDES:
            for k, j in enumerate(JOINTS):
                data.qpos[model.joint(f"{s}_{j}").qposadr[0]] = q[s][k]
        mujoco.mj_forward(model, data)
        rot, pos = data.xmat[trunk].reshape(3, 3), data.xpos[trunk]
        for s in SIDES:
            site = rot.T @ (data.site_xpos[model.site(f"{s}_foot").id] - pos)
            worst = max(worst, float(np.linalg.norm(legs[s].fk(q[s]) - site)))
    print(f"max |kinematics.fk - MuJoCo| over 200 random poses: {worst * 1000:.3f} mm")
    ok = worst < 1e-4
    print("gait kinematics:", "OK" if ok else "MISMATCH — update hangoutduck/kinematics.py from the model")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

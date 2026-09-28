"""检查 HangoutDuck 模型：关节顺序、质量、站姿质心，以及用 MuJoCo 自带 PD 做 3 秒站立测试。

不需要显卡。在 xgoduck_rl 的环境里运行：
    cd rl/third_party/xgoduck_rl && uv run python ../../scripts/check_model.py

对比三个模型，重点看质心相对双脚支撑区的位置：
  xgoduck（头颈 4 个关节可动，站 home 姿态）
  HangoutDuck（头颈焊死在同一 home 角度，只有 10 个腿关节）—— 质量和质心应与上面完全一致
  去掉头的方案（仅作对比）—— 头在躯干前方，去掉后质心明显后移
"""

from __future__ import annotations

import argparse
import sys

import mjlab  # noqa: F401  先导入 mjlab：它会加载任务插件（含本包），避免循环导入
import mujoco
import numpy as np

from hangoutduck_rl import robot as hd
from mjlab_microduck.robot import xgoduck_constants as xgo

XGODUCK_HOME = {
    **hd.HOME_JOINT_POS,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
}


def add_floor(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    return spec


def set_home(model, data, home: dict, z: float) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [0.0, 0.0, z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, q in home.items():
        data.qpos[model.joint(name).qposadr[0]] = q
    for k in range(model.nu):
        data.ctrl[k] = data.qpos[model.jnt_qposadr[model.actuator_trnid[k, 0]]]
    mujoco.mj_forward(model, data)


def foot_centers(model, data) -> np.ndarray:
    return np.array([data.site(n).xpos for n in ("left_foot", "right_foot")])


def report(label: str, spec: mujoco.MjSpec, home: dict, seconds: float) -> dict:
    model = add_floor(spec).compile()
    data = mujoco.MjData(model)
    hinge = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    actuated = [model.joint(model.actuator_trnid[k, 0]).name for k in range(model.nu)]
    set_home(model, data, home, hd.STAND_Z)
    trunk = model.body("trunk_base").id
    com = data.subtree_com[trunk].copy()
    feet = foot_centers(model, data)
    support_x = feet[:, 0].mean()
    total = float(model.body_subtreemass[trunk])

    # 用 MJCF 里 HLS1910 的伴随 PD 保持站姿（训练用的是 BAM，这里只做静力学粗查）
    steps = int(seconds / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    up = data.xmat[trunk].reshape(3, 3)[:, 2]
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
    z = float(data.xpos[trunk][2])

    print(f"\n== {label} ==")
    print(f"actuated joints ({len(actuated)}): {actuated}")
    print(f"total mass {total:.3f} kg, trunk_base {model.body_mass[trunk]:.3f} kg")
    print(f"COM at home (world): x={com[0]*1000:+.1f} mm  y={com[1]*1000:+.1f} mm  z={com[2]*1000:.1f} mm")
    print(f"feet centre x={support_x*1000:+.1f} mm  -> COM ahead of feet by {(com[0]-support_x)*1000:+.1f} mm")
    print(f"after {seconds:g} s standing with position PD: trunk z={z*1000:.1f} mm, tilt={tilt:.1f} deg "
          f"-> {'stands' if tilt < 15 and z > 0.08 else 'FALLS'}")
    return {"hinge": hinge, "actuated": actuated, "tilt": tilt, "z": z, "com": com}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    ref = report("xgoduck (head joints actuated, upstream)", xgo.get_walk_spec(), XGODUCK_HOME, args.seconds)
    r = report("HangoutDuck (head fixed, 10 leg joints)", hd.get_walk_spec(), hd.HOME_JOINT_POS, args.seconds)
    report("head removed (comparison only)", hd.remove_head(xgo.get_walk_spec()), hd.HOME_JOINT_POS, args.seconds)

    ok = True
    if tuple(r["actuated"]) != hd.LEG_JOINTS:
        print(f"\nERROR: actuator order {r['actuated']} != LEG_JOINTS {hd.LEG_JOINTS}")
        ok = False
    if set(r["hinge"]) != set(hd.LEG_JOINTS):
        print(f"\nERROR: unexpected joints left in model: {sorted(set(r['hinge']) - set(hd.LEG_JOINTS))}")
        ok = False
    if hd.TRUNK_MASS_KG is None and hd.HEAD_MASS_KG is None and not np.allclose(r["com"], ref["com"], atol=1e-4):
        print(f"\nERROR: fixed-head COM {r['com']} differs from xgoduck at home {ref['com']}; head frozen at wrong angle?")
        ok = False
    if hd.TRUNK_MASS_KG is None or hd.HEAD_MASS_KG is None:
        print("\nnote: TRUNK_MASS_KG / HEAD_MASS_KG not set; masses are still xgoduck's. Weigh the real trunk and head "
              "and set them in rl/hangoutduck_rl/robot.py")
    print("\nmodel check:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

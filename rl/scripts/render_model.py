"""把 HangoutDuck 模型渲染成图片/GIF，用于汇报和标定参考。不需要显卡。

没有显示器的机器用虚拟显示运行：
    cd rl/third_party/xgoduck_rl
    MUJOCO_GL=glfw xvfb-run -a -s "-screen 0 1920x1080x24" \
        uv run python ../../scripts/render_model.py --out ../../../docs/sim

输出：
  hangoutduck_home.png   HangoutDuck（头颈固定，10 个腿关节），站姿（home）
  hangoutduck_zero.png   腿部关节全部为 0 的机械零位（标定时要摆的姿态）
  compare_side.png       侧视对比：左 保留头（固定），右 去掉头；红线 = 质心，蓝点 = 双脚中心
  stand_test.gif         两者用位置 PD 保持站姿 3 秒（左 保留头，右 去掉头）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mjlab  # noqa: F401  先导入 mjlab：它会加载任务插件（含本包），避免循环导入
import mujoco
import numpy as np
from PIL import Image

from hangoutduck_rl import robot as hd
from mjlab_microduck.robot import xgoduck_constants as xgo

BODY_RGBA = np.array([0.91, 0.58, 0.23, 1.0])  # 躯干：橙
LEG_RGBA = np.array([0.86, 0.85, 0.82, 1.0])  # 腿：暖灰
HEAD_RGBA = np.array([0.35, 0.45, 0.55, 1.0])  # 头颈（只有 xgoduck 有）


def build(spec: mujoco.MjSpec, width: int, height: int, alpha: float = 1.0) -> mujoco.MjModel:
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.quality.shadowsize = 4096
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.93, 0.92, 0.88], rgb2=[0.86, 0.85, 0.80], width=512, height=512)
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[40, 40], reflectance=0.0)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.97, 0.96, 0.93], rgb2=[0.85, 0.87, 0.90], width=512, height=512)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], material="grid")
    spec.worldbody.add_light(pos=[0.3, -0.4, 1.2], dir=[-0.15, 0.25, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=1,
                             diffuse=[0.7, 0.7, 0.7])
    model = spec.compile()
    trunk = model.body("trunk_base").id
    neck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "neck")

    def in_head(b: int) -> bool:
        while b > 0:
            if b == neck:
                return True
            b = model.body_parentid[b]
        return False

    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        if b == 0:
            continue
        model.geom_rgba[g] = BODY_RGBA if b == trunk else HEAD_RGBA if in_head(b) else LEG_RGBA
        model.geom_rgba[g][3] = alpha
    return model


def set_pose(model, data, pose: dict, z: float) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [0.0, 0.0, z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, q in pose.items():
        data.qpos[model.joint(name).qposadr[0]] = q
    for k in range(model.nu):
        data.ctrl[k] = data.qpos[model.jnt_qposadr[model.actuator_trnid[k, 0]]]
    mujoco.mj_forward(model, data)


def camera(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def add_com_markers(scene, model, data) -> None:
    """红线：质心到地面的竖线；蓝点：两只脚 site 的中点（投影到地面）。"""
    trunk = model.body("trunk_base").id
    com = data.subtree_com[trunk].copy()
    feet = np.mean([data.site(n).xpos for n in ("left_foot", "right_foot")], axis=0)
    red = np.array([0.85, 0.15, 0.15, 1.0], dtype=np.float32)
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), red)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, com, np.array([com[0], com[1], 0.001]))
    scene.ngeom += 1
    for pos, rgba, size in ((com, [0.85, 0.15, 0.15, 1.0], 0.009),
                            (np.array([feet[0], feet[1], 0.004]), [0.18, 0.39, 0.58, 1.0], 0.009)):
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]), pos, np.eye(3).flatten(),
                            np.array(rgba, dtype=np.float32))
        scene.ngeom += 1


def plain(scene) -> None:
    """侧视图关掉阴影和反射：地面接近掠射角时阴影贴图会出噪点。"""
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False


def render(model, data, cam, width, height, markers=False) -> np.ndarray:
    with mujoco.Renderer(model, height, width) as r:
        r.update_scene(data, camera=cam)
        if markers:
            plain(r.scene)
            add_com_markers(r.scene, model, data)
        return r.render().copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sim_renders")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    W, H = 1600, 1200
    hd_model = build(hd.get_walk_spec(), W, H)
    hd_data = mujoco.MjData(hd_model)

    set_pose(hd_model, hd_data, hd.HOME_JOINT_POS, hd.STAND_Z)
    Image.fromarray(render(hd_model, hd_data, camera([0.0, 0.0, 0.15], 0.66, 135, -14), W, H)).save(out / "hangoutduck_home.png")

    set_pose(hd_model, hd_data, {n: 0.0 for n in hd.LEG_JOINTS}, hd.STAND_Z + 0.02)
    Image.fromarray(render(hd_model, hd_data, camera([0.0, 0.0, 0.16], 0.66, 135, -10), W, H)).save(out / "hangoutduck_zero.png")

    # 侧视对比（从机器人右侧看，x 朝前 = 画面向左）
    SW, SH = 900, 1100
    xg_model = build(hd.get_walk_spec(), SW, SH, alpha=0.35)
    xg_data = mujoco.MjData(xg_model)
    hd_side = build(hd.remove_head(xgo.get_walk_spec()), SW, SH, alpha=0.35)
    hd_side_data = mujoco.MjData(hd_side)
    set_pose(xg_model, xg_data, hd.HOME_JOINT_POS, hd.STAND_Z)
    set_pose(hd_side, hd_side_data, hd.HOME_JOINT_POS, hd.STAND_Z)
    side = camera([0.02, 0.0, 0.10], 0.48, -90, -8)
    left = render(xg_model, xg_data, side, SW, SH, markers=True)
    right = render(hd_side, hd_side_data, side, SW, SH, markers=True)
    gap = np.full((SH, 24, 3), 255, dtype=np.uint8)
    Image.fromarray(np.concatenate([left, gap, right], axis=1)).save(out / "compare_side.png")

    # 站立测试 GIF
    GW, GH = 480, 560
    xg_g = build(hd.get_walk_spec(), GW, GH, alpha=0.45)
    hd_g = build(hd.remove_head(xgo.get_walk_spec()), GW, GH, alpha=0.45)
    runs = []
    for model, pose in ((xg_g, hd.HOME_JOINT_POS), (hd_g, hd.HOME_JOINT_POS)):
        data = mujoco.MjData(model)
        set_pose(model, data, pose, hd.STAND_Z)
        runs.append((model, data))
    fps = 15
    steps_per_frame = int(round(1.0 / fps / xg_g.opt.timestep))
    cam = camera([0.02, 0.0, 0.09], 0.44, -90, -8)
    frames = []
    renderers = [mujoco.Renderer(m, GH, GW) for m, _ in runs]
    for _ in range(int(args.seconds * fps)):
        imgs = []
        for (model, data), r in zip(runs, renderers):
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            r.update_scene(data, camera=cam)
            plain(r.scene)
            add_com_markers(r.scene, model, data)
            imgs.append(r.render().copy())
        frames.append(Image.fromarray(np.concatenate([imgs[0], np.full((GH, 12, 3), 255, np.uint8), imgs[1]], axis=1)))
    for r in renderers:
        r.close()
    frames[0].save(out / "stand_test.gif", save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print("written:", *sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()

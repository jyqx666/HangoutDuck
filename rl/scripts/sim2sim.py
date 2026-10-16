"""sim2sim：真机控制程序原样跑在 MuJoCo 上（不需要显卡，比实时快约 20 倍）。

在 xgoduck_rl 的环境里运行：
    cd rl/third_party/xgoduck_rl
    uv run python ../../scripts/sim2sim.py walk                       # 手写步态，默认 20 步
    uv run python ../../scripts/sim2sim.py walk --param step_length_m=0.02 --param period_s=0.45
    uv run python ../../scripts/sim2sim.py stand --duration 5
    uv run python ../../scripts/sim2sim.py policy --onnx logs/.../xxx.onnx --vx 0.2   # 自己训练的策略
    uv run python ../../scripts/sim2sim.py xgoduck --vx 0.25         # 自检：xgoduck 预训练策略

模拟硬件误差（「实物」与 joints.yaml 不一致）：
    --zero-error left_knee=3     实物零位偏 3°        --flip right_hip_roll   实物转向接反
    --kp 5                       改写 run_pd 的 P       --delay-steps 8         指令延迟 8×5 ms
    --imu-tilt 0,5,0             IMU 装歪（度）         --mass-scale 0.8        整机轻 20%
录视频（无显示器：MUJOCO_GL=glfw xvfb-run -a -s "-screen 0 1280x720x24" 放在命令前面）：
    --video walk.mp4

结束时打印：是否摔倒、走了多远、多少步、最大倾角。摔倒或程序出错返回非 0。
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # 真机程序 hangoutduck 包

import mjlab  # noqa: E402,F401  先导入 mjlab：它会加载任务插件（含 hangoutduck_rl），避免循环导入
import numpy as np  # noqa: E402

from hangoutduck.robot import Robot  # noqa: E402
from hangoutduck.runtime import CONTROLLERS, SafetyLimits, StopSignal, run  # noqa: E402
from hangoutduck_rl.sim2sim import Hardware, Sim2Sim, SimLoop, default_config, joint_order_matches  # noqa: E402

XGODUCK_ONNX = REPO_ROOT / "rl" / "third_party" / "xgoduck_runtime_arduino" / "python" / "xgoduck_walk.onnx"


def _kv(items, cast=float):
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"expected key=value, got {item!r}")
        out[key.strip()] = cast(value)
    return out


def summarize(trace, report) -> dict:
    p0, p1 = trace[0].pos, trace[-1].pos
    heading = np.array([math.cos(trace[0].yaw), math.sin(trace[0].yaw)])
    d = (p1 - p0)[:2]
    touchdowns = 0
    for k in (0, 1):
        c = [r.contacts[k] for r in trace]
        touchdowns += sum(1 for a, b in zip(c, c[1:]) if b and not a)
    fell = report.fault is not None or min(r.pos[2] for r in trace) < 0.08 or max(r.tilt_deg for r in trace) > 45
    return {
        "fell": fell,
        "time_s": trace[-1].t,
        "forward_m": float(d @ heading),
        "sideways_m": float(heading[0] * d[1] - heading[1] * d[0]),
        "yaw_change_deg": math.degrees(math.remainder(trace[-1].yaw - trace[0].yaw, 2 * math.pi)),
        "touchdowns": touchdowns,
        "max_tilt_deg": max(r.tilt_deg for r in trace),
        "min_trunk_z_mm": 1000 * min(r.pos[2] for r in trace),
    }


def make_controller(args, cfg):
    if args.mode == "stand":
        return CONTROLLERS["stand"](cfg, duration_s=1.0)
    if args.mode == "walk":
        return CONTROLLERS["walk"](cfg, steps=args.steps, **_kv(args.param))
    if args.mode == "policy":
        from hangoutduck.runtime.policy import PolicyController

        return PolicyController(cfg, onnx_path=args.onnx, command=(args.vx, args.vy, args.wz))
    if args.mode == "xgoduck":
        from hangoutduck_rl.xgoduck_policy import XgoduckWalk

        onnx = args.onnx or str(XGODUCK_ONNX)
        if not Path(onnx).exists():
            raise SystemExit(f"{onnx} not found: git clone https://github.com/LuwuDynamics/xgoduck_runtime_arduino "
                             "rl/third_party/xgoduck_runtime_arduino, or pass --onnx")
        return XgoduckWalk(cfg, onnx, command=(args.vx, args.vy, args.wz))
    raise SystemExit(f"unknown mode {args.mode}")


def attach_video(sim: Sim2Sim, path: str, fps: int = 25, size=(960, 540)):
    import imageio_ffmpeg
    import mujoco

    model, data = sim.plant.model, sim.plant.data
    model.vis.global_.offwidth, model.vis.global_.offheight = size
    renderer = mujoco.Renderer(model, height=size[1], width=size[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = sim.plant.trunk
    cam.distance, cam.elevation, cam.azimuth = 0.7, -15.0, 135.0
    writer = imageio_ffmpeg.write_frames(path, size, fps=fps, codec="libx264", pix_fmt_out="yuv420p", quality=7,
                                         macro_block_size=8)
    writer.send(None)
    every = max(1, round(50 / fps))
    count = [0]

    def grab():
        count[0] += 1
        if count[0] % every == 0:
            renderer.update_scene(data, camera=cam)
            writer.send(np.ascontiguousarray(renderer.render()))

    sim.port.on_tick.append(grab)
    return writer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["stand", "walk", "policy", "xgoduck"])
    ap.add_argument("--duration", type=float, default=None, help="仿真时长（秒）；walk 默认走完为止")
    ap.add_argument("--steps", type=int, default=20, help="walk：走多少步")
    ap.add_argument("--param", action="append", help="walk：步态参数 key=value，可重复")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--vx", type=float, default=0.2)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--joints", default=None, help="joints.yaml 路径（默认 main 分支的 hardware/joints.yaml）")
    ap.add_argument("--no-imu", action="store_true", help="不接 IMU（walk 可以不用 IMU）")
    ap.add_argument("--kp", type=int, default=None, help="覆盖 joints.yaml 的 run_pd 里的 P")
    ap.add_argument("--zero-error", action="append", help="实物零位误差 joint=度，可重复")
    ap.add_argument("--flip", action="append", default=[], help="实物转向接反的关节，可重复")
    ap.add_argument("--delay-steps", type=int, default=4)
    ap.add_argument("--imu-tilt", default="0,0,0", help="IMU 装歪的 roll,pitch,yaw（度）")
    ap.add_argument("--imu-noise", type=float, default=0.0, help="陀螺噪声 rad/s（加速度计按 10 倍 m/s²）")
    ap.add_argument("--mass-scale", type=float, default=1.0)
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--vin", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    cfg = default_config(Path(args.joints) if args.joints else None)
    for problem in joint_order_matches(cfg):
        print("ERROR:", problem)
        return 1
    if args.kp is not None:
        cfg = dataclasses.replace(cfg, run_pd=(args.kp, cfg.run_pd[1] if cfg.run_pd else 20))
    hw = Hardware(
        zero_error_deg=_kv(args.zero_error),
        flipped=tuple(args.flip),
        delay_steps=args.delay_steps,
        vin=args.vin,
        foot_friction=args.friction,
        mass_scale=args.mass_scale,
        imu_tilt_deg=tuple(float(x) for x in args.imu_tilt.split(",")),
        imu_gyro_noise=args.imu_noise,
        imu_accel_noise=10 * args.imu_noise,
        seed=args.seed,
    )
    sim = Sim2Sim.create(cfg, hw)
    controller = make_controller(args, cfg)
    writer = attach_video(sim, args.video) if args.video else None
    duration = args.duration
    if duration is None:
        duration = 5.0 if args.mode == "stand" else (10.0 if args.mode != "walk" else None)

    print(f"sim2sim: {controller.name}, run_pd={cfg.run_pd}, delay={hw.delay_steps * 5} ms, "
          f"vin={sim.plant.bam.actuator.vin:.2f} V")
    wall = time.perf_counter()
    report = run(Robot(sim.bus, cfg), controller, hz=50.0, duration_s=duration,
                 imu=None if args.no_imu else sim.imu, limits=SafetyLimits(), stop=StopSignal(listen_stdin=False),
                 loop=SimLoop(50.0), clock=lambda: sim.plant.time)
    if writer is not None:
        writer.close()
    s = summarize(sim.plant.trace, report)
    print(f"stopped: {report.reason}" + (f" ({report.fault})" if report.fault else "")
          + f"; simulated {s['time_s']:.1f} s in {time.perf_counter() - wall:.1f} s")
    print(f"{'FELL' if s['fell'] else 'ok'}: forward {s['forward_m']:+.3f} m, sideways {s['sideways_m']:+.3f} m, "
          f"yaw {s['yaw_change_deg']:+.1f} deg, {s['touchdowns']} touchdowns, max tilt {s['max_tilt_deg']:.1f} deg, "
          f"min trunk z {s['min_trunk_z_mm']:.0f} mm")
    if getattr(controller, "max_ik_error_m", 0) > 1e-4:
        print(f"warning: gait asked for unreachable foot positions (IK error {controller.max_ik_error_m * 1000:.1f} mm)")
    if args.video:
        print(f"video: {args.video}")
    return 1 if s["fell"] else 0


if __name__ == "__main__":
    sys.exit(main())

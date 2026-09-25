"""hduck —— HangoutDuck 真机调试命令行。

所有命令都支持 --fake：用内存里的假舵机/假 IMU 代替硬件，没有驱动板时也能把流程走一遍。
典型顺序见 docs/bringup.md。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import (
    DEFAULT_CALIBRATION,
    DEFAULT_JOINTS,
    RobotConfig,
    load_calibration,
    load_config,
    save_calibration,
)
from .feetech import registers as reg
from .feetech.bus import BusError, FeetechBus
from .feetech.fake import FakeSerial, FakeServo
from .feetech.protocol import decode_sign_magnitude, describe_error, parse_u16_le
from .robot import Robot

DEFAULT_PORT = "/dev/ttyUSB0"


# ---- 公共 -----------------------------------------------------------------


def _config(args) -> RobotConfig:
    return load_config(Path(args.joints), Path(args.calibration))


def _bus(args, cfg: Optional[RobotConfig] = None, fake_ids: Optional[List[int]] = None) -> FeetechBus:
    baud = args.baud or (cfg.baudrate if cfg else 1_000_000)
    if args.fake:
        if fake_ids is None:
            fake_ids = cfg.ids if cfg else [1]
        zeros = {j.id: j.zero for j in cfg.joints} if cfg else {}
        port = FakeSerial([FakeServo(i, zeros.get(i, 2048)) for i in fake_ids])
        return FeetechBus(port, reply_timeout_s=0.002)
    return FeetechBus.open(args.port, baudrate=baud, echo=args.echo)


def _confirm(args, message: str) -> bool:
    if args.yes:
        return True
    reply = input(f"{message} [y/N] ").strip().lower()
    return reply in ("y", "yes")


def _parse_ids(text: str) -> List[int]:
    ids: List[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    return ids


def _print_table(rows: List[List[str]], header: List[str]) -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(r, widths)))


# ---- 单舵机工具 ------------------------------------------------------------


def cmd_scan(args) -> int:
    ids = _parse_ids(args.ids)
    bauds = list(reg.BAUD_RATES.values()) if args.all_bauds else [args.baud or 1_000_000]
    found_any = False
    for baud in bauds:
        args.baud = baud
        with _bus(args, fake_ids=_parse_ids(args.fake_ids)) as bus:
            found = bus.scan(ids)
        if found:
            found_any = True
            print(f"{baud} bps: {found}")
        elif not args.all_bauds:
            print(f"{baud} bps: no servo answered")
    return 0 if found_any else 1


def cmd_dump(args) -> int:
    with _bus(args, fake_ids=[args.id]) as bus:
        mem = bus.read(args.id, 0, reg.TABLE_END)
    rows = []
    for r in reg.REGISTERS.values():
        value = mem[r.address] if r.size == 1 else parse_u16_le(mem, r.address)
        rows.append([str(r.address), r.name, str(value), "EEPROM" if r.eeprom else "RAM", r.note])
    _print_table(rows, ["addr", "name", "value", "area", "note"])
    print("\nraw:", " ".join(f"{b:02x}" for b in mem))
    return 0


def cmd_set_id(args) -> int:
    with _bus(args, fake_ids=[args.old if args.old is not None else 1]) as bus:
        found = bus.scan(timeout_s=0.003) if args.old is None else ([args.old] if bus.ping(args.old) else [])
        if len(found) != 1:
            print(f"need exactly one servo on the bus, found {found}. Connect one servo at a time.")
            return 1
        old = found[0]
        if old == args.new:
            print(f"servo already has id {args.new}")
            return 0
        if not _confirm(args, f"change servo id {old} -> {args.new}?"):
            return 1
        bus.change_id(old, args.new)
    print(f"ok: servo {old} is now id {args.new}. Label the servo with its new id.")
    return 0


def cmd_center(args) -> int:
    """上力并转到中位 2048，用于装舵盘（此时还没有零位标定）。"""
    cfg = _config(args)
    ids = cfg.ids if args.all else [args.id]
    if not _confirm(args, f"servos {ids} will power up and move to 2048. Horns must NOT be attached to the frame."):
        return 1
    with _bus(args, cfg) as bus:
        for sid in ids:
            bus.write(sid, reg.GOAL_POSITION, bytes([0x00, 0x08, 0, 0, 0, 0]))
            bus.write_u8(sid, reg.TORQUE_ENABLE, 1)
        time.sleep(0.5)
        for sid in ids:
            print(f"id {sid}: position {bus.read_u16(sid, reg.PRESENT_POSITION)}")
        if not args.hold:
            for sid in ids:
                bus.write_u8(sid, reg.TORQUE_ENABLE, 0)
    return 0


def cmd_torque(args) -> int:
    cfg = _config(args)
    with _bus(args, cfg) as bus:
        robot = Robot(bus, cfg)
        if args.state == "off":
            robot.torque_off()
        else:
            robot.set_torque(True)
    print(f"torque {args.state}")
    return 0


# ---- 整机 -----------------------------------------------------------------


def cmd_state(args) -> int:
    cfg = _config(args)
    with _bus(args, cfg) as bus:
        robot = Robot(bus, cfg)
        st = robot.read_state()
        ticks = robot.rad_to_ticks(st.q)
    rows = []
    for k, j in enumerate(cfg.joints):
        if not st.valid[k]:
            rows.append([j.name, str(j.id), "-", "-", "-", "-", "NO REPLY"])
            continue
        err = describe_error(bus.last_errors.get(j.id, 0))
        rows.append([
            j.name, str(j.id), str(int(ticks[k])),
            f"{math.degrees(st.q[k]) + 0.0:7.1f}", f"{j.home_deg:6.1f}",
            f"{st.temperature[k]:.0f}C {st.voltage[k]:.1f}V", ",".join(err) or "ok",
        ])
    _print_table(rows, ["joint", "id", "ticks", "deg", "home", "temp/volt", "status"])
    if not cfg.calibrated:
        print("\nnote: hardware/calibration.yaml missing or incomplete, angles use the placeholder zeros")
    return 0 if st.all_valid else 1


def cmd_calibrate(args) -> int:
    """卸力，人把关节摆到参考姿态，记录零位刻度。可以一次标全部，也可以用 --joint 逐个标。"""
    cfg = _config(args)
    joints = [cfg.joint(args.joint)] if args.joint else cfg.joints
    pose = {j.name: (0.0 if args.pose == "zero" else j.home_deg) for j in joints}
    out = Path(args.calibration)
    zeros = load_calibration(out) if out.exists() else {}
    with _bus(args, cfg) as bus:
        Robot(bus, cfg).torque_off()
        print("Torque is off. Put the joints in the reference pose:")
        print("  zero: link at its mechanical (CAD) zero, as in the xgoduck assembly guide")
        print("  home: standing pose (hip_pitch/ankle +-24 deg, hip_roll +-5 deg)")
        if not args.yes:
            input(f"pose = {args.pose}, joints = {[j.name for j in joints]}. Hold it, then press Enter...")
        ticks = {j.id: bus.read_u16(j.id, reg.PRESENT_POSITION) for j in joints}
    deg_per_tick = 360.0 / cfg.ticks_per_rev
    rows = []
    for j in joints:
        zero = int(round(ticks[j.id] - j.sign * pose[j.name] / deg_per_tick))
        zeros[j.name] = zero
        rows.append([j.name, str(j.id), str(ticks[j.id]), f"{pose[j.name]:.1f}", str(zero)])
    _print_table(rows, ["joint", "id", "ticks", "pose_deg", "zero"])
    if args.dry_run:
        print("\ndry run, nothing written")
        return 0
    save_calibration(out, zeros, note=f"last update: pose={args.pose}, {time.strftime('%Y-%m-%d %H:%M')}")
    todo = [j.name for j in cfg.joints if j.name not in zeros]
    print(f"\nwritten {out}" + (f"; still uncalibrated: {todo}" if todo else "; all joints calibrated"))
    return 0


def _run_controller(args, controller_name: str, **kwargs) -> int:
    from .imu import FakeImu, WitImu
    from .runtime import CONTROLLERS, SafetyLimits, StopSignal, run

    cfg = _config(args)
    if not cfg.calibrated and not args.fake and not args.uncalibrated:
        print("hardware/calibration.yaml is missing. Run `hduck calibrate` first (or pass --uncalibrated).")
        return 1
    imu = None
    if args.fake:
        imu = FakeImu()
    elif args.imu_port:
        imu = WitImu(args.imu_port, args.imu_baud)
        imu.start()
    controller = CONTROLLERS[controller_name](cfg, **kwargs)
    limits = SafetyLimits(max_tilt_deg=None if args.no_tilt_check else 60.0)
    log_path = None
    if not args.no_log:
        log_path = Path(args.log_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}-{controller_name}.npz"
    print(f"running {controller_name} at {args.hz:g} Hz. Press Enter or Ctrl-C to stop (torque goes off).")
    try:
        with _bus(args, cfg) as bus:
            report = run(
                Robot(bus, cfg), controller, hz=args.hz, duration_s=args.duration, imu=imu,
                limits=limits, log_path=log_path, stop=StopSignal(),
            )
    finally:
        if imu is not None:
            imu.close()
    print(f"stopped: {report.reason}" + (f" ({report.fault})" if report.fault else ""))
    print(f"loop: {report.loop}")
    print(f"bus: {report.bus}")
    if report.log_path:
        print(f"log: {report.log_path}")
    return 0 if report.fault is None else 2


def cmd_stand(args) -> int:
    return _run_controller(args, "stand", duration_s=args.ramp)


def cmd_sweep(args) -> int:
    return _run_controller(
        args, "sweep", joint=args.joint, amplitude_deg=args.amp, period_s=args.period, cycles=args.cycles,
    )


def cmd_run(args) -> int:
    return _run_controller(args, args.controller)


# ---- 测量 -----------------------------------------------------------------


def cmd_bench_bus(args) -> int:
    """验收 A2：以固定频率同步读全部舵机，统计丢包。"""
    from .runtime.loop import RateLoop

    cfg = _config(args)
    ids = cfg.ids
    with _bus(args, cfg) as bus:
        loop = RateLoop(args.hz)
        lost = 0
        cycles = 0
        read_ms = float("nan")
        t_end = time.perf_counter() + args.seconds
        loop.wait()
        while time.perf_counter() < t_end:
            t = time.perf_counter()
            got = bus.sync_read(reg.PRESENT_POSITION, reg.STATE_BLOCK_LEN, ids)
            cycles += 1
            lost += len(ids) - len(got)
            if cycles == 1:
                read_ms = (time.perf_counter() - t) * 1e3
            loop.wait()
    total = cycles * len(ids)
    loss = lost / total if total else 1.0
    print(f"{cycles} cycles x {len(ids)} servos at {args.hz:g} Hz, first sync read {read_ms:.2f} ms")
    print(f"lost replies: {lost}/{total} = {loss:.2%}  ({'PASS' if loss < 0.01 else 'FAIL'}: A2 needs < 1%)")
    print(f"loop: {loop.stats()}")
    return 0 if loss < 0.01 else 1


def cmd_check_speed(args) -> int:
    """卸力后用手来回转动舵机，对比位置差分和速度寄存器，估计速度单位。"""
    with _bus(args, fake_ids=[args.id]) as bus:
        bus.write_u8(args.id, reg.TORQUE_ENABLE, 0)
        print(f"rotate servo {args.id} back and forth by hand for {args.seconds:g} s ...")
        ts, pos, spd = [], [], []
        t_end = time.perf_counter() + args.seconds
        while time.perf_counter() < t_end:
            data = bus.read(args.id, reg.PRESENT_POSITION, 4)
            ts.append(time.perf_counter())
            pos.append(parse_u16_le(data, 0))
            spd.append(decode_sign_magnitude(parse_u16_le(data, 2), 15))
            time.sleep(0.005)
    ts, pos, spd = map(np.asarray, (ts, pos, spd))
    if len(ts) < 10:
        print("not enough samples")
        return 1
    dpos = np.gradient(pos.astype(float), ts)
    mask = np.abs(spd) > 2
    if mask.sum() < 10:
        print("servo barely moved; rotate it faster")
        return 1
    ratio = float(np.dot(dpos[mask], spd[mask]) / np.dot(spd[mask], spd[mask]))
    print(f"estimated speed unit: {ratio:.1f} ticks/s per register unit "
          f"(code assumes {reg.SPEED_UNIT_TICKS_PER_S:g}; update registers.SPEED_UNIT_TICKS_PER_S if different)")
    return 0


def cmd_bench_loop(args) -> int:
    """只测控制循环本身的定时精度（WP2：Nano 环境验收）。"""
    from .runtime.loop import RateLoop

    loop = RateLoop(args.hz)
    t_end = time.perf_counter() + args.seconds
    loop.wait()
    while time.perf_counter() < t_end:
        loop.wait()
    s = loop.stats()
    ok = s.get("jitter_p99_ms", 1e9) < 5.0
    print(f"{s}  ({'PASS' if ok else 'FAIL'}: A5 needs p99 jitter < 5 ms)")
    return 0 if ok else 1


def cmd_imu(args) -> int:
    from .imu import FakeImu, GravityEstimator, WitImu

    cfg = _config(args)
    imu = FakeImu(noise=0.01) if args.fake else WitImu(args.imu_port, args.imu_baud)
    imu.start()
    est = GravityEstimator(cfg.imu_rotation())
    print("body frame: x forward, y left, z up. Level -> gravity ~ [0, 0, -1]; nose down -> gx > 0; right side down -> gy < 0")
    try:
        t_end = time.perf_counter() + args.seconds
        last_print = 0.0
        while time.perf_counter() < t_end:
            s = imu.latest()
            if s is not None:
                g = est.update(s)
                now = time.perf_counter()
                if now - last_print > 0.1:
                    last_print = now
                    w = np.degrees(est.gyro)
                    print(f"gravity [{g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}]  tilt {est.tilt_deg():5.1f} deg  "
                          f"gyro [{w[0]:+7.1f} {w[1]:+7.1f} {w[2]:+7.1f}] deg/s")
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        imu.close()
    if not args.fake and getattr(imu, "samples", 0) == 0:
        print("no IMU data received: check port, baud rate and the IMU output settings")
        return 1
    return 0


# ---- 参数 -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hduck", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=DEFAULT_PORT, help=f"舵机驱动板串口（默认 {DEFAULT_PORT}）")
    ap.add_argument("--baud", type=int, default=None, help="波特率，默认取 joints.yaml（1000000）")
    ap.add_argument("--echo", action="store_true", help="驱动板会回显发出的字节时打开")
    ap.add_argument("--joints", default=str(DEFAULT_JOINTS), help="关节表路径")
    ap.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), help="零位标定文件路径")
    ap.add_argument("--fake", action="store_true", help="不接硬件，用假舵机/假 IMU")
    ap.add_argument("--fake-ids", default="1", help="--fake 下 scan 用的假舵机 ID（默认 1）")
    ap.add_argument("-y", "--yes", action="store_true", help="跳过确认提示")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="扫描总线上的舵机 ID")
    p.add_argument("--ids", default="0-253")
    p.add_argument("--all-bauds", action="store_true", help="依次尝试所有波特率")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("dump", help="读出一个舵机的整张寄存器表")
    p.add_argument("--id", type=int, required=True)
    p.set_defaults(func=cmd_dump)

    p = sub.add_parser("set-id", help="修改舵机 ID（总线上只接这一个舵机）")
    p.add_argument("new", type=int)
    p.add_argument("--old", type=int, default=None, help="原 ID；不填则自动扫描")
    p.set_defaults(func=cmd_set_id)

    p = sub.add_parser("center", help="上力转到中位 2048，装舵盘用")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=int)
    g.add_argument("--all", action="store_true", help="关节表里全部舵机")
    p.add_argument("--hold", action="store_true", help="转到位后保持上力")
    p.set_defaults(func=cmd_center)

    p = sub.add_parser("torque", help="全部上力/卸力")
    p.add_argument("state", choices=["on", "off"])
    p.set_defaults(func=cmd_torque)

    sub.add_parser("state", help="打印全部关节的当前角度、温度、电压").set_defaults(func=cmd_state)

    p = sub.add_parser("calibrate", help="记录零位，写入 hardware/calibration.yaml")
    p.add_argument("--pose", choices=["zero", "home"], default="zero")
    p.add_argument("--joint", default=None, help="只标这一个关节，其余保留原值")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_calibrate)

    def add_run_args(p):
        p.add_argument("--hz", type=float, default=50.0)
        p.add_argument("--duration", type=float, default=None, help="秒；不填则一直运行到按回车")
        p.add_argument("--imu-port", default=None, help="IMU 串口，例如 /dev/ttyUSB1")
        p.add_argument("--imu-baud", type=int, default=115200)
        p.add_argument("--no-tilt-check", action="store_true", help="关闭摔倒检测（吊架上调试时）")
        p.add_argument("--log-dir", default="logs")
        p.add_argument("--no-log", action="store_true")
        p.add_argument("--uncalibrated", action="store_true", help="允许在没有标定文件时运行（危险）")

    p = sub.add_parser("stand", help="平滑站到 home 姿态并保持")
    p.add_argument("--ramp", type=float, default=2.0, help="插值时长（秒）")
    add_run_args(p)
    p.set_defaults(func=cmd_stand)

    p = sub.add_parser("sweep", help="单关节正弦摆动，核对方向和零位")
    p.add_argument("--joint", required=True)
    p.add_argument("--amp", type=float, default=10.0, help="幅度（度）")
    p.add_argument("--period", type=float, default=2.0)
    p.add_argument("--cycles", type=int, default=3)
    add_run_args(p)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("run", help="运行已注册的控制器")
    p.add_argument("--controller", default="stand")
    add_run_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("bench-bus", help="验收 A2：同步读丢包率")
    p.add_argument("--hz", type=float, default=100.0)
    p.add_argument("--seconds", type=float, default=60.0)
    p.set_defaults(func=cmd_bench_bus)

    p = sub.add_parser("check-speed", help="用手转舵机，核实速度寄存器单位")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--seconds", type=float, default=5.0)
    p.set_defaults(func=cmd_check_speed)

    p = sub.add_parser("bench-loop", help="只测控制循环定时抖动")
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.set_defaults(func=cmd_bench_loop)

    p = sub.add_parser("imu", help="实时显示 IMU 在机体系下的重力方向和角速度")
    p.add_argument("--imu-port", default="/dev/ttyUSB1")
    p.add_argument("--imu-baud", type=int, default=115200)
    p.add_argument("--seconds", type=float, default=30.0)
    p.set_defaults(func=cmd_imu)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BusError as exc:
        print(f"bus error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())

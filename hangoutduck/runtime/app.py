"""运行时主循环：读状态 → 安全检查 → 控制器 → 下发目标 → 记录，固定 50 Hz。

任何异常、安全故障、Ctrl-C、或在终端按回车，都会立即卸力。
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..imu.base import GravityEstimator, Imu
from ..robot import Robot
from .controllers import Controller, ImuObs
from .logger import RunLogger
from .loop import RateLoop
from .safety import SafetyLimits, SafetyMonitor


@dataclass
class RunReport:
    reason: str
    duration_s: float
    loop: Dict[str, float]
    bus: Dict[str, int]
    log_path: Optional[Path] = None
    fault: Optional[str] = None
    extra: Dict[str, float] = field(default_factory=dict)


class StopSignal:
    """在终端按回车即可让循环正常退出（非交互环境下不启用）。"""

    def __init__(self, listen_stdin: bool = True) -> None:
        self.event = threading.Event()
        if listen_stdin and sys.stdin is not None and sys.stdin.isatty():
            threading.Thread(target=self._wait_enter, daemon=True).start()

    def _wait_enter(self) -> None:
        try:
            sys.stdin.readline()
        except Exception:
            pass
        self.event.set()

    def is_set(self) -> bool:
        return self.event.is_set()


def run(
    robot: Robot,
    controller: Controller,
    hz: float = 50.0,
    duration_s: Optional[float] = None,
    imu: Optional[Imu] = None,
    limits: Optional[SafetyLimits] = None,
    log_path: Optional[Path] = None,
    stop: Optional[StopSignal] = None,
    torque_off_at_end: bool = True,
) -> RunReport:
    stop = stop or StopSignal(listen_stdin=False)
    loop = RateLoop(hz)
    safety = SafetyMonitor(limits or SafetyLimits(), robot.cfg.names)
    logger = RunLogger(robot.cfg.names)
    estimator = GravityEstimator(robot.cfg.imu_rotation()) if imu is not None else None
    fake_port = getattr(robot.bus.port, "step", None)  # FakeSerial 需要手动推进时间

    missing = robot.ping_all()
    if missing:
        raise RuntimeError(f"servos not answering: {missing}; check wiring and ids")

    reason, fault = "finished", None
    t0 = time.perf_counter()
    t = 0.0
    try:
        robot.apply_run_pd()
        state = robot.read_state()
        controller.reset(state)
        robot.set_torque(True)
        loop.wait()
        t0 = time.perf_counter()
        period = 0.0
        while True:
            t = time.perf_counter() - t0
            state = robot.read_state()
            imu_obs = None
            if imu is not None and estimator is not None:
                sample = imu.latest()
                if sample is not None:
                    gravity = estimator.update(sample)
                    imu_obs = ImuObs(gyro=estimator.gyro.copy(), gravity=gravity.copy())
            fault = safety.check(state, estimator.tilt_deg() if imu_obs is not None else None)
            if fault:
                reason = "fault"
                break
            target = controller.step(t, state, imu_obs)
            sent = robot.command(target)
            logger.log(
                t, state, sent, period,
                gyro=imu_obs.gyro if imu_obs else None,
                gravity=imu_obs.gravity if imu_obs else None,
            )
            if stop.is_set():
                reason = "stopped by user"
                break
            if controller.done(t) or (duration_s is not None and t >= duration_s):
                break
            if fake_port is not None:
                fake_port(loop.period)
            period = loop.wait()
    except KeyboardInterrupt:
        reason = "interrupted (Ctrl-C)"
    except Exception as exc:  # 出任何错都先卸力（见 finally）再抛
        reason, fault = "error", repr(exc)
        raise
    finally:
        if torque_off_at_end or fault:
            robot.torque_off()

    saved = None
    if log_path is not None and len(logger):
        saved = logger.save(log_path, meta={"controller": controller.name, "hz": hz, "reason": reason, "fault": fault})
    return RunReport(
        reason=reason,
        duration_s=t,
        loop=loop.stats(),
        bus=robot.bus.stats.as_dict(),
        log_path=saved,
        fault=fault,
    )

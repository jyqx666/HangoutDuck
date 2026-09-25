import math

import numpy as np
import pytest

from hangoutduck.cli import main
from hangoutduck.config import DEFAULT_JOINTS, load_config
from hangoutduck.feetech import registers as reg
from hangoutduck.feetech.fake import fake_bus_with
from hangoutduck.imu import FakeImu
from hangoutduck.robot import Robot
from hangoutduck.runtime import RateLoop, SafetyLimits, Stand, Sweep, run


@pytest.fixture
def cfg():
    return load_config(DEFAULT_JOINTS, calibration_path=None)


def make_robot(cfg):
    bus, port = fake_bus_with(cfg.ids, positions={j.id: j.zero for j in cfg.joints})
    return Robot(bus, cfg), port


def test_stand_reaches_home_and_turns_torque_off(cfg, tmp_path):
    robot, port = make_robot(cfg)
    report = run(robot, Stand(cfg, duration_s=0.2), hz=200, duration_s=0.5, imu=FakeImu(),
                 log_path=tmp_path / "run.npz")
    assert report.fault is None, report.fault
    assert report.reason == "finished"
    st = robot.read_state()
    assert np.allclose(st.q, robot.clamp(cfg.home_rad), atol=3 * robot._rpt)
    assert all(s.mem[reg.TORQUE_ENABLE] == 0 for s in port.servos.values())
    log = np.load(report.log_path)
    assert log["q"].shape[1] == 10
    assert np.allclose(log["gravity"][-1], [0, 0, -1])


def test_overheat_stops_run(cfg):
    robot, port = make_robot(cfg)
    port.servos[13].temperature = 80
    report = run(robot, Stand(cfg), hz=200, duration_s=1.0)
    assert report.reason == "fault"
    assert "left_knee" in report.fault
    assert all(s.mem[reg.TORQUE_ENABLE] == 0 for s in port.servos.values())


def test_lost_servo_stops_run(cfg):
    robot, port = make_robot(cfg)

    class DropAfter(Stand):
        def step(self, t, state, imu):
            if t > 0.05:
                port.servos.pop(22, None)
            return super().step(t, state, imu)

    report = run(robot, DropAfter(cfg), hz=200, duration_s=1.0, limits=SafetyLimits(max_missed_cycles=3))
    assert report.reason == "fault"
    assert "right_hip_pitch" in report.fault


def test_missing_servo_refuses_to_start(cfg):
    robot, port = make_robot(cfg)
    del port.servos[10]
    with pytest.raises(RuntimeError, match="10"):
        run(robot, Stand(cfg), hz=100, duration_s=0.1)


def test_sweep_moves_only_selected_joint(cfg):
    sweep = Sweep(cfg, "left_knee", amplitude_deg=10, period_s=1.0, cycles=1, settle_s=0.5)
    robot, _ = make_robot(cfg)
    sweep.reset(robot.read_state())
    q = sweep.step(0.5 + 0.25, robot.read_state(), None)  # 1/4 周期：正弦峰值
    diff = np.degrees(q - cfg.home_rad)
    k = cfg.index("left_knee")
    assert diff[k] == pytest.approx(10.0)
    assert np.allclose(np.delete(diff, k), 0.0)


def test_rate_loop_timing():
    loop = RateLoop(200)
    for _ in range(40):
        loop.wait()
    s = loop.stats()
    assert s["cycles"] == 39
    assert abs(s["mean_ms"] - 5.0) < 1.0


def test_cli_fake_end_to_end(capsys, tmp_path):
    assert main(["--fake", "scan", "--ids", "0-5"]) == 0
    assert "[1]" in capsys.readouterr().out
    assert main(["--fake", "-y", "set-id", "12"]) == 0
    assert main(["--fake", "state"]) == 0
    cal = tmp_path / "cal.yaml"
    assert main(["--fake", "-y", "--calibration", str(cal), "calibrate", "--pose", "zero"]) == 0
    assert cal.exists()
    assert main(["--fake", "-y", "--calibration", str(cal), "calibrate", "--joint", "left_knee", "--pose", "home"]) == 0
    assert main(["--fake", "--calibration", str(cal), "stand", "--duration", "0.3", "--no-log"]) == 0
    assert main(["--fake", "bench-bus", "--seconds", "0.3"]) == 0
    assert main(["--fake", "dump", "--id", "10"]) == 0

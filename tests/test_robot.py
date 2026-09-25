import math

import numpy as np
import pytest

from hangoutduck.config import ConfigError, DEFAULT_JOINTS, axis_map_matrix, load_config, save_calibration
from hangoutduck.feetech import registers as reg
from hangoutduck.feetech.fake import fake_bus_with
from hangoutduck.robot import Robot


@pytest.fixture
def cfg():
    return load_config(DEFAULT_JOINTS, calibration_path=None)


def make_robot(cfg, **kwargs):
    bus, port = fake_bus_with(cfg.ids, positions={j.id: j.zero for j in cfg.joints}, **kwargs)
    return Robot(bus, cfg), port


def test_joint_table_matches_xgoduck_layout(cfg):
    assert cfg.ids == [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    assert cfg.names[0] == "left_hip_yaw" and cfg.names[-1] == "right_ankle"
    assert np.allclose(np.degrees(cfg.home_rad)[[1, 2, 4]], [-5, -24, 24])


def test_conversion_roundtrip_and_sign(cfg):
    robot, _ = make_robot(cfg)
    q = np.radians(np.linspace(-20, 20, robot.n))
    ticks = robot.rad_to_ticks(q)
    assert np.allclose(robot.ticks_to_rad(ticks), q, atol=robot._rpt)
    # sign = -1：正角度对应刻度减小
    k = cfg.index("left_knee")
    assert robot.rad_to_ticks(np.eye(robot.n)[k] * math.radians(10))[k] < cfg.joints[k].zero


def test_command_clamps_to_soft_limits(cfg):
    robot, port = make_robot(cfg)
    k = cfg.index("left_hip_roll")
    target = cfg.home_rad.copy()
    target[k] = math.radians(80)  # 超过 ±22° 的机械限位
    sent = robot.command(target)
    assert math.degrees(sent[k]) == pytest.approx(22.0 * 0.9)
    goal = port.servos[11].u16("goal_position")
    assert robot.ticks_to_rad(np.full(robot.n, goal))[k] == pytest.approx(sent[k], abs=robot._rpt)


def test_read_state_after_motion(cfg):
    robot, port = make_robot(cfg)
    robot.set_torque(True)
    robot.command(cfg.home_rad)
    for _ in range(50):
        port.step(0.02)
    st = robot.read_state()
    assert st.all_valid
    assert np.allclose(st.q, robot.clamp(cfg.home_rad), atol=2 * robot._rpt)
    assert np.all(st.temperature == 30)


def test_set_torque_holds_current_position(cfg):
    robot, port = make_robot(cfg)
    port.servos[10].position = 2500
    robot.set_torque(True)
    assert port.servos[10].u16("goal_position") == 2500
    assert all(s.mem[reg.TORQUE_ENABLE] == 1 for s in port.servos.values())
    robot.torque_off()
    assert all(s.mem[reg.TORQUE_ENABLE] == 0 for s in port.servos.values())


def test_missing_servo_marks_invalid(cfg):
    robot, port = make_robot(cfg)
    del port.servos[23]
    st = robot.read_state()
    assert not st.valid[cfg.index("right_knee")]
    assert st.valid.sum() == robot.n - 1
    with pytest.raises(RuntimeError):
        robot.set_torque(True)


def test_run_pd_written_to_hls_registers(cfg):
    robot, port = make_robot(cfg)
    robot.apply_run_pd()
    assert port.servos[10].mem[reg.TEMP_P] == 6
    assert port.servos[10].mem[reg.TEMP_D] == 20


def test_calibration_overrides_zero(tmp_path, cfg):
    cal = tmp_path / "calibration.yaml"
    save_calibration(cal, {j.name: 2000 + k for k, j in enumerate(cfg.joints)})
    cfg2 = load_config(DEFAULT_JOINTS, cal)
    assert cfg2.calibrated
    assert [j.zero for j in cfg2.joints] == [2000 + k for k in range(10)]


def test_axis_map():
    assert np.allclose(axis_map_matrix(["+z", "-x", "-y"]) @ [1, 2, 3], [3, -1, -2])
    with pytest.raises(ConfigError):
        axis_map_matrix(["+x", "+x", "+z"])

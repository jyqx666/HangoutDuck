"""腿部运动学和保底步态（不需要 MuJoCo；FK 与 MuJoCo 的对照在 rl 分支 scripts/check_gait_kinematics.py）。"""

import math

import numpy as np
import pytest

from hangoutduck.cli import main
from hangoutduck.config import DEFAULT_JOINTS, load_config
from hangoutduck.feetech.fake import fake_bus_with
from hangoutduck.imu import FakeImu
from hangoutduck.kinematics import SIDES, Leg
from hangoutduck.robot import Robot
from hangoutduck.runtime import CONTROLLERS, GaitParams, Walk, run


@pytest.fixture
def cfg():
    return load_config(DEFAULT_JOINTS, calibration_path=None)


def leg_angles(cfg, side, q):
    return np.array([q[cfg.index(f"{side}_{j}")] for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")])


def test_home_feet_are_mirror_images_below_the_hips(cfg):
    left = Leg("left").fk(leg_angles(cfg, "left", cfg.home_rad))
    right = Leg("right").fk(leg_angles(cfg, "right", cfg.home_rad))
    # 与 MuJoCo 里 left_foot / right_foot site 在 home 姿态的位置一致（mm）
    assert np.allclose(left * 1000, [16.44, 22.24, -137.39], atol=0.05)
    assert np.allclose(right * 1000, [16.44, -22.24, -137.39], atol=0.05)


@pytest.mark.parametrize("side", SIDES)
def test_ik_inverts_fk(side):
    leg = Leg(side)
    rng = np.random.default_rng(1)
    for _ in range(30):
        q = rng.uniform(-0.4, 0.4, 5)
        q[0] = rng.uniform(-0.2, 0.2)  # 髋 yaw 由调用方给定
        foot = leg.fk(q)
        sol = leg.ik(foot, leg.foot_pitch(q), hip_yaw=q[0], q_init=q + rng.normal(0, 0.05, 5))
        assert np.linalg.norm(leg.fk(sol) - foot) < 1e-6
        assert abs(leg.foot_pitch(sol) - leg.foot_pitch(q)) < 1e-9


def test_walk_starts_and_ends_at_home(cfg):
    walk = Walk(cfg, steps=4)
    walk.reset(type("S", (), {"q": cfg.home_rad.copy()})())
    p = walk.p
    t_walk = p.settle_s + 0.01
    assert np.allclose(walk.step(t_walk, None, None), cfg.home_rad, atol=1e-3)
    end = p.settle_s + (walk.walk_cycles + 2 * p.ramp_cycles) * p.period_s
    assert np.allclose(walk.step(end + 0.1, None, None), cfg.home_rad, atol=1e-9)
    assert not walk.done(end) and walk.done(end + 2 * p.period_s)
    assert walk.max_ik_error_m < 1e-6


def test_walk_foot_offsets_are_continuous_and_alternate(cfg):
    walk = Walk(cfg, steps=None, step_length_m=0.02, lift_m=0.015)
    phases = np.linspace(0.0, 1.0, 2001)
    off = {s: np.array([walk.foot_offsets(ph % 1.0, 1.0)[s] for ph in phases]) for s in SIDES}
    for s in SIDES:
        assert np.max(np.abs(np.diff(off[s], axis=0))) < 2e-4  # 无跳变
        assert np.isclose(off[s][:, 0].max() - off[s][:, 0].min(), 0.02, atol=1e-4)
    # 两只脚不会同时抬起
    assert not np.any((off["left"][:, 2] > walk.p.crouch_m + 1e-4) & (off["right"][:, 2] > walk.p.crouch_m + 1e-4))
    # 左脚在前半周期抬起，右脚在后半周期
    assert off["left"][500, 2] > off["right"][500, 2]
    assert off["right"][1500, 2] > off["left"][1500, 2]


def test_unknown_gait_parameter_is_rejected(cfg):
    with pytest.raises(ValueError, match="unknown gait parameters"):
        Walk(cfg, stride=0.1)
    assert GaitParams.from_dict({"lift_m": 0.02}).lift_m == 0.02


def test_walk_runs_on_fake_servos(cfg):
    assert CONTROLLERS["walk"] is Walk
    bus, port = fake_bus_with(cfg.ids, positions={j.id: j.zero for j in cfg.joints})
    walk = Walk(cfg, steps=2, settle_s=0.2, ramp_cycles=0.5, period_s=0.4)
    report = run(Robot(bus, cfg), walk, hz=200, imu=FakeImu())
    assert report.fault is None and report.reason == "finished"


def test_hduck_walk_cli_with_fake_servos(capsys):
    code = main(["--fake", "walk", "--steps", "2", "--no-log", "--hz", "200",
                 "--param", "settle_s=0.2", "--param", "ramp_cycles=0.5"])
    assert code == 0
    assert "stopped: finished" in capsys.readouterr().out
    assert main(["--fake", "walk", "--param", "nope"]) == 1


def test_imu_feedback_pushes_against_the_tilt(cfg):
    walk = Walk(cfg, steps=None, imu_roll_gain=1.0, imu_pitch_gain=1.0, step_roll_gain=0.0, step_pitch_gain=0.0,
                settle_s=0.0)
    walk.reset(type("S", (), {"q": cfg.home_rad.copy()})())
    from hangoutduck.runtime import ImuObs

    roll = math.radians(5)  # 机身左侧抬高（绕 +x 转 5°）
    tilted = ImuObs(gyro=np.zeros(3), gravity=np.array([0.0, -math.sin(roll), -math.cos(roll)]))
    level = ImuObs(gyro=np.zeros(3), gravity=np.array([0.0, 0.0, -1.0]))
    q_tilt = walk.step(1.0, None, tilted)
    q_level = walk.step(1.0, None, level)
    for side in SIDES:
        k = cfg.index(f"{side}_hip_roll")
        assert q_tilt[k] - q_level[k] == pytest.approx(roll, abs=1e-6)


def test_foot_placement_feedback_steps_toward_the_fall(cfg):
    walk = Walk(cfg, steps=None, step_pitch_gain=0.05, step_roll_gain=0.05)
    level = walk.foot_offsets(0.25, 1.0)  # 左脚摆动到一半
    leaning = walk.foot_offsets(0.25, 1.0, pitch=0.1, roll=0.1)  # 前倾、左侧抬高（往右倒）
    d = leaning["left"] - level["left"]
    assert d[0] > 0  # 往前倒 → 摆动脚往前迈
    assert d[1] < 0  # 往右倒 → 摆动脚往右迈
    # 修正量在摆动开始时为 0，轨迹不跳变
    start = walk.foot_offsets(walk.p.double_support / 4, 1.0, pitch=0.1, roll=0.1)["left"]
    assert np.allclose(start[:2], walk.foot_offsets(walk.p.double_support / 4, 1.0)["left"][:2], atol=1e-9)

import numpy as np
import pytest

from hangoutduck.config import DEFAULT_JOINTS, load_config
from hangoutduck.feetech.fake import fake_bus_with
from hangoutduck.imu import FakeImu
from hangoutduck.robot import Robot
from hangoutduck.runtime import ImuObs, run
from hangoutduck.runtime.policy import ACTION_DIM, OBS_DIM, PolicyController

# 与 rl 分支导出的 ONNX 元数据格式相同（逗号分隔）
SIM_META = {
    "joint_names": "left_hip_yaw,left_hip_roll,left_hip_pitch,left_knee,left_ankle,"
    "right_hip_yaw,right_hip_roll,right_hip_pitch,right_knee,right_ankle",
    "default_joint_pos": "0.000,-0.087,-0.419,-0.005,0.419,0.000,0.087,0.419,0.005,-0.419",
    "observation_names": "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command",
}


@pytest.fixture
def cfg():
    return load_config(DEFAULT_JOINTS, calibration_path=None)


def test_obs_layout_and_action_mapping(cfg):
    seen = []

    def infer(obs):
        seen.append(obs.copy())
        return np.full(ACTION_DIM, 0.1)

    ctl = PolicyController(cfg, infer=infer, meta=SIM_META, command=(0.2, 0.0, -0.5))
    bus, _ = fake_bus_with(cfg.ids, positions={j.id: j.zero for j in cfg.joints})
    state = Robot(bus, cfg).read_state()
    imu = ImuObs(gyro=np.array([0.1, 0.2, 0.3]), gravity=np.array([0.0, 0.0, -1.0]))
    target = ctl.step(0.0, state, imu)
    ctl.step(0.02, state, imu)

    obs0, obs1 = seen
    assert obs0.shape == (OBS_DIM,)
    assert np.allclose(obs0[0:3], [0.1, 0.2, 0.3])
    assert np.allclose(obs0[3:6], [0, 0, -1])
    assert np.allclose(obs0[6:16], state.q - cfg.home_rad)
    assert np.allclose(obs0[26:36], 0.0)
    assert np.allclose(obs1[26:36], 0.1)  # 上一步的原始动作
    assert np.allclose(obs0[36:39], [0.2, 0.0, -0.5])
    assert np.allclose(target, cfg.home_rad + 0.1)


def test_rejects_policy_with_wrong_joint_order(cfg):
    bad = dict(SIM_META, joint_names="right_ankle," + SIM_META["joint_names"].rsplit(",", 1)[0])
    with pytest.raises(ValueError, match="joint order"):
        PolicyController(cfg, infer=lambda o: np.zeros(ACTION_DIM), meta=bad)


def test_rejects_policy_with_different_obs_terms(cfg):
    bad = dict(SIM_META, observation_names="base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command")
    with pytest.raises(ValueError, match="observation terms"):
        PolicyController(cfg, infer=lambda o: np.zeros(ACTION_DIM), meta=bad)


def test_policy_runs_in_fake_runtime(cfg):
    bus, port = fake_bus_with(cfg.ids, positions={j.id: j.zero for j in cfg.joints})
    ctl = PolicyController(cfg, infer=lambda o: np.zeros(ACTION_DIM), meta=SIM_META)
    report = run(Robot(bus, cfg), ctl, hz=200, duration_s=0.3, imu=FakeImu())
    assert report.fault is None

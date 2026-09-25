import math

import numpy as np
import pytest

from hangoutduck.imu.base import G, GravityEstimator, ImuSample
from hangoutduck.imu.wit import TYPE_ACCEL, TYPE_GYRO, WitImu, encode_packet, raw_accel, raw_gyro


def test_wit_parser_decodes_accel_and_gyro():
    imu = WitImu("/dev/null")
    stream = (
        b"\x00\x55"  # 噪声
        + encode_packet(TYPE_ACCEL, raw_accel([0.0, 0.0, G]))
        + encode_packet(TYPE_GYRO, raw_gyro([0.0, 0.0, math.radians(90)]))
    )
    for i in range(0, len(stream), 4):
        imu.handle_bytes(stream[i : i + 4], t=1.0)
    s = imu.latest()
    assert s is not None
    assert np.allclose(s.accel, [0, 0, G], atol=0.01)
    assert np.allclose(s.gyro, [0, 0, math.radians(90)], atol=1e-3)


def test_gravity_level_and_axis_map():
    est = GravityEstimator(np.eye(3))
    g = est.update(ImuSample(t=0.0, gyro=np.zeros(3), accel=np.array([0.0, 0.0, G])))
    assert np.allclose(g, [0, 0, -1])
    assert est.tilt_deg() == 0.0
    # xgoduck 的装法：机体 = [imu_z, -imu_x, -imu_y]，IMU 的 -y 轴朝上
    rot = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    est = GravityEstimator(rot)
    g = est.update(ImuSample(t=0.0, gyro=np.zeros(3), accel=np.array([0.0, -G, 0.0])))
    assert np.allclose(g, [0, 0, -1])


def test_gravity_tracks_nose_down_pitch():
    """机头朝下转 30°：gx 应变为正，与仿真 projected_gravity 约定一致。"""
    est = GravityEstimator(np.eye(3), tau_s=0.3)
    theta = math.radians(30)
    # 机头朝下 = 绕机体 y 轴正向转动；此时比力在机体系为 [-sin, 0, cos] * G
    accel = np.array([-math.sin(theta), 0.0, math.cos(theta)]) * G
    t = 0.0
    est.update(ImuSample(t=t, gyro=np.zeros(3), accel=np.array([0.0, 0.0, G])))
    for _ in range(200):
        t += 0.01
        g = est.update(ImuSample(t=t, gyro=np.zeros(3), accel=accel))
    assert g[0] > 0.45
    assert est.tilt_deg() == pytest.approx(30.0, abs=1.0)


def test_gyro_prediction_between_accel_updates():
    """加速度计不可信（冲击中）时靠陀螺积分。"""
    est = GravityEstimator(np.eye(3))
    t = 0.0
    est.update(ImuSample(t=t, gyro=np.zeros(3), accel=np.array([0.0, 0.0, G])))
    w = np.array([0.0, math.radians(90), 0.0])  # 绕 y 轴 90°/s
    for _ in range(50):  # 0.5 s -> 45°
        t += 0.01
        g = est.update(ImuSample(t=t, gyro=w, accel=np.array([0.0, 0.0, 3 * G])))
    assert est.tilt_deg() == pytest.approx(45.0, abs=1.5)
    assert g[0] > 0



def test_repeated_sample_keeps_estimate():
    est = GravityEstimator(np.eye(3))
    s0 = ImuSample(t=0.0, gyro=np.zeros(3), accel=np.array([0.0, 0.0, G]))
    est.update(s0)
    s1 = ImuSample(t=0.01, gyro=np.array([0.0, 1.0, 0.0]), accel=np.array([0.0, 0.0, 3 * G]))
    g1 = est.update(s1).copy()
    g2 = est.update(s1)  # 同一个样本再来一次：不应重置成加速度计读数
    assert np.allclose(g1, g2)
    assert g2[0] > 0

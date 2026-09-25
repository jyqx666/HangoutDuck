"""维特智能（WIT）标准协议 IMU，例如 WT901 系列的 USB 版本。

每个数据包 11 字节：0x55 TYPE D0..D7 SUM，SUM 为前 10 字节之和的低 8 位；
D0..D7 是 4 个小端 int16。
  0x51 加速度  raw / 32768 * 16 g
  0x52 角速度  raw / 32768 * 2000 °/s
  0x53 角度    raw / 32768 * 180 °（本驱动不使用）
出厂波特率常见为 9600 或 115200，输出频率建议用厂家上位机设为 100–200 Hz。
"""

from __future__ import annotations

import math
import struct
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from .base import G, Imu, ImuSample

PACKET_LEN = 11
TYPE_ACCEL = 0x51
TYPE_GYRO = 0x52


class WitParser:
    def __init__(self) -> None:
        self._buf = bytearray()
        self.checksum_errors = 0

    def feed(self, data: bytes) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """返回 [(type, (v0, v1, v2, v3)), ...]。"""
        self._buf += data
        out = []
        buf = self._buf
        while True:
            start = buf.find(b"\x55")
            if start < 0:
                buf.clear()
                return out
            if start:
                del buf[:start]
            if len(buf) < PACKET_LEN:
                return out
            if sum(buf[:10]) & 0xFF != buf[10]:
                self.checksum_errors += 1
                del buf[0]
                continue
            out.append((buf[1], struct.unpack_from("<4h", buf, 2)))
            del buf[:PACKET_LEN]


def accel_from_raw(v) -> np.ndarray:
    return np.array(v[:3], dtype=float) / 32768.0 * 16.0 * G


def gyro_from_raw(v) -> np.ndarray:
    return np.radians(np.array(v[:3], dtype=float) / 32768.0 * 2000.0)


class WitImu(Imu):
    def __init__(self, device: str, baudrate: int = 115200) -> None:
        self.device = device
        self.baudrate = baudrate
        self.parser = WitParser()
        self.samples = 0
        self._lock = threading.Lock()
        self._accel: Optional[np.ndarray] = None
        self._latest: Optional[ImuSample] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._port = None

    def start(self) -> None:
        import serial

        self._port = serial.Serial(self.device, baudrate=self.baudrate, timeout=0.01)
        self._thread = threading.Thread(target=self._run, name="wit-imu", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            data = self._port.read(max(1, self._port.in_waiting))
            if data:
                self.handle_bytes(data, time.perf_counter())

    def handle_bytes(self, data: bytes, t: float) -> None:
        for kind, values in self.parser.feed(data):
            if kind == TYPE_ACCEL:
                self._accel = accel_from_raw(values)
            elif kind == TYPE_GYRO and self._accel is not None:
                # 每收到一次角速度就和最近的加速度组成一个样本
                with self._lock:
                    self._latest = ImuSample(t=t, gyro=gyro_from_raw(values), accel=self._accel)
                    self.samples += 1

    def latest(self) -> Optional[ImuSample]:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._port is not None:
            self._port.close()


def encode_packet(kind: int, values) -> bytes:
    """生成一个 WIT 数据包（测试和假 IMU 用）。"""
    body = bytes([0x55, kind]) + struct.pack("<4h", *[int(v) for v in values])
    return body + bytes([sum(body) & 0xFF])


def raw_accel(accel_ms2) -> List[int]:
    return [int(round(a / G / 16.0 * 32768.0)) for a in accel_ms2] + [0]


def raw_gyro(gyro_rad_s) -> List[int]:
    return [int(round(math.degrees(w) / 2000.0 * 32768.0)) for w in gyro_rad_s] + [0]

"""飞特总线驱动：在一个串口（USB 转半双工 TTL 驱动板）上收发指令。

传输层只需要 pyserial.Serial 的这几个接口：
``write(bytes)``、``read(n)``、``in_waiting``、``reset_input_buffer()``、``close()``。
测试和 ``--fake`` 模式用 :class:`hangoutduck.feetech.fake.FakeSerial` 代替真串口。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

from . import protocol as p
from . import registers as reg


class BusError(RuntimeError):
    pass


class BusTimeout(BusError):
    pass


@dataclass
class BusStats:
    sent: int = 0
    replies: int = 0
    timeouts: int = 0
    servo_errors: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


class FeetechBus:
    """一条飞特总线。

    ``echo=True`` 用于会把自己发出的字节回显到接收端的驱动板：
    发送后先丢弃同样长度的字节再解析应答。
    """

    def __init__(self, port, reply_timeout_s: float = 0.006, echo: bool = False) -> None:
        self.port = port
        self.reply_timeout_s = reply_timeout_s
        self.echo = echo
        self.stats = BusStats()
        self._parser = p.PacketParser()
        # 最近一次应答里的舵机 ERROR 字节（非 0 表示舵机自报故障）
        self.last_errors: Dict[int, int] = {}

    @classmethod
    def open(cls, device: str, baudrate: int = 1_000_000, **kwargs) -> "FeetechBus":
        import serial  # 延迟导入：纯离线测试不需要 pyserial

        port = serial.Serial(device, baudrate=baudrate, timeout=0.001, write_timeout=0.05)
        return cls(port, **kwargs)

    def close(self) -> None:
        self.port.close()

    def __enter__(self) -> "FeetechBus":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- 底层收发 -------------------------------------------------------

    def _send(self, packet: bytes) -> None:
        self.port.reset_input_buffer()
        self._parser.reset()
        self.port.write(packet)
        self.stats.sent += 1
        if self.echo:
            self._discard(len(packet))

    def _discard(self, n: int) -> None:
        deadline = time.perf_counter() + self.reply_timeout_s
        while n > 0 and time.perf_counter() < deadline:
            n -= len(self.port.read(n))

    def _collect(self, ids: Iterable[int], timeout_s: Optional[float] = None) -> Dict[int, p.Packet]:
        """等待指定 ID 的应答包，超时即返回已收到的部分。"""
        pending = set(ids)
        got: Dict[int, p.Packet] = {}
        deadline = time.perf_counter() + (timeout_s if timeout_s is not None else self.reply_timeout_s)
        while pending and time.perf_counter() < deadline:
            chunk = self.port.read(max(1, self.port.in_waiting))
            if not chunk:
                continue
            for pkt in self._parser.feed(chunk):
                if pkt.servo_id in pending:
                    pending.discard(pkt.servo_id)
                    got[pkt.servo_id] = pkt
                    self.last_errors[pkt.servo_id] = pkt.error
                    if pkt.error:
                        self.stats.servo_errors += 1
        self.stats.replies += len(got)
        self.stats.timeouts += len(pending)
        return got

    def _transact(self, servo_id: int, packet: bytes) -> p.Packet:
        self._send(packet)
        got = self._collect([servo_id])
        if servo_id not in got:
            raise BusTimeout(f"servo {servo_id} did not reply")
        return got[servo_id]

    # ---- 单个舵机 -------------------------------------------------------

    def ping(self, servo_id: int, timeout_s: Optional[float] = None) -> bool:
        self._send(p.ping_packet(servo_id))
        return servo_id in self._collect([servo_id], timeout_s)

    def read(self, servo_id: int, address: int, length: int) -> bytes:
        pkt = self._transact(servo_id, p.read_packet(servo_id, address, length))
        if len(pkt.params) != length:
            raise BusError(f"servo {servo_id}: expected {length} bytes, got {len(pkt.params)}")
        return pkt.params

    def write(self, servo_id: int, address: int, data: bytes) -> None:
        packet = p.write_packet(servo_id, address, data)
        if servo_id == p.BROADCAST_ID:
            self._send(packet)  # 广播不应答
            return
        self._transact(servo_id, packet)

    def read_u8(self, servo_id: int, address: int) -> int:
        return self.read(servo_id, address, 1)[0]

    def read_u16(self, servo_id: int, address: int) -> int:
        return p.parse_u16_le(self.read(servo_id, address, 2))

    def write_u8(self, servo_id: int, address: int, value: int) -> None:
        self.write(servo_id, address, bytes([value & 0xFF]))

    def write_u16(self, servo_id: int, address: int, value: int) -> None:
        self.write(servo_id, address, p.u16_le(value))

    def write_eeprom(self, servo_id: int, address: int, data: bytes) -> None:
        """写 EEPROM 区：先解锁，写完再上锁。改 ID 请用 :meth:`change_id`。"""
        if address == reg.ID:
            raise ValueError("use change_id() to change a servo id")
        self.write_u8(servo_id, reg.LOCK, 0)
        try:
            self.write(servo_id, address, data)
        finally:
            self.write_u8(servo_id, reg.LOCK, 1)

    def change_id(self, old_id: int, new_id: int) -> None:
        """改 ID 并掉电保存。应答可能来自旧 ID 也可能来自新 ID，两者都接受。"""
        if not 0 <= new_id <= p.MAX_ID:
            raise ValueError(f"new id out of range: {new_id}")
        self.write_u8(old_id, reg.LOCK, 0)
        self._send(p.write_packet(old_id, reg.ID, bytes([new_id])))
        self._collect([old_id, new_id])
        self.write_u8(new_id, reg.LOCK, 1)
        if not self.ping(new_id):
            raise BusError(f"servo did not answer on new id {new_id}")

    # ---- 多个舵机 -------------------------------------------------------

    def sync_write(self, address: int, length: int, data: Mapping[int, bytes]) -> None:
        if data:
            self._send(p.sync_write_packet(address, length, data))

    def sync_read(self, address: int, length: int, ids: List[int]) -> Dict[int, bytes]:
        """同步读：每个舵机按顺序各回一个包。缺失的 ID 不出现在结果里。"""
        self._send(p.sync_read_packet(address, length, ids))
        # 每个应答包约 (6 + length) 字节，1 Mbps 下约 10 us/字节，再加每个舵机的返回延迟
        timeout = self.reply_timeout_s + len(ids) * 0.0005
        got = self._collect(ids, timeout)
        return {i: pkt.params for i, pkt in got.items() if len(pkt.params) == length}

    def scan(self, ids: Iterable[int] = range(0, p.MAX_ID + 1), timeout_s: float = 0.003) -> List[int]:
        return [i for i in ids if self.ping(i, timeout_s)]

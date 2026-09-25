"""飞特 SCS/STS 串行总线协议：打包、解包、数值编码。

包格式（半双工 TTL，默认 1 Mbps）::

    指令包: FF FF ID LEN INSTR PARAM... CHK
    应答包: FF FF ID LEN ERROR PARAM... CHK

LEN = 参数个数 + 2；CHK = ~(ID + LEN + INSTR/ERROR + 参数之和) & 0xFF。
STS/HLS 系列多字节数据为小端（低字节在前）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional

HEADER = b"\xff\xff"
BROADCAST_ID = 0xFE
MAX_ID = 0xFD


class Instruction:
    PING = 0x01
    READ = 0x02
    WRITE = 0x03
    REG_WRITE = 0x04
    ACTION = 0x05
    SYNC_READ = 0x82
    SYNC_WRITE = 0x83


# 应答包 ERROR 字节各位含义（飞特 STS 手册）
ERROR_BITS = {
    0: "voltage",
    1: "angle_sensor",
    2: "overheat",
    3: "overcurrent",
    5: "overload",
}


def describe_error(error: int) -> List[str]:
    """把 ERROR 字节翻译成可读的故障列表；0 表示无故障。"""
    return [name for bit, name in ERROR_BITS.items() if error & (1 << bit)]


def checksum(body: Iterable[int]) -> int:
    """body 为 ID 起到最后一个参数为止的字节。"""
    return (~sum(body)) & 0xFF


def build_packet(servo_id: int, instruction: int, params: bytes = b"") -> bytes:
    if not 0 <= servo_id <= BROADCAST_ID:
        raise ValueError(f"servo id out of range: {servo_id}")
    length = len(params) + 2
    if length > 0xFF:
        raise ValueError(f"packet too long: {len(params)} parameter bytes")
    body = bytes([servo_id, length, instruction]) + params
    return HEADER + body + bytes([checksum(body)])


def ping_packet(servo_id: int) -> bytes:
    return build_packet(servo_id, Instruction.PING)


def read_packet(servo_id: int, address: int, length: int) -> bytes:
    return build_packet(servo_id, Instruction.READ, bytes([address, length]))


def write_packet(servo_id: int, address: int, data: bytes) -> bytes:
    return build_packet(servo_id, Instruction.WRITE, bytes([address]) + data)


def sync_read_packet(address: int, length: int, ids: Iterable[int]) -> bytes:
    return build_packet(BROADCAST_ID, Instruction.SYNC_READ, bytes([address, length, *ids]))


def sync_write_packet(address: int, length: int, data: Mapping[int, bytes]) -> bytes:
    params = bytearray([address, length])
    for servo_id, payload in data.items():
        if len(payload) != length:
            raise ValueError(f"id {servo_id}: expected {length} bytes, got {len(payload)}")
        params.append(servo_id)
        params += payload
    return build_packet(BROADCAST_ID, Instruction.SYNC_WRITE, bytes(params))


@dataclass(frozen=True)
class Packet:
    """解析出的一个包。对应答包而言 code 是 ERROR 字节，对指令包而言是 INSTR。"""

    servo_id: int
    code: int
    params: bytes

    @property
    def error(self) -> int:
        return self.code


class PacketParser:
    """增量解析器：喂入任意切分的字节流，吐出校验通过的完整包。

    遇到噪声或校验失败时丢弃一个字节后重新找包头，保证不会卡死。
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self.checksum_errors = 0

    def reset(self) -> None:
        self._buf.clear()

    def feed(self, data: bytes) -> List[Packet]:
        self._buf += data
        out: List[Packet] = []
        while True:
            pkt = self._next()
            if pkt is None:
                return out
            out.append(pkt)

    def _next(self) -> Optional[Packet]:
        buf = self._buf
        while True:
            start = buf.find(HEADER)
            if start < 0:
                # 保留可能是半个包头的最后一个 0xFF
                keep = 1 if buf[-1:] == b"\xff" else 0
                del buf[: len(buf) - keep]
                return None
            if start:
                del buf[:start]
            # 连续多个 0xFF 时，把包头对齐到最后两个
            while len(buf) >= 3 and buf[2] == 0xFF:
                del buf[0]
            if len(buf) < 4:
                return None
            length = buf[3]
            if length < 2:
                del buf[0]
                continue
            total = 4 + length
            if len(buf) < total:
                return None
            body = buf[2 : total - 1]
            if checksum(body) != buf[total - 1]:
                self.checksum_errors += 1
                del buf[0]
                continue
            pkt = Packet(servo_id=buf[2], code=buf[4], params=bytes(buf[5 : total - 1]))
            del buf[:total]
            return pkt


def u16_le(value: int) -> bytes:
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"u16 out of range: {value}")
    return bytes([value & 0xFF, value >> 8])


def parse_u16_le(data: bytes, offset: int = 0) -> int:
    return data[offset] | (data[offset + 1] << 8)


def decode_sign_magnitude(raw: int, sign_bit: int) -> int:
    """飞特的有符号量用"符号位 + 幅值"表示（不是补码），sign_bit 为符号位位置。"""
    magnitude = raw & ((1 << sign_bit) - 1)
    return -magnitude if raw & (1 << sign_bit) else magnitude


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    limit = (1 << sign_bit) - 1
    magnitude = min(abs(int(value)), limit)
    return magnitude | (1 << sign_bit) if value < 0 else magnitude

"""在内存里模拟一条挂着若干飞特舵机的总线。

用途：单元测试；没有驱动板时用 ``--fake`` 把工具和运行时整条链路跑通。
模拟的是协议和寄存器，不是电机动力学：上力后当前位置按限速追目标位置。
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, Iterable, List, Optional

from . import protocol as p
from . import registers as reg


class FakeServo:
    def __init__(self, servo_id: int, position: int = 2048) -> None:
        self.mem = bytearray(256)
        self.mem[reg.ID] = servo_id
        self.mem[reg.REGISTERS["firmware_major"].address] = 3
        self.mem[reg.REGISTERS["model_major"].address] = 9
        self.mem[reg.REGISTERS["model_minor"].address] = 19
        self.mem[reg.REGISTERS["max_temperature"].address] = 70
        self._set_u16("max_angle_limit", 4095)
        self._set_u16("max_torque", 1000)
        self._set_u16("torque_limit", 1000)
        self.mem[reg.REGISTERS["p_coefficient"].address] = 32
        self.mem[reg.LOCK] = 1
        self.mem[reg.REGISTERS["present_voltage"].address] = 78
        self.mem[reg.REGISTERS["present_temperature"].address] = 30
        self._set_u16("present_position", position)
        self._set_u16("goal_position", position)
        self.max_speed_ticks_per_s = 4000.0
        self.error = 0

    @property
    def id(self) -> int:
        return self.mem[reg.ID]

    def _set_u16(self, name: str, value: int) -> None:
        a = reg.REGISTERS[name].address
        self.mem[a : a + 2] = p.u16_le(value)

    def u16(self, name: str) -> int:
        return p.parse_u16_le(self.mem, reg.REGISTERS[name].address)

    @property
    def position(self) -> int:
        return self.u16("present_position")

    @position.setter
    def position(self, ticks: int) -> None:
        self._set_u16("present_position", max(0, min(4095, int(round(ticks)))))

    @property
    def temperature(self) -> int:
        return self.mem[reg.REGISTERS["present_temperature"].address]

    @temperature.setter
    def temperature(self, value: int) -> None:
        self.mem[reg.REGISTERS["present_temperature"].address] = value

    def write(self, address: int, data: bytes) -> None:
        # 不模拟 EEPROM 锁：真舵机上锁时写 EEPROM 区掉电会丢，模拟里一律直接生效
        self.mem[address : address + len(data)] = data

    def step(self, dt: float) -> None:
        """推进 dt 秒：上力时位置按最大速度追目标，速度寄存器同步更新。"""
        goal = self.u16("goal_position")
        pos = self.position
        speed_units = 0
        if self.mem[reg.TORQUE_ENABLE]:
            max_step = self.max_speed_ticks_per_s * dt
            delta = max(-max_step, min(max_step, goal - pos))
            self.position = pos + delta
            if dt > 0:
                speed_units = int(round(delta / dt / reg.SPEED_UNIT_TICKS_PER_S))
        self._set_u16("present_speed", p.encode_sign_magnitude(speed_units, 15))


class FakeSerial:
    """与 pyserial 接口兼容的假串口，把指令包分发给挂在上面的 FakeServo。"""

    def __init__(
        self,
        servos: Iterable[FakeServo] = (),
        drop_rate: float = 0.0,
        echo: bool = False,
        seed: int = 0,
    ) -> None:
        self.servos: Dict[int, FakeServo] = {}
        for s in servos:
            self.add(s)
        self.drop_rate = drop_rate
        self.echo = echo
        self._rng = random.Random(seed)
        self._rx: deque = deque()
        self._parser = p.PacketParser()
        self.written: List[bytes] = []
        self.is_open = True

    def add(self, servo: FakeServo) -> FakeServo:
        self.servos[servo.id] = servo
        return servo

    def _rekey(self) -> None:
        self.servos = {s.id: s for s in self.servos.values()}

    # ---- pyserial 接口 ------------------------------------------------

    @property
    def in_waiting(self) -> int:
        return len(self._rx)

    def reset_input_buffer(self) -> None:
        self._rx.clear()

    def read(self, n: int = 1) -> bytes:
        out = bytearray()
        while self._rx and len(out) < n:
            out.append(self._rx.popleft())
        return bytes(out)

    def write(self, data: bytes) -> int:
        self.written.append(bytes(data))
        if self.echo:
            self._rx.extend(data)
        for pkt in self._parser.feed(data):
            self._handle(pkt)
        return len(data)

    def close(self) -> None:
        self.is_open = False

    # ---- 协议处理 --------------------------------------------------------

    def _reply(self, servo: FakeServo, params: bytes = b"") -> None:
        if self.drop_rate and self._rng.random() < self.drop_rate:
            return
        body = bytes([servo.id, len(params) + 2, servo.error]) + params
        self._rx.extend(p.HEADER + body + bytes([p.checksum(body)]))

    def _handle(self, pkt: p.Packet) -> None:
        instr, params = pkt.code, pkt.params
        if instr == p.Instruction.SYNC_WRITE:
            address, length = params[0], params[1]
            body = params[2:]
            for k in range(0, len(body), length + 1):
                servo = self.servos.get(body[k])
                if servo:
                    servo.write(address, body[k + 1 : k + 1 + length])
            return
        if instr == p.Instruction.SYNC_READ:
            address, length = params[0], params[1]
            for sid in params[2:]:
                servo = self.servos.get(sid)
                if servo:
                    self._reply(servo, bytes(servo.mem[address : address + length]))
            return

        targets = list(self.servos.values()) if pkt.servo_id == p.BROADCAST_ID else [self.servos.get(pkt.servo_id)]
        for servo in targets:
            if servo is None:
                continue
            respond = pkt.servo_id != p.BROADCAST_ID
            if instr == p.Instruction.PING:
                if respond:
                    self._reply(servo)
            elif instr == p.Instruction.READ:
                address, length = params[0], params[1]
                if respond:
                    self._reply(servo, bytes(servo.mem[address : address + length]))
            elif instr == p.Instruction.WRITE:
                servo.write(params[0], params[1:])
                if params[0] == reg.ID:
                    self._rekey()
                if respond:
                    self._reply(servo)

    def step(self, dt: float) -> None:
        for s in self.servos.values():
            s.step(dt)


def fake_bus_with(ids: Iterable[int], positions: Optional[Dict[int, int]] = None, **kwargs):
    """便捷函数：返回挂着指定 ID 舵机的 (FeetechBus, FakeSerial)。"""
    from .bus import FeetechBus

    positions = positions or {}
    port = FakeSerial([FakeServo(i, positions.get(i, 2048)) for i in ids], **kwargs)
    return FeetechBus(port, reply_timeout_s=0.002, echo=port.echo), port

import pytest

from hangoutduck.feetech import registers as reg
from hangoutduck.feetech.bus import BusTimeout, FeetechBus
from hangoutduck.feetech.fake import FakeSerial, FakeServo, fake_bus_with


def test_ping_and_scan():
    bus, _ = fake_bus_with([10, 11, 20])
    assert bus.ping(10)
    assert not bus.ping(12)
    assert bus.scan(range(0, 30)) == [10, 11, 20]


def test_read_write_roundtrip():
    bus, port = fake_bus_with([10])
    bus.write_u16(10, reg.GOAL_POSITION, 3000)
    assert bus.read_u16(10, reg.GOAL_POSITION) == 3000
    assert port.servos[10].u16("goal_position") == 3000


def test_read_missing_servo_raises_timeout():
    bus, _ = fake_bus_with([10])
    with pytest.raises(BusTimeout):
        bus.read_u16(99, reg.PRESENT_POSITION)
    assert bus.stats.timeouts == 1


def test_sync_read_returns_only_present_ids():
    bus, _ = fake_bus_with([10, 11], positions={10: 1000, 11: 3000})
    got = bus.sync_read(reg.PRESENT_POSITION, 2, [10, 11, 12])
    assert set(got) == {10, 11}
    assert got[10] == bytes([1000 & 0xFF, 1000 >> 8])


def test_sync_write_reaches_every_servo():
    bus, port = fake_bus_with([10, 11])
    bus.sync_write(reg.GOAL_POSITION, 2, {10: b"\x10\x00", 11: b"\x20\x00"})
    assert port.servos[10].u16("goal_position") == 0x10
    assert port.servos[11].u16("goal_position") == 0x20


def test_echoing_adapter():
    port = FakeSerial([FakeServo(10, 1234)], echo=True)
    bus = FeetechBus(port, reply_timeout_s=0.002, echo=True)
    assert bus.read_u16(10, reg.PRESENT_POSITION) == 1234


def test_change_id_and_eeprom_lock():
    bus, port = fake_bus_with([1])
    bus.change_id(1, 12)
    assert bus.ping(12)
    assert not bus.ping(1)
    assert port.servos[12].mem[reg.LOCK] == 1
    with pytest.raises(ValueError):
        bus.write_eeprom(12, reg.ID, b"\x05")


def test_servo_error_byte_is_recorded():
    bus, port = fake_bus_with([10])
    port.servos[10].error = 0b100  # overheat
    bus.read_u8(10, reg.TORQUE_ENABLE)
    assert bus.last_errors[10] == 0b100
    assert bus.stats.servo_errors == 1

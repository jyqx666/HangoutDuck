import pytest

from hangoutduck.feetech import protocol as p


def test_ping_packet_matches_datasheet_example():
    # 飞特手册示例：PING ID 1 -> FF FF 01 02 01 FB
    assert p.ping_packet(1) == bytes.fromhex("ffff010201fb")


def test_read_packet_checksum():
    pkt = p.read_packet(1, 56, 2)
    assert pkt[:5] == bytes([0xFF, 0xFF, 1, 4, p.Instruction.READ])
    assert pkt[-1] == (~(1 + 4 + 2 + 56 + 2)) & 0xFF


def test_sync_write_layout():
    pkt = p.sync_write_packet(42, 2, {10: b"\x00\x08", 11: b"\xff\x07"})
    # FF FF FE LEN 83 addr len [id d d] [id d d] chk
    assert pkt[2] == p.BROADCAST_ID
    assert pkt[3] == 2 * 3 + 4
    assert pkt[4] == p.Instruction.SYNC_WRITE
    assert pkt[5:7] == bytes([42, 2])
    assert pkt[7:13] == bytes([10, 0x00, 0x08, 11, 0xFF, 0x07])


def test_sync_write_rejects_wrong_length():
    with pytest.raises(ValueError):
        p.sync_write_packet(42, 2, {10: b"\x00"})


def test_parser_handles_noise_split_and_bad_checksum():
    good = p.build_packet(12, 0, b"\x01\x02")
    bad = bytearray(p.build_packet(13, 0, b"\x03"))
    bad[-1] ^= 0xFF
    stream = b"\x00\x12\xff" + bytes(bad) + good + b"\xff\xff\xff" + good
    parser = p.PacketParser()
    out = []
    for i in range(0, len(stream), 3):  # 任意切分
        out += parser.feed(stream[i : i + 3])
    assert [(x.servo_id, x.params) for x in out] == [(12, b"\x01\x02"), (12, b"\x01\x02")]
    assert parser.checksum_errors >= 1


def test_sign_magnitude_roundtrip():
    for v in (0, 1, -1, 1234, -1234):
        assert p.decode_sign_magnitude(p.encode_sign_magnitude(v, 15), 15) == v
    assert p.decode_sign_magnitude(0x8005, 15) == -5
    assert p.decode_sign_magnitude(0x0405, 10) == -5


def test_describe_error():
    assert p.describe_error(0) == []
    assert p.describe_error(0b100100) == ["overheat", "overload"]

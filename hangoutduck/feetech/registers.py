"""飞特 STS 系列控制表（HLS1910 按 xgoduck 的说法使用 STS3215 固件）。

地址来自飞特 STS3215 手册和 xgoduck 运行时（sketch/duck_config.h）。
到货后请用 `hduck dump --id <ID>` 读出整张表核对，尤其是标注为
「HLS1910 专有」的地址。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Register:
    name: str
    address: int
    size: int  # 字节数：1 或 2
    eeprom: bool  # True: 掉电保存，写之前需要先解锁（LOCK=0）
    note: str = ""


_TABLE: List[Register] = [
    Register("firmware_major", 0, 1, True),
    Register("firmware_minor", 1, 1, True),
    Register("model_major", 3, 1, True),
    Register("model_minor", 4, 1, True),
    Register("id", 5, 1, True),
    Register("baud_rate", 6, 1, True, "0=1M 1=500k 2=250k 3=128k 4=115200 5=76800 6=57600 7=38400"),
    Register("return_delay", 7, 1, True, "单位 2 us，建议 0"),
    Register("response_level", 8, 1, True),
    Register("min_angle_limit", 9, 2, True, "刻度"),
    Register("max_angle_limit", 11, 2, True, "刻度"),
    Register("max_temperature", 13, 1, True, "°C"),
    Register("max_voltage", 14, 1, True, "0.1 V"),
    Register("min_voltage", 15, 1, True, "0.1 V"),
    Register("max_torque", 16, 2, True, "0.1 %"),
    Register("phase", 18, 1, True),
    Register("unload_condition", 19, 1, True),
    Register("led_alarm_condition", 20, 1, True),
    Register("p_coefficient", 21, 1, True, "位置环 P（永久）"),
    Register("d_coefficient", 22, 1, True, "位置环 D（永久）"),
    Register("i_coefficient", 23, 1, True),
    Register("min_startup_force", 24, 2, True),
    Register("cw_dead_zone", 26, 1, True),
    Register("ccw_dead_zone", 27, 1, True),
    Register("protection_current", 28, 2, True),
    Register("angular_resolution", 30, 1, True),
    Register("position_offset", 31, 2, True, "bit11 为符号位"),
    Register("mode", 33, 1, True, "0=位置 1=速度 2=开环 3=步进"),
    Register("protective_torque", 34, 1, True),
    Register("protection_time", 35, 1, True),
    Register("overload_torque", 36, 1, True),
    Register("speed_p", 37, 1, True),
    Register("overcurrent_time", 38, 1, True),
    Register("speed_i", 39, 1, True),
    Register("torque_enable", 40, 1, False, "0=卸力 1=上力"),
    Register("acceleration", 41, 1, False),
    Register("goal_position", 42, 2, False, "刻度，4096/圈"),
    Register("goal_time", 44, 2, False, "0=不限制"),
    Register("goal_speed", 46, 2, False, "0=最大速度"),
    Register("torque_limit", 48, 2, False, "0.1 %"),
    Register("temp_p", 50, 1, False, "HLS1910 专有：临时位置环 P（xgoduck 运行值 6）"),
    Register("temp_d", 51, 1, False, "HLS1910 专有：临时位置环 D（xgoduck 运行值 20）"),
    Register("lock", 55, 1, False, "0=解锁 EEPROM 1=上锁"),
    Register("present_position", 56, 2, False, "刻度"),
    Register("present_speed", 58, 2, False, "bit15 为方向位"),
    Register("present_load", 60, 2, False, "bit10 为方向位"),
    Register("present_voltage", 62, 1, False, "0.1 V"),
    Register("present_temperature", 63, 1, False, "°C"),
    Register("async_write_flag", 64, 1, False),
    Register("status", 65, 1, False),
    Register("moving", 66, 1, False),
    Register("present_current", 69, 2, False),
]

REGISTERS: Dict[str, Register] = {r.name: r for r in _TABLE}
TABLE_END = max(r.address + r.size for r in _TABLE)

# 常用地址
ID = REGISTERS["id"].address
BAUD_RATE = REGISTERS["baud_rate"].address
RETURN_DELAY = REGISTERS["return_delay"].address
MIN_ANGLE_LIMIT = REGISTERS["min_angle_limit"].address
MAX_ANGLE_LIMIT = REGISTERS["max_angle_limit"].address
TORQUE_ENABLE = REGISTERS["torque_enable"].address
GOAL_POSITION = REGISTERS["goal_position"].address
TEMP_P = REGISTERS["temp_p"].address
TEMP_D = REGISTERS["temp_d"].address
LOCK = REGISTERS["lock"].address
PRESENT_POSITION = REGISTERS["present_position"].address

# 控制循环一次同步写：goal_position(2) + goal_time(2) + goal_speed(2)
GOAL_BLOCK_LEN = 6
# 控制循环一次同步读：位置(2) 速度(2) 负载(2) 电压(1) 温度(1)
STATE_BLOCK_LEN = 8

BAUD_RATES = {0: 1_000_000, 1: 500_000, 2: 250_000, 3: 128_000, 4: 115_200, 5: 76_800, 6: 57_600, 7: 38_400}

TICKS_PER_REV = 4096
# 速度寄存器 1 个单位对应的刻度/秒。xgoduck 在 HLS1910 上用的是 50，
# 请用 `hduck bench-bus --check-speed` 对照位置差分核实。
SPEED_UNIT_TICKS_PER_S = 50.0

"""读取 sim_params.yaml：仿真 / 训练里所有需要按实物确认的参数。

只依赖 PyYAML，不依赖 mjlab，所以主仓库的测试（CI 里的 Python 3.8–3.12、树莓派上的 3.11）也能校验这个文件。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

PARAMS_PATH = Path(__file__).with_name("sim_params.yaml")
STATUSES = ("assumed", "measured", "decided")
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
_RANGE_KEYS = ("supply_voltage_v", "foot_friction", "cmd_lin_vel_x", "cmd_lin_vel_y", "cmd_ang_vel_z")
_KEYS = ("trunk_mass_kg", "trunk_com_offset_mm", "head_mass_kg", "head_fixed_deg", "servo_kp",
         "command_delay_steps") + _RANGE_KEYS


@dataclass(frozen=True)
class SimParams:
    version: str
    trunk_mass_kg: Optional[float]
    trunk_com_offset_m: Tuple[float, float, float]
    head_mass_kg: Optional[float]
    head_fixed_rad: Dict[str, float]
    servo_kp: float
    supply_voltage_v: Tuple[float, float]
    command_delay_steps: Tuple[int, int]
    foot_friction: Tuple[float, float]
    cmd_lin_vel_x: Tuple[float, float]
    cmd_lin_vel_y: Tuple[float, float]
    cmd_ang_vel_z: Tuple[float, float]
    status: Dict[str, str]
    raw: Dict[str, dict]

    def unconfirmed(self) -> List[str]:
        return [k for k in _KEYS if self.status[k] == "assumed"]


def _range(name: str, v) -> Tuple[float, float]:
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise ValueError(f"{name}: expected [low, high], got {v!r}")
    lo, hi = float(v[0]), float(v[1])
    if lo > hi:
        raise ValueError(f"{name}: low {lo} > high {hi}")
    return lo, hi


def _mass(name: str, v) -> Optional[float]:
    if v is None:
        return None
    if float(v) <= 0:
        raise ValueError(f"{name}: mass must be positive, got {v}")
    return float(v)


def load(path: Path = PARAMS_PATH) -> SimParams:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    version = str(doc.get("version", ""))
    # version 会拼进训练日志目录名，只允许文件名安全的字符
    if not re.fullmatch(r"[A-Za-z0-9._-]+", version):
        raise ValueError(f"version {version!r}: use letters, digits, '.', '_' or '-'")
    params = doc.get("params") or {}
    missing = [k for k in _KEYS if k not in params]
    unknown = [k for k in params if k not in _KEYS]
    if missing or unknown:
        raise ValueError(f"sim_params.yaml: missing {missing}, unknown {unknown}")
    status = {}
    for k in _KEYS:
        entry = params[k]
        if not isinstance(entry, dict) or "value" not in entry:
            raise ValueError(f"{k}: expected a mapping with 'value'")
        if entry.get("status") not in STATUSES:
            raise ValueError(f"{k}: status must be one of {STATUSES}, got {entry.get('status')!r}")
        status[k] = entry["status"]
    v = {k: params[k]["value"] for k in _KEYS}

    com = v["trunk_com_offset_mm"]
    if not (isinstance(com, (list, tuple)) and len(com) == 3):
        raise ValueError(f"trunk_com_offset_mm: expected [x, y, z], got {com!r}")
    head = v["head_fixed_deg"]
    if not isinstance(head, dict) or set(head) != set(HEAD_JOINTS):
        raise ValueError(f"head_fixed_deg: expected keys {HEAD_JOINTS}, got {head!r}")
    if float(v["servo_kp"]) <= 0:
        raise ValueError(f"servo_kp must be positive, got {v['servo_kp']}")
    delay = v["command_delay_steps"]
    if not (isinstance(delay, (list, tuple)) and len(delay) == 2 and all(isinstance(d, int) for d in delay)
            and 0 <= delay[0] <= delay[1]):
        raise ValueError(f"command_delay_steps: expected [low, high] integers, got {delay!r}")
    if _range("supply_voltage_v", v["supply_voltage_v"])[0] <= 0:
        raise ValueError("supply_voltage_v must be positive")

    return SimParams(
        version=version,
        trunk_mass_kg=_mass("trunk_mass_kg", v["trunk_mass_kg"]),
        trunk_com_offset_m=tuple(float(c) / 1000.0 for c in com),
        head_mass_kg=_mass("head_mass_kg", v["head_mass_kg"]),
        head_fixed_rad={j: math.radians(float(head[j])) for j in HEAD_JOINTS},
        servo_kp=float(v["servo_kp"]),
        supply_voltage_v=_range("supply_voltage_v", v["supply_voltage_v"]),
        command_delay_steps=(int(delay[0]), int(delay[1])),
        foot_friction=_range("foot_friction", v["foot_friction"]),
        cmd_lin_vel_x=_range("cmd_lin_vel_x", v["cmd_lin_vel_x"]),
        cmd_lin_vel_y=_range("cmd_lin_vel_y", v["cmd_lin_vel_y"]),
        cmd_ang_vel_z=_range("cmd_ang_vel_z", v["cmd_ang_vel_z"]),
        status=status,
        raw={k: dict(params[k]) for k in _KEYS},
    )


def summary(p: SimParams) -> str:
    lines = [f"sim params {p.version}: {len(_KEYS) - len(p.unconfirmed())}/{len(_KEYS)} confirmed"]
    for k in _KEYS:
        e = p.raw[k]
        lines.append(f"  [{e['status']:8s}] {k:20s} = {e['value']!r}  ({e.get('check', '')})")
    return "\n".join(lines)

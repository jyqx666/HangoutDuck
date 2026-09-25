"""加载 hardware/joints.yaml（关节表）和 hardware/calibration.yaml（每台机器的零位）。

joints.yaml 手写、带注释、进版本库；calibration.yaml 由 `hduck calibrate` 生成，
只存零位刻度，加载时覆盖 joints.yaml 里的 zero。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JOINTS = REPO_ROOT / "hardware" / "joints.yaml"
DEFAULT_CALIBRATION = REPO_ROOT / "hardware" / "calibration.yaml"

_AXES = {"x": 0, "y": 1, "z": 2}


class ConfigError(ValueError):
    pass


@dataclass
class JointConfig:
    name: str
    id: int
    sign: int
    zero: int
    home_deg: float
    min_deg: float
    max_deg: float


@dataclass
class RobotConfig:
    joints: List[JointConfig]
    baudrate: int = 1_000_000
    ticks_per_rev: int = 4096
    run_pd: Optional[Tuple[int, int]] = None
    soft_limit_factor: float = 0.9
    imu_axis_map: Sequence[str] = ("+x", "+y", "+z")
    calibrated: bool = False
    source: Optional[Path] = None

    @property
    def names(self) -> List[str]:
        return [j.name for j in self.joints]

    @property
    def ids(self) -> List[int]:
        return [j.id for j in self.joints]

    def joint(self, key) -> JointConfig:
        for j in self.joints:
            if j.name == key or j.id == key:
                return j
        raise KeyError(key)

    def index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def home_rad(self) -> np.ndarray:
        return np.radians([j.home_deg for j in self.joints])

    def soft_limits_rad(self) -> Tuple[np.ndarray, np.ndarray]:
        """把 [min, max] 围绕中点按 soft_limit_factor 收缩，和仿真的做法一致。"""
        lo = np.radians([j.min_deg for j in self.joints])
        hi = np.radians([j.max_deg for j in self.joints])
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * self.soft_limit_factor
        return mid - half, mid + half

    def imu_rotation(self) -> np.ndarray:
        """3x3 矩阵 R，机体向量 = R @ IMU 向量。"""
        return axis_map_matrix(self.imu_axis_map)


def axis_map_matrix(axis_map: Sequence[str]) -> np.ndarray:
    if len(axis_map) != 3:
        raise ConfigError(f"imu.axis_map needs 3 entries, got {axis_map!r}")
    rot = np.zeros((3, 3))
    for row, spec in enumerate(axis_map):
        spec = spec.strip().lower()
        sign = -1.0 if spec.startswith("-") else 1.0
        axis = spec.lstrip("+-")
        if axis not in _AXES:
            raise ConfigError(f"bad imu axis {spec!r}; use +x/-x/+y/-y/+z/-z")
        rot[row, _AXES[axis]] = sign
    if abs(abs(np.linalg.det(rot)) - 1.0) > 1e-9:
        raise ConfigError(f"imu.axis_map {axis_map!r} uses an axis twice")
    return rot


def _validate(cfg: RobotConfig) -> None:
    names, ids = cfg.names, cfg.ids
    if len(set(names)) != len(names):
        raise ConfigError("duplicate joint names")
    if len(set(ids)) != len(ids):
        raise ConfigError("duplicate servo ids")
    for j in cfg.joints:
        if j.sign not in (-1, 1):
            raise ConfigError(f"{j.name}: sign must be +1 or -1")
        if not 0 <= j.zero < cfg.ticks_per_rev:
            raise ConfigError(f"{j.name}: zero {j.zero} outside 0..{cfg.ticks_per_rev - 1}")
        if not j.min_deg < j.max_deg:
            raise ConfigError(f"{j.name}: min_deg must be < max_deg")
        if not j.min_deg <= j.home_deg <= j.max_deg:
            raise ConfigError(f"{j.name}: home_deg {j.home_deg} outside limits")
    if not 0 < cfg.soft_limit_factor <= 1:
        raise ConfigError("soft_limit_factor must be in (0, 1]")
    axis_map_matrix(cfg.imu_axis_map)


def load_config(
    joints_path: Path = DEFAULT_JOINTS,
    calibration_path: Optional[Path] = DEFAULT_CALIBRATION,
) -> RobotConfig:
    raw = yaml.safe_load(Path(joints_path).read_text(encoding="utf-8"))
    servo = raw.get("servo", {})
    joints = [
        JointConfig(
            name=str(j["name"]),
            id=int(j["id"]),
            sign=int(j["sign"]),
            zero=int(j["zero"]),
            home_deg=float(j["home_deg"]),
            min_deg=float(j["min_deg"]),
            max_deg=float(j["max_deg"]),
        )
        for j in raw["joints"]
    ]
    run_pd = servo.get("run_pd")
    cfg = RobotConfig(
        joints=joints,
        baudrate=int(servo.get("baudrate", 1_000_000)),
        ticks_per_rev=int(servo.get("ticks_per_rev", 4096)),
        run_pd=(int(run_pd[0]), int(run_pd[1])) if run_pd else None,
        soft_limit_factor=float(raw.get("soft_limit_factor", 0.9)),
        imu_axis_map=tuple(raw.get("imu", {}).get("axis_map", ("+x", "+y", "+z"))),
        source=Path(joints_path),
    )
    if calibration_path is not None and Path(calibration_path).exists():
        zeros = load_calibration(calibration_path)
        for j in cfg.joints:
            if j.name in zeros:
                j.zero = zeros[j.name]
        cfg.calibrated = all(j.name in zeros for j in cfg.joints)
    _validate(cfg)
    return cfg


def load_calibration(path: Path) -> Dict[str, int]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(k): int(v) for k, v in (raw.get("zero") or {}).items()}


def save_calibration(path: Path, zeros: Dict[str, int], note: str = "") -> None:
    header = (
        "# 由 `hduck calibrate` 生成：每个关节在 joint_deg = 0 时的舵机刻度。\n"
        "# 换舵机、重装舵盘后必须重新标定。\n"
    )
    if note:
        header += f"# {note}\n"
    body = yaml.safe_dump({"zero": {k: int(v) for k, v in zeros.items()}}, sort_keys=False)
    Path(path).write_text(header + body, encoding="utf-8")


def deg_per_tick(cfg: RobotConfig) -> float:
    return 360.0 / cfg.ticks_per_rev


def rad_per_tick(cfg: RobotConfig) -> float:
    return 2 * math.pi / cfg.ticks_per_rev

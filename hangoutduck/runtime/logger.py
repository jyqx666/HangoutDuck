"""把每个控制周期的数据存成 .npz，方便事后画图、和仿真对比。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..robot import JointState


class RunLogger:
    def __init__(self, joint_names: List[str]) -> None:
        self.joint_names = joint_names
        self._rows: Dict[str, List] = {k: [] for k in (
            "t", "q", "dq", "target", "load", "voltage", "temperature", "valid", "gyro", "gravity", "period",
        )}

    def log(
        self,
        t: float,
        state: JointState,
        target: np.ndarray,
        period: float,
        gyro: Optional[np.ndarray] = None,
        gravity: Optional[np.ndarray] = None,
    ) -> None:
        r = self._rows
        r["t"].append(t)
        r["q"].append(state.q.copy())
        r["dq"].append(state.dq.copy())
        r["target"].append(np.asarray(target, dtype=float).copy())
        r["load"].append(state.load.copy())
        r["voltage"].append(state.voltage.copy())
        r["temperature"].append(state.temperature.copy())
        r["valid"].append(state.valid.copy())
        r["gyro"].append(np.full(3, np.nan) if gyro is None else np.asarray(gyro, dtype=float))
        r["gravity"].append(np.full(3, np.nan) if gravity is None else np.asarray(gravity, dtype=float))
        r["period"].append(period)

    def __len__(self) -> int:
        return len(self._rows["t"])

    def save(self, path: Path, meta: Optional[dict] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.asarray(v) for k, v in self._rows.items()}
        arrays["joint_names"] = np.asarray(self.joint_names)
        arrays["meta"] = np.asarray(json.dumps(meta or {}, ensure_ascii=False))
        np.savez_compressed(path, **arrays)
        return path

"""rl/hangoutduck_rl/sim_params.yaml 的格式和默认值（不需要 mjlab）。"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sim_params", ROOT / "rl" / "hangoutduck_rl" / "sim_params.py")
sim_params = importlib.util.module_from_spec(_spec)
sys.modules["sim_params"] = sim_params  # dataclass 需要能从 sys.modules 找到所在模块
_spec.loader.exec_module(sim_params)


def _write(tmp_path, mutate):
    doc = yaml.safe_load(sim_params.PARAMS_PATH.read_text(encoding="utf-8"))
    mutate(doc)
    path = tmp_path / "sim_params.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return path


def test_shipped_file_matches_xgoduck_defaults():
    p = sim_params.load()
    assert p.trunk_mass_kg is None and p.head_mass_kg is None
    assert p.trunk_com_offset_m == (0.0, 0.0, 0.0)
    assert p.head_fixed_rad["neck_pitch"] == pytest.approx(0.3491, abs=1e-3)
    assert p.head_fixed_rad["head_pitch"] == pytest.approx(0.3491, abs=1e-3)
    assert p.head_fixed_rad["head_yaw"] == 0.0 and p.head_fixed_rad["head_roll"] == 0.0
    assert p.servo_kp == 5
    assert p.supply_voltage_v == (7.4, 8.0)
    assert p.command_delay_steps == (3, 6)
    assert p.foot_friction == (0.4, 1.5)
    assert (p.cmd_lin_vel_x, p.cmd_lin_vel_y, p.cmd_ang_vel_z) == ((-0.4, 0.4), (-0.3, 0.3), (-1.0, 1.0))


def test_measured_values_are_used(tmp_path):
    def mutate(doc):
        doc["version"] = "v1-weighed"
        doc["params"]["trunk_mass_kg"].update(value=0.21, status="measured")
        doc["params"]["head_fixed_deg"]["value"]["neck_pitch"] = 15.0
    p = sim_params.load(_write(tmp_path, mutate))
    assert p.version == "v1-weighed"
    assert p.trunk_mass_kg == 0.21
    assert p.head_fixed_rad["neck_pitch"] == pytest.approx(math.radians(15.0))
    assert "trunk_mass_kg" not in p.unconfirmed()
    assert "head_mass_kg" in p.unconfirmed()
    assert "1/" in sim_params.summary(p).splitlines()[0]


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(version="v1 weighed"), "version"),
    (lambda d: d["params"]["foot_friction"].update(value=[1.5, 0.4]), "foot_friction"),
    (lambda d: d["params"]["servo_kp"].update(status="maybe"), "status"),
    (lambda d: d["params"].pop("cmd_lin_vel_y"), "missing"),
    (lambda d: d["params"].update(extra={"value": 1, "status": "assumed"}), "unknown"),
    (lambda d: d["params"]["head_mass_kg"].update(value=-0.1), "positive"),
    (lambda d: d["params"]["command_delay_steps"].update(value=[6, 3]), "command_delay_steps"),
    (lambda d: d["params"]["head_fixed_deg"].update(value={"neck_pitch": 20.0}), "head_fixed_deg"),
])
def test_invalid_files_are_rejected(tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        sim_params.load(_write(tmp_path, mutate))

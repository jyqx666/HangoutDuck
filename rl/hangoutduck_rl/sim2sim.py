"""sim2sim：真机控制程序原样跑在 MuJoCo 上。

真机：    hduck 运行时 → FeetechBus → 串口 → 驱动板 → 10 个飞特 1910；IMU 单独一个串口
sim2sim： hduck 运行时 → FeetechBus → SimSerial（同一套飞特协议包，解析沿用 FakeSerial）
                                     → MuJoCo 里的 HangoutDuck（BAM 舵机模型）
          IMU → SimImu（读躯干上 imu site 的陀螺仪和加速度计）

Robot、主循环、安全检查、控制器、RL 策略一行不改。和训练仿真相同的部分：同一个模型
（含 sim_params.yaml）、5 ms 物理步长、碰撞设置、HLS1910 的 BAM 舵机模型（固件 P、电压和压降）、
指令延迟。舵机的固件 P 取部署程序实际写进寄存器的值（run_pd），所以 P 配错在这里也看得出来。

「实物」和 joints.yaml 可以故意不一致（零位偏、方向反、IMU 装歪、延迟更大……），
用来测试控制器能承受多大的硬件误差。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np
from bam.model import load_model
from bam.mujoco import MujocoController

from hangoutduck.config import RobotConfig, axis_map_matrix, rad_per_tick
from hangoutduck.feetech import protocol as p
from hangoutduck.feetech import registers as reg
from hangoutduck.feetech.bus import FeetechBus
from hangoutduck.feetech.fake import FakeSerial, FakeServo
from hangoutduck.imu.base import Imu, ImuSample
from mjlab_microduck.robot import xgoduck_constants as xgo

from . import robot as hd

PHYSICS_DT = 0.005  # 与训练相同
BAM_JSON = Path(xgo._HLS1910_JSON)
FACTORY_P = 32  # 舵机 EEPROM 里的位置环 P（FakeServo 的出厂值）；运行时 run_pd 会写临时 P 覆盖它
SETTLE_S = 0.5  # 开始前让机器人在地上站稳的时间


@dataclass
class Hardware:
    """仿真里「实物」的样子。默认与 joints.yaml / sim_params.yaml 完全一致；改这里模拟硬件误差。"""

    zero_error_deg: Dict[str, float] = field(default_factory=dict)  # 实物零位比 joints.yaml 多转了多少度
    flipped: Sequence[str] = ()  # 实物转向与 joints.yaml 的 sign 相反的关节
    delay_steps: int = 4  # 指令从发出到舵机生效的延迟，物理步（5 ms）；训练里是 3–6
    vin: Optional[float] = None  # 供电电压；None 取 sim_params 范围的中点
    vin_drop_gain: float = 0.1  # 带载压降（训练里在 0–0.2 之间随机）
    foot_friction: float = 1.0
    mass_scale: float = 1.0  # 整机质量缩放（所有部件同比例）
    trunk_com_error_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 实物躯干质心比模型偏多少（躯干系）
    imu_tilt_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # IMU 实际安装比 axis_map 多转的 roll/pitch/yaw
    imu_gyro_noise: float = 0.0  # 陀螺噪声标准差，rad/s
    imu_accel_noise: float = 0.0  # 加速度计噪声标准差，m/s²
    imu_delay_steps: int = 0  # IMU 数据晚几个控制周期
    seed: int = 0


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def build_spec(hw: Hardware) -> Tuple[mujoco.MjSpec, float]:
    """HangoutDuck 模型 + 地面，执行器按训练里的 BAM 方式改成力矩模式。返回 (spec, 力矩上限)。"""
    spec = hd.get_walk_spec()  # 头颈固定、质量和质心来自 sim_params.yaml
    xgo.XGODUCK_COLLISION.edit_spec(spec)  # 只有 *_collision 参与碰撞，脚底 condim 3、优先级 1
    for g in spec.geoms:
        if g.name in ("left_foot_collision", "right_foot_collision"):
            g.friction = [hw.foot_friction, 0.005, 0.0001]
    if hw.mass_scale != 1.0:
        for b in spec.bodies:
            if b.mass > 0:
                b.mass = b.mass * hw.mass_scale
                b.fullinertia = np.asarray(b.fullinertia) * hw.mass_scale

    if any(hw.trunk_com_error_mm):
        trunk = spec.body("trunk_base")
        trunk.ipos = np.asarray(trunk.ipos) + np.asarray(hw.trunk_com_error_mm) / 1000.0

    # 外观（只影响录视频，不影响物理）：浅色棋盘格地面、渐变天空
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.93, 0.92, 0.88], rgb2=[0.84, 0.83, 0.78], width=512, height=512)
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[20, 20], reflectance=0.0)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.97, 0.96, 0.93], rgb2=[0.82, 0.85, 0.89], width=512, height=512)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], material="grid")
    spec.worldbody.add_light(pos=[0.3, -0.4, 1.2], dir=[-0.15, 0.25, -1.0],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=True, diffuse=[0.7, 0.7, 0.7])

    bam = load_model(str(BAM_JSON))
    vin_hi = max(hd.PARAMS.supply_voltage_v)
    force_limit = vin_hi * bam.kt.value / bam.R.value
    armature = bam.actuator.get_extra_inertia()
    legs = set(hd.LEG_JOINTS)
    for act in spec.actuators:
        if act.target in legs:
            act.set_to_motor()
            act.forcelimited = True
            act.forcerange = [-force_limit, force_limit]
            act.gear = [1.0, 0, 0, 0, 0, 0]
    for j in spec.joints:
        if j.name in legs:
            j.armature = float(armature)
            j.damping = np.zeros((3, 1))
            j.frictionloss = 0.0
    return spec, force_limit


@dataclass
class TraceRow:
    """每个控制周期记一行「实物」的真实状态，用来评估走得怎么样。"""

    t: float
    pos: np.ndarray  # 躯干位置，世界系
    yaw: float  # 躯干朝向，rad
    tilt_deg: float  # 躯干倾角
    contacts: Tuple[bool, bool]  # 左、右脚是否着地


class Plant:
    """MuJoCo 里的机器人本体：10 个 BAM 舵机，按关节名收目标角，按物理步推进。"""

    def __init__(self, cfg: RobotConfig, hw: Optional[Hardware] = None) -> None:
        self.cfg = cfg
        self.hw = hw or Hardware()
        spec, self.force_limit = build_spec(self.hw)
        self.model = spec.compile()
        opt = self.model.opt
        opt.timestep = PHYSICS_DT
        opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        opt.iterations, opt.ls_iterations = 30, 50
        self.data = mujoco.MjData(self.model)

        self.names = list(cfg.names)
        acts = []
        for name in self.names:
            jid = self.model.joint(name).id
            acts.append(next(self.model.actuator(a).name for a in range(self.model.nu)
                             if self.model.actuator_trnid[a, 0] == jid))
        self.bam = load_model(str(BAM_JSON))
        lo, hi = hd.PARAMS.supply_voltage_v
        self.bam.actuator.vin = self.hw.vin if self.hw.vin is not None else 0.5 * (lo + hi)
        self.servos = MujocoController(self.bam, acts, self.model, self.data,
                                       vin_drop_gain=self.hw.vin_drop_gain, vin_min=min(7.0, lo - 0.4))
        self.qadr = self.servos.qpos_indexes
        self.dadr = self.servos.dof_indexes
        self.trunk = self.model.body("trunk_base").id
        self.imu_site = self.model.site("imu").id
        self._gyro_adr = self.model.sensor_adr[self.model.sensor("imu_ang_vel").id]
        self._acc_adr = self.model.sensor_adr[self.model.sensor("imu_accel").id]
        self.feet = [self.model.geom(f"{s}_foot_collision").id for s in ("left", "right")]
        self.floor = self.model.geom("floor").id
        self._rng = np.random.default_rng(self.hw.seed)
        self.reset()

    # ---- 状态 --------------------------------------------------------------

    def reset(self, home: Optional[np.ndarray] = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0.0, 0.0, hd.STAND_Z]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        q0 = self.cfg.home_rad if home is None else np.asarray(home, dtype=float)
        self.data.qpos[self.qadr] = q0
        mujoco.mj_forward(self.model, self.data)
        self.servos.reset(self.data.qpos)
        self.bam.actuator.q_target_smooth = q0.copy()
        # 先让机器人在地上站稳（舵机保持这个姿态 SETTLE_S 秒），再把时钟清零：
        # 真机上电时也是静止放在地上，IMU 第一帧读到的是重力，而不是自由落体
        self.bam.actuator.kp = np.full(len(self.names), float(FACTORY_P))
        for _ in range(int(SETTLE_S / PHYSICS_DT)):
            self.servos.update()
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        self.servos.last_ts = 0.0
        self.torque = np.zeros(len(self.names))
        self.torque_on = np.zeros(len(self.names), dtype=bool)
        self._pending: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray]] = deque()
        self._imu_hist: Deque[Tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=self.hw.imu_delay_steps + 1)
        self.trace: List[TraceRow] = []
        self._record_imu()

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.data.qvel[self.dadr].copy()

    def trunk_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.trunk].copy(), self.data.xmat[self.trunk].reshape(3, 3).copy()

    def tilt_deg(self) -> float:
        up = self.data.xmat[self.trunk].reshape(3, 3)[2, 2]
        return math.degrees(math.acos(max(-1.0, min(1.0, float(up)))))

    def foot_contacts(self) -> Tuple[bool, bool]:
        touching = [False, False]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for k, g in enumerate(self.feet):
                if {c.geom1, c.geom2} == {g, self.floor}:
                    touching[k] = True
        return touching[0], touching[1]

    # ---- 推进 --------------------------------------------------------------

    def advance(self, dt: float, q_target: np.ndarray, torque_on: np.ndarray, kp: np.ndarray) -> None:
        """推进 dt 秒。目标、上力状态和固件 P 经过 delay_steps 个物理步才在舵机上生效。"""
        cmd = (np.asarray(q_target, dtype=float).copy(), np.asarray(torque_on, dtype=bool).copy(),
               np.asarray(kp, dtype=float).copy())
        for _ in range(max(1, int(round(dt / PHYSICS_DT)))):
            self._pending.append(cmd)
            while len(self._pending) > self.hw.delay_steps + 1:
                self._pending.popleft()
            target, on, gains = self._pending[0]
            # 卸力的舵机：目标跟着当前位置走，重新上力时不会跳
            target = np.where(on, target, self.q)
            self.servos.q_target[:] = target
            self.bam.actuator.kp = gains
            self.servos.update()
            ctrl = self.data.ctrl[self.servos.act_indexes]
            ctrl[~on] = 0.0
            self.data.ctrl[self.servos.act_indexes] = ctrl
            self.torque = ctrl.copy()
            self.torque_on = on
            mujoco.mj_step(self.model, self.data)
        self._record_imu()

    # ---- IMU ---------------------------------------------------------------

    def _record_imu(self) -> None:
        mujoco.mj_forward(self.model, self.data)  # 让传感器、接触与当前状态一致
        gyro = self.data.sensordata[self._gyro_adr:self._gyro_adr + 3].copy()
        acc = self.data.sensordata[self._acc_adr:self._acc_adr + 3].copy()
        self._imu_hist.append((self.time, gyro, acc))
        pos, rot = self.trunk_pose()
        self.trace.append(TraceRow(self.time, pos, math.atan2(rot[1, 0], rot[0, 0]), self.tilt_deg(),
                                   self.foot_contacts()))

    def imu_reading(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """躯干系（imu site 与躯干同向）下的 (时刻, 角速度, 比力)，含 imu_delay_steps 的延迟。"""
        return self._imu_hist[0]


class SimServo(FakeServo):
    pass


class SimSerial(FakeSerial):
    """飞特协议的假串口，背后是 Plant。FeetechBus 看不出区别。

    每个控制周期运行时调用 step(dt)：读寄存器里的目标位置 / 上力 / 临时 P，
    按「实物」的零位和方向换成关节角交给 Plant，推进物理后把位置、速度、负载、电压写回寄存器。
    """

    def __init__(self, plant: Plant) -> None:
        cfg = plant.cfg
        self.plant = plant
        hw = plant.hw
        self._rpt = rad_per_tick(cfg)
        self._ids = list(cfg.ids)
        flipped = set(hw.flipped)
        self._sign = np.array([-j.sign if j.name in flipped else j.sign for j in cfg.joints], dtype=float)
        err = np.array([math.radians(hw.zero_error_deg.get(j.name, 0.0)) for j in cfg.joints])
        # 实物零位（刻度）：关节角为 0 时编码器的读数
        self._zero = np.array([j.zero for j in cfg.joints], dtype=float) - err / self._rpt * self._sign
        servos = []
        for k, sid in enumerate(self._ids):
            s = SimServo(sid, self._ticks(plant.q[k], k))
            s.mem[reg.REGISTERS["p_coefficient"].address] = FACTORY_P
            servos.append(s)
        super().__init__(servos)
        self.on_tick: List[Callable[[], None]] = []  # 每个控制周期推进完物理后调用（录视频等）
        self._sync_registers()

    def _ticks(self, q: float, k: int) -> int:
        t = self._zero[k] + q / self._rpt * self._sign[k]
        return int(max(0, min(4095, round(t))))

    def step(self, dt: float) -> None:
        servos = [self.servos[i] for i in self._ids]
        goal = np.array([s.u16("goal_position") for s in servos], dtype=float)
        on = np.array([bool(s.mem[reg.TORQUE_ENABLE]) for s in servos])
        kp = np.array([s.mem[reg.TEMP_P] or s.mem[reg.REGISTERS["p_coefficient"].address] for s in servos], dtype=float)
        target = (goal - self._zero) * self._rpt * self._sign
        self.plant.advance(dt, target, on, kp)
        self._sync_registers()
        for hook in self.on_tick:
            hook()

    def _sync_registers(self) -> None:
        q, dq = self.plant.q, self.plant.dq
        vin = float(self.plant.bam.actuator.vin)
        for k, sid in enumerate(self._ids):
            s = self.servos[sid]
            s.position = self._ticks(q[k], k)
            speed = int(round(dq[k] / self._rpt * self._sign[k] / reg.SPEED_UNIT_TICKS_PER_S))
            s._set_u16("present_speed", p.encode_sign_magnitude(max(-32767, min(32767, speed)), 15))
            load = int(round(1000.0 * self.plant.torque[k] / self.plant.force_limit * self._sign[k]))
            s._set_u16("present_load", p.encode_sign_magnitude(max(-1000, min(1000, load)), 10))
            s.mem[reg.REGISTERS["present_voltage"].address] = int(round(vin * 10))


class SimImu(Imu):
    """仿真里的 IMU：输出「IMU 芯片坐标系」下的读数，经 joints.yaml 的 axis_map 换回躯干系。"""

    def __init__(self, plant: Plant) -> None:
        self.plant = plant
        hw = plant.hw
        # 运行时用 axis_map 把 IMU 读数转到躯干系：v_trunk = R_map @ v_imu。实物若装歪了，
        # 真实关系是 v_trunk = R_map @ R_tilt @ v_imu
        mount = axis_map_matrix(plant.cfg.imu_axis_map) @ _rpy_matrix(*np.radians(hw.imu_tilt_deg))
        self._to_imu = mount.T
        self._rng = np.random.default_rng(hw.seed + 1)

    def latest(self) -> Optional[ImuSample]:
        t, gyro, acc = self.plant.imu_reading()
        hw = self.plant.hw
        g = self._to_imu @ gyro
        a = self._to_imu @ acc
        if hw.imu_gyro_noise:
            g = g + self._rng.normal(0.0, hw.imu_gyro_noise, 3)
        if hw.imu_accel_noise:
            a = a + self._rng.normal(0.0, hw.imu_accel_noise, 3)
        return ImuSample(t=t, gyro=g, accel=a)


class SimLoop:
    """不睡眠的 RateLoop：仿真按控制周期推进，比实时快。"""

    def __init__(self, hz: float) -> None:
        self.period = 1.0 / hz
        self.cycles = 0

    def wait(self) -> float:
        self.cycles += 1
        return self.period if self.cycles > 1 else 0.0

    def stats(self) -> Dict[str, float]:
        return {"cycles": self.cycles, "simulated": 1}


@dataclass
class Sim2Sim:
    """一套接好的仿真：plant + 总线 + IMU。`bus`、`imu` 直接交给 hangoutduck.runtime.run。"""

    plant: Plant
    port: SimSerial
    bus: FeetechBus
    imu: SimImu

    @classmethod
    def create(cls, cfg: RobotConfig, hw: Optional[Hardware] = None) -> "Sim2Sim":
        plant = Plant(cfg, hw)
        port = SimSerial(plant)
        return cls(plant=plant, port=port, bus=FeetechBus(port, reply_timeout_s=0.002), imu=SimImu(plant))


def default_config(joints_path: Optional[Path] = None) -> RobotConfig:
    """读 main 分支的 hardware/joints.yaml（不读标定文件：仿真里零位就是 joints.yaml 写的值）。"""
    from hangoutduck.config import DEFAULT_JOINTS, load_config

    return load_config(joints_path or DEFAULT_JOINTS, None)


def joint_order_matches(cfg: RobotConfig) -> List[str]:
    """joints.yaml 和仿真模型的关节名是否一致；返回不一致的说明。"""
    problems = []
    if tuple(cfg.names) != hd.LEG_JOINTS:
        problems.append(f"joints.yaml order {cfg.names} != model {list(hd.LEG_JOINTS)}")
    return problems

# HangoutDuck

小组复刻 [Microduck](https://github.com/pollen-robotics/microduck) 的双足小鸭子，舵机换成国产**飞特 1910**，
机械和关节布局参考已经用 1910 做过一遍的 [xgoduck](https://github.com/LuwuDynamics/xgoduck_hardware)。

**当前阶段（展示前）**：只做下半身（两条腿 × 5 自由度 = 10 个舵机），Jetson Nano 放桌上拴线控制。
展示底线是手写步态在平地连续走 ≥10 步；RL 策略并行推进，展示时在仿真里能走即可。
完整的范围、非目标和验收标准见 [需求规格](docs/deep-interview-hangoutduck-phase1.md)。

## 分支

| 分支 | 放什么 |
|---|---|
| `main` | 真机相关：舵机驱动、IMU、Jetson Nano 运行时、调试工具、硬件文档 |
| `rl` | 仿真与强化学习：下半身 10 自由度模型、训练任务、策略导出与部署适配 |

## 目录

```
hardware/
  joints.yaml        关节表（ID、方向、零位、home、限位）—— 全仓库唯一真源
  calibration.yaml   零位标定结果（hduck calibrate 生成）
  BOM.md             物料与采购建议
  wiring.md          接线图、ID 表
hangoutduck/         Python 包（Python ≥ 3.8，Nano 可用）
  feetech/           飞特总线协议、驱动、FakeSerial（无硬件测试用）
  imu/               WIT 协议 IMU 驱动、重力方向互补滤波
  robot.py           关节角 ↔ 舵机刻度换算、整机同步读写
  runtime/           50 Hz 控制循环、安全监控、日志、控制器接口
  cli.py             hduck 命令行
tests/               pytest，全部基于假舵机，不需要硬件
docs/
  deep-interview-hangoutduck-phase1.md   第一阶段需求规格
  jetson_nano_setup.md                   Nano 环境（WP2）
  bringup.md                             从单舵机到整机站立（WP4–WP6）
```

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate    # Nano 上用 python3.8，见 docs/jetson_nano_setup.md
pip install -e '.[dev]'
pytest -q

# 不接硬件，把整条链路走一遍
hduck --fake state
hduck --fake stand --duration 5
hduck --fake sweep --joint left_knee
hduck bench-loop --hz 50 --seconds 10
```

接上驱动板后按 [docs/bringup.md](docs/bringup.md) 一步步来。常用命令：

| 命令 | 用途 |
|---|---|
| `hduck scan --all-bauds` | 找总线上的舵机 |
| `hduck dump --id 1` | 读整张寄存器表，核对 1910 的寄存器 |
| `hduck set-id 10` | 改 ID（总线上只接一个） |
| `hduck center --id 10` | 转到中位装舵盘 |
| `hduck calibrate --joint left_knee` | 逐关节标零位 |
| `hduck state` | 各关节角度、温度、电压 |
| `hduck sweep --joint left_knee` | 单关节摆动，核对方向 |
| `hduck stand` | 平滑站到 home 并保持 |
| `hduck bench-bus` / `bench-loop` | 验收 A2 / A5 |
| `hduck imu` | 核对 IMU 装向 |

运行中按回车或 Ctrl-C 立即卸力；温度、电压、丢包、倾角越界也会自动卸力。

## 约定

- **角度**：关节角用弧度，0 = MJCF 里的 CAD 零位，与 `xgoduck_rl` 仿真完全一致；
  `joint_deg = (ticks - zero) × 360 / 4096 × sign`。
- **关节顺序**：左腿 hip_yaw, hip_roll, hip_pitch, knee, ankle，然后右腿同序。
- **机体坐标系**：x 朝前、y 朝左、z 朝上；projected gravity 平放为 `[0, 0, -1]`。
- 新控制器（保底步态、RL 策略）实现 `hangoutduck.runtime.Controller`，在 `CONTROLLERS` 里注册即可用 `hduck run --controller <名字>` 运行。

## 进度

| 工作包 | 状态 |
|---|---|
| WP0 采购驱动板、IMU | 待下单 |
| WP1 仓库、关节表、文档 | ✅ |
| WP2 Nano 环境 | 文档就绪，待上机 |
| WP4 舵机驱动层 | ✅ 代码与测试完成，待真机验证 |
| WP5 IMU 接入 | ✅ WIT 驱动与滤波完成，待到货 |
| WP9 Nano 运行时 | ✅ 框架完成（stand / sweep） |
| WP3 下半身仿真模型 | ✅ 见 `rl/`（躯干质量待实称） |
| WP8 RL 下半身策略 | 任务与部署链路已通，待在显卡上正式训练，见 `rl/README.md` |
| WP6 装配标定、WP7 保底步态 | 未开始 |

## 致谢与许可

关节命名、舵机 ID、home 姿态、1910 的寄存器用法参考了 LuwuDynamics 的
[xgoduck_hardware](https://github.com/LuwuDynamics/xgoduck_hardware)、
[xgoduck_runtime_arduino](https://github.com/LuwuDynamics/xgoduck_runtime_arduino)、
[xgoduck_rl](https://github.com/LuwuDynamics/xgoduck_rl)，以及 Pollen Robotics 的 Microduck。
Microduck 的 3D 模型为 CC BY-SA-NC，xgoduck_hardware 未声明许可证；本仓库目前只用于学习，公开发布前需核对。

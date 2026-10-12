# HangoutDuck RL（`rl` 分支）

在 [xgoduck_rl](https://github.com/LuwuDynamics/xgoduck_rl)（Microduck 配方 + 飞特 HLS1910 的 BAM 执行器模型）
的基础上，做 HangoutDuck 的行走策略：**10 个腿部关节可动，头颈保留为固定件**。对应需求规格里的 WP3（仿真模型）和 WP8（RL 策略）。

## 和 xgoduck 的区别

| | xgoduck | HangoutDuck |
|---|---|---|
| 可动关节 | 14（腿 10 + 颈/头 4） | **10**（只有腿） |
| 头颈 | 4 个舵机驱动 | **保留几何和质量，焊死在 xgoduck 的 home 角度**；不装舵机、不装传感器 |
| 模型 | `robot_walk.xml` | 同一个文件，加载时删掉 4 个颈/头关节和执行器（`hangoutduck_rl/robot.py` 的 `freeze_head`） |
| 整机质量 | 0.800 kg | 0.800 kg（躯干、头颈质量都还是 xgoduck 的，待实称） |
| 策略观测 | 61 维（含头部指令 4 + 身体姿态指令 6） | **39 维**：角速度 3、重力 3、关节角 10、关节速度 10、上一步动作 10、速度指令 3 |
| 策略动作 | 14 | **10**：目标角 = home + action |
| 奖励/随机化/课程 | — | 沿用 xgoduck，只去掉头颈关节相关的项；头颈质心随机化保留（`hangoutduck_rl/tasks.py`） |

关节顺序与 `main` 分支的 `hardware/joints.yaml` 一致（左腿 5 个，再右腿 5 个），真机运行时会从 ONNX 元数据里核对。

## 为什么保留头

头大约 0.25 kg，装在躯干前方。`scripts/check_model.py` 的对比：

| | 质心相对双脚中心 | PD 站立 3 s 后仰 |
|---|---|---|
| xgoduck（头可动） | 后方 7 mm | 1.2° |
| **HangoutDuck（头固定）** | **后方 7 mm**（与 xgoduck 完全一致） | 2.1° |
| 去掉头（旧方案，仅对比） | 后方 16 mm | 5.4° |

脚跟只在踝关节后约 20 mm，去掉头会让质心明显后移。所以仿真保留头颈，**真机也要装一个固定的头**
（打印件，不装舵机），质量和位置尽量接近仿真，否则仿真和真机对不上。

## 仿真参数：先用已知数据训练，确认后改参数重跑

所有需要按实物确认的参数集中在 **`hangoutduck_rl/sim_params.yaml`**，现在全部是 xgoduck 的已知值（`status: assumed`）：

| 参数 | 现在的值 | 影响 |
|---|---|---|
| `trunk_mass_kg` / `trunk_com_offset_mm` | xgoduck 躯干（0.238 kg，含电池和主控） | 模型质量、质心 |
| `head_mass_kg` / `head_fixed_deg` | xgoduck 头颈（0.251 kg，含 5 个舵机）/ 20°、20° | 模型质量、质心 |
| `servo_kp` | 5 | 舵机模型；**必须等于真机 `hardware/joints.yaml` 的 run_pd 第一个数** |
| `supply_voltage_v` | 7.4–8.0 V | 舵机模型（速度、力矩随电压变） |
| `command_delay_steps` | 3–6 个物理步（15–30 ms） | 指令延迟随机化 |
| `foot_friction` | 0.4–1.5 | 地面摩擦随机化 |
| `cmd_lin_vel_x` / `cmd_lin_vel_y` / `cmd_ang_vel_z` | ±0.4 / ±0.3 m/s，±1.0 rad/s | 训练的速度指令范围 |

流程：

1. 现在直接训练（下面「训练」一节），得到第一版策略 `v0-xgoduck-defaults`；
2. 某项实测或定下来后，改它的 `value`，`status` 改成 `measured` 或 `decided`，`source` 写明来源；
3. 把文件开头的 `version` 换个新名字（如 `v1-weighed`），跑 `check_model.py` 看整机质量、质心和各项状态；
4. 重新训练。新的 run 目录名和导出的 ONNX 元数据（`sim_params_version`、`sim_params_unconfirmed`、`servo_kp`）
   都带新 version，旧策略不会混淆。

质量为 `null` 表示沿用 xgoduck；填了实称值后，该部件的质量和惯量按比例缩放。如果头太轻导致质心后移，可以在头里加配重。
文件格式由主仓库的测试 `tests/test_sim_params.py` 检查。

## 环境（训练电脑：Linux + NVIDIA 显卡）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # 装 uv（已有就跳过）
git clone -b rl https://github.com/jyqx666/HangoutDuck.git
cd HangoutDuck
rl/scripts/setup.sh
```

`setup.sh` 会把 xgoduck_rl 克隆到 `rl/third_party/xgoduck_rl`、切到验证过的版本（`326d77a`）、`uv sync`，
再把 `hangoutduck_rl` 装进同一个环境，最后跑模型检查并列出任务。xgoduck_rl 的 lock 文件用的是上海交大 PyPI 镜像，国内下载一般没问题。

装好后先自检（CPU 也能跑，约 1–2 分钟，大部分时间在编译内核）：

```bash
cd rl/third_party/xgoduck_rl
uv run python ../../scripts/check_model.py   # 关节顺序、质量、质心、站立，以及仿真参数各项状态
uv run python ../../scripts/smoke_test.py    # 观测 39 维、动作 10 维、观测各段顺序
```

## 渲染图片（汇报 / 标定参考）

```bash
cd rl/third_party/xgoduck_rl
MUJOCO_GL=glfw xvfb-run -a -s "-screen 0 1920x1080x24" \
    uv run python ../../scripts/render_model.py --out ../../../docs/sim
```

生成站姿图、机械零位图（标定时要摆的姿态）、保留头与去掉头的侧视对比（红线为质心），以及 3 秒站立测试 GIF。
有显示器的电脑去掉 `xvfb-run ...` 那一段即可。仓库里已放了一份：`docs/sim/`。

## 训练

```bash
rl/scripts/train.sh                                        # Mjlab-Velocity-Flat-HangoutDuck，4096 个环境
rl/scripts/train.sh --env.scene.num-envs 2048              # 显存不够时减少环境数
TASK=Mjlab-Velocity-Rough-HangoutDuck rl/scripts/train.sh  # 不平地面

# 对照组：xgoduck 原版（带头），确认环境本身没问题
TASK=Mjlab-Velocity-Flat-XgoDuck rl/scripts/train.sh
```

上游说 4096 个环境训出可用步态约 1–2 小时。日志和 TensorBoard 在
`rl/third_party/xgoduck_rl/logs/rsl_rl/hangoutduck_velocity/<run>/`：

```bash
cd rl/third_party/xgoduck_rl
uv run tensorboard --logdir logs/rsl_rl/hangoutduck_velocity
uv run play Mjlab-Velocity-Flat-HangoutDuck                # 看最新的 checkpoint；没有显示器时自动开网页查看器（viser）
```

**展示用**：录一段 `play` 的仿真行走视频就满足验收 A8（10 关节策略在仿真里按指令走 ≥10 s）。

## 导出与上真机

训练过程中每次保存 checkpoint 都会在 run 目录里自动导出同名 `.onnx`。也可以手动导出指定的 checkpoint：

```bash
cd rl/third_party/xgoduck_rl
uv run python scripts/export.py Mjlab-Velocity-Flat-HangoutDuck \
    --checkpoint-file logs/rsl_rl/hangoutduck_velocity/<run>/model_XXXX.pt \
    --onnx-file logs/rsl_rl/hangoutduck_velocity/<run>/<run>.onnx
```

拷到 Nano 上（Nano 需要 `pip install onnxruntime`），先吊架、再落地：

```bash
hduck policy --onnx walk.onnx --imu-port /dev/hduck_imu --no-tilt-check   # 吊架上，零速度指令
hduck policy --onnx walk.onnx --imu-port /dev/hduck_imu --vx 0.1          # 落地，慢速前进
```

`hduck policy` 启动时会检查 ONNX 元数据：关节顺序、观测各段顺序必须一致，home 姿态差 1° 以内，否则拒绝运行。
真机 RL 是展示后的加分项（A9），前提是 IMU 装向（`hduck imu`）和关节方向（`hduck sweep`）都已核对。

## 已验证 / 未验证

在没有显卡的机器上（CPU）验证过：

- 头颈固定的模型能编译，只剩 10 个腿关节，质量、质心与 xgoduck 在 home 姿态完全一致；
- 任务能建出来并步进，观测 39 维、动作 10 维、各段顺序正确；
- 训练 2 次迭代正常，checkpoint 和 ONNX 正常导出，ONNX 元数据符合真机运行时的检查；
- 该 ONNX 在 `hduck --fake policy` 里以 50 Hz 跑通；
- xgoduck 原版任务同样能训练（对照）；
- `sim_params.yaml` 的质量、质心偏移、舵机 P / 电压 / 延迟、摩擦、指令范围都会写进模型和任务；
  run 目录名和 ONNX 元数据带参数 version。

**没有验证**：在显卡上完整训练出能走的策略（这里没有 GPU），以及任何 sim2real 效果。

## 目录

```
rl/
  hangoutduck_rl/sim_params.yaml  需要按实物确认的参数（现在是 xgoduck 的值）
  hangoutduck_rl/sim_params.py    读取并校验上面的文件
  hangoutduck_rl/robot.py    模型（头颈固定）、home 姿态、执行器
  hangoutduck_rl/tasks.py    任务注册：Mjlab-Velocity-{Flat,Rough}-HangoutDuck；摩擦、指令范围、ONNX 元数据
  scripts/setup.sh           准备环境
  scripts/train.sh           训练
  scripts/check_model.py     模型检查（质量、质心、站立）
  scripts/smoke_test.py      任务自检（维度、顺序）
  scripts/render_model.py    渲染模型图片和站立 GIF
  third_party/               setup.sh 克隆的 xgoduck_rl（不进版本库）
hangoutduck/runtime/policy.py  真机上运行 ONNX 策略（hduck policy）
```

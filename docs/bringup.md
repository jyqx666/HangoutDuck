# 真机上电流程（WP4 → WP6）

从驱动板到货到机器人站稳，按顺序做。每一步都有对应的验收项（编号见需求规格）。
命令里的串口按实际情况替换，所有命令加 `--fake` 可以先不接硬件练一遍。

## 0. 准备

- Nano 环境已按 `docs/jetson_nano_setup.md` 装好。
- 电源设为 **7.4 V、限流 2 A**，先只接驱动板，不接舵机。
- 吊架搭好：整机测试时机身吊起，脚离地。

## 1. 单舵机验收（A1）

总线上**只接一个舵机**。

```bash
hduck scan --all-bauds                 # 出厂一般是 ID 1、1 Mbps
hduck dump --id 1                      # 读出整张寄存器表
hduck check-speed --id 1               # 卸力，用手来回拧 5 秒
```

核对 `dump` 的结果：

- `baud_rate` 为 0（1 Mbps）。不是的话，用厂家调试软件改成 1 Mbps。
- `temp_p` / `temp_d`（地址 50/51）：xgoduck 说 HLS1910 在这里放临时 PD。如果你们的舵机读出来全是 0、写入无效，
  删掉 `hardware/joints.yaml` 里的 `run_pd`。
- 把完整输出保存到 `docs/servo_dump.txt` 提交，方便以后对照。

`check-speed` 会估计速度寄存器的单位（代码默认 50 刻度/秒）。差得多的话改
`hangoutduck/feetech/registers.py` 里的 `SPEED_UNIT_TICKS_PER_S`。

## 2. 设 ID

还是一次只接一个舵机，按 `hardware/wiring.md` 的表依次设：

```bash
hduck set-id 10        # 自动找到总线上唯一的舵机，改成 10
```

设完立刻贴标签。10 个都设完后全部串上总线：

```bash
hduck scan --ids 0-30  # 应该正好是 [10..14, 20..24]
```

## 3. 回中位、装舵盘

```bash
hduck center --id 10   # 上力转到 2048，然后卸力
```

在舵机停在 2048 时，按 xgoduck 装配手册的角度装舵盘，让关节尽量接近机械零位
（xgoduck 那台机器标定出来的零位都在 2048 ± 200 以内，见 `joints.yaml` 的 `xgoduck_ref_zero`）。

## 4. 整机装配与总线验收（A2）

```bash
hduck state                           # 10 个关节都有应答，温度电压正常
hduck bench-bus --hz 100 --seconds 60 # 丢包 < 1% 为通过
```

## 5. 零位标定（A4）

机身吊起，卸力。逐个关节把连杆摆到机械零位（和仿真里全零姿态一致），记录：

```bash
hduck calibrate --joint left_hip_yaw  # 摆好后回车
hduck calibrate --joint left_hip_roll
...                                   # 10 个关节都标完
hduck state                           # 各关节角度应接近 0
```

也可以做一个站姿夹具，一次标完：`hduck calibrate --pose home`。
标定结果在 `hardware/calibration.yaml`。机器只有一台，这个文件提交到仓库，组员共享。

## 6. 方向核对（A4）

吊架上逐个关节做小幅正弦摆动，看实际转向是否与仿真一致：

```bash
hduck sweep --joint left_knee --amp 10 --no-tilt-check
```

仿真里正角度是什么方向，以 `rl` 分支的模型为准。转反了就把 `joints.yaml` 里该关节的 `sign` 改成相反数，
然后**重新标定该关节**。

## 7. 站立（A5、A6）

```bash
hduck stand --no-tilt-check --duration 60                      # 吊架上
hduck stand --imu-port /dev/hduck_imu --duration 60            # 落地，开摔倒检测
```

结束后看输出里的 `loop`（抖动）和日志 `logs/*.npz`（温度、跟踪误差）。
运行中**按回车或 Ctrl-C 立即卸力**，温度、电压、丢包、倾角任一越界也会自动卸力。

## 出问题时

| 现象 | 先查 |
|---|---|
| scan 找不到舵机 | 电源是否打开；波特率（加 `--all-bauds`）；驱动板信号线是否接在 TTL 口 |
| 偶发 `no reply` | USB 线质量；FTDI 延迟（见 Nano 文档）；驱动板是否回显（试 `--echo`） |
| 上力瞬间抖一下 | 正常情况下不会：上力前会先把目标设成当前位置。若仍抖，降低 `run_pd` 的 P |
| 关节越走越偏 | 舵盘打滑或零位标错，重新标该关节 |

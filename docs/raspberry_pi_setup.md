# 树莓派 5 环境（WP2）

主控是**树莓派 5（1 GB）**，和电池一起装在机身上，不拴线；电脑通过 Wi-Fi 用 SSH 登录树莓派运行 `hduck`。
目标：装好 `hangoutduck`，确认供电不欠压、50 Hz 控制循环的定时精度达标（验收 A5 的一半，另一半要接上舵机再测）。

仿真、训练、sim2sim 都在电脑上做，树莓派上只跑真机程序。

## 1. 系统

用 Raspberry Pi Imager 烧 **Raspberry Pi OS Lite (64-bit)**（Bookworm，不带桌面，省内存）：

- 必须是 **64 位**：RL 策略要用的 onnxruntime 只有 64 位 ARM 的安装包；
- 在 Imager 的设置里预先填好主机名（例如 `hduck`）、用户名、Wi-Fi，并**打开 SSH**。

上电后在电脑上：

```bash
ssh <用户名>@hduck.local
uname -m            # aarch64
python3 --version   # Bookworm 自带 3.11
free -h             # 1 GB
```

## 2. 安装本项目

Bookworm 不允许往系统 Python 里直接 `pip install`，用虚拟环境：

```bash
sudo apt update
sudo apt install -y python3-venv git tmux
python3 -m venv ~/hduck-venv
source ~/hduck-venv/bin/activate

git clone https://github.com/jyqx666/HangoutDuck.git
cd HangoutDuck
pip install -e '.[dev,policy]'     # policy = onnxruntime，跑 RL 策略才需要；手写步态不需要
pytest -q                          # 全部通过即可
```

以后每次登录先 `source ~/hduck-venv/bin/activate`。

## 3. 舵机驱动板的串口

**USB 驱动板**（微雪 Bus Servo Adapter、飞特 FE-URT-1 等）：

```bash
sudo usermod -aG dialout $USER     # 重新登录后生效
ls /dev/ttyUSB* /dev/ttyACM*       # 插上驱动板后应多出设备
dmesg | tail                       # 看不到设备时查这里
```

**插在 GPIO 上的驱动扩展板**（走树莓派的硬件串口，少一根 USB 线）：在 `/boot/firmware/config.txt` 末尾加
`dtparam=uart0=on`，用 `sudo raspi-config` → Interface Options → Serial Port 关掉串口登录、打开串口硬件，
重启后设备是 `/dev/ttyAMA0`。买之前确认扩展板支持飞特 SCS/STS 协议和 1 Mbps。

## 4. 固定设备名（用 USB 设备时推荐）

插拔顺序不同，`ttyUSB0/1` 会互换。用 udev 规则按 USB 的厂商号 / 产品号固定名字：

```bash
udevadm info -a -n /dev/ttyUSB0 | grep -m3 -E 'idVendor|idProduct|serial'
```

把查到的值填进 `/etc/udev/rules.d/99-hangoutduck.rules`（只有用 USB IMU 时才需要第二行）：

```
SUBSYSTEM=="tty", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="yyyy", SYMLINK+="hduck_servo"
SUBSYSTEM=="tty", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="zzzz", SYMLINK+="hduck_imu"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

之后统一用 `--port /dev/hduck_servo`。如果驱动板是 FTDI 芯片，把 USB 串口延迟从默认 16 ms 调到 1 ms：

```bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

## 5. 供电和温度检查（装在机身上必须做）

树莓派由机身上的 5 V 降压模块从电池取电（见 `hardware/wiring.md`）。舵机突然发力时电源会往下掉，树莓派欠压会降频甚至重启；
电池快没电时更容易发生。**让舵机站立、来回摆动的同时**检查，满电和快没电的电池各测一次：

```bash
vcgencmd get_throttled     # 必须是 throttled=0x0；不是 0 说明欠压或过热过
vcgencmd measure_temp      # 长时间运行后低于 70 °C
```

欠压就换电流更大的降压模块、加粗电源线、在降压模块输入端并大电容。树莓派 5 至少要贴散热片。

## 6. 降低定时抖动

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor   # 锁最高频率（重启后失效）
```

抖动仍然偏大时，可以在 `/boot/firmware/cmdline.txt` 末尾加 `isolcpus=3` 空出一个核，再用
`taskset -c 3 chrt -f 50 hduck ...` 把控制程序固定在这个核上、用实时优先级运行。

## 7. 不接硬件的检查

```bash
hduck bench-loop --hz 50 --seconds 30        # 目标：jitter_p99_ms < 5
hduck --fake stand --duration 5              # 整条运行时链路用假舵机跑一遍
hduck --fake walk --steps 10                 # 保底步态，顺便看每周期计算时间
```

接上舵机后再跑 `hduck bench-bus`（验收 A2），流程见 `docs/bringup.md`。

## 8. 远程运行的注意事项

- 程序在树莓派上跑，**Wi-Fi 断了它不会停**。在 `tmux` 里运行（断线后 `tmux attach` 回来），
  并且**机身上的电源开关就是急停**：电池是舵机和树莓派的唯一电源，关掉它所有舵机立即卸力。
  机器人走动时要有人跟在旁边，手能够到开关；别走出 Wi-Fi 信号好的范围。
- 电池电压用到约 7.0 V（`hduck state` 看舵机测到的电压）就停下充电；低于 6.5 V 程序会自动卸力。
- 直接断电可能损坏 SD 卡上的文件，平时停止用回车或 Ctrl-C，正常关机用 `sudo poweroff`。
- 1 GB 内存跑 `hduck` 足够；不要在树莓派上装 mjlab / 训练环境。

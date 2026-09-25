# Jetson Nano 环境（WP2）

目标：在 Nano 上装好 `hangoutduck`，并确认 50 Hz 控制循环的定时精度达标（验收 A5 的一半，另一半要接上舵机再测）。

## 1. 确认系统版本

```bash
cat /etc/nv_tegra_release      # R32.x 对应 JetPack 4.x（Ubuntu 18.04）
python3 --version              # JetPack 4 自带 3.6，太旧
free -h                        # 2 GB 还是 4 GB 版本
```

把结果记到 `docs/` 下的周报里，需求规格里这是一个待确认项。

## 2. 安装 Python 3.8 和本项目

Ubuntu 18.04 的 universe 源里有 python3.8，不用自己编译：

```bash
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev git
python3.8 -m venv ~/hduck-venv
source ~/hduck-venv/bin/activate
pip install --upgrade pip          # 自带的 pip 太旧，装不了 aarch64 的 numpy wheel

git clone https://github.com/jyqx666/HangoutDuck.git
cd HangoutDuck
pip install -e '.[dev]'
pytest -q                          # 全部通过即可
```

以后每次开终端先 `source ~/hduck-venv/bin/activate`。

## 3. 串口权限

```bash
sudo usermod -aG dialout $USER     # 重新登录后生效
ls /dev/ttyUSB* /dev/ttyACM*       # 插上驱动板/IMU 后应多出设备
dmesg | tail                       # 看不到设备时查这里；WCH 芯片的板子可能要装厂家驱动
```

## 4. 固定设备名（推荐）

插拔顺序不同，`ttyUSB0/1` 会互换。用 udev 规则按 USB 的厂商号/产品号固定名字：

```bash
udevadm info -a -n /dev/ttyUSB0 | grep -m3 -E 'idVendor|idProduct|serial'
```

把查到的值填进 `/etc/udev/rules.d/99-hangoutduck.rules`（两个设备如果芯片相同，再加 `ATTRS{serial}` 区分）：

```
SUBSYSTEM=="tty", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="yyyy", SYMLINK+="hduck_servo"
SUBSYSTEM=="tty", ATTRS{idVendor}=="xxxx", ATTRS{idProduct}=="zzzz", SYMLINK+="hduck_imu"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

之后统一用 `--port /dev/hduck_servo --imu-port /dev/hduck_imu`。

## 5. 性能模式

```bash
sudo nvpmodel -m 0      # 最大功耗模式（需要 5 V 4 A 的 DC 供电，别用 micro-USB）
sudo jetson_clocks      # 锁最高频率，减少定时抖动
```

如果驱动板是 FTDI 芯片，把 USB 串口延迟从默认 16 ms 调到 1 ms：

```bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

## 6. 不接硬件的检查

```bash
hduck bench-loop --hz 50 --seconds 30        # 目标：jitter_p99_ms < 5
hduck --fake stand --duration 5              # 整条运行时链路用假舵机跑一遍
hduck --fake sweep --joint left_knee
```

接上舵机后再跑 `hduck bench-bus`（验收 A2），流程见 `docs/bringup.md`。

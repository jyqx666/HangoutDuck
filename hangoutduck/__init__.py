"""HangoutDuck 真机软件包。

分层（下层不依赖上层）：
  feetech/  飞特总线舵机协议与驱动（含可替代真机的 FakeBus）
  imu/      IMU 驱动与重力方向估计
  config    关节表 hardware/joints.yaml 与标定文件的加载
  robot     关节角（弧度）与舵机刻度之间的换算、整机读写
  runtime/  50 Hz 控制循环、安全监控、日志、控制器
  cli       hduck 命令行工具
"""

__version__ = "0.1.0"

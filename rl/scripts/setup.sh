#!/usr/bin/env bash
# 在训练电脑（Linux + NVIDIA 显卡）上准备 RL 环境：
#   1. 把 xgoduck_rl 克隆到 rl/third_party/xgoduck_rl 并切到验证过的版本
#   2. 用 uv 装好它的环境（mjlab、MuJoCo Warp、torch、BAM）
#   3. 把本目录的 hangoutduck_rl 以可编辑方式装进同一个环境，注册 HangoutDuck 任务
#   4. 把仓库根目录的真机程序 hangoutduck 也装进去（sim2sim 用）
#   5. 克隆 xgoduck 的运行时仓库，里面有预训练策略（sim2sim 自检用）
#   6. 跑一遍模型检查，列出任务
set -euo pipefail

XGODUCK_RL_REPO="${XGODUCK_RL_REPO:-https://github.com/LuwuDynamics/xgoduck_rl.git}"
# 2026-09-22 的 main，本分支的任务代码基于这个版本写成并验证
XGODUCK_RL_REV="${XGODUCK_RL_REV:-326d77a1122870bdefa2c36403937502c958e69c}"

XGODUCK_RUNTIME_REPO="${XGODUCK_RUNTIME_REPO:-https://github.com/LuwuDynamics/xgoduck_runtime_arduino.git}"
XGODUCK_RUNTIME_REV="${XGODUCK_RUNTIME_REV:-a6dc9f98eae6f8cc8f2bac81a97dde5a7a8bd0d2}"

RL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$RL_DIR/.." && pwd)"
DEST="$RL_DIR/third_party/xgoduck_rl"
RUNTIME_DEST="$RL_DIR/third_party/xgoduck_runtime_arduino"

command -v uv >/dev/null || { echo "需要 uv：curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

if [ ! -d "$DEST/.git" ]; then
    git clone "$XGODUCK_RL_REPO" "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$XGODUCK_RL_REV"

cd "$DEST"
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" uv sync
uv pip install --no-deps -e "$RL_DIR"
uv pip install --no-deps -e "$REPO_ROOT"  # 真机程序 hangoutduck（numpy、pyyaml 环境里已经有）

if [ ! -d "$RUNTIME_DEST/.git" ]; then
    git clone --quiet "$XGODUCK_RUNTIME_REPO" "$RUNTIME_DEST" || echo "（可选）xgoduck 运行时仓库克隆失败，sim2sim 的 xgoduck 自检用不了"
fi
if [ -d "$RUNTIME_DEST/.git" ]; then
    git -C "$RUNTIME_DEST" checkout --quiet "$XGODUCK_RUNTIME_REV" || true
fi

uv run python "$RL_DIR/scripts/check_model.py"
uv run python "$RL_DIR/scripts/check_gait_kinematics.py"
echo
echo "已注册的 HangoutDuck 任务："
uv run list-envs | grep -i hangoutduck || { echo "没有找到 HangoutDuck 任务，检查上面的报错"; exit 1; }

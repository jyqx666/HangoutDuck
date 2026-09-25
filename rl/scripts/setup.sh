#!/usr/bin/env bash
# 在训练电脑（Linux + NVIDIA 显卡）上准备 RL 环境：
#   1. 把 xgoduck_rl 克隆到 rl/third_party/xgoduck_rl 并切到验证过的版本
#   2. 用 uv 装好它的环境（mjlab、MuJoCo Warp、torch、BAM）
#   3. 把本目录的 hangoutduck_rl 以可编辑方式装进同一个环境，注册 HangoutDuck 任务
#   4. 跑一遍模型检查，列出任务
set -euo pipefail

XGODUCK_RL_REPO="${XGODUCK_RL_REPO:-https://github.com/LuwuDynamics/xgoduck_rl.git}"
# 2026-09-22 的 main，本分支的任务代码基于这个版本写成并验证
XGODUCK_RL_REV="${XGODUCK_RL_REV:-326d77a1122870bdefa2c36403937502c958e69c}"

RL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$RL_DIR/third_party/xgoduck_rl"

command -v uv >/dev/null || { echo "需要 uv：curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

if [ ! -d "$DEST/.git" ]; then
    git clone "$XGODUCK_RL_REPO" "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$XGODUCK_RL_REV"

cd "$DEST"
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" uv sync
uv pip install --no-deps -e "$RL_DIR"

uv run python "$RL_DIR/scripts/check_model.py"
echo
echo "已注册的 HangoutDuck 任务："
uv run list-envs | grep -i hangoutduck || { echo "没有找到 HangoutDuck 任务，检查上面的报错"; exit 1; }

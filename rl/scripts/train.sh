#!/usr/bin/env bash
# 用法：rl/scripts/train.sh [额外参数]
#   rl/scripts/train.sh                                   # 平地行走，4096 个并行环境
#   TASK=Mjlab-Velocity-Rough-HangoutDuck rl/scripts/train.sh
#   rl/scripts/train.sh --env.scene.num-envs 2048         # 显存不够时减少环境数
set -euo pipefail
RL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TASK="${TASK:-Mjlab-Velocity-Flat-HangoutDuck}"
cd "$RL_DIR/third_party/xgoduck_rl"
if [ "$#" -eq 0 ]; then
    set -- --env.scene.num-envs 4096
fi
exec uv run train "$TASK" "$@"

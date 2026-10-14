#!/usr/bin/env bash
# Stage-1 training on the single-focal human embryo dataset (SFHE).
# Usage: bash scripts/train_stage1_SFHE.sh [gpu_id] [exp_name]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="$REPO_ROOT/configs/train_stage1_SFHE.json"
GPU_ID="${1:-0}"
EXP_NAME="${2:-stage1_SFHE}"

cd "$REPO_ROOT"

echo "Running Stage-1 training"
echo "  config: $CONFIG_PATH"
echo "  gpu   : $GPU_ID"
echo "  exp   : $EXP_NAME"

python -m scripts.train_stage1 \
    --config "$CONFIG_PATH" \
    --device "$GPU_ID" \
    --exp_name "$EXP_NAME"


#!/usr/bin/env bash
# Stage-2 training on the multi-focal human embryo dataset (MFHE).
# Usage: bash scripts/train_stage2_MFHE.sh [gpu_id] [exp_name]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="$REPO_ROOT/configs/train_stage2_MFHE.json"
GPU_ID="${1:-0}"
EXP_NAME="${2:-stage2_MFHE}"

cd "$REPO_ROOT"

echo "Running Stage-2 training"
echo "  config: $CONFIG_PATH"
echo "  gpu   : $GPU_ID"
echo "  exp   : $EXP_NAME"

python -m scripts.train_stage2 \
    --config "$CONFIG_PATH" \
    --device "$GPU_ID" \
    --exp_name "$EXP_NAME"


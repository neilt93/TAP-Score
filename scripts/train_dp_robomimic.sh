#!/bin/bash
# Train Diffusion Policy (BC-RNN) on robomimic lowdim tasks.
# Usage: bash scripts/train_dp_robomimic.sh [lift|can|all]
#
# Checkpoints go to baselines/diffusion_policy/data/outputs/...
# After training, find the best checkpoint path in the output directory.
set -euo pipefail

TASK="${1:-all}"
SEED="${SEED:-42}"

DP_DIR="baselines/diffusion_policy"

if [ ! -d "$DP_DIR" ]; then
  echo "ERROR: Diffusion Policy not found at $DP_DIR"
  echo "Run: bash scripts/setup_runpod.sh"
  exit 1
fi

train_task() {
  local task_name="$1"
  echo ""
  echo "=== Training DP BC-RNN on ${task_name}_lowdim (seed=$SEED) ==="
  echo ""
  cd "$DP_DIR"
  python train.py --config-name=train_robomimic_lowdim_workspace \
    task="${task_name}_lowdim" \
    training.seed="$SEED"
  cd ../..
  echo ""
  echo "=== Done: ${task_name}_lowdim ==="
  echo "Checkpoints in: $DP_DIR/data/outputs/*${task_name}*/"
}

case "$TASK" in
  lift)
    train_task lift
    ;;
  can)
    train_task can
    ;;
  all)
    train_task lift
    train_task can
    ;;
  *)
    echo "Usage: $0 [lift|can|all]"
    exit 1
    ;;
esac

#!/bin/bash
set -euo pipefail

# Robomimic headroom audit for Lift + Can (lowdim).
# Checks whether best-of-K oracle reranking has meaningful gain over K=1.
#
# Usage:
#   LIFT_CKPT=path/to/lift.ckpt CAN_CKPT=path/to/can.ckpt \
#   bash scripts/run_robomimic_headroom_audit.sh
#
# Optional env vars:
#   N_EPISODES (default 50), K (default 8), DEVICE (default cuda),
#   DECISION_INTERVAL (default 5), SKIP_FIRST (default 2)

: "${LIFT_CKPT:?Set LIFT_CKPT to lift checkpoint path}"
: "${CAN_CKPT:?Set CAN_CKPT to can checkpoint path}"

N_EPISODES="${N_EPISODES:-50}"
K="${K:-8}"
DEVICE="${DEVICE:-cuda}"
DECISION_INTERVAL="${DECISION_INTERVAL:-5}"
SKIP_FIRST="${SKIP_FIRST:-2}"

mkdir -p eval_results

echo "=== Robomimic Headroom Audit ==="
echo "  Lift ckpt: $LIFT_CKPT"
echo "  Can ckpt:  $CAN_CKPT"
echo "  K=$K  N=$N_EPISODES  device=$DEVICE"
echo ""

echo "--- Running Lift ---"
python scripts/robomimic_headroom_audit.py \
  --dp_checkpoint "$LIFT_CKPT" \
  --n_episodes "$N_EPISODES" \
  --K "$K" \
  --decision_interval "$DECISION_INTERVAL" \
  --skip_first "$SKIP_FIRST" \
  --device "$DEVICE" \
  --output "eval_results/robomimic_headroom_lift_K${K}_n${N_EPISODES}.json"

echo ""
echo "--- Running Can ---"
python scripts/robomimic_headroom_audit.py \
  --dp_checkpoint "$CAN_CKPT" \
  --n_episodes "$N_EPISODES" \
  --K "$K" \
  --decision_interval "$DECISION_INTERVAL" \
  --skip_first "$SKIP_FIRST" \
  --device "$DEVICE" \
  --output "eval_results/robomimic_headroom_can_K${K}_n${N_EPISODES}.json"

echo ""
echo "=== Done ==="
echo "Results:"
echo "  eval_results/robomimic_headroom_lift_K${K}_n${N_EPISODES}.json"
echo "  eval_results/robomimic_headroom_can_K${K}_n${N_EPISODES}.json"
echo ""
echo "Check 'fork_decision' field in each JSON to see if reranking is worth pursuing."

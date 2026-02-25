#!/bin/bash
# Run lift and can headroom audits IN PARALLEL with reduced episodes.
# Usage: nohup bash scripts/run_parallel_audit.sh > pipeline.log 2>&1 &
set -uo pipefail
cd /workspace/TAP-Score
export MUJOCO_GL=egl

N=10
K=8

LIFT_CKPT="data/robomimic/checkpoints/lift/latest.ckpt"
CAN_CKPT="data/robomimic/checkpoints/can/latest.ckpt"

echo "=============================================="
echo "  Parallel Headroom Audit (N=$N, K=$K)"
echo "  Started: $(date)"
echo "=============================================="

mkdir -p eval_results

# Launch both in parallel
echo ""
echo "=== Launching LIFT audit (background) ==="
python scripts/robomimic_headroom_audit.py \
    --dp_checkpoint "$LIFT_CKPT" \
    --n_episodes $N --K $K \
    --decision_interval 5 --skip_first 2 \
    --device cuda \
    --output eval_results/robomimic_headroom_lift_K${K}_n${N}.json \
    > eval_results/lift_audit.log 2>&1 &
LIFT_PID=$!
echo "  PID: $LIFT_PID"

echo ""
echo "=== Launching CAN audit (background) ==="
python scripts/robomimic_headroom_audit.py \
    --dp_checkpoint "$CAN_CKPT" \
    --n_episodes $N --K $K \
    --decision_interval 5 --skip_first 2 \
    --device cuda \
    --output eval_results/robomimic_headroom_can_K${K}_n${N}.json \
    > eval_results/can_audit.log 2>&1 &
CAN_PID=$!
echo "  PID: $CAN_PID"

echo ""
echo "Waiting for both to finish..."
echo "  Monitor: tail -f eval_results/lift_audit.log"
echo "  Monitor: tail -f eval_results/can_audit.log"

LIFT_OK=0
CAN_OK=0
wait $LIFT_PID && LIFT_OK=1 || echo "LIFT audit failed (exit $?)"
wait $CAN_PID && CAN_OK=1 || echo "CAN audit failed (exit $?)"

echo ""
echo "=============================================="
echo "  COMPLETE: $(date)"
echo "=============================================="

if [ $LIFT_OK -eq 1 ]; then
    echo ""
    echo "--- LIFT results ---"
    python -c "
import json
d = json.load(open('eval_results/robomimic_headroom_lift_K${K}_n${N}.json'))
for k in ['fork_decision','summary']:
    if k in d: print(f'{k}: {json.dumps(d[k], indent=2)}')
"
else
    echo "LIFT: FAILED - check eval_results/lift_audit.log"
fi

if [ $CAN_OK -eq 1 ]; then
    echo ""
    echo "--- CAN results ---"
    python -c "
import json
d = json.load(open('eval_results/robomimic_headroom_can_K${K}_n${N}.json'))
for k in ['fork_decision','summary']:
    if k in d: print(f'{k}: {json.dumps(d[k], indent=2)}')
"
else
    echo "CAN: FAILED - check eval_results/can_audit.log"
fi

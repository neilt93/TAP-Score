#!/bin/bash
# Wait for pytorch3d build, then run the full pipeline.
# Usage: nohup bash scripts/run_after_pytorch3d.sh > pipeline.log 2>&1 &
set -euo pipefail
cd /workspace/TAP-Score
export MUJOCO_GL=egl

echo "Waiting for pytorch3d build to finish..."
while ps aux | grep -q "[p]ip install git+https://github.com/facebookresearch/pytorch3d"; do
    sleep 10
done
echo "pytorch3d build done (or not running). Verifying import..."
python -c "import pytorch3d; print('pytorch3d OK')" || {
    echo "pytorch3d import failed. Attempting install from source..."
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    python -c "import pytorch3d; print('pytorch3d OK')"
}

LIFT_CKPT="data/robomimic/checkpoints/lift/latest.ckpt"
CAN_CKPT="data/robomimic/checkpoints/can/latest.ckpt"

echo ""
echo "=============================================="
echo "  TAP-Score Robomimic Full Pipeline"
echo "  Started: $(date)"
echo "=============================================="

# Step 1: Stochasticity check (lift)
echo ""
echo "=== STEP 1/4: Stochasticity check (lift) ==="
python scripts/check_policy_stochasticity.py \
    --dp_checkpoint "$LIFT_CKPT" --K 8 --n_obs 5
echo "STEP 1 DONE: $(date)"

# Step 2: Stochasticity check (can)
echo ""
echo "=== STEP 2/4: Stochasticity check (can) ==="
python scripts/check_policy_stochasticity.py \
    --dp_checkpoint "$CAN_CKPT" --K 8 --n_obs 5
echo "STEP 2 DONE: $(date)"

# Step 3: Headroom audit (lift)
echo ""
echo "=== STEP 3/4: Headroom audit (lift) ==="
mkdir -p eval_results
python scripts/robomimic_headroom_audit.py \
    --dp_checkpoint "$LIFT_CKPT" \
    --n_episodes 50 --K 8 \
    --decision_interval 5 --skip_first 2 \
    --device cuda \
    --output eval_results/robomimic_headroom_lift_K8_n50.json
echo "STEP 3 DONE: $(date)"

# Step 4: Headroom audit (can)
echo ""
echo "=== STEP 4/4: Headroom audit (can) ==="
python scripts/robomimic_headroom_audit.py \
    --dp_checkpoint "$CAN_CKPT" \
    --n_episodes 50 --K 8 \
    --decision_interval 5 --skip_first 2 \
    --device cuda \
    --output eval_results/robomimic_headroom_can_K8_n50.json
echo "STEP 4 DONE: $(date)"

# Summary
echo ""
echo "=============================================="
echo "  PIPELINE COMPLETE: $(date)"
echo "=============================================="
echo "Results:"
echo "  eval_results/robomimic_headroom_lift_K8_n50.json"
echo "  eval_results/robomimic_headroom_can_K8_n50.json"

for f in eval_results/robomimic_headroom_*.json; do
    if [ -f "$f" ]; then
        echo ""
        echo "--- $(basename $f) ---"
        python -c "import json; d=json.load(open('$f')); print(json.dumps({k:v for k,v in d.items() if 'fork' in k.lower() or 'summary' in k.lower()}, indent=2))"
    fi
done

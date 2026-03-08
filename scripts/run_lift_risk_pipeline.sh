#!/bin/bash
# Lift risk model pipeline: train → evaluate
# Data collection runs separately; this starts after it finishes.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LIFT_CKPT="data/robomimic/checkpoints/lift_ph_diffusion_policy_cnn.ckpt"
DEVICE="cuda"
REGIMES="clean zero80 zero80_jitter dropout80_p02 dropout80_p04"

mkdir -p logs

echo "============================================================"
echo "LIFT RISK MODEL PIPELINE — $(date)"
echo "============================================================"

# ── Phase 1: Wait for collection to finish ──────────────────────
ROLLOUT_DIR="data/risk_rollouts/lift"
EXPECTED=250
echo ""
echo "=== Checking rollout data ==="
FOUND=$(ls "$ROLLOUT_DIR"/ep_*.npz 2>/dev/null | wc -l)
echo "Found $FOUND / $EXPECTED episodes"
if [ "$FOUND" -lt "$EXPECTED" ]; then
    echo "ERROR: Need at least $EXPECTED episodes. Run collect_risk_rollouts.py first."
    exit 1
fi

# ── Phase 2: Train risk model ─────────────────────────────────────
echo ""
echo "=== PHASE 2: Train Lift risk model (50 epochs) ==="
python train_risk_tap.py \
    --rollout_dir "$ROLLOUT_DIR" \
    --task lift --epochs 50 --batch_size 64 \
    --lr 1e-3 --fail_horizon 64 --hard_mining_epoch 15 \
    --action_chunk 8 \
    --regimes $REGIMES \
    --device "$DEVICE" 2>&1 | tee logs/train_risk_lift.log

# ── Phase 3: Detection evaluation ────────────────────────────────
echo ""
echo "=== PHASE 3: Evaluate Lift detection (n=50, zero_object onset=31) ==="
python scripts/eval_tap_detection.py \
    --dp_checkpoint "$LIFT_CKPT" \
    --tap_checkpoint checkpoints_contrastive/robomimic_lift_lowdim/risk_tap_best.pt \
    --risk_model \
    --n_episodes 50 \
    --perturb zero_object --perturb_start_step 31 \
    --output eval_results/risk_detection_lift_zero31_v2_n50.json \
    --device "$DEVICE" 2>&1 | tee logs/eval_risk_lift.log

echo ""
echo "============================================================"
echo "LIFT PIPELINE COMPLETE — $(date)"
echo "============================================================"
echo "Results: eval_results/risk_detection_lift_zero31_v2_n50.json"

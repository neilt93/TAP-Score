#!/bin/bash
# Lift risk model v3: only failure-prone regimes (higher fail rate)
# Previous v2 failed because dropout regimes added 0% failures, diluting signal
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LIFT_CKPT="data/robomimic/checkpoints/lift_ph_diffusion_policy_cnn.ckpt"
DEVICE="cuda"
# Only regimes that actually cause failures on Lift
REGIMES="zero80 zero80_jitter"

mkdir -p logs

echo "============================================================"
echo "LIFT HIGH-FAIL RISK PIPELINE v3 — $(date)"
echo "============================================================"
echo "Regimes: $REGIMES (no clean, no dropout — those add 0% failures)"

# ── Phase 1: Collect rollout data ────────────────────────────────
echo ""
echo "=== PHASE 1: Collect 250 Lift rollouts (high-fail regimes only) ==="
python scripts/collect_risk_rollouts.py \
    --dp_checkpoint "$LIFT_CKPT" \
    --task lift --n_episodes 250 \
    --output_dir data/risk_rollouts/lift_highfail/ \
    --regimes $REGIMES \
    --device "$DEVICE" 2>&1 | tee logs/collect_lift_highfail.log

# ── Phase 2: Train risk model ─────────────────────────────────────
echo ""
echo "=== PHASE 2: Train Lift risk model v3 (50 epochs, high-fail data) ==="
python train_risk_tap.py \
    --rollout_dir data/risk_rollouts/lift_highfail/ \
    --task lift --epochs 50 --batch_size 64 \
    --lr 1e-3 --fail_horizon 64 --hard_mining_epoch 15 \
    --action_chunk 8 \
    --regimes $REGIMES \
    --device "$DEVICE" 2>&1 | tee logs/train_risk_lift_highfail.log

# Move checkpoint so we don't overwrite v1
BEST_CKPT="checkpoints_contrastive/robomimic_lift_lowdim/risk_tap_best.pt"
V3_CKPT="checkpoints_contrastive/robomimic_lift_lowdim/risk_tap_v3_highfail.pt"
cp "$BEST_CKPT" "$V3_CKPT"
echo "Saved v3 checkpoint: $V3_CKPT"

# ── Phase 3: Detection evaluation ────────────────────────────────
echo ""
echo "=== PHASE 3: Evaluate Lift v3 detection (n=50, zero_object onset=31) ==="
python scripts/eval_tap_detection.py \
    --dp_checkpoint "$LIFT_CKPT" \
    --tap_checkpoint "$V3_CKPT" \
    --risk_model \
    --n_episodes 50 \
    --perturb zero_object --perturb_start_step 31 \
    --output eval_results/risk_detection_lift_zero31_v3_highfail_n50.json \
    --device "$DEVICE" 2>&1 | tee logs/eval_risk_lift_v3.log

echo ""
echo "============================================================"
echo "LIFT HIGH-FAIL PIPELINE COMPLETE — $(date)"
echo "============================================================"
echo "Results: eval_results/risk_detection_lift_zero31_v3_highfail_n50.json"

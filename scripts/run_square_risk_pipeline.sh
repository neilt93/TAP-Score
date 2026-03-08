#!/bin/bash
# Square risk model: full pipeline collect → train → evaluate
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SQUARE_CKPT="data/robomimic/checkpoints/square_ph_diffusion_policy_cnn.ckpt"
DEVICE="cuda"
REGIMES="clean zero80 zero80_jitter dropout80_p02 dropout80_p04"

mkdir -p logs

echo "============================================================"
echo "SQUARE RISK MODEL PIPELINE — $(date)"
echo "============================================================"

# ── Phase 1: Collect rollout data ────────────────────────────────
echo ""
echo "=== PHASE 1: Collect 250 Square rollouts ==="
python scripts/collect_risk_rollouts.py \
    --dp_checkpoint "$SQUARE_CKPT" \
    --task square --n_episodes 250 \
    --output_dir data/risk_rollouts/square/ \
    --regimes $REGIMES \
    --device "$DEVICE" 2>&1 | tee logs/collect_square.log

# ── Phase 2: Train risk model ─────────────────────────────────────
echo ""
echo "=== PHASE 2: Train Square risk model (50 epochs) ==="
python train_risk_tap.py \
    --rollout_dir data/risk_rollouts/square/ \
    --task square --epochs 50 --batch_size 64 \
    --lr 1e-3 --fail_horizon 64 --hard_mining_epoch 15 \
    --action_chunk 8 \
    --regimes $REGIMES \
    --device "$DEVICE" 2>&1 | tee logs/train_risk_square.log

# ── Phase 3: Detection evaluation ────────────────────────────────
echo ""
echo "=== PHASE 3: Evaluate Square detection (n=50, zero_object onset=80) ==="
python scripts/eval_tap_detection.py \
    --dp_checkpoint "$SQUARE_CKPT" \
    --tap_checkpoint checkpoints_contrastive/robomimic_square_lowdim/risk_tap_best.pt \
    --risk_model \
    --n_episodes 50 \
    --perturb zero_object --perturb_start_step 80 \
    --output eval_results/risk_detection_square_zero80_n50.json \
    --device "$DEVICE" 2>&1 | tee logs/eval_risk_square.log

echo ""
echo "============================================================"
echo "SQUARE PIPELINE COMPLETE — $(date)"
echo "============================================================"
echo "Results: eval_results/risk_detection_square_zero80_n50.json"

#!/bin/bash
# 10-hour task queue: runs sequentially on GPU
# Task 1: Lift high-fail pipeline (~2.5h)
# Task 2: Can n=100 eval (~1h)
# Task 3: Lift v1 n=100 eval (~1.5h)
# Task 4: Can freeze80 cross-perturbation eval (~1h)
# Task 5: Can dropout cross-perturbation eval (~1h)
# Task 6: Square n=100 eval if model exists (~1h)
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CAN_CKPT="data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt"
LIFT_CKPT="data/robomimic/checkpoints/lift_ph_diffusion_policy_cnn.ckpt"
SQUARE_CKPT="data/robomimic/checkpoints/square_ph_diffusion_policy_cnn.ckpt"
CAN_TAP="checkpoints_contrastive/robomimic_can_lowdim/risk_tap_best.pt"
LIFT_TAP_V1="checkpoints_contrastive/robomimic_lift_lowdim/risk_tap_best.pt"
SQUARE_TAP="checkpoints_contrastive/robomimic_square_lowdim/risk_tap_best.pt"
DEVICE="cuda"

mkdir -p logs eval_results

echo "============================================================"
echo "10-HOUR TASK QUEUE — $(date)"
echo "============================================================"

# ── TASK 1: Lift high-fail pipeline ────────────────────────────
echo ""
echo ">>> TASK 1/6: Lift high-fail pipeline (~2.5h)"
bash scripts/run_lift_highfail_pipeline.sh 2>&1 | tee logs/queue_task1_lift_highfail.log
echo ">>> TASK 1 COMPLETE — $(date)"

# ── TASK 2: Can n=100 eval (tighter CIs) ──────────────────────
echo ""
echo ">>> TASK 2/6: Can n=100 eval (~1h)"
python scripts/eval_tap_detection.py \
    --dp_checkpoint "$CAN_CKPT" \
    --tap_checkpoint "$CAN_TAP" \
    --risk_model \
    --n_episodes 100 \
    --perturb zero_object --perturb_start_step 80 \
    --output eval_results/risk_detection_can_zero80_n100.json \
    --device "$DEVICE" 2>&1 | tee logs/queue_task2_can_n100.log
echo ">>> TASK 2 COMPLETE — $(date)"

# ── TASK 3: Lift v1 n=100 eval (tighter CIs) ──────────────────
echo ""
echo ">>> TASK 3/6: Lift v1 n=100 eval (~1.5h)"
python scripts/eval_tap_detection.py \
    --dp_checkpoint "$LIFT_CKPT" \
    --tap_checkpoint "$LIFT_TAP_V1" \
    --risk_model \
    --n_episodes 100 \
    --perturb zero_object --perturb_start_step 31 \
    --output eval_results/risk_detection_lift_zero31_v1_n100.json \
    --device "$DEVICE" 2>&1 | tee logs/queue_task3_lift_n100.log
echo ">>> TASK 3 COMPLETE — $(date)"

# ── TASK 4: Can freeze80 cross-perturbation ────────────────────
echo ""
echo ">>> TASK 4/6: Can freeze_object eval (~1h)"
python scripts/eval_tap_detection.py \
    --dp_checkpoint "$CAN_CKPT" \
    --tap_checkpoint "$CAN_TAP" \
    --risk_model \
    --n_episodes 50 \
    --perturb freeze_object --perturb_start_step 80 \
    --output eval_results/risk_detection_can_freeze80_n50.json \
    --device "$DEVICE" 2>&1 | tee logs/queue_task4_can_freeze.log
echo ">>> TASK 4 COMPLETE — $(date)"

# ── TASK 5: Can dropout cross-perturbation ─────────────────────
echo ""
echo ">>> TASK 5/6: Can intermittent_dropout eval (~1h)"
python scripts/eval_tap_detection.py \
    --dp_checkpoint "$CAN_CKPT" \
    --tap_checkpoint "$CAN_TAP" \
    --risk_model \
    --n_episodes 50 \
    --perturb intermittent_dropout --perturb_start_step 80 \
    --output eval_results/risk_detection_can_dropout80_n50.json \
    --device "$DEVICE" 2>&1 | tee logs/queue_task5_can_dropout.log
echo ">>> TASK 5 COMPLETE — $(date)"

# ── TASK 6: Square n=100 eval (if model exists) ───────────────
echo ""
echo ">>> TASK 6/6: Square n=100 eval (~1h)"
if [ -f "$SQUARE_TAP" ]; then
    python scripts/eval_tap_detection.py \
        --dp_checkpoint "$SQUARE_CKPT" \
        --tap_checkpoint "$SQUARE_TAP" \
        --risk_model \
        --n_episodes 100 \
        --perturb zero_object --perturb_start_step 80 \
        --output eval_results/risk_detection_square_zero80_n100.json \
        --device "$DEVICE" 2>&1 | tee logs/queue_task6_square_n100.log
    echo ">>> TASK 6 COMPLETE — $(date)"
else
    echo ">>> TASK 6 SKIPPED — Square model not found at $SQUARE_TAP"
fi

echo ""
echo "============================================================"
echo "10-HOUR QUEUE COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Results generated:"
ls -la eval_results/risk_detection_*.json 2>/dev/null

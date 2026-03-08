#!/usr/bin/env bash
# Can perturbation onset sweep — zero_object and freeze_object
# Matches the Lift sweep points for cross-task comparison.
set -euo pipefail

CKPT="data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt"
COMMON="--n_episodes 20 --K 1 --decision_interval 999 --skip_first 999999 \
        --action_mode abs --reset_to_dataset_init \
        --no-eval_oracle_policy --no-log_candidates --kp 500"

echo "=== Can perturbation onset sweep ==="
echo "Started: $(date)"
echo ""

# Clean baseline
echo "--- Can clean (n=20) ---"
python scripts/robomimic_headroom_audit.py --dp_checkpoint "$CKPT" $COMMON \
    --output eval_results/perturb_can_clean.json

# zero_object onset sweep
for ONSET in 0 25 50 75 100 150; do
    echo ""
    echo "--- Can zero_object onset=$ONSET ---"
    python scripts/robomimic_headroom_audit.py --dp_checkpoint "$CKPT" $COMMON \
        --perturb zero_object --perturb_start_step "$ONSET" \
        --output "eval_results/perturb_can_zero${ONSET}.json"
done

# freeze_object onset sweep
for ONSET in 0 25 50 75 100; do
    echo ""
    echo "--- Can freeze_object onset=$ONSET ---"
    python scripts/robomimic_headroom_audit.py --dp_checkpoint "$CKPT" $COMMON \
        --perturb freeze_object --perturb_start_step "$ONSET" \
        --output "eval_results/perturb_can_freeze${ONSET}.json"
done

echo ""
echo "=== Sweep complete ==="
echo "Finished: $(date)"

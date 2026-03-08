#!/bin/bash
# Comprehensive perturbation sweep for Lift and Can.
# Estimated runtime: ~10 hours on RTX 4080.
# All runs use --resume so already-completed conditions skip instantly.
set -e

cd "$(dirname "$0")/.."

LIFT_CKPT="data/robomimic/checkpoints/lift_ph_diffusion_policy_cnn.ckpt"
CAN_CKPT="data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt"
COMMON="--K 1 --decision_interval 999 --skip_first 999999 --action_mode abs --reset_to_dataset_init --no-eval_oracle_policy --no-log_candidates --kp 500 --resume"

run() {
    local label="$1"; shift
    # Find --output value to check if already done.
    local outfile=""
    local args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[$i]}" == "--output" ]]; then
            outfile="${args[$((i+1))]}"
            break
        fi
    done
    # Skip if output file already exists (completed in a previous sweep).
    if [[ -n "$outfile" && -f "$outfile" ]]; then
        echo "  $(date '+%H:%M:%S')  $label — SKIP (already done)"
        return 0
    fi
    echo ""
    echo "======================================================================"
    echo "  $(date '+%H:%M:%S')  $label"
    echo "======================================================================"
    python scripts/robomimic_headroom_audit.py "$@" $COMMON
    echo "  >>> $label DONE at $(date '+%H:%M:%S')"
}

echo "Perturbation sweep started at $(date)"
echo "======================================================================"

# ── LIFT: Clean baselines ────────────────────────────────────────────
run "Lift clean (n=20)" \
    --dp_checkpoint "$LIFT_CKPT" --n_episodes 20 --perturb none \
    --output eval_results/perturb_lift_clean.json

run "Lift clean (n=50)" \
    --dp_checkpoint "$LIFT_CKPT" --n_episodes 50 --perturb none \
    --output eval_results/perturb_lift_clean_n50.json

# ── LIFT: Zero object (onset sweep) ─────────────────────────────────
for step in 0 25 50 100 150; do
    run "Lift zero@${step} (n=20)" \
        --dp_checkpoint "$LIFT_CKPT" --n_episodes 20 \
        --perturb zero_object --perturb_start_step $step \
        --output eval_results/perturb_lift_zero${step}.json
done

run "Lift zero@0 (n=50)" \
    --dp_checkpoint "$LIFT_CKPT" --n_episodes 50 \
    --perturb zero_object --perturb_start_step 0 \
    --output eval_results/perturb_lift_zero0_n50.json

# ── LIFT: Freeze object (onset sweep) ───────────────────────────────
for step in 0 25 50 100 150; do
    run "Lift freeze@${step} (n=20)" \
        --dp_checkpoint "$LIFT_CKPT" --n_episodes 20 \
        --perturb freeze_object --perturb_start_step $step \
        --output eval_results/perturb_lift_freeze${step}.json
done

# ── LIFT: Noise (severity sweep) ────────────────────────────────────
for std in 0.01 0.02 0.05 0.1 0.2 0.5; do
    tag=$(echo $std | tr '.' '')
    run "Lift noise std=${std} (n=20)" \
        --dp_checkpoint "$LIFT_CKPT" --n_episodes 20 \
        --perturb noise_object --perturb_noise_std $std \
        --output eval_results/perturb_lift_noise${tag}.json
done

run "Lift noise std=0.05 (n=50)" \
    --dp_checkpoint "$LIFT_CKPT" --n_episodes 50 \
    --perturb noise_object --perturb_noise_std 0.05 \
    --output eval_results/perturb_lift_noise005_n50.json

# ── CAN: Clean baselines ────────────────────────────────────────────
run "Can clean (n=20)" \
    --dp_checkpoint "$CAN_CKPT" --n_episodes 20 --perturb none \
    --output eval_results/perturb_can_clean.json

run "Can clean (n=50)" \
    --dp_checkpoint "$CAN_CKPT" --n_episodes 50 --perturb none \
    --output eval_results/perturb_can_clean_n50.json

# ── CAN: Zero object (onset sweep) ──────────────────────────────────
for step in 0 25 50 100 150; do
    run "Can zero@${step} (n=20)" \
        --dp_checkpoint "$CAN_CKPT" --n_episodes 20 \
        --perturb zero_object --perturb_start_step $step \
        --output eval_results/perturb_can_zero${step}.json
done

run "Can zero@0 (n=50)" \
    --dp_checkpoint "$CAN_CKPT" --n_episodes 50 \
    --perturb zero_object --perturb_start_step 0 \
    --output eval_results/perturb_can_zero0_n50.json

# ── CAN: Freeze object (onset sweep) ────────────────────────────────
for step in 0 25 50 100 150; do
    run "Can freeze@${step} (n=20)" \
        --dp_checkpoint "$CAN_CKPT" --n_episodes 20 \
        --perturb freeze_object --perturb_start_step $step \
        --output eval_results/perturb_can_freeze${step}.json
done

# ── CAN: Noise (severity sweep) ─────────────────────────────────────
for std in 0.01 0.02 0.05 0.1 0.2 0.5; do
    tag=$(echo $std | tr '.' '')
    run "Can noise std=${std} (n=20)" \
        --dp_checkpoint "$CAN_CKPT" --n_episodes 20 \
        --perturb noise_object --perturb_noise_std $std \
        --output eval_results/perturb_can_noise${tag}.json
done

run "Can noise std=0.05 (n=50)" \
    --dp_checkpoint "$CAN_CKPT" --n_episodes 50 \
    --perturb noise_object --perturb_noise_std 0.05 \
    --output eval_results/perturb_can_noise005_n50.json

echo ""
echo "======================================================================"
echo "  SWEEP COMPLETE at $(date)"
echo "======================================================================"

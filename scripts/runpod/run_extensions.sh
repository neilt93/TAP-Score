#!/bin/bash
# Run all TAP-Score extensions on RunPod:
#   1. Train obs-only baselines (Can + Lift)
#   2. Evaluate obs-only detection (n=100 each)
#   3. Run intervention evals (baseline, resample K=4, restart R=2)
#
# Prerequisites:
#   - setup_runpod.sh already run
#   - Rollout data exists in data/risk_rollouts/{can,lift}/
#   - Existing risk model checkpoints in checkpoints_contrastive/
#
# Usage: bash scripts/runpod/run_extensions.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

CAN_CKPT="data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt"
LIFT_CKPT="data/robomimic/checkpoints/lift_ph_diffusion_policy_cnn.ckpt"
CAN_TAP="checkpoints_contrastive/robomimic_can_lowdim/risk_tap_best.pt"
LIFT_TAP="checkpoints_contrastive/robomimic_lift_lowdim/risk_tap_best.pt"
CAN_TAP_OBS="checkpoints_contrastive/robomimic_can_lowdim/risk_tap_obs_only_best.pt"
LIFT_TAP_OBS="checkpoints_contrastive/robomimic_lift_lowdim/risk_tap_obs_only_best.pt"
DEVICE="cuda"

mkdir -p logs eval_results

echo "============================================================"
echo "TAP-SCORE EXTENSIONS PIPELINE — $(date)"
echo "============================================================"

# ── Preflight checks ─────────────────────────────────────────────
echo ""
echo "=== Preflight Checks ==="
CAN_ROLLOUTS=$(ls data/risk_rollouts/can/ep_*.npz 2>/dev/null | wc -l)
LIFT_ROLLOUTS=$(ls data/risk_rollouts/lift/ep_*.npz 2>/dev/null | wc -l)
echo "  Can rollouts:  $CAN_ROLLOUTS"
echo "  Lift rollouts: $LIFT_ROLLOUTS"
echo "  Can DP ckpt:   $(test -f "$CAN_CKPT" && echo 'OK' || echo 'MISSING')"
echo "  Lift DP ckpt:  $(test -f "$LIFT_CKPT" && echo 'OK' || echo 'MISSING')"
echo "  Can TAP ckpt:  $(test -f "$CAN_TAP" && echo 'OK' || echo 'MISSING')"
echo "  Lift TAP ckpt: $(test -f "$LIFT_TAP" && echo 'OK' || echo 'MISSING')"

if [ "$CAN_ROLLOUTS" -lt 200 ]; then
    echo "ERROR: Need >= 200 Can rollouts for training. Have $CAN_ROLLOUTS."
    echo "Run: python scripts/collect_risk_rollouts.py --task can --n_episodes 500"
    exit 1
fi
if [ "$LIFT_ROLLOUTS" -lt 100 ]; then
    echo "ERROR: Need >= 100 Lift rollouts for training. Have $LIFT_ROLLOUTS."
    echo "Run: python scripts/collect_risk_rollouts.py --task lift --n_episodes 250"
    exit 1
fi

# ── 1. Train obs-only baselines ──────────────────────────────────
echo ""
echo "============================================================"
echo "  STEP 1: Train obs-only baselines"
echo "============================================================"

REGIMES="clean zero80 zero80_jitter dropout80_p02 dropout80_p04"

if [ -f "$CAN_TAP_OBS" ]; then
    echo "  Can obs-only already trained, skipping."
else
    echo ""
    echo "--- Training Can obs-only (onset_window, 50 epochs) ---"
    python train_risk_tap.py \
        --rollout_dir data/risk_rollouts/can/ \
        --task can --epochs 50 --batch_size 64 \
        --lr 1e-3 --fail_horizon 64 --hard_mining_epoch 15 \
        --action_chunk 8 \
        --label_mode onset_window \
        --regimes $REGIMES \
        --obs_only \
        --device "$DEVICE" 2>&1 | tee logs/train_obs_only_can.log
fi

if [ -f "$LIFT_TAP_OBS" ]; then
    echo "  Lift obs-only already trained, skipping."
else
    echo ""
    echo "--- Training Lift obs-only (episode_window, balanced, 50 epochs) ---"
    python train_risk_tap.py \
        --rollout_dir data/risk_rollouts/lift/ \
        --task lift --epochs 50 --batch_size 64 \
        --lr 1e-3 --fail_horizon 128 --hard_mining_epoch 15 \
        --action_chunk 8 \
        --label_mode episode_window \
        --balanced --post_onset_only \
        --regimes $REGIMES \
        --obs_only \
        --device "$DEVICE" 2>&1 | tee logs/train_obs_only_lift.log
fi

# ── 2. Evaluate obs-only detection ───────────────────────────────
echo ""
echo "============================================================"
echo "  STEP 2: Evaluate obs-only detection (n=100)"
echo "============================================================"

CAN_OBS_OUT="eval_results/detection_can_obs_only_n100"
LIFT_OBS_OUT="eval_results/detection_lift_obs_only_n100"

if [ -f "${CAN_OBS_OUT}.json" ]; then
    echo "  Can obs-only eval already done, skipping."
else
    echo ""
    echo "--- Can obs-only detection (n=100, zero80 onset=80) ---"
    python scripts/eval_tap_detection.py \
        --dp_checkpoint "$CAN_CKPT" \
        --tap_checkpoint "$CAN_TAP_OBS" \
        --risk_model \
        --n_episodes 100 \
        --perturb zero_object --perturb_start_step 80 \
        --output "${CAN_OBS_OUT}.csv" \
        --device "$DEVICE" 2>&1 | tee logs/eval_obs_only_can.log
fi

if [ -f "${LIFT_OBS_OUT}.json" ]; then
    echo "  Lift obs-only eval already done, skipping."
else
    echo ""
    echo "--- Lift obs-only detection (n=100, zero31 onset=31) ---"
    python scripts/eval_tap_detection.py \
        --dp_checkpoint "$LIFT_CKPT" \
        --tap_checkpoint "$LIFT_TAP_OBS" \
        --risk_model \
        --n_episodes 100 \
        --perturb zero_object --perturb_start_step 31 \
        --output "${LIFT_OBS_OUT}.csv" \
        --device "$DEVICE" 2>&1 | tee logs/eval_obs_only_lift.log
fi

# ── 3. Intervention evals ────────────────────────────────────────
echo ""
echo "============================================================"
echo "  STEP 3: Intervention evaluation"
echo "============================================================"

run_intervention() {
    local TASK="$1" DP="$2" TAP="$3" ONSET="$4" LABEL="$5" OUT="$6"
    shift 6  # remaining args are mode-specific flags

    if [ -f "$OUT" ]; then
        echo "  $TASK $LABEL already done, skipping."
        return 0
    fi
    echo ""
    echo "--- $TASK $LABEL (n=100) ---"
    python scripts/eval_intervention.py \
        --dp_checkpoint "$DP" \
        --tap_checkpoint "$TAP" \
        --task "$TASK" \
        --n_episodes 100 \
        --perturb zero_object --perturb_start_step "$ONSET" \
        --output "$OUT" \
        "$@" \
        --device "$DEVICE" 2>&1 | tee "logs/intervention_${TASK}_${LABEL}.log"
}

# Can interventions
run_intervention can "$CAN_CKPT" "$CAN_TAP" 80 baseline \
    eval_results/intervention_can_baseline_n100.json \
    --mode baseline

run_intervention can "$CAN_CKPT" "$CAN_TAP" 80 resample_K4 \
    eval_results/intervention_can_resample_K4_n100.json \
    --mode resample --K 4 --risk_threshold 0.5

run_intervention can "$CAN_CKPT" "$CAN_TAP" 80 restart_R2 \
    eval_results/intervention_can_restart_R2_n100.json \
    --mode restart --max_restarts 2 --risk_threshold 0.5 --risk_window 3

# Lift interventions
run_intervention lift "$LIFT_CKPT" "$LIFT_TAP" 31 baseline \
    eval_results/intervention_lift_baseline_n100.json \
    --mode baseline

run_intervention lift "$LIFT_CKPT" "$LIFT_TAP" 31 resample_K4 \
    eval_results/intervention_lift_resample_K4_n100.json \
    --mode resample --K 4 --risk_threshold 0.5

run_intervention lift "$LIFT_CKPT" "$LIFT_TAP" 31 restart_R2 \
    eval_results/intervention_lift_restart_R2_n100.json \
    --mode restart --max_restarts 2 --risk_threshold 0.5 --risk_window 3

# ── 4. Analysis ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  STEP 4: Analysis"
echo "============================================================"

echo "--- Failure density analysis ---"
python scripts/analyze_failure_density.py 2>&1 | tee logs/failure_density.log

echo ""
echo "--- Selective execution (with obs-only if available) ---"
OBS_ARGS=""
if [ -f "${CAN_OBS_OUT}.json" ]; then
    OBS_ARGS="$OBS_ARGS --can_obs_only ${CAN_OBS_OUT}.json"
fi
if [ -f "${LIFT_OBS_OUT}.json" ]; then
    OBS_ARGS="$OBS_ARGS --lift_obs_only ${LIFT_OBS_OUT}.json"
fi
python scripts/analyze_selective_execution.py $OBS_ARGS 2>&1 | tee logs/selective_execution.log

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  RESULTS SUMMARY"
echo "============================================================"
python3 -c "
import json, os

print()
print('=== Obs-Only Baseline ===')
for task, path in [('Can', 'eval_results/detection_can_obs_only_n100.json'),
                   ('Lift', 'eval_results/detection_lift_obs_only_n100.json')]:
    if os.path.exists(path):
        d = json.load(open(path))
        print(f'  {task}: AUROC(mean)={d.get(\"auroc_mean\",\"?\"):.3f}  '
              f'n={d[\"n_episodes\"]} ({d[\"n_success\"]} ok, {d[\"n_fail\"]} fail)')
    else:
        print(f'  {task}: not found')

print()
print('=== Intervention ===')
for task in ['can', 'lift']:
    for mode in ['baseline', 'resample_K4', 'restart_R2']:
        path = f'eval_results/intervention_{task}_{mode}_n100.json'
        if os.path.exists(path):
            d = json.load(open(path))
            extra = ''
            if 'total_resamples' in d:
                extra = f'  resamples={d[\"total_resamples\"]}'
            elif 'total_restarts' in d:
                extra = f'  restarts={d[\"total_restarts\"]}'
            print(f'  {task:4s} {mode:14s}: {d[\"success_rate\"]:.1%} '
                  f'({d[\"n_success\"]}/{d[\"n_episodes\"]}){extra}')
        else:
            print(f'  {task:4s} {mode:14s}: not found')
"

echo ""
echo "============================================================"
echo "EXTENSIONS PIPELINE COMPLETE — $(date)"
echo "============================================================"
echo "Copy results back with:"
echo "  scp -r eval_results/detection_*obs_only* eval_results/intervention_* artifacts/ <local>:TAP-Score/"

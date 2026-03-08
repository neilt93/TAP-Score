#!/bin/bash
# Essential Can perturbation experiments for cross-task story.
# First pass: coarse onset grid (0, 25, 50, 75, 100).
# Second pass: adaptive densification around the cliff.
set -e

cd "$(dirname "$0")/.."

CAN_CKPT="data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt"
COMMON="--K 1 --decision_interval 999 --skip_first 999999 --action_mode abs --reset_to_dataset_init --no-eval_oracle_policy --no-log_candidates --kp 500 --resume"

run() {
    local label="$1"; shift
    # Find --output value
    local outfile=""
    local args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[$i]}" == "--output" ]]; then
            outfile="${args[$((i+1))]}"
            break
        fi
    done
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

report() {
    echo ""
    echo "======================================================================"
    echo "  RESULTS SO FAR"
    echo "======================================================================"
    python -c "
import json, os, numpy as np

results = []
for f in sorted(os.listdir('eval_results')):
    if not f.startswith('perturb_can_') or not f.endswith('.json') or '.ckpt.' in f:
        continue
    try:
        d = json.load(open(os.path.join('eval_results', f)))
        bl = d.get('baseline_episodes', d.get('episodes', []))
        planned = d.get('config', {}).get('n_episodes', len(bl))
        n = len(bl)
        sr = sum(1 for e in bl if e['episode_success'])
        mr = np.array([e['episode_return'] for e in bl]).mean() if n > 0 else 0
        results.append((f, planned, n, sr, mr))
    except: pass

print('{:<45} {:>12} {:>12} {:>10}'.format('Condition', 'Completed', 'Success', 'Mean Ret'))
print('-' * 82)
for f, planned, n, sr, mr in sorted(results):
    pct = 100*sr/n if n > 0 else 0
    print('{:<45} {:>4}/{:<4}    {:>4}/{:<4} ({:3.0f}%) {:>8.1f}'.format(f, n, planned, sr, n, pct, mr))
"
}

echo "Can essential experiments started at $(date)"
echo "======================================================================"

# ── A) Can clean sanity ──────────────────────────────────────────────
run "Can clean (n=20)" \
    --dp_checkpoint "$CAN_CKPT" --n_episodes 20 --perturb none \
    --output eval_results/perturb_can_clean.json

# ── B) Can zero onset — first pass ──────────────────────────────────
for step in 0 25 50 75 100; do
    run "Can zero@${step} (n=20)" \
        --dp_checkpoint "$CAN_CKPT" --n_episodes 20 \
        --perturb zero_object --perturb_start_step $step \
        --output eval_results/perturb_can_zero${step}.json
done

# ── C) Can freeze onset — first pass ────────────────────────────────
for step in 0 25 50 75 100; do
    run "Can freeze@${step} (n=20)" \
        --dp_checkpoint "$CAN_CKPT" --n_episodes 20 \
        --perturb freeze_object --perturb_start_step $step \
        --output eval_results/perturb_can_freeze${step}.json
done

report

# ── Adaptive densification ──────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  ADAPTIVE DENSIFICATION"
echo "======================================================================"

python -c "
import json, os, numpy as np

def get_sr(fname):
    try:
        d = json.load(open(os.path.join('eval_results', fname)))
        bl = d.get('baseline_episodes', d.get('episodes', []))
        n = len(bl)
        return sum(1 for e in bl if e['episode_success']) / max(n, 1)
    except:
        return None

for ptype in ('zero', 'freeze'):
    onsets = [0, 25, 50, 75, 100]
    srs = []
    for s in onsets:
        sr = get_sr(f'perturb_can_{ptype}{s}.json')
        srs.append(sr if sr is not None else -1)

    # Find adjacent pairs with biggest jump (cliff)
    densify = []
    for i in range(len(onsets) - 1):
        if srs[i] < 0 or srs[i+1] < 0:
            continue
        gap = abs(srs[i+1] - srs[i])
        if gap >= 0.2:  # 20pp+ jump = cliff
            lo, hi = onsets[i], onsets[i+1]
            step_size = (hi - lo) // 6
            if step_size < 1:
                step_size = 1
            for s in range(lo + step_size, hi, step_size):
                densify.append(s)

    if densify:
        print(f'DENSIFY_{ptype.upper()}={\" \".join(str(s) for s in densify)}')
    else:
        print(f'DENSIFY_{ptype.upper()}=')
" > /tmp/densify_vars.sh

source /tmp/densify_vars.sh

if [[ -n "$DENSIFY_ZERO" ]]; then
    echo "  Zero cliff densification: $DENSIFY_ZERO"
    for step in $DENSIFY_ZERO; do
        run "Can zero@${step} densify (n=10)" \
            --dp_checkpoint "$CAN_CKPT" --n_episodes 10 \
            --perturb zero_object --perturb_start_step $step \
            --output eval_results/perturb_can_zero${step}.json
    done
else
    echo "  No zero cliff found (no adjacent pair with 20pp+ gap)"
fi

if [[ -n "$DENSIFY_FREEZE" ]]; then
    echo "  Freeze cliff densification: $DENSIFY_FREEZE"
    for step in $DENSIFY_FREEZE; do
        run "Can freeze@${step} densify (n=10)" \
            --dp_checkpoint "$CAN_CKPT" --n_episodes 10 \
            --perturb freeze_object --perturb_start_step $step \
            --output eval_results/perturb_can_freeze${step}.json
    done
else
    echo "  No freeze cliff found (no adjacent pair with 20pp+ gap)"
fi

report

echo ""
echo "======================================================================"
echo "  CAN ESSENTIAL SWEEP COMPLETE at $(date)"
echo "======================================================================"

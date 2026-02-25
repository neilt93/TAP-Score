#!/bin/bash
# Run headroom audit with 1 process per episode across all CPU cores.
# 256 cores means we can run all episodes simultaneously.
# Usage: nohup bash scripts/run_massively_parallel_audit.sh > pipeline.log 2>&1 &
set -uo pipefail
cd /workspace/TAP-Score
export MUJOCO_GL=egl

N=10  # episodes per task
K=8

LIFT_CKPT="data/robomimic/checkpoints/lift/latest.ckpt"
CAN_CKPT="data/robomimic/checkpoints/can/latest.ckpt"

echo "=============================================="
echo "  Massively Parallel Headroom Audit"
echo "  N=$N episodes/task, K=$K, $(nproc) CPU cores"
echo "  Started: $(date)"
echo "=============================================="

mkdir -p eval_results/episodes

PIDS=()

# Launch N processes for lift (1 episode each)
for i in $(seq 0 $((N-1))); do
    python scripts/robomimic_headroom_audit.py \
        --dp_checkpoint "$LIFT_CKPT" \
        --n_episodes 1 --K $K \
        --seed_offset $i \
        --decision_interval 5 --skip_first 2 \
        --device cuda \
        --output "eval_results/episodes/lift_ep${i}.json" \
        > "eval_results/episodes/lift_ep${i}.log" 2>&1 &
    PIDS+=($!)
    echo "  Lift ep $i -> PID $!"
done

# Launch N processes for can (1 episode each)
for i in $(seq 0 $((N-1))); do
    python scripts/robomimic_headroom_audit.py \
        --dp_checkpoint "$CAN_CKPT" \
        --n_episodes 1 --K $K \
        --seed_offset $i \
        --decision_interval 5 --skip_first 2 \
        --device cuda \
        --output "eval_results/episodes/can_ep${i}.json" \
        > "eval_results/episodes/can_ep${i}.log" 2>&1 &
    PIDS+=($!)
    echo "  Can ep $i -> PID $!"
done

echo ""
echo "Launched ${#PIDS[@]} parallel workers. Waiting..."
echo "  Monitor: ls eval_results/episodes/*.json | wc -l"

# Wait for all
FAILED=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAILED=$((FAILED+1))
done

echo ""
echo "All workers done. Failed: $FAILED"
echo "Finished: $(date)"

# Merge results
echo ""
echo "=== Merging results ==="
python3 -c "
import json, glob, numpy as np

for task in ['lift', 'can']:
    files = sorted(glob.glob(f'eval_results/episodes/{task}_ep*.json'))
    if not files:
        print(f'{task}: NO RESULTS')
        continue

    all_episodes = []
    for f in files:
        try:
            d = json.load(open(f))
            if 'episodes' in d:
                all_episodes.extend(d['episodes'])
            elif 'baseline_episodes' in d:
                all_episodes.extend(d['baseline_episodes'])
        except Exception as e:
            print(f'  Warning: {f}: {e}')

    if not all_episodes:
        print(f'{task}: no episode data found in {len(files)} files')
        continue

    successes = [ep.get('episode_success', 0) for ep in all_episodes]
    returns = [ep.get('episode_return', 0) for ep in all_episodes]
    spreads = [ep.get('mean_candidate_spread', 0) for ep in all_episodes]

    # Try to get oracle headroom
    oracle_gains = []
    for ep in all_episodes:
        if 'oracle_gain_over_baseline' in ep:
            oracle_gains.append(ep['oracle_gain_over_baseline'])
        elif 'decisions' in ep:
            for dec in ep.get('decisions', []):
                if 'oracle_gain' in dec:
                    oracle_gains.append(dec['oracle_gain'])

    merged = {
        'task': task,
        'n_episodes': len(all_episodes),
        'success_rate': float(np.mean(successes)),
        'mean_return': float(np.mean(returns)),
        'mean_spread': float(np.mean(spreads)) if spreads else 0,
        'episodes': all_episodes,
    }
    if oracle_gains:
        merged['mean_oracle_gain'] = float(np.mean(oracle_gains))

    out = f'eval_results/robomimic_headroom_{task}_K${K}_n${N}.json'
    json.dump(merged, open(out, 'w'), indent=2)
    print(f'{task}: {len(all_episodes)} episodes, success={np.mean(successes):.2f}, return={np.mean(returns):.2f}, spread={np.mean(spreads):.4f}')
    if oracle_gains:
        print(f'  oracle_gain={np.mean(oracle_gains):.4f}')
"

echo ""
echo "=============================================="
echo "  PIPELINE COMPLETE: $(date)"
echo "=============================================="
echo "Results:"
echo "  eval_results/robomimic_headroom_lift_K${K}_n${N}.json"
echo "  eval_results/robomimic_headroom_can_K${K}_n${N}.json"

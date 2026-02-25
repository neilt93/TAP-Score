#!/bin/bash
# Run headroom audit: 8 workers at a time (GPU limit), two waves.
# Wave 1: lift 0-4, can 0-4 (8 workers)
# Wave 2: lift 5-9, can 5-9 (8 workers, wait for 2 to finish first)
# Usage: nohup bash scripts/run_batched_audit.sh > pipeline.log 2>&1 &
set -uo pipefail
cd /workspace/TAP-Score
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N=10
K=8
BATCH=8  # max concurrent GPU workers

LIFT_CKPT="data/robomimic/checkpoints/lift/latest.ckpt"
CAN_CKPT="data/robomimic/checkpoints/can/latest.ckpt"

echo "=============================================="
echo "  Batched Headroom Audit (N=$N, K=$K, batch=$BATCH)"
echo "  Started: $(date)"
echo "=============================================="

mkdir -p eval_results/episodes

run_episode() {
    local task=$1
    local ckpt=$2
    local seed=$3
    python scripts/robomimic_headroom_audit.py \
        --dp_checkpoint "$ckpt" \
        --n_episodes 1 --K $K \
        --seed_offset $seed \
        --decision_interval 5 --skip_first 2 \
        --device cuda \
        --output "eval_results/episodes/${task}_ep${seed}.json" \
        > "eval_results/episodes/${task}_ep${seed}.log" 2>&1
}

# Wave 1: 4 lift + 4 can = 8 workers
echo ""
echo "=== Wave 1: lift 0-3, can 0-3 (8 workers) ==="
PIDS=()
for i in 0 1 2 3; do
    run_episode lift "$LIFT_CKPT" $i &
    PIDS+=($!)
    echo "  lift ep $i -> PID $!"
done
for i in 0 1 2 3; do
    run_episode can "$CAN_CKPT" $i &
    PIDS+=($!)
    echo "  can ep $i -> PID $!"
done

echo "Waiting for wave 1..."
for pid in "${PIDS[@]}"; do wait $pid; done
echo "Wave 1 done: $(date)"

# Wave 2: remaining episodes
echo ""
echo "=== Wave 2: lift 4-9, can 4-9 ==="
# Split into 2 sub-waves of 8
PIDS=()
for i in 4 5 6 7; do
    run_episode lift "$LIFT_CKPT" $i &
    PIDS+=($!)
    echo "  lift ep $i -> PID $!"
done
for i in 4 5 6 7; do
    run_episode can "$CAN_CKPT" $i &
    PIDS+=($!)
    echo "  can ep $i -> PID $!"
done

echo "Waiting for wave 2..."
for pid in "${PIDS[@]}"; do wait $pid; done
echo "Wave 2 done: $(date)"

# Wave 3: last 2 each
echo ""
echo "=== Wave 3: lift 8-9, can 8-9 ==="
PIDS=()
for i in 8 9; do
    run_episode lift "$LIFT_CKPT" $i &
    PIDS+=($!)
    echo "  lift ep $i -> PID $!"
done
for i in 8 9; do
    run_episode can "$CAN_CKPT" $i &
    PIDS+=($!)
    echo "  can ep $i -> PID $!"
done

echo "Waiting for wave 3..."
for pid in "${PIDS[@]}"; do wait $pid; done
echo "Wave 3 done: $(date)"

# Check for failures
echo ""
echo "=== Checking results ==="
for task in lift can; do
    ok=0; fail=0
    for i in $(seq 0 $((N-1))); do
        if [ -f "eval_results/episodes/${task}_ep${i}.json" ]; then
            ok=$((ok+1))
        else
            fail=$((fail+1))
            echo "  MISSING: ${task}_ep${i}"
        fi
    done
    echo "  $task: $ok/$N succeeded, $fail failed"
done

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
        except Exception as e:
            print(f'  Warning: {f}: {e}')

    if not all_episodes:
        print(f'{task}: no episode data in {len(files)} files')
        continue

    successes = [ep.get('episode_success', 0) for ep in all_episodes]
    returns = [ep.get('episode_return', 0) for ep in all_episodes]
    spreads = [ep.get('mean_candidate_spread', 0) for ep in all_episodes]

    merged = {
        'task': task,
        'n_episodes': len(all_episodes),
        'success_rate': float(np.mean(successes)),
        'mean_return': float(np.mean(returns)),
        'mean_spread': float(np.mean(spreads)) if spreads else 0,
        'episodes': all_episodes,
    }

    out = f'eval_results/robomimic_headroom_{task}_K${K}_n${N}.json'
    json.dump(merged, open(out, 'w'), indent=2)
    print(f'{task}: {len(all_episodes)} eps, success={np.mean(successes):.2f}, return={np.mean(returns):.2f}, spread={np.mean(spreads):.4f}')

print()
print('Done. Check fork_decision in output JSONs.')
"

echo ""
echo "=============================================="
echo "  PIPELINE COMPLETE: $(date)"
echo "=============================================="

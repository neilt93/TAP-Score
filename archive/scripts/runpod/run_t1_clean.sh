#!/bin/bash
cd /root/TAP-Score
export PYTHONPATH=/root/TAP-Score
DP=baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt

for K in 2 4 8; do
  for L in 1 5; do
    echo "=== Clean K=$K L=$L n=200 ==="
    python scripts/reranking_experiment.py --dp_checkpoint $DP \
      --K $K --L $L --n_episodes 200 --seed_offset 0 --perturb_seed 123 \
      --output outputs/t1_headroom_clean_K${K}_L${L}_n200.json --device cuda
  done
done

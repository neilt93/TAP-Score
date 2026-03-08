#!/bin/bash
cd /root/TAP-Score
export PYTHONPATH=/root/TAP-Score
DP=baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt

for K in 2 4 8; do
  for L in 1 5; do
    echo "=== Occ p=0.1 K=$K L=$L n=200 ==="
    python scripts/reranking_experiment.py --dp_checkpoint $DP \
      --K $K --L $L --n_episodes 200 --seed_offset 0 \
      --perturb occlusion --patch_size 20 --perturb_prob 0.1 --perturb_seed 123 --occlusion_mode episode \
      --output outputs/t1_headroom_occ_K${K}_L${L}_n200.json --device cuda
  done
done

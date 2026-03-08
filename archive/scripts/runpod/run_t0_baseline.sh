#!/bin/bash
cd /root/TAP-Score
export PYTHONPATH=/root/TAP-Score
DP=baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt

echo "=== T0 Baseline Clean n=200 ==="
python scripts/reranking_experiment.py --dp_checkpoint $DP \
  --K 1 --L 1 --n_episodes 200 --seed_offset 0 --perturb_seed 123 \
  --output outputs/t0_baseline_clean_n200.json --device cuda

echo "=== T0 Baseline Occ p=0.1 n=200 ==="
python scripts/reranking_experiment.py --dp_checkpoint $DP \
  --K 1 --L 1 --n_episodes 200 --seed_offset 0 \
  --perturb occlusion --patch_size 20 --perturb_prob 0.1 --perturb_seed 123 --occlusion_mode episode \
  --output outputs/t0_baseline_occ_p01_n200.json --device cuda

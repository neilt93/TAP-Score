#!/bin/bash
cd /root/TAP-Score
export PYTHONPATH=/root/TAP-Score
DP=baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt
COMMON="--K 1 --L 1 --n_episodes 50 --seed_offset 0 --perturb_seed 123 --device cuda"

echo "=== Occlusion p=0.1 ==="
python scripts/reranking_experiment.py --dp_checkpoint $DP $COMMON \
  --perturb occlusion --patch_size 20 --perturb_prob 0.1 --occlusion_mode episode \
  --output outputs/t0_cal_occ_p01_n50.json

echo "=== Occlusion p=0.2 ==="
python scripts/reranking_experiment.py --dp_checkpoint $DP $COMMON \
  --perturb occlusion --patch_size 20 --perturb_prob 0.2 --occlusion_mode episode \
  --output outputs/t0_cal_occ_p02_n50.json

echo "=== Occlusion p=0.3 ==="
python scripts/reranking_experiment.py --dp_checkpoint $DP $COMMON \
  --perturb occlusion --patch_size 20 --perturb_prob 0.3 --occlusion_mode episode \
  --output outputs/t0_cal_occ_p03_n50.json

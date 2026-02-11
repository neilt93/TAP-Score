#!/bin/bash
cd /root/TAP-Score
export PYTHONPATH=/root/TAP-Score
DP=baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt
TAP=checkpoints_contrastive/pusht/contrastive_tap_h8_best.pt

# Fill in K and L after T1 results — defaulting to K=8 L=5 for now
K=8
L=5

echo "=== T2 TAP Reranking Occ p=0.1 K=$K L=$L n=200 ==="
python scripts/reranking_experiment.py --dp_checkpoint $DP --tap_checkpoint $TAP \
  --K $K --L $L --n_episodes 200 --seed_offset 0 \
  --perturb occlusion --patch_size 20 --perturb_prob 0.1 --perturb_seed 123 --occlusion_mode episode \
  --tap_sees_perturb \
  --output outputs/t2_tap_occ_K${K}_L${L}_n200.json --device cuda

echo "=== T2 TAP Clean Sanity K=$K L=$L n=50 ==="
python scripts/reranking_experiment.py --dp_checkpoint $DP --tap_checkpoint $TAP \
  --K $K --L $L --n_episodes 50 --seed_offset 0 --perturb_seed 123 \
  --tap_sees_perturb \
  --output outputs/t2_tap_clean_K${K}_L${L}_n50.json --device cuda

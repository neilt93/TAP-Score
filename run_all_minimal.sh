#!/bin/bash
cd /root/TAP-Score
export PYTHONPATH=/root/TAP-Score
DP=baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt
TAP=checkpoints_contrastive/pusht/contrastive_tap_h8_best.pt

echo "=== [1/3] T1 Headroom Clean K=8 L=5 n=200 ==="
python scripts/reranking_experiment.py --dp_checkpoint $DP \
  --K 8 --L 5 --n_episodes 200 --seed_offset 0 --perturb_seed 123 \
  --output outputs/t1_headroom_clean_K8_L5_n200.json --device cuda

echo "=== [2/3] T1 Headroom Occ p=0.1 K=8 L=5 n=200 ==="
python scripts/reranking_experiment.py --dp_checkpoint $DP \
  --K 8 --L 5 --n_episodes 200 --seed_offset 0 \
  --perturb occlusion --patch_size 20 --perturb_prob 0.1 --perturb_seed 123 --occlusion_mode episode \
  --output outputs/t1_headroom_occ_K8_L5_n200.json --device cuda

echo "=== [3/3] T2 TAP Reranking Occ p=0.1 K=8 L=5 n=200 ==="
python scripts/reranking_experiment.py --dp_checkpoint $DP --tap_checkpoint $TAP \
  --K 8 --L 5 --n_episodes 200 --seed_offset 0 \
  --perturb occlusion --patch_size 20 --perturb_prob 0.1 --perturb_seed 123 --occlusion_mode episode \
  --tap_sees_perturb \
  --output outputs/t2_tap_occ_K8_L5_n200.json --device cuda

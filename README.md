# TAP-Score: Detecting Off-Manifold Actions in Diffusion Policies

**Author:** Neil Tripathi

---

## Overview

An investigation into when and where intervention on Diffusion Policy (Chi et al., 2024) action proposals actually helps. Two main findings:

### 1. TAP-Score: Contrastive Failure Detection
A contrastive scorer (InfoNCE) trained on expert demonstrations detects when a diffusion policy proposes off-manifold actions. AUROC 0.998 on held-out failure types, 94.3% TPR at 1% FPR, with early warning from the first 70% of an episode.

### 2. Reranking Headroom Diagnosis
A counterfactual branching framework ("label spread gate") that measures oracle best-of-K improvement *before* training any ranker. Key finding: **under clean conditions, there is nothing to rank** (+0.009 mean improvement). **Under degraded conditions (50% occlusion), headroom explodes 12x** (+0.109 mean improvement, 59% of decisions improvable).

The policy doesn't need help when it's succeeding — it needs a safety net when it's failing.

---

## Key Results

### Reranking Headroom (the main finding)

| Metric | Clean | 50% Occlusion | Change |
|--------|-------|---------------|--------|
| Mean oracle improvement | +0.009 | +0.109 | **12x** |
| Median oracle improvement | ~0 | +0.027 | 0 to meaningful |
| Frac > 0.01 | 11% | 59% | **5x** |
| Frac > 0.1 | 1% | 35% | **28x** |

This independently corroborates the memorization hypothesis (Chi et al., 2025): clean observations trigger reliable recall of memorized sequences (candidates converge); corrupted observations cause retrieval failure (candidates scatter).

### TAP-Score Detection

| Metric | Value |
|--------|-------|
| Held-Out Failure AUROC | 0.998 |
| Prefix-Only AUROC | 0.997 |
| TPR @ 1% FPR | 94.3% |
| TPR @ 5% FPR | 100.0% |

---

## Reproduction

```bash
# Setup
./setup.sh
./scripts/download_data.sh

# Train TAP-Score
python train_contrastive_tap.py --benchmark pusht

# Evaluate TAP-Score
python eval_tap_final.py --benchmark pusht

# Tier 0 baseline: vanilla DP clean vs perturbed (no TAP reranking)
python scripts/reranking_experiment.py \
  --dp_checkpoint baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt \
  --K 1 --L 5 --n_episodes 200 --seed_offset 0 --perturb_seed 123 \
  --output outputs/baseline_clean_n200.json
python scripts/reranking_experiment.py \
  --dp_checkpoint baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt \
  --K 1 --L 5 --n_episodes 200 --seed_offset 0 \
  --perturb occlusion --patch_size 24 --perturb_prob 0.5 --perturb_seed 123 \
  --output outputs/baseline_occ24_n200.json
python scripts/compare_tier0_baseline.py \
  --clean outputs/baseline_clean_n200.json \
  --perturbed outputs/baseline_occ24_n200.json \
  --output outputs/baseline_tier0_compare.json

# Headroom diagnostic (clean)
python scripts/headroom_diagnostic.py \
  --dp_checkpoint baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt \
  --n_episodes 20 --K 8 --L 5 --device cuda

# Headroom diagnostic (occluded — the key contrast)
python scripts/headroom_diagnostic.py \
  --dp_checkpoint baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt \
  --n_episodes 20 --K 8 --L 5 --perturb occlusion --perturb_prob 0.5 --device cuda
```

---

## Project Structure

```
├── tap/                        # Core TAP-Score implementation
│   ├── contrastive.py          # Contrastive model (InfoNCE)
│   ├── dataset.py              # Positive/negative sampling
│   ├── benchmarks.py           # Multi-benchmark registry
│   ├── env_wrapper.py          # State save/restore for counterfactual branching
│   └── data_loaders.py         # Zarr/NPY loading
├── scripts/
│   ├── headroom_diagnostic.py  # Label spread gate + oracle best-of-K
│   ├── label_spread_gate.py    # K/L sweep diagnostic
│   └── test_state_restore.py   # Determinism verification
├── train_contrastive_tap.py    # Training
├── eval_tap_final.py           # Evaluation
├── eval_results/               # All experimental results (JSON)
├── papers/                     # Paper drafts
│   └── paper_b_reranking_headroom.md
└── tweets/                     # Social media materials
    └── reranking_headroom/
```

---

## References

- Chi et al. (2024). Diffusion Policy: Visuomotor Policy Learning via Action Score Gradients. *RSS 2024*.
- Chi et al. (2025). Demystifying Diffusion Policy. *arXiv*.

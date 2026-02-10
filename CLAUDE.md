# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TAP-Score (Temporal Action-Proposal Scoring) - Off-manifold action detection in visuomotor imitation learning. Penn State research project for Prof. Huijuan Xu's Vision-Language Lab.

**Core Idea:** Train a contrastive scorer on expert demonstrations to detect when a policy proposes actions inconsistent with expert behavior, enabling early failure prediction.

## Architecture Overview

### Two-Encoder Contrastive Architecture

TAP-Score uses **InfoNCE contrastive ranking** instead of binary classification:

- **Observation Encoder:** CNN for image observations (Push-T) or MLP for state observations (Block Push, Kitchen)
- **Action Encoder:** MLP that processes action chunks
- **Scoring:** Dot product of normalized embeddings → logits for ranking
- **Loss:** InfoNCE - model must rank correct action above M negatives
- **Inference:** Log-margin score = `log_prob(correct) - logsumexp(log_prob(negatives))`

Key files:
- `tap/contrastive.py`: Core model (`ContrastiveTAPScore`, `ContrastiveObsEncoder`, `ContrastiveActionEncoder`)
- `tap/dataset.py`: Positive/negative sampling with hard negatives (permutation, mirroring, noise, random)
- `tap/benchmarks.py`: Multi-benchmark registry (pusht, blockpush, kitchen)

### Training Strategy

**Negative sampling** (in training):
- Random actions from other episodes
- Hard negatives: permuted, mirrored, noisy versions of correct action
- Ratio controlled by `--hard_negative_ratio`

**Held-out failure families** (testing only):
- Scaling, bias, stuck, delayed
- OR magnitude-preserving: rotation, axis swap, sign flip, time warp
- These are NEVER seen during training → tests true generalization

### Multi-Benchmark Support

The codebase supports 4 benchmarks with different observation and action spaces:

| Benchmark | Obs Type | Action Dim | Description |
|-----------|----------|------------|-------------|
| `pusht` | image (96x96x3) | 2 | Push T-block task |
| `blockpush` | state (16D) | 2 | Block pushing task |
| `kitchen` | state (60D) | 9 | Kitchen manipulation |

Config lives in `tap/benchmarks.py`. Use `--benchmark <name>` to select.

## Common Commands

### Environment Setup

```bash
# Initial setup (creates venv, installs deps)
./setup.sh

# Activate environment
source venv/bin/activate

# Download benchmark data
./scripts/download_data.sh              # Push-T only
./scripts/download_benchmarks.sh all    # All benchmarks
```

### Training

```bash
# Train contrastive TAP-Score (recommended)
python train_contrastive_tap.py --benchmark pusht --epochs 30 --batch_size 64

# Train on different benchmarks
python train_contrastive_tap.py --benchmark blockpush --use_deltas
python train_contrastive_tap.py --benchmark kitchen

# Legacy BCE-based training (not recommended - suffers from collapse)
python train_tap.py --data_dir data/processed/pusht --epochs 20
```

Key flags:
- `--benchmark`: Which benchmark to train on (pusht, blockpush, kitchen)
- `--n_negatives`: Number of negatives per sample (default: 15)
- `--hard_negative_ratio`: Fraction of hard negatives (default: 0.5)
- `--use_deltas`: For state obs, concatenate (state, delta) features
- `--temperature`: InfoNCE temperature (default: 0.1)

Output: `checkpoints_contrastive/<benchmark>/contrastive_tap_best.pt`

### Evaluation

```bash
# Main evaluation (AUROC, operating points, M ablation, prefix-only)
python eval_tap_final.py --benchmark pusht --n_episodes 40

# Magnitude-preserving failures (harder test)
python eval_tap_final.py --benchmark pusht --magnitude_preserving

# Cross-benchmark comparison
python scripts/eval_all_benchmarks.py

# Episode-level analysis (score traces over time)
python eval_episode_tap.py --checkpoint checkpoints_contrastive/pusht/best.pt
```

Output: `eval_results/<benchmark>_eval_results.json`

### Baseline Replication

```bash
# Clone and verify Diffusion Policy baseline
./scripts/run_replication.sh
python scripts/verify_dp_data.py
```

## Key Metrics

Evaluation produces:
- **AUROC**: Success vs held-out failure separation (target: >0.95)
- **Operating points**: TPR @ 1%, 5%, 10% FPR
- **M ablation**: Stability across different numbers of negatives (7, 15, 31)
- **Prefix-only AUROC**: Early warning using first 70% of episode

## Data Format

### Zarr Format (Push-T, Block Push)
```python
data/
  meta/
    episode_ends: [206, 412, ...]  # Cumulative episode boundaries
  data/
    img or obs: (N, H, W, C) or (N, obs_dim)  # Observations
    action: (N, action_dim)                    # Actions
```

### NPY Format (Kitchen)
```
train.npy: List of dicts with keys ['obs', 'action']
```

Loading handled by `tap/data_loaders.py` - auto-detects format based on benchmark config.

## Project Structure

```
pennstate-project/
├── tap/                    # Core TAP-Score implementation
│   ├── contrastive.py      # Contrastive model (InfoNCE)
│   ├── dataset.py          # Positive/negative sampling
│   ├── benchmarks.py       # Multi-benchmark registry
│   ├── data_loaders.py     # Zarr/NPY loading utilities
│   ├── perturbations.py    # Augmentation functions
│   └── resampling.py       # Intervention logic
├── train_contrastive_tap.py   # Main training script (use this)
├── train_tap.py                # Legacy BCE training
├── eval_tap_final.py           # Main evaluation script
├── eval_*.py                   # Specialized evaluations
├── scripts/
│   ├── download_data.sh        # Download Push-T
│   ├── download_benchmarks.sh  # Download all benchmarks
│   ├── run_replication.sh      # Clone Diffusion Policy
│   └── eval_all_benchmarks.py  # Cross-benchmark comparison
├── report/                     # Requirements documentation
│   ├── 1_sota.md               # SOTA identification
│   ├── 2_replication.md        # Baseline replication
│   ├── 3_ideas.md              # Proposed ideas
│   └── 4_method_and_results.md # TAP-Score results
├── checkpoints_contrastive/    # Trained models by benchmark
├── eval_results/               # Evaluation outputs (JSON)
└── baselines/
    └── diffusion_policy/       # Baseline repo (cloned)
```

## Important Implementation Details

### Why Contrastive > BCE

The original `train_tap.py` used binary cross-entropy (BCE) to classify positive/negative action pairs. This **collapsed** - the model learned to ignore observations and output constant scores because:
1. Any fixed score achieves 50% accuracy on balanced data
2. No incentive to use observations when labels are binary

**Solution:** Contrastive ranking forces the model to rank the correct action above M negatives. A constant score fails because negatives vary across observations.

### Score Computation at Inference

```python
# Training: InfoNCE over (1 correct + M negatives)
logits = model(obs, candidates)  # (B, M+1)
loss = -log_softmax(logits)[:, 0]

# Inference: Log-margin scoring
margin = logits[0] - logsumexp(logits[1:])  # Higher = more consistent with expert
```

Lower margins → action is off-manifold.

### Calibration

Set threshold τ on validation success episodes to achieve target FPR (e.g., 5%). At deployment:
- `score < τ` → predict failure, trigger intervention
- `score ≥ τ` → predict success, execute action

### State Observations (Block Push, Kitchen)

For state-based benchmarks, use MLP encoder instead of CNN. Optional delta encoding:
```bash
python train_contrastive_tap.py --benchmark blockpush --use_deltas
```

This concatenates `(s_t, s_t - s_{t-1})` to help model learn that actions cause state changes.

## Tech Stack

- Python 3.13
- PyTorch 2.0+
- Zarr (data storage)
- scikit-learn (metrics)
- Diffusion Policy baseline (baselines/diffusion_policy)

## Requirements Deliverables

1. `report/1_sota.md` - SOTA identification (Diffusion Policy)
2. `report/2_replication.md` - Partial replication (transparent)
3. `report/3_ideas.md` - Proposed ideas (TAP-Score, phase-conditioning, etc.)
4. `report/4_method_and_results.md` - TAP-Score method and results (AUROC 0.998)

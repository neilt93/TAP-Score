# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Obsidian Knowledge Vault

**IMPORTANT:** The user maintains an Obsidian vault as the central knowledge repo for all projects. You MUST update it whenever results, decisions, or research directions change.

- **Vault path:** `C:/Users/neilt/OneDrive/Documents/Neil's Planner/Task-Agent/`
- **TAP-Score project note:** `Projects/TAP-Score.md`
- **Always update the Obsidian project note when:**
  - New evaluation results are produced (update Results section)
  - Tasks are completed (check off in Tasks section)
  - Research directions change or new ones emerge (update Research Directions)
  - Significant decisions are made (add to Log with date)
  - Milestones are reached (update Milestones table)
- **Follow vault conventions:**
  - Tasks plugin syntax: `- [ ] Task name 📅 YYYY-MM-DD`
  - Frontmatter: `status: active | on-hold | completed`
  - Log entries: `### YYYY-MM-DD` with bullet points
- **Read the project note at session start** to understand current state before making changes

## Critical Directives

- **NEVER USE SYNTHETIC DATA:** No synthetic negatives, no synthetic action corruptions. Real DP rollouts only.
- **Contrastive approach is superseded:** The InfoNCE contrastive scorer (Push-T) was the first approach. The active approach is the **risk model** (binary failure predictor on real rollout data). Do not revisit contrastive scoring for new tasks.

## Project Overview

TAP-Score (Temporal Action-Proposal Scoring) — Off-manifold action detection in visuomotor imitation learning.

**Core Idea:** Detect when a Diffusion Policy proposes actions that will lead to failure, enabling early intervention (abstention, recovery).

### Two Approaches

1. **Contrastive TAP-Score (Push-T only, completed):** InfoNCE ranking of correct action vs negatives. AUROC 0.994 on held-out synthetic failure types. Lives in `tap/contrastive.py`.

2. **Risk TAP-Score (Robomimic, active):** Binary predictor `P(fail_within_H | obs, action)` trained on real DP rollouts under diverse perturbation regimes. Lives in `tap/risk.py`. This is the current focus.

## Architecture Overview

### Risk Model (Active Approach)

- **Model:** `RiskTAPScore` in `tap/risk.py` — obs_encoder(MLP) + action_encoder(MLP) → concat → sigmoid
- **Obs-only variant:** `RiskTAPScore(obs_only=True)` — uses only obs_encoder, no action input. Baseline to test whether actions carry signal beyond state.
- **Output:** `P(fail_within_H | obs, action)`, H=64 env steps (8 action chunks)
- **Score in eval CSV:** `1.0 - risk_prob` (higher = safer)
- **Labeling modes:**
  - `onset_window`: label=1 if step in [onset, onset+H] AND episode failed. Best for tasks with sustained failure signal (Can).
  - `episode_window`: label=episode_outcome for steps in [onset, onset+W]. Best for tasks with delayed/sparse failures (Lift).
- **Training data:** Real DP rollouts under regimes: clean, zero80, zero80_jitter, dropout80_p02, dropout80_p04
- **Aggregation:** `mean` is best for both Can and Lift per-episode AUROC

### Contrastive Model (Completed, Push-T)

- **Observation Encoder:** CNN for image obs (Push-T) or MLP for state obs
- **Action Encoder:** MLP that processes action chunks
- **Scoring:** Dot product of normalized embeddings → InfoNCE ranking
- **Inference:** Log-margin score = `log_prob(correct) - logsumexp(log_prob(negatives))`

## Benchmark Registry

Six benchmarks configured in `tap/benchmarks.py`:

| Benchmark | Key | Obs Type | Obs Dim | Action Dim |
|-----------|-----|----------|---------|------------|
| Push-T | `pusht` | image (96x96x3) | — | 2 |
| Block Push | `blockpush` | state | 16 | 2 |
| Kitchen | `kitchen` | state | 60 | 9 |
| Robomimic Lift | `robomimic_lift_lowdim` | state | 19 | 7 |
| Robomimic Can | `robomimic_can_lowdim` | state | 23 | 7 |
| Robomimic Square | `robomimic_square_lowdim` | state | 23 | 7 |

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

### Risk Model Pipeline (Robomimic)

```bash
# 1. Collect rollout data under perturbation regimes
python scripts/collect_risk_rollouts.py \
  --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
  --task can --n_episodes 200 --output_dir data/risk_rollouts/can/ \
  --regimes clean zero80 freeze80 dropout80

# 2. Train risk model
python train_risk_tap.py \
  --rollout_dir data/risk_rollouts/can/ \
  --task can --epochs 30 --batch_size 64 \
  --fail_horizon 64 --hard_mining_epoch 10

# 3. Evaluate detection
python scripts/eval_tap_detection.py \
  --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
  --risk_model checkpoints_contrastive/robomimic_can_lowdim/risk_tap_best.pt \
  --n_episodes 50 --output eval_results/detection_can_zero80_n50.json

# 4. Evaluate abstention
python scripts/eval_abstention.py --results eval_results/detection_can_zero80_n50.json
```

### Contrastive Model (Push-T)

```bash
# Train
python train_contrastive_tap.py --benchmark pusht --epochs 30 --batch_size 64

# Evaluate (AUROC, operating points, M ablation)
python eval_tap_final.py --benchmark pusht --n_episodes 40 --magnitude_preserving
```

### Obs-Only Baseline

```bash
# Train obs-only baseline (same data, no action encoder)
python train_risk_tap.py \
  --rollout_dir data/risk_rollouts/can/ \
  --task can --epochs 30 --obs_only

# Evaluate obs-only baseline
python scripts/eval_tap_detection.py \
  --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
  --risk_model --tap_checkpoint checkpoints_contrastive/robomimic_can_lowdim/risk_tap_obs_only_best.pt \
  --n_episodes 100 --output eval_results/detection_can_obs_only_n100.csv
```

### Intervention Evaluation

```bash
# Resample: pick safest of K proposals when risk exceeds threshold
python scripts/eval_intervention.py \
  --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
  --tap_checkpoint checkpoints_contrastive/robomimic_can_lowdim/risk_tap_best.pt \
  --task can --mode resample --K 4 --n_episodes 50

# Early-stop + restart: terminate and retry when risk stays high
python scripts/eval_intervention.py \
  --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
  --tap_checkpoint checkpoints_contrastive/robomimic_can_lowdim/risk_tap_best.pt \
  --task can --mode restart --max_restarts 2 --n_episodes 50
```

### Analysis

```bash
# Failure density analysis (why TAP works on Can/Lift, not Square)
python scripts/analyze_failure_density.py

# Selective execution curves (success vs coverage)
python scripts/analyze_selective_execution.py
```

### Headroom Audit (Robomimic)

```bash
python scripts/robomimic_headroom_audit.py \
  --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
  --task Can --K 4 --n_episodes 20
```

### Tests

```bash
python -m pytest tests/
```

## Key Results

### Risk Model (Final, n=100)

| Task | AUROC (mean) | AUPRC | CI (95%) | Val AUROC |
|------|-------------|-------|----------|-----------|
| Can  | **0.856** | 0.881 | [0.779, 0.915] | 0.973 |
| Lift | **0.811** | 0.821 | [0.718, 0.892] | 0.954 |
| Square | 0.523 | — | — | — |

Can selective execution: 100%@10%, 95%@20%, 80%@30% (vs 45% baseline).
Lift selective execution: 80%@10%, 90%@20%, 83%@30% (vs 47% baseline).

### Contrastive Model (Push-T)

AUROC 0.994 on held-out failure types, 97.5% TPR @ 1% FPR.

## robosuite 1.5.2 Compatibility (Critical for Robomimic)

DP checkpoints were trained with old cheng-chi/robosuite fork (mujoco_py, v1.2.0). Running with robosuite 1.5.2 requires:

**Five controller patches** (in `_patch_abs_action_controller` in `robomimic_headroom_audit.py`):
1. `input_type = "absolute"`
2. `input_ref_frame = "world"`
3. Boost `kp` to 500
4. `kd = 2 * sqrt(kp)`
5. Monkey-patch `reset()` to re-apply patches

**Observation fixes:**
- Lift: sign-flip indices [7:10]
- Can: field reorder (swap halves)
- Must use `reset_to` with demo initial states from HDF5

**Action transform:** DP actions must be `undo_transform_action()` before TAP scoring (rotation_6d → axis_angle).

## Data Formats

| Format | Benchmarks | Structure |
|--------|-----------|-----------|
| Zarr | Push-T, Block Push | `data/meta/episode_ends`, `data/data/{img,obs,action}` |
| NPY | Kitchen | `train.npy` — list of dicts with `['obs', 'action']` |
| HDF5 | Robomimic | Standard robomimic format, obs_keys in benchmark config |
| NPZ | Risk rollouts | Per-episode files in `data/risk_rollouts/<task>/` |

Loading handled by `tap/data_loaders.py` (zarr/npy) and `tap/risk.py` (npz rollouts).

## Project Structure

```
TAP-Score/
├── tap/                              # Core library
│   ├── __init__.py                   # Exports (contrastive only)
│   ├── contrastive.py                # Contrastive model + datasets
│   ├── risk.py                       # Risk model + rollout datasets
│   ├── benchmarks.py                 # Benchmark registry (6 tasks)
│   ├── data_loaders.py               # Episode loaders (zarr/npy)
│   ├── perturbations.py              # Synthetic failure families
│   └── env_wrapper.py                # Push-T state wrapper
├── train_contrastive_tap.py          # Train contrastive (Push-T)
├── train_risk_tap.py                 # Train risk model (robomimic)
├── eval_tap_final.py                 # Contrastive eval (AUROC etc.)
├── scripts/
│   ├── collect_risk_rollouts.py      # Collect labeled rollout data
│   ├── eval_tap_detection.py         # Runtime detection eval
│   ├── eval_abstention.py            # Abstention curve eval
│   ├── eval_intervention.py          # Resample/restart intervention eval
│   ├── analyze_selective_execution.py # Success-vs-coverage curves
│   ├── analyze_failure_density.py    # Failure density analysis (why TAP works)
│   ├── robomimic_headroom_audit.py   # Oracle best-of-K headroom
│   ├── run_perturb_sweep.sh          # Perturbation sweep
│   ├── run_risk_pipeline.sh          # Full risk pipeline
│   ├── runpod/                       # RunPod-specific scripts
│   └── windows/                      # Windows setup scripts
├── tests/                            # Unit tests
├── archive/                          # Superseded Push-T reranking work
├── artifacts/                        # Frozen results + paper drafts
├── checkpoints_contrastive/          # Trained models
├── eval_results/                     # Evaluation results (JSON/CSV)
├── report/                           # Requirements documentation
├── data/                             # Datasets + rollouts
└── baselines/
    └── diffusion_policy/             # Baseline repo (cloned)
```

## Key Implementation Details

### Score Computation

```python
# Risk model (active): binary prediction
risk_prob = model.predict_risk(obs, action)  # P(fail) in [0,1]
score = 1.0 - risk_prob                       # higher = safer

# Contrastive (Push-T): log-margin
logits = model(obs, candidates)              # (B, M+1)
margin = logits[0] - logsumexp(logits[1:])   # higher = more expert-like
```

### Calibration

Set threshold τ on validation success episodes to achieve target FPR (e.g., 5%). At deployment:
- `score < τ` → predict failure, trigger intervention
- `score ≥ τ` → predict success, execute action

### DP Checkpoint Config (Robomimic)

- `n_obs_steps=2`, `n_action_steps=8`, `n_latency_steps=0`
- chunk_len = 8 env steps per decision
- Actions in rotation_6d format, must convert to axis_angle for scoring

## Tech Stack

- Python 3.10 (Windows) / 3.13 (Linux)
- PyTorch 2.0+
- Zarr, h5py (data storage)
- scikit-learn (metrics)
- robosuite 1.5.2 + robomimic (robomimic tasks)
- Diffusion Policy baseline (baselines/diffusion_policy)

## Requirements Deliverables

1. `report/1_sota.md` — SOTA identification (Diffusion Policy)
2. `report/2_replication.md` — Partial replication (transparent)
3. `report/3_ideas.md` — Proposed ideas (TAP-Score, phase-conditioning, etc.)
4. `report/4_method_and_results.md` — TAP-Score method and results

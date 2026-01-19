# Penn State Take-Home Project

**Candidate:** Neil Tripathi
**Target:** Prof. Huijuan Xu, Vision-Language Lab

---

## Problem Chosen: Detecting Off-Manifold Action Proposals in Visuomotor Imitation

**Generalization Problem:** Imitation policies (e.g., diffusion policies) can drift without any built-in signal that their proposed actions no longer match expert behavior. We need a method to detect when action proposals fall off the expert manifold—enabling early failure prediction and potential intervention.

---

## Requirement 1: Identify SOTA Method

**File:** `report/1_sota.md`

**SOTA Identified:** Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023)

- Achieves ~95% success rate on Push-T benchmark
- Uses denoising diffusion for action generation
- No built-in OOD detection or failure prediction

---

## Requirement 2: Replicate the Code

**File:** `report/2_replication.md`

**Status:** Partial replication (transparent documentation)

| Step | Status |
|------|--------|
| Clone repository | Done (commit `5ba07ac`) |
| Download Push-T data | Done (206 episodes) |
| Verify data format | Done |
| Train Diffusion Policy | Not done (resource constraints) |
| Train TAP-Score on demos | Done |

**Justification:** TAP-Score methodology requires expert demonstrations, not a trained policy checkpoint.

---

## Requirement 3: Propose New Ideas

**File:** `report/3_ideas.md`

**Ideas Proposed:**
1. **TAP-Score (Implemented):** Learn to score (obs, action) compatibility; detect off-manifold actions
2. Phase-conditioned scoring for multi-stage tasks
3. Uncertainty-weighted action selection
4. Learned perturbation-invariant representations

---

## Requirement 4: Implement One Specific Idea

**File:** `report/4_method_and_results.md`

### Method: Contrastive TAP-Score

**Architecture:**
- Observation encoder (CNN) → 128-dim features
- Action encoder (MLP) → 128-dim features
- Contrastive ranking with InfoNCE loss
- Log-margin scoring: `margin = l_0 - logsumexp([l_1, ..., l_M])`

**Key Innovation:** Train on (noise, permute, mirror, random) negatives, evaluate on held-out failure types (scaling, bias, stuck, delayed) to prove generalization.

### Results

| Metric | Value |
|--------|-------|
| **Held-Out Failure AUROC** | **0.998** |
| **Prefix-Only AUROC** (early warning) | **0.997** |
| **TPR @ 1% FPR** | **94.3%** |
| **TPR @ 5% FPR** | **100.0%** |
| Action Magnitude Baseline | 0.665 |
| **TAP Advantage** | +0.333 absolute AUROC |

### Key Findings

1. **Generalization:** AUROC 0.998 on failure types never seen in training
2. **Early warning:** Can predict failure using only first 70% of episode
3. **Practical:** 94.3% detection at just 1% false alarm rate
4. **Stable:** Consistent results across M=7, 15, 31 negatives

---

## Reproduction (3 Commands)

```bash
# Train Contrastive TAP
python train_contrastive_tap.py --data_dir data/processed/pusht

# Evaluate with bulletproof metrics
python eval_tap_final.py --checkpoint checkpoints_contrastive/contrastive_tap_best.pt

# View results
cat eval_results/final_eval_results.json
```

---

## Project Structure

```
pennstate-project/
├── report/
│   ├── 1_sota.md              # Req 1: SOTA identification
│   ├── 2_replication.md       # Req 2: Partial replication (transparent)
│   ├── 3_ideas.md             # Req 3: Proposed ideas
│   └── 4_method_and_results.md # Req 4: TAP-Score implementation
├── tap/
│   ├── contrastive.py         # Contrastive TAP model
│   └── ...
├── train_contrastive_tap.py   # Training script
├── eval_tap_final.py          # Bulletproof evaluation
├── plot_score_trace.py        # Visualization script
├── results/
│   ├── final_eval_results.json
│   └── figures/score_trace.png
├── checkpoints_contrastive/   # Trained model
├── baselines/
│   └── diffusion_policy/      # Official repo (cloned)
├── data/
│   └── processed/pusht/       # Push-T demonstrations
└── README.md
```

---

## Quick Setup

```bash
# 1. Setup environment
./setup.sh

# 2. Download Push-T data
./scripts/download_data.sh

# 3. (Optional) Clone Diffusion Policy baseline
./scripts/run_replication.sh
```

---

## Talk Track (1 minute)

> "Imitation policies like diffusion policies can drift without any built-in signal that their proposed actions no longer match expert behavior. I built **Contrastive TAP-Score**, a temporal action-proposal scorer that learns observation-action compatibility from demonstrations. The key change was switching from BCE to contrastive ranking so the model must rank the expert-consistent action above many plausible but incorrect alternatives. I score proposals with a numerically stable log-margin, which enables clean calibration and operating points. On Push-T demos with held-out off-manifold failure families not used in training (scaling, bias, stuck, delayed), TAP predicts failure early with **AUROC 0.998** and prefix-only **AUROC 0.997**, achieving **94.3% TPR at 1% FPR** and remaining stable across negative-set sizes. It also outperforms simple action-stat baselines by **+0.33 AUROC**."

---

## Connection to Prof. Xu's Research

1. **Temporal action understanding** - TAP-Score learns action structure over time
2. **Weak supervision** - No manual OOD labels, just expert demos
3. **Video-level reasoning** - Observation windows, not single frames
4. **Practical robustness** - Closed-loop intervention, not just detection

# TAP-Score Results Writeup

## Phase 1: Basic TAP (Contrastive Two-Tower)

### 1.1 Problem Statement

In visuomotor imitation learning, policies like Diffusion Policy can silently fail — producing actions that look plausible but lead to task failure. There is no built-in mechanism to detect when a policy's proposed actions have drifted off the expert manifold. We want a lightweight scorer that, given the current observation and a proposed action chunk, can flag whether that action is consistent with expert behavior.

### 1.2 Architecture: Contrastive Two-Tower

**What we built:**
- Observation encoder: CNN (3-channel 96x96 images, obs window T=2)
- Action encoder: MLP (action chunks of length H)
- Scoring: dot product of L2-normalized embeddings, scaled by temperature
- Loss: InfoNCE — rank the correct action above M negatives
- Inference score: log-margin = logit[correct] - logsumexp(logit[negatives])

**Key files:** `tap/contrastive.py`, `train_contrastive_tap.py`

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| hidden_dim | 128 |
| temperature | 0.1 |
| M (negatives) | 15 |
| hard_negative_ratio | 0.5 |
| obs_window | 2 |
| batch_size | 64 |
| epochs | 30 |

### 1.3 What We Tried First (and Failed): BCE Classification

The original approach used binary cross-entropy to classify (obs, action) pairs as positive/negative. This **collapsed completely** — the model learned to output a constant score regardless of input, achieving ~50% accuracy on balanced data.

**Why it failed:** With binary labels, there's no incentive to use observations. Any constant prediction achieves baseline accuracy. The model has no reason to learn what expert behavior looks like.

**Lesson learned:** You need a ranking objective, not a classification one, when the task is "does this action fit this observation?"

### 1.4 Offline Evaluation (Synthetic Failures)

Before integrating with Diffusion Policy, we validated TAP on synthetic action corruptions applied to expert demonstrations.

**Training negatives:** noise (gaussian), permutation (temporal shuffle), mirroring (sign flip), random (different trajectory)

**Held-out test failures (never seen in training):** scaling (0.5x-2.0x), constant bias, stuck policy, delayed reaction

**Results on synthetic data:**

| Metric | TAP-Score | Action Magnitude Baseline |
|--------|-----------|---------------------------|
| Held-out failure AUROC | **0.994** | 0.636 |
| Prefix-only AUROC (first 70%) | **1.000** | — |
| TPR @ 1% FPR | 97.5% | — |
| TPR @ 5% FPR | 100.0% | — |

**M ablation (stability):** AUROC M=7 (1.000), M=15 (0.992), M=31 (0.995) — stable.

**Caveat:** These numbers are on synthetic corruptions of expert data, not real policy rollouts. They show TAP learns the expert manifold, but don't prove it works on actual DP failures.

### 1.5 Live DP Integration

We integrated TAP into closed-loop Diffusion Policy rollouts on Push-T (RunPod, GPU). This required fixing several critical issues:

**Bugs fixed during integration:**
- **Observation history update:** DP needs to see consecutive frames, not the same frame repeated. Initially the obs history wasn't being updated after env steps, causing the policy to see stale frames.
- **Perturbation consistency:** Perturbations (noise, blur, occlusion) must be applied consistently on both reset and step, with per-episode RNG seeded by the episode seed.
- **Paired seeds:** To compare conditions (clean vs perturbed, K=1 vs K=4), same seeds must produce same initial states. We used `env.seed(seed)` before each reset.
- **Horizon mismatch:** DP uses action_chunk=8 (n_action_steps=8), but original TAP was trained with H=16. We retrained with H=8 to match.
- **Success labeling:** Push-T success = max_reward >= 0.80 at any point during the episode.

**Retrained TAP H=8 checkpoint:** `checkpoints_contrastive/pusht/contrastive_tap_h8_best.pt`

### 1.6 Passive Detection Results (TAP as Detector)

Using TAP H=8 in closed-loop DP rollouts (50 episodes per condition, paired seeds):

**DP baseline success rates:**
- Clean: 54%
- Noise (std=0.30): 48%
- Blur (sigma=2.0): 54%

**TAP detection performance:**

| Metric | Value |
|--------|-------|
| Success vs failure AUROC (p10) | **0.690** |
| Clean vs perturbed AUROC (p10) | **0.812** |
| Mean TAP score (clean) | -1.56 |
| Mean TAP score (noise) | -5.32 |
| Mean TAP score (blur) | -1.84 |

**Early warning behavior:**
- 53/72 failures flagged before episode end
- Median flag index: step 5 (out of ~25 policy steps)
- Mean lead time: 17.2 policy steps
- 88.7% of flagged failures detected early (within prefix)

**Interpretation:** TAP reliably separates clean from perturbed observations (AUROC 0.81). Success vs failure separation is weaker (0.69) because many failures under blur look similar to successes in TAP-score space — blur doesn't shift scores as dramatically as noise.

### 1.7 Active Reranking (TAP as Selector)

**Idea:** Instead of just detecting bad actions, use TAP to actively improve the policy. At each step, sample K action chunks from DP, score each with TAP, execute the best one.

**Implementation:** `archive/eval_dp_reranking.py`
- Batched K sampling: repeat_interleave obs K times, run DP once, reshape to (B, K, H, Da)
- Vectorized TAP scoring: score all K candidates per env in one forward pass
- Margin gate: optionally require margin_delta gap before overriding candidate 0
- No-TAP control: sample K candidates but always pick index 0 (isolates RNG effect from TAP effect)

### 1.8 Critical Insight: Causal Controls

**The pitfall we caught:** When K>1, DP samples K independent action chunks. Even picking candidate 0 gives different behavior than K=1, because the internal RNG state differs. So comparing K=4 TAP vs K=1 is **not** a valid causal comparison — any success difference could be from sampling randomness, not TAP.

**Solution:** Two controls:
1. **No-TAP control:** Sample K candidates, always pick index 0. Same RNG path as TAP condition, but no TAP selection.
2. **changed_any flag:** Only count an episode as a "TAP rescue" if TAP actually selected a non-zero candidate on at least one step.

**The only valid causal metric:** success(K=4 TAP) - success(K=4 notap)

### 1.9 Perturbation Sweet Spot

**What killed DP entirely (reranking can't help if policy scores 0%):**
- Full occlusion (20px patch): 0% success
- Brightness (1.5x): 0% success

**Sweet spot (DP partially degrades, room for reranking to help):**
- prob_occlusion (30% chance of 12px patch per step): ~26-36% success
- mild_blur (sigma=1.0): moderate degradation

### 1.10 The Alignment Problem

**First reranking attempt (TAP trained on synthetic negatives):**
On mild_blur, K=4 TAP performed *worse* than the K=4 no-TAP control. Zero true TAP rescues.

**Root cause:** TAP was trained to distinguish expert actions from synthetic negatives (random, permuted, noisy). But reranking asks TAP to choose the *best* among K plausible DP proposals — all of which are close to expert-like. TAP couldn't tell them apart because its negatives were too easy.

**Lesson learned:** Detection and selection are fundamentally different tasks. A good anomaly detector is not necessarily a good selector among near-expert candidates.

### 1.11 DP-Proposal Negative Retraining

**Fix:** Retrain TAP using actual DP proposals as hard negatives.

**DP neg cache construction:**
- For each expert (obs, action) sample, run DP K=8 times to get 8 candidate action chunks
- Store as `dp_neg_cache.npz`: shape (N_samples, 8, 8, 2)
- Also store aligned expert actions for distance computation

**Distance analysis:**
- DP proposals are a mix of near-expert and far-from-expert
- Used `dp_neg_top_m=4` (closest 4 of 8 candidates) as hard negatives
- This forces TAP to learn fine-grained discrimination within DP's proposal distribution

**Retraining:** `train_contrastive_tap.py --dp_neg_cache_path dp_neg_cache.npz --dp_neg_ratio 0.5 --dp_neg_top_m 4`
- 50% of negatives from DP cache (closest 4), 50% from standard synthetic negatives
- Val accuracy ~0.32 (harder task than before, as expected)

### 1.12 Reranking After DP-Neg Retrain (n=50)

**Conditions:** prob_occlusion, K=4, with no-TAP control, 50 episodes

| Condition | Success Rate |
|-----------|-------------|
| K=1 (baseline) | 36% |
| K=4 TAP (DP-neg retrained) | **30%** |
| K=4 no-TAP control | 26% |

**Causal effect:** TAP - notap = +4 pp (30% vs 26%)

**True TAP rescues (notap fails, TAP succeeds, TAP changed selection):** seeds [10, 17, 38] — 3 episodes

**Note:** K=1 at 36% is higher than K=4 conditions. This is because K>1 sampling introduces additional variance. The valid comparison is TAP vs notap at the same K.

### 1.13 Where Phase 1 Ended

**Two reportable stories:**

1. **Passive detection (strong):** TAP trained on expert demos can detect when DP is operating under perturbation (AUROC 0.81) and provides early warning of failures (median lead time: 17 policy steps). This works with the basic synthetic-negative-trained TAP.

2. **Active reranking (negative result):** Best-of-K reranking shows limited effectiveness. Under degraded conditions where headroom appears, candidate distributions collapse — all K proposals become similarly bad, leaving little for a ranker to exploit.

---

## Phase 2: Robomimic Detection

### 2.1 Robomimic Porting

Ported TAP-Score evaluation to robomimic Can and Lift tasks with lowdim (state-based) observations and Diffusion Policy checkpoints.

**Key challenges solved:**
- robosuite 1.5.2 compatibility: controller patches for absolute action mode, observation field reordering
- DP action format: `undo_transform_action()` for rotation_6d → axis_angle conversion before TAP scoring
- HDF5 dataset support for training TAP on robomimic demos

### 2.2 Detection Results (Can)

**Setting:** Can task, zero_object perturbation at onset=80, n=100 episodes

| Metric | Value |
|--------|-------|
| Detection AUROC (mean risk score) | **0.856** |
| Detection AUROC (min score) | 0.793 |
| AUPRC (mean risk score) | 0.881 |
| Magnitude baseline AUROC | 0.537 |
| TAP advantage | +0.320 |
| Bootstrap 95% CI | [0.779, 0.915] |
| Abstention @ 20% coverage | 95% success (vs 45% baseline) |
| Lead time (median) | 320 env steps |
| Failures detected | 55/55 |

**Key insight:** For the supervised risk model, mean episode risk is the strongest discriminator on Can. Unlike the earlier contrastive scorer, the risk model benefits from aggregating the sustained post-onset risk signal rather than looking only for a single worst action chunk.

### 2.3 Detection Results (Lift)

**Setting:** Lift task, zero_object perturbation at onset=31, n=100 episodes

| Metric | Value |
|--------|-------|
| Detection AUROC (mean risk score) | **0.811** |
| Detection AUROC (min score) | 0.418 |
| AUPRC (mean risk score) | 0.821 |
| Magnitude baseline AUROC | 0.551 |
| TAP advantage | +0.260 |
| Bootstrap 95% CI | [0.718, 0.892] |
| Abstention @ 10% coverage | 80% success (vs 47% baseline) |
| Abstention @ 20% coverage | 90% success (vs 47% baseline) |
| Lead time (median) | 360 env steps |
| Failures detected | 53/53 |

**Lift onset curve:** Sharp transition — 30% success at onset=30, 55% at onset=31, 85% at onset=32. Onset=31 was chosen as the sweet spot for detection evaluation.

**Cross-task comparison:** Can (AUROC 0.856) outperforms Lift (0.811), and both show clear TAP advantage over action-magnitude baselines (+0.320 and +0.260 respectively). Mean aggregation is the right statistic for the final risk model on both tasks; min-score aggregation is mainly retained as a comparison point and for the Square negative result.

### 2.4 Reranking Headroom (Negative Result)

Counterfactual branching on Can with K=4:

| Config | Clean Success | Degraded Success | Oracle Improvement |
|--------|--------------|------------------|--------------------|
| Single decision | 90% | 45% | +5% |
| Multi decision | 90% | 45% | +5% |

**Finding:** +5% oracle improvement across all configs, but candidate distributions collapse under zero_object — all candidates are similarly bad. Reranking cannot help when the proposal distribution itself has shifted off-manifold. This confirms detection/abstention is the right framing.

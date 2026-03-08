# TAP-Score Paper Section Drafts

## 1. Stress Regime Construction (Onset Curves)

We construct controlled partial-observability regimes by zeroing object-state
observations at a specified onset step during rollouts of pretrained Diffusion
Policy checkpoints. This simulates sudden occlusion where the policy loses
access to task-critical information mid-execution.

**Can (zero_object onset curve, n=20 per point):**

| Onset Step | Success Rate | Mean Return |
|------------|-------------|-------------|
| 0 | 0/20 (0%) | 0.0 |
| 25 | 0/20 (0%) | 0.0 |
| 50 | 0/20 (0%) | 0.0 |
| 75 | 2/20 (10%) | 31.9 |
| 80 | 9/20 (45%) | 140.6 |
| 85 | 5/10 (50%) | 155.7 |
| 90 | 8/10 (80%) | 245.6 |
| 100 | 17/20 (85%) | 258.9 |
| 150 | 19/20 (95%) | 288.7 |
| clean | 18/20 (90%) | 273.8 |

The onset curve exhibits a sharp phase transition between steps 75-100. We
select onset=80 (~45% success) as the primary stress regime: policies fail
roughly half the time, providing a balanced mix for detection evaluation.

**Lift (zero_object onset curve, n=20 per point):**

| Onset Step | Success Rate | Mean Return |
|------------|-------------|-------------|
| 0 | 0/20 (0%) | 0.0 |
| 25 | 6/20 (30%) | 1.8 |
| 50 | 20/20 (100%) | 24.8 |
| 100 | 20/20 (100%) | 73.0 |
| clean | 20/20 (100%) | 339.2 |

Lift's transition is earlier (25-50 steps), consistent with its simpler
dynamics: the policy commits to a lift trajectory early and needs object
information only briefly.


## 2. Oracle Headroom Under Occlusion (Negative Result)

Before pivoting to detection, we evaluated whether best-of-K reranking could
rescue policy performance under occlusion. We tested oracle selection (choosing
the candidate that maximizes ground-truth return) across four configurations:

| Configuration | K | Decision | Baseline SR | Oracle SR | Gain |
|--------------|---|----------|-------------|-----------|------|
| Can zero_80, multi-decision (di=2) | 4 | every 2 chunks | 9/20 (45%) | 10/20 (50%) | +1/20 |
| Can zero_80, single decision | 4 | at onset only | 9/20 (45%) | 10/20 (50%) | +1/20 |
| Can zero_80, anticipatory (step 72) | 8 | before onset | 9/20 (45%) | 10/20 (50%) | +1/20 |
| Can freeze_80, multi-decision (di=2) | 4 | every 2 chunks | 19/20 (95%) | 20/20 (100%) | +1/20 |

All four configurations show exactly +1/20 episodes gained under oracle
selection. The maximum possible gain from any reranking strategy is 5
percentage points.

**Root cause: candidate distribution collapse.** Under zero_object occlusion,
the policy loses access to the object's position, quaternion, and relative
pose. Since all K candidates are generated from the same (degraded)
observation, they propose similarly misguided actions. Sampling diversity is
insufficient to compensate for the missing signal --- the candidates are "all
bad in similar ways." This is a structural limitation of reranking under
information loss: selection cannot rescue a policy that lacks the evidence to
generate any good candidate.

This finding motivates the pivot from reranking to detection: TAP-Score cannot
fix a degraded policy, but it can flag when the policy's proposed actions
diverge from expert behavior.


## 3. TAP-Score Failure Detection

### 3.1 Push-T (Canonical Benchmark)

TAP-Score achieves AUROC 0.994 on Push-T for distinguishing successful
trajectories from held-out failure modes (scaling, bias, stuck, delayed),
substantially outperforming an action-magnitude baseline (AUROC 0.636). The
detector remains stable across the number of contrastive negatives per query
(M=7: 1.000; M=15: 0.992; M=31: 0.995). Using only the first 70% prefix of
each trajectory yields AUROC 1.000, indicating strong early-warning capability
before failure becomes evident from environment state alone.

| Metric | Value |
|--------|-------|
| AUROC (held-out failures) | 0.994 |
| Action magnitude baseline | 0.636 |
| Prefix-only AUROC (70%) | 1.000 |
| TPR @ 1% FPR | 97.5% |
| TPR @ 5% FPR | 97.5% |
| TPR @ 10% FPR | 97.5% |

*40 success + 40 failure episodes. Failure modes held out from training
(trained on noise, permute, mirror, random; tested on scaling, bias, stuck,
delayed).*

### 3.2 Robomimic Can Under Occlusion

Under zero_object onset=80 (the stress regime where Diffusion Policy succeeds
~45% of the time), TAP-Score detects episode-level failure with AUROC 0.750
using the minimum per-episode score as the detection statistic.

| Metric | Value | 95% CI |
|--------|-------|--------|
| Episodes | 50 (20 success, 30 fail) | --- |
| AUROC (min score) | 0.750 | [0.604, 0.882] |
| AUROC (mean score) | 0.593 | --- |
| Lead time (median) | 264 env steps | [252, 292] |
| Failed eps detected | 22/30 (73%) | --- |

The performance gap between Push-T (0.994) and Can (0.750) reflects the
nature of the failure mechanism. In Push-T, held-out perturbations produce
action chunks that are structurally incompatible with expert behavior
(wrong scale, stuck, delayed). Under occlusion in Can, the policy still
generates "action-plausible" predictions --- the actions have correct magnitude
and timing but target the wrong spatial location because the object position is
missing. TAP-Score partially detects this degradation through the observation
encoder's sensitivity to the zeroed fields, but the action encoder cannot
distinguish spatially-correct from spatially-incorrect manipulation actions
without object context.

### 3.3 Abstention Improves Reliability

By ranking episodes by their minimum TAP-Score and executing only the
top-scoring fraction, we convert the detector into an abstention trigger that
improves reliability at the cost of coverage:

**Can zero_object onset=80 abstention curve:**

| Coverage | N Kept | Success Rate | 95% CI |
|----------|--------|-------------|--------|
| 10% | 5 | 60% | --- |
| 20% | 10 | 70% | [30%, 90%] |
| 50% | 25 | 64% | --- |
| 100% | 50 | 40% (baseline) | --- |

At 20% coverage, abstention lifts success rate from 40% to 70% (+30pp). The
wide confidence interval (30-90%) reflects the small sample size and motivates
future work with larger evaluation sets.

The median lead time of 264 env steps (out of 400 total, with perturbation
onset at step 80) means the detector flags failures roughly 264 steps before
the episode terminates --- providing substantial advance warning for an
intervention policy or human operator.


## 4. Reranking Headroom Collapses Under Hard Occlusion

(One-paragraph version for related work or discussion)

We evaluated oracle best-of-K reranking under zero_object occlusion on
robomimic Can (onset=80, baseline 45% success). Across four configurations
(K=4 multi-decision, K=4 single-decision, K=8 anticipatory, K=4 freeze), the
maximum oracle gain was +1 episode out of 20 in every case. The candidate
distribution collapses when key observation signals are missing: all K
candidates propose similarly misguided actions, so selection among them cannot
rescue performance. This structural limitation motivates using TAP-Score as a
failure detector rather than a reranker under partial observability.

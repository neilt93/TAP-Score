# Proposed Ideas (Requirement 3)

## Core Insight

> When observations shift, imitation policies propose actions that deviate from the expert manifold. A learned "actionness" score can detect this deviation and optionally correct it.

---

## Proposed Ideas

### Idea 1: TAP-Score (Temporal Action-Proposal Score) ← **IMPLEMENTED**

**What:** Train a classifier to score (observation window, action chunk) pairs as expert-like or not.

**Training:**
- Positive: (obs, matching expert action)
- Negative: (obs, mismatched action from different time/trajectory)
- Negative: (obs, noisy action)

**Usage:**
- Low score → OOD detected
- Resample K actions, pick highest score → improved robustness

**Why it works:** Learns the expert action manifold without corruption labels. Under perturbations, policy actions drift off-manifold → score drops.

---

### Idea 2: Conformal Calibration

**What:** Choose threshold τ so that clean false-alarm rate ≤ α (e.g., 5%).

**How:** On clean validation set, τ = quantile(TAP scores, α).

**Benefit:** Principled control of false alarms, safety-minded design.

---

### Idea 3: Ensemble Disagreement

**What:** Train 3 TAP models, use variance/disagreement as uncertainty.

**Benefit:** Often more robust than single-model confidence.

---

### Idea 4: Temporal Consistency Auxiliary

**What:** Add auxiliary loss predicting next observation features.

**Benefit:** High prediction error indicates drift/OOD.

---

### Idea 5: Phase-Conditioned Scoring

**What:** Learn coarse phase (reach/push/finish), condition TAP on phase.

**Benefit:** Reduces false alarms by accounting for expected action variation.

---

### Idea 6: Adaptive Action Horizon

**What:** When TAP score is low, shorten action horizon (more frequent replanning).

**Benefit:** Conservative behavior under uncertainty.

---

## Selected Idea: TAP-Score + Conformal Calibration

**Reasons:**
1. Clean training procedure (no corruption labels)
2. Direct connection to action manifold reasoning
3. Optional intervention (resampling) for closed-loop benefit
4. Calibration gives controlled false-alarm rate
5. Feasible to implement end-to-end

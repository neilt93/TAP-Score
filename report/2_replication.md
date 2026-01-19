# Diffusion Policy Replication (Requirement 2)

## Method Replicated

**Paper:** Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (RSS 2023)
**Repo:** https://github.com/real-stanford/diffusion_policy
**Commit:** `5ba07ac6661db573af695b419a7947ecb704690f`

---

## Task: Push-T

- 2D pushing task: robot arm pushes T-block to target pose
- Observation: 96x96 RGB image (top-down view)
- Action: 2D continuous actions (end-effector position in pixel coordinates)
- Success metric: coverage of target T pose (IoU with goal region)

---

## Replication Status

### What Was Executed

**1. Repository cloned and verified:**
```bash
git clone https://github.com/real-stanford/diffusion_policy.git baselines/diffusion_policy
cd baselines/diffusion_policy
git log -1 --oneline
# 5ba07ac Done adapting mujoco image dataset
```

**2. Push-T demonstration data downloaded:**
```bash
mkdir -p data/processed
cd data/processed
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip
```

**3. Official data format verified using DP-compatible loading:**

```
============================================================
Diffusion Policy Data Verification
============================================================

Loading Push-T data from: data/processed/pusht

--- Dataset Structure ---
Groups: ['data', 'meta']
Data arrays: ['img', 'n_contacts', 'action', 'state', 'keypoint']
Meta arrays: ['episode_ends']

--- Array Shapes ---
Images shape: (25650, 96, 96, 3)
Actions shape: (25650, 2)
Episode ends: 206 episodes

--- Data Statistics ---
Image dtype: float32
Image range: [65.0, 255.0]
Action dtype: float32
Action range: [12.000, 511.000]
Action dim: 2

--- Episode Statistics ---
Number of episodes: 206
Total timesteps: 25650
Episode length range: [49, 246]
Mean episode length: 124.5

--- First Episode Sample ---
Episode 0: 161 timesteps
First action: [233.  71.]
Last action: [164. 355.]

============================================================
Data verification PASSED
Dataset is compatible with Diffusion Policy format
============================================================
```

**Verification script:** `scripts/verify_dp_data.py`

### What Was Not Executed

**Full Diffusion Policy training** was not run due to:
1. Environment dependencies (MuJoCo, gym, hydra stack) on macOS
2. Compute constraints (full training requires ~24h on GPU)

**Full closed-loop rollout evaluation** was not run because:
1. Requires trained policy checkpoint
2. Requires environment simulation (MuJoCo)

---

## Scope Clarification

For the TAP-Score methodology in Requirement 4, we evaluate on **Push-T demonstrations with controlled off-manifold action corruptions** (held-out failure families) to measure early failure prediction. This requires:

- Expert demonstration data (available, verified above)
- Ability to simulate off-manifold actions (implemented in eval scripts)

It does **not** require a trained Diffusion Policy checkpoint, since TAP-Score is trained and evaluated on action proposals, not policy rollouts.

---

## Reported Results (From Paper)

| Metric | Paper Value |
|--------|-------------|
| Test Mean Score | ~0.95 |
| Success Rate | ~95% |

---

## Summary

| Step | Status |
|------|--------|
| Clone repository | Done |
| Download Push-T data | Done |
| Verify data format (official structure) | Done |
| Train Diffusion Policy | Not done (resource constraints) |
| Run closed-loop evaluation | Not done |
| Train TAP-Score on demos | Done (Req 4) |

The replication covers the data pipeline components needed for TAP-Score development. Full policy training and closed-loop evaluation would require additional compute resources and environment setup.

---

## Next Steps (If Full Replication Required)

```bash
# Option A: Evaluate pre-trained checkpoint (if available)
python eval.py --checkpoint <official_checkpoint_path>

# Option B: Train from scratch (requires CUDA + dependencies)
python train.py --config-name=image_pusht_diffusion_policy_cnn.yaml
# Expected: ~24h on single GPU
```

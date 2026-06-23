# TAP-Score Final Artifacts

## Frozen Settings

- **Primary robomimic setting**: Can, zero_object, onset=80
- **Detection statistic**: episode mean risk score for Can/Lift; episode min score for the Square negative result
- **Negative pool**: 2000 expert action chunks, M=15 negatives per scoring

## Model Checkpoints

Checkpoint directories are ignored by git because they are large. The paths below
are the expected local/generated locations used by the saved result metadata, not
guaranteed tracked repository files.

| Benchmark | Expected Path | Config |
|-----------|---------------|--------|
| Push-T | `checkpoints_contrastive/pusht/contrastive_tap_best.pt` | image obs, action_dim=2, action_chunk=16 |
| Can (lowdim) | `checkpoints_contrastive/robomimic_can_lowdim/risk_tap_best.pt` | state obs, obs_dim=23, action_dim=7, action_chunk=16 |
| Lift (lowdim) | `checkpoints_contrastive/robomimic_lift_lowdim/risk_tap_best.pt` | state obs, action_dim=7, action_chunk=16 |
| Square (lowdim) | `checkpoints_contrastive/robomimic_square_lowdim/risk_tap_best.pt` | state obs, action_dim=7, action_chunk=16 |

## Cited Result Files

| Result | File | Key Numbers |
|--------|------|-------------|
| Push-T AUROC | `eval_results/pusht_eval_results.json` | AUROC 0.994, prefix 1.000, mag baseline 0.636 |
| Can risk detection (n=100) | `eval_results/risk_detection_can_zero80_onset_n100.json` | AUROC 0.856, abstention@20% -> 95%, baseline 45% |
| Can risk detection CSV | `eval_results/risk_detection_can_zero80_onset_n100.csv` | Per-episode scores, 100 rows |
| Lift risk detection (n=100) | `eval_results/risk_detection_lift_zero31_epwin_n100.json` | AUROC 0.811, abstention@20% -> 90%, baseline 47% |
| Lift risk detection CSV | `eval_results/risk_detection_lift_zero31_epwin_n100.csv` | Per-episode scores, 100 rows |
| Square risk detection (n=50) | `eval_results/risk_detection_square_zero80_n50.json` | AUROC 0.523, negative result |
| Can onset curve | `eval_results/perturb_can_zero*.json` | 0% at 0-50, 50% at 80, 90% at clean |

## Reproduction Commands

### Push-T: Train + Evaluate

```bash
# Train contrastive TAP-Score on Push-T (30 epochs, ~5 min on RTX 4080)
python train_contrastive_tap.py --benchmark pusht --epochs 30 --batch_size 64 \
  --data_dir data/processed/pusht/pusht_cchi_v7_replay.zarr

# Evaluate (AUROC + operating points + M ablation + prefix)
python eval_tap_final.py --benchmark pusht --n_episodes 40 --device cuda \
  --data_dir data/processed/pusht/pusht_cchi_v7_replay.zarr
```

### Can: Train Risk TAP-Score

```bash
python train_risk_tap.py \
  --rollout_dir data/risk_rollouts/can/ \
  --task can --epochs 30 --batch_size 64 \
  --fail_horizon 64 --hard_mining_epoch 10
```

### Can: Detection Evaluation (n=100, zero_object onset=80)

```bash
python scripts/eval_tap_detection.py \
  --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
  --risk_model checkpoints_contrastive/robomimic_can_lowdim/risk_tap_best.pt \
  --n_episodes 100 --perturb zero_object --perturb_start_step 80 \
  --output eval_results/risk_detection_can_zero80_onset_n100.json --device cuda

# Includes magnitude baseline, abstention curve, lead times, and bootstrap 95% CIs.
```

### Can: Perturbation Onset Sweep (headroom audit infrastructure)

```bash
# Example: zero_object onset=80, K=1 baseline, n=20
python scripts/robomimic_headroom_audit.py \
  --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
  --n_episodes 20 --K 1 --perturb zero_object --perturb_start_step 80 --kp 500
```

## Bootstrap CIs (Final risk detection)

| Metric | Point | 95% CI |
|--------|-------|--------|
| Can AUROC (mean risk score) | 0.856 | [0.779, 0.915] |
| Can success @ 20% coverage | 0.950 | [0.700, 1.000] |
| Can lead time median (env steps) | 320 | [320, 320] |
| Lift AUROC (mean risk score) | 0.811 | [0.718, 0.892] |
| Lift success @ 20% coverage | 0.900 | [0.700, 1.000] |
| Lift lead time median (env steps) | 360 | [360, 368] |

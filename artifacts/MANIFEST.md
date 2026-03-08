# TAP-Score Final Artifacts

## Frozen Settings

- **Primary robomimic setting**: Can, zero_object, onset=80
- **Detection statistic**: episode min TAP-Score (not mean)
- **Negative pool**: 2000 expert action chunks, M=15 negatives per scoring

## Model Checkpoints

| Benchmark | Path | Config |
|-----------|------|--------|
| Push-T | `checkpoints_contrastive/pusht/contrastive_tap_best.pt` | image obs, action_dim=2, action_chunk=16 |
| Can (lowdim) | `checkpoints_contrastive/robomimic_can_lowdim/contrastive_tap_best.pt` | state obs, obs_dim=23, action_dim=7, action_chunk=16 |

## Cited Result Files

| Result | File | Key Numbers |
|--------|------|-------------|
| Push-T AUROC | `eval_results/pusht_eval_results.json` | AUROC 0.994, prefix 1.000, mag baseline 0.636 |
| Can detection (n=50) | `eval_results/detection_can_zero80_n50.json` | AUROC 0.750, abstention@20% -> 70% |
| Can detection CSV | `eval_results/detection_can_zero80_n50.csv` | Per-step scores, 2000 rows |
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

### Can: Train TAP-Score

```bash
python train_contrastive_tap.py --benchmark robomimic_can_lowdim --epochs 30 --batch_size 64
```

### Can: Detection Evaluation (n=50, zero_object onset=80)

```bash
python scripts/eval_tap_detection.py \
  --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
  --tap_checkpoint checkpoints_contrastive/robomimic_can_lowdim/contrastive_tap_best.pt \
  --n_episodes 50 --perturb zero_object --perturb_start_step 80 \
  --output eval_results/detection_can_zero80_n50.csv --device cuda

# Now includes: magnitude baseline AUROC + bootstrap 95% CIs
```

### Can: Perturbation Onset Sweep (headroom audit infrastructure)

```bash
# Example: zero_object onset=80, K=1 baseline, n=20
python scripts/robomimic_headroom_audit.py \
  --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
  --n_episodes 20 --K 1 --perturb zero_object --perturb_start_step 80 --kp 500
```

## Bootstrap CIs (Can detection, n=50, 10000 bootstrap resamples)

| Metric | Point | 95% CI |
|--------|-------|--------|
| AUROC (min score) | 0.750 | [0.604, 0.882] |
| Success @ 20% coverage | 0.700 | [0.300, 0.900] |
| Lead time median (env steps) | 264 | [252, 292] |

# TAP-Score: Runtime Risk Monitor for Diffusion Policy

**Author:** Neil Tripathi

TAP-Score is a lightweight runtime monitor that predicts when a Diffusion Policy will fail, enabling selective execution — abstaining on risky actions to boost success rates. A simple risk MLP trained on real rollout data achieves AUROC 0.856 on robomimic Can and 0.811 on Lift, nearly doubling success rates at 20% coverage compared to unmonitored execution.

## Key Results

### Per-Episode Failure Detection (n=100)

| Task | AUROC | AUPRC | 95% CI | Val AUROC |
|------|-------|-------|--------|-----------|
| Can (zero80) | **0.856** | 0.881 | [0.779, 0.915] | 0.973 |
| Lift (zero31) | **0.811** | 0.821 | [0.718, 0.892] | 0.954 |
| Square (zero80) | 0.523 | — | — | — |

### Selective Execution (success rate at coverage level)

| Coverage | Can (TAP) | Can (baseline) | Lift (TAP) | Lift (baseline) |
|----------|-----------|----------------|------------|-----------------|
| 10% | 100% | 45% | 80% | 47% |
| 20% | 95% | 45% | 90% | 47% |
| 30% | 80% | 45% | 83% | 47% |
| 50% | 70% | 45% | 68% | 47% |

## How It Works

1. **Collect rollouts** under diverse perturbation regimes (clean, zero-object, dropout, freeze) to get labeled success/failure episodes with per-step observations and actions.
2. **Train a risk MLP** — `P(fail within H steps | obs, action)` — on the collected data. The model is a pair of small encoders (obs + action) whose outputs are concatenated and passed through a classifier head.
3. **Score at runtime** — at each policy step, the risk model outputs a failure probability. Episodes with high mean risk are flagged for abstention.

## Reproduction

```bash
# Setup
./setup.sh
source venv/bin/activate

# 1. Collect rollout data
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
  --n_episodes 100 --output eval_results/risk_detection_can_zero80_onset_n100.json

# 4. Selective execution analysis
python scripts/eval_abstention.py --results eval_results/risk_detection_can_zero80_onset_n100.json
```

Or run the full pipeline: `bash scripts/run_risk_pipeline.sh`

## Project Structure

```
TAP-Score/
├── tap/                              # Core library
│   ├── risk.py                       # Risk model + rollout datasets
│   ├── contrastive.py                # Contrastive model (Push-T)
│   ├── benchmarks.py                 # Benchmark registry (6 tasks)
│   ├── data_loaders.py               # Episode loaders (zarr/npy)
│   ├── perturbations.py              # Synthetic perturbations
│   └── env_wrapper.py                # Push-T state wrapper
├── train_risk_tap.py                 # Train risk model (robomimic)
├── train_contrastive_tap.py          # Train contrastive model (Push-T)
├── eval_tap_final.py                 # Contrastive eval (Push-T)
├── scripts/
│   ├── collect_risk_rollouts.py      # Collect labeled rollout data
│   ├── eval_tap_detection.py         # Runtime detection eval
│   ├── eval_abstention.py            # Abstention curve eval
│   ├── analyze_selective_execution.py # Selective execution analysis
│   ├── plot_early_warning.py         # Early warning plots
│   ├── robomimic_headroom_audit.py   # Oracle best-of-K headroom
│   └── run_risk_pipeline.sh          # Full pipeline script
├── archive/                          # Superseded work (reranking, Phase 1)
├── artifacts/                        # Frozen figures + paper drafts
├── eval_results/                     # Final evaluation results
├── report/                           # Method writeup
├── data/                             # Datasets + rollouts
└── baselines/                        # Diffusion Policy repo
```

## Push-T Prototype (Phase 1)

An earlier contrastive scorer (InfoNCE) on Push-T achieves AUROC 0.994 on held-out synthetic failure types (97.5% TPR @ 1% FPR). See `tap/contrastive.py` and `eval_tap_final.py`.

## References

- Chi et al. (2024). *Diffusion Policy: Visuomotor Policy Learning via Action Score Gradients.* RSS 2024.
- Chi et al. (2025). *Demystifying Diffusion Policy.* arXiv.

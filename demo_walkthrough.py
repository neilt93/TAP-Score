#!/usr/bin/env python3
"""
TAP-Score Project Walkthrough
Run this to get a guided tour of every file and what it does.
Usage: python demo_walkthrough.py
"""

import os
import time

# ANSI colors
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"

def section(title):
    print(f"\n{BOLD}{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}{RESET}\n")

def file_entry(path, description, key_detail=None):
    print(f"  {GREEN}{path:<45}{RESET} {description}")
    if key_detail:
        print(f"  {DIM}{' '*45} {key_detail}{RESET}")

def pause():
    input(f"\n  {DIM}[Press Enter to continue]{RESET}")

def main():
    print(f"""
{BOLD}{CYAN}
  ╔══════════════════════════════════════════════════════════════╗
  ║        TAP-Score: Temporal Action-Proposal Scoring          ║
  ║        Off-Manifold Detection in Imitation Learning         ║
  ╚══════════════════════════════════════════════════════════════╝
{RESET}
  Detects when a visuomotor policy proposes actions inconsistent
  with expert behavior. Trained on demos, tested on live Diffusion
  Policy rollouts.
""")
    pause()

    # ----------------------------------------------------------------
    section("1. CORE MODEL  (tap/)")
    print("  The TAP-Score model: two-encoder contrastive architecture.\n")

    file_entry("tap/contrastive.py",
               "Core model: ContrastiveTAPScore",
               "CNN obs encoder + MLP action encoder + InfoNCE loss")

    file_entry("tap/dataset.py",
               "Training data: positive/negative sampling",
               "Hard negatives: permute, mirror, noise, random")

    file_entry("tap/benchmarks.py",
               "Benchmark registry: pusht, blockpush, kitchen",
               "Each defines obs_type, action_dim, data paths")

    file_entry("tap/data_loaders.py",
               "Unified data loading for zarr and npy formats")

    file_entry("tap/perturbations.py",
               "Observation perturbation functions",
               "Gaussian noise, blur, brightness, occlusion")

    file_entry("tap/resampling.py",
               "Intervention logic (resample from expert)")

    pause()

    # ----------------------------------------------------------------
    section("2. TRAINING")
    print("  How TAP-Score is trained.\n")

    file_entry("train_contrastive_tap.py",
               "Main training script (USE THIS)",
               "InfoNCE contrastive ranking, --action_chunk 8 for DP")

    file_entry("train_tap.py",
               "Legacy BCE training (DO NOT USE)",
               "Collapsed to constant output -- kept for reference")

    print(f"""
  {YELLOW}Key training command:{RESET}
    python train_contrastive_tap.py --benchmark pusht --action_chunk 8 --epochs 30

  {YELLOW}What happens:{RESET}
    1. Load Push-T expert demos (206 episodes, zarr format)
    2. For each sample: 1 correct action + 15 negatives
    3. Model learns to rank correct action highest (InfoNCE)
    4. Saves best checkpoint to checkpoints_contrastive/pusht/
""")
    pause()

    # ----------------------------------------------------------------
    section("3. EVALUATION -- Synthetic Failures (Offline)")
    print("  Test TAP on expert data with synthetic action corruptions.\n")

    file_entry("eval_tap_final.py",
               "Main synthetic eval: AUROC, operating points, M ablation",
               "Held-out failures: scaling, bias, stuck, delayed")

    file_entry("eval_contrastive_tap.py",
               "Contrastive-specific evaluation")

    file_entry("eval_ablations.py",
               "Ablation studies (M negatives, temperature)")

    file_entry("eval_tap_robust.py",
               "Robustness evaluation across perturbation types")

    file_entry("eval_episode_tap.py",
               "Episode-level analysis with score traces over time")

    file_entry("eval_resampling_intervention.py",
               "Test intervention: resample action when TAP flags")

    file_entry("plot_score_trace.py",
               "Visualization: TAP score traces for success vs failure")

    print(f"""
  {YELLOW}Key result (synthetic):{RESET}
    Held-out Failure AUROC:  0.998
    Prefix-Only AUROC:       0.997  (early warning)
    TPR @ 1% FPR:            94.3%
""")
    pause()

    # ----------------------------------------------------------------
    section("4. EVALUATION -- Live Diffusion Policy (End-to-End)")
    print("  The real test: score a live DP in closed-loop Push-T.\n")

    file_entry("eval_dp_with_tap.py",
               "MAIN INTEGRATION SCRIPT",
               "Batched DP rollouts + TAP scoring + detection metrics")

    print(f"""
  {YELLOW}What this script does:{RESET}
    1. Load Diffusion Policy checkpoint + TAP-Score checkpoint
    2. Run 50 clean + 50 perturbed rollouts (paired seeds)
    3. Score every policy proposal with TAP log-margin
    4. Compute AUROC, operating points, early warning metrics

  {YELLOW}Key design choices:{RESET}
    - Paired seeds: same initial states for clean vs perturbed
    - Clean-only calibration: threshold set on clean successes only
    - Trigger-style detection: flag if min(scores) < tau
    - H=8 action chunk: matches DP horizon, no stitching needed

  {YELLOW}Key result (live DP, H=8):{RESET}
    Clean vs Perturbed AUROC:   0.812
    Success vs Failure AUROC:   0.690
    TPR @ 5% FPR:               73.6%
    Failures flagged early:     88.7%
    Mean lead time:             17.2 policy steps
""")
    pause()

    # ----------------------------------------------------------------
    section("5. REPORT")
    print("  Written deliverables.\n")

    file_entry("report/1_sota.md",
               "SOTA identification: Diffusion Policy (RSS 2023)")

    file_entry("report/2_replication.md",
               "Baseline replication: data verification, partial training")

    file_entry("report/3_ideas.md",
               "Proposed ideas: TAP-Score, phase-conditioning, etc.")

    file_entry("report/4_method_and_results.md",
               "Full method writeup + all results")

    file_entry("main.pdf",
               "Paper PDF")

    pause()

    # ----------------------------------------------------------------
    section("6. DATA & CHECKPOINTS")

    file_entry("data/pusht/",
               "Push-T expert demonstrations (206 episodes, zarr)")

    file_entry("checkpoints_contrastive/pusht/",
               "TAP-Score checkpoints",
               "contrastive_tap_h8_best.pt + contrastive_tap_h16_old.pt")

    file_entry("eval_results/dp_with_tap/",
               "Live DP integration results (JSON)",
               "dp_with_tap_h8.json + dp_with_tap_h16.json")

    file_entry("results/",
               "Synthetic eval results + score trace figures")

    pause()

    # ----------------------------------------------------------------
    section("7. UTILITIES")

    file_entry("scripts/download_data.sh",
               "Download Push-T dataset")

    file_entry("scripts/download_benchmarks.sh",
               "Download all benchmarks (pusht, blockpush, kitchen)")

    file_entry("scripts/run_replication.sh",
               "Clone Diffusion Policy baseline repo")

    file_entry("scripts/verify_dp_data.py",
               "Verify downloaded DP data format")

    file_entry("scripts/eval_all_benchmarks.py",
               "Cross-benchmark comparison script")

    file_entry("setup.sh",
               "Environment setup (venv + deps)")

    file_entry("requirements.txt",
               "Python dependencies")

    # ----------------------------------------------------------------
    section("SUMMARY")
    print(f"""
  {BOLD}The story in 30 seconds:{RESET}

  1. Diffusion Policy generates action chunks but has no failure signal
  2. TAP-Score learns expert action manifold via contrastive ranking
  3. At inference: score each proposal against negatives (log-margin)
  4. If score drops below threshold -> flag failure, intervene early

  {BOLD}Synthetic results:{RESET}  AUROC 0.998, 94.3% TPR @ 1% FPR
  {BOLD}Live DP results:{RESET}    AUROC 0.812 (clean vs perturbed)
                        73.6% TPR @ 5% FPR
                        88.7% failures caught early

  {CYAN}Total files: 10 Python scripts, 7 tap/ modules, 4 report docs{RESET}
""")


if __name__ == "__main__":
    main()

"""
Evaluate TAP-Score for Off-Manifold Detection

Usage:
    python eval_tap.py --checkpoint checkpoints/tap_best.pt --data_dir data/processed/pusht
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from tap import build_tap_model, TAPDataset, TAPCalibrator, evaluate_detection
from tap.perturbations import PERTURBATIONS, get_perturbation
from tap.resampling import evaluate_resampling


def score_dataset(model, dataset, device, perturbation=None, n_samples=500, positives_only=True):
    """Score samples from dataset, optionally with perturbation.

    Args:
        model: TAP-Score model
        dataset: TAPDataset instance
        device: torch device
        perturbation: Optional perturbation function to apply
        n_samples: Number of samples to score
        positives_only: If True, only score positive (expert) samples.
                       This is the correct setting for clean baseline evaluation.
    """
    model.eval()
    scores = []

    if positives_only:
        # Only sample from positive indices (first half of dataset = expert pairs)
        num_positives = len(dataset.valid_indices)
        indices = np.random.choice(num_positives, min(n_samples, num_positives), replace=False)
    else:
        indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Scoring", leave=False):
            if positives_only:
                # Directly get positive sample to ensure we're scoring expert pairs
                sample = dataset._get_positive(idx)
            else:
                sample = dataset[idx]

            obs = sample['obs'].unsqueeze(0)  # (1, T, C, H, W)

            # Apply perturbation if specified
            if perturbation is not None:
                obs_np = obs.numpy()
                obs_np = perturbation(obs_np)
                obs = torch.from_numpy(obs_np)

            obs = obs.to(device)
            actions = sample['actions'].unsqueeze(0).to(device)

            score = model(obs, actions).item()
            scores.append(score)

    return np.array(scores)


def score_control_expert_action(model, dataset, device, perturbation, n_samples=500):
    """Control: perturbed obs + MATCHING expert action.

    If TAP is just detecting pixel shift, this will collapse to ~0.
    If TAP learned the action manifold, this should stay reasonably high.
    """
    model.eval()
    scores = []
    num_positives = len(dataset.valid_indices)
    indices = np.random.choice(num_positives, min(n_samples, num_positives), replace=False)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Expert-action control", leave=False):
            sample = dataset._get_positive(idx)
            obs = sample['obs'].unsqueeze(0)

            # Apply perturbation to observation
            obs_np = obs.numpy()
            obs_np = perturbation(obs_np)
            obs = torch.from_numpy(obs_np).to(device)

            # Keep the MATCHING expert action (not mismatched)
            actions = sample['actions'].unsqueeze(0).to(device)

            score = model(obs, actions).item()
            scores.append(score)

    return np.array(scores)


def score_control_action_only(model, dataset, device, n_samples=500):
    """Control: CLEAN obs + mismatched action.

    TAP should still score this low, confirming it learned the action manifold.
    """
    model.eval()
    scores = []
    num_positives = len(dataset.valid_indices)
    indices = np.random.choice(num_positives, min(n_samples, num_positives), replace=False)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Action-only control", leave=False):
            sample = dataset._get_positive(idx)
            obs = sample['obs'].unsqueeze(0).to(device)  # Clean observation

            # Get MISMATCHED action from different sample
            other_idx = (idx + num_positives // 2) % num_positives
            other_sample = dataset._get_positive(other_idx)
            actions = other_sample['actions'].unsqueeze(0).to(device)

            score = model(obs, actions).item()
            scores.append(score)

    return np.array(scores)


def plot_score_distributions(results, output_path):
    """Plot score distributions for clean vs perturbed."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Only include perturbations that have 'auroc' key (exclude controls)
    perturbations = [k for k in results.keys() if k not in ['clean', 'control_action_only'] and 'auroc' in results.get(k, {})]

    for idx, pert_name in enumerate(perturbations[:4]):
        ax = axes[idx]

        clean_scores = results['clean']['scores']
        pert_scores = results[pert_name]['scores']

        ax.hist(clean_scores, bins=30, alpha=0.6, label='Clean', color='blue', density=True)
        ax.hist(pert_scores, bins=30, alpha=0.6, label=pert_name, color='red', density=True)

        auroc = results[pert_name]['auroc']
        ax.set_title(f'{pert_name} (AUROC: {auroc:.3f})')
        ax.set_xlabel('TAP Score')
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved distribution plot to {output_path}")
    plt.close()


def plot_summary_table(results, output_path):
    """Create summary results figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Only include perturbations that have 'auroc' key (exclude controls)
    perturbations = [k for k in results.keys() if k not in ['clean', 'control_action_only'] and 'auroc' in results.get(k, {})]

    data = []
    for pert in perturbations:
        data.append({
            'Perturbation': pert,
            'Clean Mean': f"{results['clean']['mean']:.3f}",
            'Pert Mean': f"{results[pert]['mean']:.3f}",
            'AUROC': f"{results[pert]['auroc']:.3f}",
        })

    # Create table
    columns = ['Perturbation', 'Clean Mean', 'Pert Mean', 'AUROC']
    cell_text = [[d[c] for c in columns] for d in data]

    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    ax.set_title('TAP-Score Detection Results', fontsize=14, fontweight='bold', pad=20)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary table to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate TAP-Score for off-manifold detection")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tap_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed/pusht")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("TAP-Score Evaluation")
    print("=" * 60)

    device = torch.device(args.device)
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

    model = build_tap_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} (val_acc: {checkpoint['val_acc']:.3f})")

    # Load dataset
    print(f"\nLoading data from {args.data_dir}...")
    dataset = TAPDataset(
        args.data_dir,
        obs_window=config['obs_window'],
        action_chunk=config['action_chunk'],
    )

    # Evaluate clean
    print("\nScoring clean samples...")
    clean_scores = score_dataset(model, dataset, device, perturbation=None, n_samples=args.n_samples)

    results = {
        'clean': {
            'scores': clean_scores.tolist(),
            'mean': float(np.mean(clean_scores)),
            'std': float(np.std(clean_scores)),
        }
    }

    # Calibrate
    calibrator = TAPCalibrator(target_false_alarm=0.05)
    calibrator.fit(clean_scores)
    threshold = calibrator.threshold

    # CONTROL 1: Action-only (clean obs + mismatched action)
    print("\n" + "-" * 40)
    print("CONTROL: Clean obs + mismatched action")
    print("-" * 40)
    action_only_scores = score_control_action_only(model, dataset, device, n_samples=args.n_samples)
    results['control_action_only'] = {
        'scores': action_only_scores.tolist(),
        'mean': float(np.mean(action_only_scores)),
        'std': float(np.std(action_only_scores)),
    }
    print(f"  Mean score: {np.mean(action_only_scores):.3f} (should be LOW if TAP learned action manifold)")

    # Evaluate perturbations with controls
    perturbation_names = ['brightness_low', 'noise_high', 'occlusion_large', 'translation']

    for pert_name in perturbation_names:
        print(f"\nScoring {pert_name}...")
        perturbation = get_perturbation(pert_name)

        # Original: perturbed obs + expert action (what we had before)
        pert_scores = score_dataset(model, dataset, device, perturbation=perturbation, n_samples=args.n_samples)

        # CONTROL 2: Expert-action control (perturbed obs + MATCHING expert action)
        expert_action_scores = score_control_expert_action(model, dataset, device, perturbation, n_samples=args.n_samples)

        detection_results = evaluate_detection(clean_scores, pert_scores, threshold)

        results[pert_name] = {
            'scores': pert_scores.tolist(),
            'mean': float(np.mean(pert_scores)),
            'std': float(np.std(pert_scores)),
            'expert_action_control_mean': float(np.mean(expert_action_scores)),
            'expert_action_control_std': float(np.std(expert_action_scores)),
            **detection_results,
        }

        print(f"  AUROC: {detection_results['auroc']:.3f}")
        print(f"  Pert+expert action mean: {np.mean(pert_scores):.3f}")
        print(f"  Expert-action control: {np.mean(expert_action_scores):.3f} (should stay HIGH if TAP learned action manifold)")

    # Print summary with controls
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nClean expert pairs mean: {results['clean']['mean']:.3f}")
    print(f"Mismatched action control: {results['control_action_only']['mean']:.3f} (verifies action manifold learning)")

    print(f"\n{'Perturbation':<18} {'AUROC':<8} {'Pert Mean':<12} {'Expert-Act Ctrl':<16}")
    print("-" * 60)
    for pert_name in perturbation_names:
        r = results[pert_name]
        print(f"{pert_name:<18} {r['auroc']:<8.3f} {r['mean']:<12.3f} {r['expert_action_control_mean']:<16.3f}")

    print("\nInterpretation:")
    print("- If 'Expert-Act Ctrl' stays HIGH: TAP learned action manifold, not just pixel detection")
    print("- If 'Expert-Act Ctrl' collapses to ~0: TAP is mostly detecting observation shift")

    # Evaluate TAP-guided resampling
    print("\n" + "=" * 60)
    print("TAP-GUIDED RESAMPLING EVALUATION")
    print("=" * 60)

    print("\nClean conditions:")
    resample_clean = evaluate_resampling(model, dataset, device, perturbation=None, n_samples=200, k=5)
    results['resampling_clean'] = resample_clean
    print(f"  TAP-selected mean: {resample_clean['tap_mean_score']:.3f}")
    print(f"  Random selected mean: {resample_clean['random_mean_score']:.3f}")
    print(f"  TAP selected expert rate: {resample_clean['tap_selected_expert_rate']:.1%}")
    print(f"  Improvement over random: {resample_clean['improvement_over_random']:.3f}")

    for pert_name in perturbation_names:
        print(f"\n{pert_name}:")
        perturbation = get_perturbation(pert_name)
        resample_pert = evaluate_resampling(model, dataset, device, perturbation=perturbation, n_samples=200, k=5)
        results[f'resampling_{pert_name}'] = resample_pert
        print(f"  TAP-selected mean: {resample_pert['tap_mean_score']:.3f}")
        print(f"  Random selected mean: {resample_pert['random_mean_score']:.3f}")
        print(f"  TAP selected expert rate: {resample_pert['tap_selected_expert_rate']:.1%}")
        print(f"  Improvement over random: {resample_pert['improvement_over_random']:.3f}")

    # Save results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Generate plots
    print("\nGenerating plots...")
    plot_score_distributions(results, output_dir / "figures" / "score_distributions.png")
    plot_summary_table(results, output_dir / "figures" / "results_summary.png")

    print(f"\nResults saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

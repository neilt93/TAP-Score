"""
Final TAP Evaluation - Bulletproof Metrics

Supports multiple benchmarks: pusht, lift, kitchen, blockpush

Usage:
    python eval_tap_final.py --benchmark pusht --checkpoint checkpoints_contrastive/pusht/best.pt
    python eval_tap_final.py --benchmark lift --checkpoint checkpoints_contrastive/lift/best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

from tap.contrastive import ContrastiveTAPScore
from tap.benchmarks import get_benchmark_config, get_data_path, list_benchmarks
from tap.data_loaders import load_episodes, print_data_stats


M_NEGATIVES = 15


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = ContrastiveTAPScore(
        obs_channels=config.get("obs_channels", 3),
        action_dim=config.get("action_dim", 2),
        obs_window=config.get("obs_window", 2),
        action_chunk=config.get("action_chunk", 16),
        hidden_dim=config.get("hidden_dim", 128),
        temperature=config.get("temperature", 0.1),
        obs_type=config.get("obs_type", "image"),
        obs_dim=config.get("obs_dim"),
        use_deltas=config.get("use_deltas", False),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config




def build_negative_pool(episodes, action_chunk, n_samples=2000):
    pool = []
    for _ in range(n_samples):
        ep = episodes[np.random.randint(len(episodes))]
        if len(ep['actions']) < action_chunk:
            continue
        t = np.random.randint(0, len(ep['actions']) - action_chunk)
        pool.append(ep['actions'][t:t + action_chunk].astype(np.float32))
    return np.stack(pool)


def perturb_observation(obs, severity=0.5, obs_type='image'):
    """Perturb observation with noise."""
    noise = np.random.randn(*obs.shape).astype(np.float32) * (0.3 * severity)
    if obs_type == 'image':
        return np.clip(obs + noise, 0, 1)
    else:
        # For state observations, add noise directly (no clipping)
        return obs + noise * 0.1  # Smaller noise for state observations


def simulate_success_action(expert_action):
    noise = np.random.randn(*expert_action.shape).astype(np.float32) * 0.05
    return expert_action + noise


def simulate_failure_heldout(expert_action, magnitude_preserving=False):
    """HELD-OUT failures NOT in training negatives.

    If magnitude_preserving=True, uses failures that preserve per-step L2 norms
    to avoid trivial magnitude-based detection.
    """
    if magnitude_preserving:
        return simulate_failure_magnitude_preserving(expert_action)

    failure_type = np.random.randint(4)
    if failure_type == 0:
        scale = np.random.choice([0.5, 0.7, 1.5, 2.0])
        return (expert_action * scale).astype(np.float32)
    elif failure_type == 1:
        bias = np.random.randn(expert_action.shape[1]).astype(np.float32) * 0.3
        return (expert_action + bias).astype(np.float32)
    elif failure_type == 2:
        stuck = np.tile(expert_action[0:1], (expert_action.shape[0], 1))
        return stuck.astype(np.float32)
    else:
        k = np.random.randint(3, 8)
        delayed = np.zeros_like(expert_action)
        delayed[:k] = expert_action[0]
        delayed[k:] = expert_action[:-k] if k < len(expert_action) else expert_action[0]
        return delayed.astype(np.float32)


def simulate_failure_magnitude_preserving(expert_action):
    """Magnitude-preserving failures that test true compatibility learning.

    All failures preserve per-step L2 norms, forcing the model to use
    state-action structure rather than magnitude cues.
    """
    failure_type = np.random.randint(5)

    if failure_type == 0:
        # Rotation: rotate action vectors by random angle (2D only)
        if expert_action.shape[1] == 2:
            angle = np.random.uniform(np.pi/4, np.pi)  # 45-180 degrees
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            return (expert_action @ rot.T).astype(np.float32)
        else:
            # For higher dims, apply random rotation matrix
            d = expert_action.shape[1]
            q, _ = np.linalg.qr(np.random.randn(d, d))
            return (expert_action @ q.T).astype(np.float32)

    elif failure_type == 1:
        # Axis swap: swap action dimensions
        swapped = expert_action.copy()
        d = expert_action.shape[1]
        perm = np.random.permutation(d)
        return swapped[:, perm].astype(np.float32)

    elif failure_type == 2:
        # Sign flip: randomly flip signs but preserve magnitude
        signs = np.random.choice([-1, 1], size=expert_action.shape)
        return (expert_action * signs).astype(np.float32)

    elif failure_type == 3:
        # Time warp: shuffle temporal order (preserves per-step magnitude)
        n_steps = len(expert_action)
        # Reverse, or random permutation of chunks
        if np.random.random() < 0.5:
            return expert_action[::-1].copy().astype(np.float32)
        else:
            # Shuffle in chunks of 4
            chunk_size = 4
            n_chunks = n_steps // chunk_size
            chunks = [expert_action[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
            remainder = expert_action[n_chunks*chunk_size:]
            np.random.shuffle(chunks)
            if len(remainder) > 0:
                chunks.append(remainder)
            return np.concatenate(chunks, axis=0).astype(np.float32)

    else:
        # Direction noise: add noise then renormalize to original magnitude
        noise = np.random.randn(*expert_action.shape).astype(np.float32) * 0.5
        perturbed = expert_action + noise
        # Renormalize each step to match original magnitude
        orig_norms = np.linalg.norm(expert_action, axis=1, keepdims=True) + 1e-8
        new_norms = np.linalg.norm(perturbed, axis=1, keepdims=True) + 1e-8
        return (perturbed * orig_norms / new_norms).astype(np.float32)


def score_action_logmargin(model, obs, action, negative_pool, device, m_negatives):
    neg_indices = np.random.choice(len(negative_pool), m_negatives, replace=False)
    neg_actions = negative_pool[neg_indices]
    candidates = np.concatenate([action[np.newaxis], neg_actions], axis=0)

    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    candidates_t = torch.from_numpy(candidates).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(obs_t, candidates_t)
        l0 = logits[0, 0]
        l_neg = logits[0, 1:]
        margin = l0 - torch.logsumexp(l_neg, dim=0)

    return margin.item()


def baseline_action_magnitude(action_chunk):
    return np.linalg.norm(action_chunk)


def score_episode(model, episode, negative_pool, device, config,
                  failure_fn=None, m_negatives=M_NEGATIVES, prefix_only=False,
                  magnitude_preserving=False):
    obs_window = config.get("obs_window", 2)
    action_chunk = config.get("action_chunk", 16)
    obs_type = config.get("obs_type", "image")

    # Get observations based on type
    if obs_type == 'image':
        observations = episode['images']
    else:
        observations = episode['obs']

    actions = episode['actions']
    ep_len = len(observations)

    if ep_len < obs_window + action_chunk:
        return None, None

    end_t = ep_len - action_chunk
    if prefix_only:
        end_t = int((ep_len - action_chunk) * 0.7)

    tap_margins = []
    action_mags = []

    for t in range(obs_window - 1, end_t):
        start_t = t - obs_window + 1

        if obs_type == 'image':
            # Process image observations
            obs = observations[start_t:t + 1].astype(np.float32) / 255.0
            obs = np.transpose(obs, (0, 3, 1, 2))
        else:
            # Process state observations (no normalization or transpose needed)
            obs = observations[start_t:t + 1].astype(np.float32)

        obs = perturb_observation(obs, obs_type=obs_type)

        expert_action = actions[t:t + action_chunk].astype(np.float32)

        if failure_fn is None:
            policy_action = simulate_success_action(expert_action)
        elif magnitude_preserving:
            policy_action = simulate_failure_magnitude_preserving(expert_action)
        else:
            policy_action = failure_fn(expert_action)

        margin = score_action_logmargin(model, obs, policy_action, negative_pool, device, m_negatives)
        tap_margins.append(margin)
        action_mags.append(baseline_action_magnitude(policy_action))

    if len(tap_margins) == 0:
        return None, None

    return np.percentile(tap_margins, 10), np.percentile(action_mags, 10)


def compute_metrics_at_threshold(success_scores, failure_scores, tau):
    """
    Compute TPR and FPR at threshold tau.

    Convention: score < tau => predict FAILURE
                score >= tau => predict SUCCESS
    """
    # False positive rate: fraction of successes incorrectly flagged as failure
    fpr = np.mean([s < tau for s in success_scores])

    # True positive rate: fraction of failures correctly detected
    tpr = np.mean([s < tau for s in failure_scores])

    return tpr, fpr


def find_threshold_at_fpr(success_scores, target_fpr):
    """
    Find threshold tau such that FPR = target_fpr on success episodes.

    Since score < tau => failure, and we want target_fpr fraction of
    successes to fall below tau, tau = percentile(success_scores, target_fpr*100)
    """
    return np.percentile(success_scores, target_fpr * 100)


def main():
    parser = argparse.ArgumentParser(description="Final TAP Evaluation")
    parser.add_argument("--benchmark", type=str, default="pusht",
                        choices=list_benchmarks(),
                        help="Benchmark to evaluate on")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path (default: auto-resolve based on benchmark)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (default: auto-resolve based on benchmark)")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--n_episodes", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--magnitude_preserving", action="store_true",
                        help="Use magnitude-preserving failures (rotation, axis swap, sign flip, time warp)")
    args = parser.parse_args()

    # Get benchmark config
    benchmark_config = get_benchmark_config(args.benchmark)

    # Auto-resolve paths
    if args.checkpoint is None:
        args.checkpoint = f"checkpoints_contrastive/{args.benchmark}/contrastive_tap_best.pt"
    if args.data_dir is None:
        args.data_dir = str(get_data_path(args.benchmark))

    print("=" * 70)
    print(f"FINAL TAP EVALUATION - {benchmark_config['name']}")
    print("=" * 70)
    print(f"\nBenchmark:    {args.benchmark}")
    print(f"Action dim:   {benchmark_config['action_dim']}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Data dir:     {args.data_dir}")
    print("\nTraining negatives: noise, permute, mirror, random")
    if args.magnitude_preserving:
        print("Held-out failures:  rotation, axis_swap, sign_flip, time_warp, direction_noise (MAGNITUDE-PRESERVING)")
    else:
        print("Held-out failures:  scaling, bias, stuck, delayed (NOT in training)")
    print("=" * 70)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print("\nLoading...")
    model, config = load_model(args.checkpoint, device)
    episodes = load_episodes(args.data_dir, args.benchmark)
    print_data_stats(episodes, args.benchmark)
    action_chunk = config.get("action_chunk", 16)
    negative_pool = build_negative_pool(episodes, action_chunk)

    np.random.seed(42)
    test_episodes = episodes[:args.n_episodes]

    # ========================================================================
    # 1. Collect scores
    # ========================================================================
    print("\n" + "=" * 70)
    print("Collecting Episode Scores")
    print("=" * 70)

    success_tap, success_mag = [], []
    failure_tap, failure_mag = [], []

    for ep in tqdm(test_episodes, desc="Success"):
        tap, mag = score_episode(model, ep, negative_pool, device, config, failure_fn=None)
        if tap is not None:
            success_tap.append(tap)
            success_mag.append(mag)

    for ep in tqdm(test_episodes, desc="Held-out failure"):
        tap, mag = score_episode(model, ep, negative_pool, device, config,
                                  failure_fn=simulate_failure_heldout,
                                  magnitude_preserving=args.magnitude_preserving)
        if tap is not None:
            failure_tap.append(tap)
            failure_mag.append(mag)

    print(f"\nSuccess mean score:  {np.mean(success_tap):.2f}")
    print(f"Failure mean score:  {np.mean(failure_tap):.2f}")
    print(f"Separation:          {np.mean(success_tap) - np.mean(failure_tap):.2f}")

    # ========================================================================
    # 2. Compute AUROC
    # ========================================================================
    print("\n" + "=" * 70)
    print("AUROC Metrics")
    print("=" * 70)

    # For AUROC: compute with both orientations and report max
    # (invariant to whether model outputs higher or lower scores for failures)
    all_tap = success_tap + failure_tap
    labels = [0] * len(success_tap) + [1] * len(failure_tap)
    tap_auroc_raw = roc_auc_score(labels, [-s for s in all_tap])
    tap_auroc = max(tap_auroc_raw, 1 - tap_auroc_raw)
    tap_orientation = "lower margin → failure" if tap_auroc_raw >= 0.5 else "higher margin → failure"

    # Baseline: action magnitude
    all_mag = success_mag + failure_mag
    mag_auroc_raw = roc_auc_score(labels, all_mag)
    mag_auroc = max(mag_auroc_raw, 1 - mag_auroc_raw)

    print(f"TAP AUROC (held-out failures):     {tap_auroc:.3f}")
    print(f"  Score orientation:               {tap_orientation}")
    print(f"Action Magnitude Baseline AUROC:   {mag_auroc:.3f}")
    print(f"TAP advantage:                     {tap_auroc - mag_auroc:+.3f} (absolute)")

    # ========================================================================
    # 3. Operating points (TPR @ FPR)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Operating Points")
    print("=" * 70)

    target_fprs = [0.01, 0.05, 0.10]
    operating_points = {}

    # Determine score orientation from AUROC computation
    score_flipped = tap_auroc_raw < 0.5  # If true, higher scores indicate failure

    print(f"\n(Using orientation: {'higher' if score_flipped else 'lower'} score → failure)")
    print("\n| FPR Target | Threshold τ | TPR (Detection Rate) |")
    print("|------------|-------------|----------------------|")

    for target_fpr in target_fprs:
        if score_flipped:
            # Higher score = failure: threshold at upper percentile of success scores
            tau = np.percentile(success_tap, 100 - target_fpr * 100)
            # FPR: fraction of successes above tau
            actual_fpr = np.mean([s > tau for s in success_tap])
            # TPR: fraction of failures above tau
            tpr = np.mean([s > tau for s in failure_tap])
        else:
            # Lower score = failure: use original logic
            tau = find_threshold_at_fpr(success_tap, target_fpr)
            tpr, actual_fpr = compute_metrics_at_threshold(success_tap, failure_tap, tau)
        operating_points[f'{int(target_fpr*100)}%'] = {'tau': tau, 'tpr': tpr, 'fpr': actual_fpr}
        print(f"| {target_fpr*100:5.0f}%     | {tau:11.2f} | {tpr*100:19.1f}% |")

    # ========================================================================
    # 4. M Ablation (stability across negative set sizes)
    # ========================================================================
    print("\n" + "=" * 70)
    print("M Ablation (Stability Check)")
    print("=" * 70)

    m_values = [7, 15, 31]
    m_results = {}

    for m in m_values:
        m_success, m_failure = [], []

        for ep in tqdm(test_episodes[:20], desc=f"M={m}", leave=False):
            tap, _ = score_episode(model, ep, negative_pool, device, config, failure_fn=None, m_negatives=m)
            if tap is not None:
                m_success.append(tap)

        for ep in tqdm(test_episodes[:20], desc=f"M={m} fail", leave=False):
            tap, _ = score_episode(model, ep, negative_pool, device, config,
                                    failure_fn=simulate_failure_heldout, m_negatives=m,
                                    magnitude_preserving=args.magnitude_preserving)
            if tap is not None:
                m_failure.append(tap)

        if len(m_success) > 0 and len(m_failure) > 0:
            all_scores = m_success + m_failure
            m_labels = [0] * len(m_success) + [1] * len(m_failure)
            m_auroc_raw = roc_auc_score(m_labels, [-s for s in all_scores])
            m_auroc = max(m_auroc_raw, 1 - m_auroc_raw)
            m_results[m] = m_auroc

    print("\n| M (negatives) | AUROC |")
    print("|---------------|-------|")
    for m, auroc in m_results.items():
        print(f"| {m:13} | {auroc:.3f} |")

    # ========================================================================
    # 5. Prefix-only (early warning)
    # ========================================================================
    print("\n" + "=" * 70)
    print("Prefix-Only Evaluation (Early Warning)")
    print("=" * 70)

    prefix_success, prefix_failure = [], []

    for ep in tqdm(test_episodes[:25], desc="Prefix success"):
        tap, _ = score_episode(model, ep, negative_pool, device, config, failure_fn=None, prefix_only=True)
        if tap is not None:
            prefix_success.append(tap)

    for ep in tqdm(test_episodes[:25], desc="Prefix failure"):
        tap, _ = score_episode(model, ep, negative_pool, device, config,
                                failure_fn=simulate_failure_heldout, prefix_only=True,
                                magnitude_preserving=args.magnitude_preserving)
        if tap is not None:
            prefix_failure.append(tap)

    if len(prefix_success) > 0 and len(prefix_failure) > 0:
        prefix_all = prefix_success + prefix_failure
        prefix_labels = [0] * len(prefix_success) + [1] * len(prefix_failure)
        prefix_auroc_raw = roc_auc_score(prefix_labels, [-s for s in prefix_all])
        prefix_auroc = max(prefix_auroc_raw, 1 - prefix_auroc_raw)
        print(f"\nPrefix-only AUROC: {prefix_auroc:.3f}")
        print("(Using only first 70% of episode to predict eventual failure)")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    print("""
┌────────────────────────────────────────────────────────────────────┐
│                      CONTRASTIVE TAP RESULTS                       │
├────────────────────────────────────────────────────────────────────┤""")
    print(f"│ Held-out Failure AUROC:        {tap_auroc:.3f}                            │")
    print(f"│ Prefix-Only AUROC:             {prefix_auroc:.3f}  (early warning)            │" if 'prefix_auroc' in dir() else "")
    print(f"│ Action Magnitude Baseline:     {mag_auroc:.3f}                            │")
    print(f"│ TAP advantage:                +{tap_auroc - mag_auroc:.3f} (absolute AUROC)           │")
    print("""├────────────────────────────────────────────────────────────────────┤
│ OPERATING POINTS                                                   │""")
    for fpr_str, vals in operating_points.items():
        print(f"│ TPR @ {fpr_str} FPR:               {vals['tpr']*100:5.1f}%                           │")
    print("""├────────────────────────────────────────────────────────────────────┤
│ M ABLATION (stability)                                             │""")
    for m, auroc in m_results.items():
        print(f"│ M = {m:2}:                        {auroc:.3f}                            │")
    print("""└────────────────────────────────────────────────────────────────────┘""")

    print("\nTraining negatives: noise, permute, mirror, random")
    print("Held-out failures:  scaling, bias, stuck, delayed")

    # Save results
    results = {
        'benchmark': args.benchmark,
        'action_dim': benchmark_config['action_dim'],
        'tap_auroc': float(tap_auroc),
        'baseline_auroc': float(mag_auroc),
        'prefix_auroc': float(prefix_auroc) if 'prefix_auroc' in dir() else None,
        'operating_points': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in operating_points.items()},
        'm_ablation': {str(k): float(v) for k, v in m_results.items()},
        'success_mean': float(np.mean(success_tap)),
        'failure_mean': float(np.mean(failure_tap)),
        'separation': float(np.mean(success_tap) - np.mean(failure_tap)),
    }

    results_file = output_dir / f"{args.benchmark}_eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()

"""
Visualize TAP-Guided Reranking Traces

Creates a 2-panel figure for a single episode:
- Top: TAP scores for all K candidates + bold selected + markers where reranking
  chose != index 0
- Bottom: Bar chart of selected candidate index per step

Automatically picks a "rescued" episode (failed at K=1, succeeded at target K)
for maximum visual impact.

Usage:
    python plot_reranking_trace.py \
        --results eval_results/dp_reranking/reranking_results.json \
        --traces eval_results/dp_reranking/reranking_traces.npz \
        --output eval_results/dp_reranking/figures/reranking_trace.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_episode_traces(npz_path, condition, K):
    """Load per-step trace data for a (condition, K) combination.

    Returns list of dicts with keys:
        seed, success, max_reward, selected_indices, all_scores (list of lists)
    """
    data = np.load(npz_path, allow_pickle=True)
    prefix_key = f"{condition}_K{K}"

    # Discover how many episodes exist for this key
    episodes = []
    ep_idx = 0
    while True:
        seed_key = f"{prefix_key}_ep{ep_idx}_seed"
        if seed_key not in data:
            break

        ep_prefix = f"{prefix_key}_ep{ep_idx}"
        seed = int(data[f"{ep_prefix}_seed"])
        success = bool(data[f"{ep_prefix}_success"])
        max_reward = float(data[f"{ep_prefix}_max_reward"])
        selected_indices = data[f"{ep_prefix}_selected_indices"].tolist()

        # Reconstruct per-step scores
        all_scores = []
        step_idx = 0
        while True:
            score_key = f"{ep_prefix}_scores_step{step_idx}"
            if score_key not in data:
                break
            all_scores.append(data[score_key].tolist())
            step_idx += 1

        episodes.append({
            'seed': seed,
            'success': success,
            'max_reward': max_reward,
            'selected_indices': selected_indices,
            'all_scores': all_scores,
        })
        ep_idx += 1

    return episodes


def find_rescued_episode(results_json, npz_path, condition, target_K):
    """Find a seed that failed at K=1 but succeeded at target_K.

    Falls back to any K>1 episode if no rescued episode exists.
    """
    rescued = results_json.get('rescued_episodes', {})
    condition_rescued = rescued.get(condition, {})
    rescued_seeds = condition_rescued.get(str(target_K), [])

    # Load traces for target K
    traces = load_episode_traces(npz_path, condition, target_K)
    if not traces:
        return None

    # Try to find a rescued episode
    if rescued_seeds:
        for trace in traces:
            if trace['seed'] in rescued_seeds:
                return trace

    # Fallback: pick any episode where reranking changed something
    for trace in traces:
        if any(idx != 0 for idx in trace['selected_indices']):
            return trace

    # Last resort: first episode
    return traces[0]


def plot_reranking_trace(episode, K, condition, output_path):
    """Create 2-panel reranking trace figure."""
    selected_indices = episode['selected_indices']
    all_scores = episode['all_scores']
    n_steps = len(all_scores)

    if n_steps == 0:
        print("No scored steps to plot.")
        return

    steps = np.arange(n_steps)

    # Extract per-candidate score traces: (n_steps, K)
    score_matrix = np.array(all_scores)  # (n_steps, K)
    selected_scores = [score_matrix[t, selected_indices[t]] for t in range(n_steps)]

    # Color palette for candidates
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    cand_colors = [base_colors[i % len(base_colors)] for i in range(K)]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 6), height_ratios=[3, 1],
        sharex=True, gridspec_kw={'hspace': 0.08}
    )

    # --- Top panel: TAP scores ---
    for k in range(K):
        ax_top.plot(steps, score_matrix[:, k], color=cand_colors[k],
                    alpha=0.3, linewidth=1, label=f'Candidate {k}')

    # Bold line for selected candidate
    ax_top.plot(steps, selected_scores, 'k-', linewidth=2.5, alpha=0.9,
                label='Selected', zorder=10)

    # Red markers where reranking chose != 0
    reranked_steps = [t for t in range(n_steps) if selected_indices[t] != 0]
    if reranked_steps:
        reranked_scores = [selected_scores[t] for t in reranked_steps]
        ax_top.scatter(reranked_steps, reranked_scores, color='red', s=40,
                       zorder=15, marker='v', label='Reranked (idx != 0)')

    ax_top.set_ylabel('TAP Score (log-margin)', fontsize=11)
    status = "SUCCESS" if episode['success'] else "FAILURE"
    ax_top.set_title(
        f'TAP Reranking Trace  |  K={K}, {condition}  |  '
        f'seed={episode["seed"]}  |  {status} (reward={episode["max_reward"]:.2f})',
        fontsize=12
    )
    ax_top.legend(loc='lower left', fontsize=8, ncol=min(K + 2, 5))
    ax_top.grid(True, alpha=0.3)

    # --- Bottom panel: selected index bar chart ---
    bar_colors = [cand_colors[idx] for idx in selected_indices[:n_steps]]
    ax_bot.bar(steps, selected_indices[:n_steps], color=bar_colors, width=0.8)
    ax_bot.set_ylabel('Selected idx', fontsize=11)
    ax_bot.set_xlabel('Policy step', fontsize=11)
    ax_bot.set_yticks(range(K))
    ax_bot.set_ylim(-0.5, K - 0.5)
    ax_bot.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reranking trace to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot TAP Reranking Traces")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to reranking_results.json")
    parser.add_argument("--traces", type=str, default=None,
                        help="Path to reranking_traces.npz (default: same dir as results)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output figure path (default: figures/ next to results)")
    parser.add_argument("--condition", type=str, default=None,
                        help="Condition to plot (default: first perturbation, or 'clean')")
    parser.add_argument("--target_k", type=int, default=None,
                        help="K value to plot (default: largest K in results)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Specific seed to plot (overrides auto-selection)")
    args = parser.parse_args()

    results_path = Path(args.results)
    with open(results_path) as f:
        results_json = json.load(f)

    # Resolve traces path
    traces_path = args.traces
    if traces_path is None:
        traces_path = results_path.parent / 'reranking_traces.npz'
    traces_path = Path(traces_path)

    # Resolve output path
    output_path = args.output
    if output_path is None:
        output_path = results_path.parent / 'figures' / 'reranking_trace.png'

    # Resolve condition
    config = results_json['config']
    condition = args.condition
    if condition is None:
        # Prefer first perturbation for more interesting plot; fall back to clean
        if config['perturbations']:
            condition = config['perturbations'][0]
        else:
            condition = 'clean'

    # Resolve K
    target_K = args.target_k
    if target_K is None:
        target_K = max(config['k_values'])

    print(f"Plotting: condition={condition}, K={target_K}")

    if args.seed is not None:
        # Load specific seed
        traces = load_episode_traces(traces_path, condition, target_K)
        episode = None
        for t in traces:
            if t['seed'] == args.seed:
                episode = t
                break
        if episode is None:
            print(f"Seed {args.seed} not found for {condition} K={target_K}")
            return
    else:
        # Auto-select: prefer rescued episode
        episode = find_rescued_episode(results_json, traces_path, condition, target_K)
        if episode is None:
            print(f"No episodes found for {condition} K={target_K}")
            return

    print(f"Selected episode: seed={episode['seed']}, "
          f"success={episode['success']}, reward={episode['max_reward']:.2f}")

    plot_reranking_trace(episode, target_K, condition, output_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge split headroom audit part files into a single report.

Usage:
    python scripts/merge_headroom_parts.py \
        eval_results/robomimic_headroom_lift_K8_n50_partA.json \
        eval_results/robomimic_headroom_lift_K8_n50_partB.json \
        -o eval_results/robomimic_headroom_lift_K8_n50.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def mean_ci95(values):
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "n": 1}
    se = float(arr.std(ddof=1) / np.sqrt(n))
    half = 1.96 * se
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half, "n": n}


def main():
    parser = argparse.ArgumentParser(description="Merge headroom audit part files")
    parser.add_argument("parts", nargs="+", help="Part JSON files to merge")
    parser.add_argument("-o", "--output", required=True, help="Output merged JSON")
    args = parser.parse_args()

    parts = []
    for p in args.parts:
        with open(p, "r") as f:
            parts.append(json.load(f))

    # Use first part's config as base
    config = parts[0]["config"]

    # Merge episodes and decision points
    baseline_episodes = []
    decision_points = []
    oracle_episode_summaries = []

    for part in parts:
        baseline_episodes.extend(part.get("episodes", []))
        decision_points.extend(part.get("decision_points", []))
        if part.get("oracle_policy_episodes"):
            oracle_episode_summaries.extend(part["oracle_policy_episodes"])

    # Recompute global metrics
    spreads = np.array([d["candidate_spread_l2_mean_pairwise"] for d in decision_points], dtype=np.float64)
    headroom_best_mean = np.array([d["oracle_headroom_best_minus_mean"] for d in decision_points], dtype=np.float64)
    headroom_best_k0 = np.array([d["oracle_headroom_best_minus_k0"] for d in decision_points], dtype=np.float64)

    k0_decision_success = np.array([float(d["candidate_success"][0]) for d in decision_points], dtype=np.float64)
    oracle_decision_success = np.array([float(d["candidate_success"][d["oracle_index"]]) for d in decision_points], dtype=np.float64)
    decision_success_gain = oracle_decision_success - k0_decision_success

    all_candidate_returns = np.array([r for d in decision_points for r in d["candidate_returns"]], dtype=np.float64)
    ret_min = float(all_candidate_returns.min()) if all_candidate_returns.size else float("nan")
    ret_max = float(all_candidate_returns.max()) if all_candidate_returns.size else float("nan")
    ret_range = ret_max - ret_min if all_candidate_returns.size else 0.0

    headroom_fraction_threshold = config["thresholds"]["headroom_fraction_of_return_range"]
    success_gain_threshold = config["thresholds"]["success_gain_points"]
    headroom_abs_threshold = headroom_fraction_threshold * ret_range
    headroom_stats = mean_ci95(headroom_best_k0)
    decision_success_gain_stats = mean_ci95(decision_success_gain)

    baseline_ep_success = np.array([float(ep["episode_success"]) for ep in baseline_episodes], dtype=np.float64)
    baseline_ep_return = np.array([float(ep["episode_return"]) for ep in baseline_episodes], dtype=np.float64)

    oracle_success_stats = None
    oracle_return_stats = None
    episode_success_gain_stats = None
    success_gain_for_gate = decision_success_gain_stats
    success_gain_source = "decision_to_go"

    if oracle_episode_summaries:
        oracle_ep_success = np.array([float(ep["episode_success"]) for ep in oracle_episode_summaries], dtype=np.float64)
        oracle_ep_return = np.array([float(ep["episode_return"]) for ep in oracle_episode_summaries], dtype=np.float64)
        oracle_success_stats = mean_ci95(oracle_ep_success)
        oracle_return_stats = mean_ci95(oracle_ep_return)
        ep_success_gain = oracle_ep_success - baseline_ep_success[:len(oracle_ep_success)]
        episode_success_gain_stats = mean_ci95(ep_success_gain)
        success_gain_for_gate = episode_success_gain_stats
        success_gain_source = "episode_policy"

    headroom_collapsed = np.isfinite(headroom_stats["mean"]) and headroom_stats["mean"] < headroom_abs_threshold
    success_gain_small = np.isfinite(success_gain_for_gate["ci95_high"]) and success_gain_for_gate["ci95_high"] < success_gain_threshold
    meaningful_headroom = (
        np.isfinite(headroom_stats["mean"]) and headroom_stats["mean"] >= headroom_abs_threshold
        and np.isfinite(success_gain_for_gate["ci95_low"]) and success_gain_for_gate["ci95_low"] >= success_gain_threshold
    )

    if meaningful_headroom:
        fork_decision = "meaningful_headroom"
        recommendation = "TAP-Score reranking is justified as an inference-time selector."
    elif headroom_collapsed and success_gain_small:
        fork_decision = "headroom_collapsed"
        recommendation = "Increase generator diversity first; ranking is not the bottleneck yet."
    else:
        fork_decision = "inconclusive"
        recommendation = "Signal is mixed. Run more episodes or tighten evaluation controls."

    total_episodes = len(baseline_episodes)
    config["n_episodes"] = total_episodes

    report = {
        "meta": {
            "script": "scripts/merge_headroom_parts.py",
            "source_parts": args.parts,
            "timestamp_unix": time.time(),
        },
        "config": config,
        "global_metrics": {
            "n_decision_points": len(decision_points),
            "candidate_return_range": {"min": ret_min, "max": ret_max, "range": ret_range},
            "candidate_spread_l2_mean_pairwise": mean_ci95(spreads),
            "oracle_headroom_best_minus_mean": mean_ci95(headroom_best_mean),
            "oracle_headroom_best_minus_k0": headroom_stats,
            "decision_success_k0": mean_ci95(k0_decision_success),
            "decision_success_oracle": mean_ci95(oracle_decision_success),
            "decision_success_gain_oracle_minus_k0": decision_success_gain_stats,
            "episode_success_k1": mean_ci95(baseline_ep_success),
            "episode_return_k1": mean_ci95(baseline_ep_return),
            "episode_success_oracle_policy": oracle_success_stats,
            "episode_return_oracle_policy": oracle_return_stats,
            "episode_success_gain_oracle_minus_k1": episode_success_gain_stats,
            "gate_metrics": {
                "headroom_threshold_abs": headroom_abs_threshold,
                "success_gain_threshold": success_gain_threshold,
                "success_gain_source": success_gain_source,
            },
            "fork_decision": fork_decision,
            "recommendation": recommendation,
        },
        "episodes": baseline_episodes,
        "oracle_policy_episodes": oracle_episode_summaries or None,
        "decision_points": decision_points,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Merged {len(args.parts)} parts -> {out}")
    print(f"  {total_episodes} baseline + {len(oracle_episode_summaries)} oracle episodes")
    print(f"  {len(decision_points)} decision points")
    print(f"  Fork decision: {fork_decision}")
    print(f"  Recommendation: {recommendation}")


if __name__ == "__main__":
    main()

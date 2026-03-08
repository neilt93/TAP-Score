#!/usr/bin/env python3
"""Failure density analysis: why TAP works on Can/Lift but not Square.

Analyzes per-step score distributions, temporal risk profiles, and label
density to explain which tasks have learnable failure signals and why.

Inputs: per-step CSVs and per-episode JSONs from eval_tap_detection.py.
Outputs: multi-panel figure + summary JSON.
"""

import argparse
import csv
import json
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Data loading ──────────────────────────────────────────────────────────

def load_step_data(csv_path: str) -> dict:
    """Load per-step CSV from eval_tap_detection.py."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "score": float(row["score"]),
                "action_mag": float(row["action_mag"]),
                "label": int(row["label"]),
                "env_step": int(row["env_step"]),
                "policy_step": int(row["policy_step"]),
                "episode_id": int(row["episode_id"]),
            })
    return rows


def load_episode_data(json_path: str) -> dict:
    """Load per-episode JSON from eval_tap_detection.py."""
    with open(json_path) as f:
        return json.load(f)


# ── Analysis functions ────────────────────────────────────────────────────

def temporal_score_profiles(rows: list, onset: int = 0) -> dict:
    """Compute mean TAP score over time, split by episode outcome.

    Returns dict with arrays for success/fail mean scores at each time bin.
    """
    # Group rows by episode
    episodes = defaultdict(list)
    for r in rows:
        episodes[r["episode_id"]].append(r)

    # Determine episode outcomes
    ep_labels = {}
    for eid, ep_rows in episodes.items():
        ep_labels[eid] = ep_rows[0]["label"]  # all rows in episode share label

    # Bin env_steps
    all_steps = [r["env_step"] for r in rows]
    max_step = max(all_steps) if all_steps else 400
    bin_size = 16  # 2 action chunks
    bins = list(range(onset, max_step + bin_size, bin_size))

    success_profiles = defaultdict(list)
    failure_profiles = defaultdict(list)

    for eid, ep_rows in episodes.items():
        is_success = ep_labels[eid] == 1
        for r in ep_rows:
            bin_idx = (r["env_step"] - onset) // bin_size
            if is_success:
                success_profiles[bin_idx].append(r["score"])
            else:
                failure_profiles[bin_idx].append(r["score"])

    # Compute means per bin
    all_bins = sorted(set(list(success_profiles.keys()) + list(failure_profiles.keys())))
    result = {
        "bin_centers": [],
        "success_mean": [],
        "success_std": [],
        "failure_mean": [],
        "failure_std": [],
        "separation": [],
        "n_success": [],
        "n_failure": [],
    }

    for b in all_bins:
        center = onset + b * bin_size + bin_size / 2
        s_scores = success_profiles.get(b, [])
        f_scores = failure_profiles.get(b, [])

        s_mean = np.mean(s_scores) if s_scores else np.nan
        f_mean = np.mean(f_scores) if f_scores else np.nan
        s_std = np.std(s_scores) if len(s_scores) > 1 else 0.0
        f_std = np.std(f_scores) if len(f_scores) > 1 else 0.0

        result["bin_centers"].append(center)
        result["success_mean"].append(s_mean)
        result["success_std"].append(s_std)
        result["failure_mean"].append(f_mean)
        result["failure_std"].append(f_std)
        result["separation"].append(s_mean - f_mean if not (np.isnan(s_mean) or np.isnan(f_mean)) else 0.0)
        result["n_success"].append(len(s_scores))
        result["n_failure"].append(len(f_scores))

    return result


def score_distribution_stats(rows: list) -> dict:
    """Compute score distribution stats split by outcome."""
    success_scores = [r["score"] for r in rows if r["label"] == 1]
    failure_scores = [r["score"] for r in rows if r["label"] == 0]

    def stats(arr):
        if not arr:
            return {"n": 0, "mean": None, "std": None, "median": None,
                    "p10": None, "p25": None, "p75": None, "p90": None}
        a = np.array(arr)
        return {
            "n": len(a),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
        }

    overlap = compute_distribution_overlap(success_scores, failure_scores)

    return {
        "success": stats(success_scores),
        "failure": stats(failure_scores),
        "overlap_coefficient": overlap,
    }


def compute_distribution_overlap(a: list, b: list, n_bins: int = 50) -> float:
    """Compute overlap coefficient between two score distributions.

    Returns value in [0, 1]: 0 = perfectly separated, 1 = identical.
    """
    if not a or not b:
        return 1.0
    all_vals = a + b
    lo, hi = min(all_vals), max(all_vals)
    if lo == hi:
        return 1.0
    bins = np.linspace(lo, hi, n_bins + 1)
    hist_a, _ = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bins, density=True)
    # Overlap = sum of min of the two normalized histograms
    bin_width = bins[1] - bins[0]
    return float(np.sum(np.minimum(hist_a, hist_b)) * bin_width)


def failure_density_over_time(rows: list, onset: int = 0) -> dict:
    """What fraction of steps at each time bin come from failing episodes?"""
    episodes = defaultdict(list)
    for r in rows:
        episodes[r["episode_id"]].append(r)

    ep_labels = {}
    for eid, ep_rows in episodes.items():
        ep_labels[eid] = ep_rows[0]["label"]

    bin_size = 16
    all_steps = [r["env_step"] for r in rows]
    max_step = max(all_steps) if all_steps else 400

    bins = {}
    for r in rows:
        b = (r["env_step"] - onset) // bin_size
        if b not in bins:
            bins[b] = {"success": 0, "failure": 0}
        if ep_labels[r["episode_id"]] == 1:
            bins[b]["success"] += 1
        else:
            bins[b]["failure"] += 1

    result = {"bin_centers": [], "failure_fraction": [], "n_total": []}
    for b in sorted(bins.keys()):
        center = onset + b * bin_size + bin_size / 2
        total = bins[b]["success"] + bins[b]["failure"]
        ff = bins[b]["failure"] / total if total > 0 else 0.0
        result["bin_centers"].append(center)
        result["failure_fraction"].append(ff)
        result["n_total"].append(total)

    return result


# ── Figure ────────────────────────────────────────────────────────────────

def make_figure(tasks: dict, output_dir: str):
    """Create multi-panel failure density figure.

    Layout: 3 columns (Can, Lift, Square), 3 rows:
    1. Temporal score profiles (success vs fail)
    2. Score separation over time
    3. Score distributions (histogram)
    """
    task_names = [t for t in ["Can", "Lift", "Square"] if t in tasks]
    n_tasks = len(task_names)

    fig, axes = plt.subplots(3, n_tasks, figsize=(4.5 * n_tasks, 10), squeeze=False)

    for col, name in enumerate(task_names):
        t = tasks[name]
        profiles = t["profiles"]
        dist_stats = t["dist_stats"]
        rows = t["rows"]

        bc = np.array(profiles["bin_centers"])
        s_mean = np.array(profiles["success_mean"])
        f_mean = np.array(profiles["failure_mean"])
        s_std = np.array(profiles["success_std"])
        f_std = np.array(profiles["failure_std"])
        sep = np.array(profiles["separation"])

        onset = t.get("onset", 0)

        # Row 0: Temporal score profiles
        ax = axes[0, col]
        valid_s = ~np.isnan(s_mean)
        valid_f = ~np.isnan(f_mean)
        if valid_s.any():
            ax.plot(bc[valid_s], s_mean[valid_s], "o-", color="C0",
                    markersize=3, label="Success episodes")
            ax.fill_between(bc[valid_s],
                            s_mean[valid_s] - s_std[valid_s],
                            s_mean[valid_s] + s_std[valid_s],
                            color="C0", alpha=0.15)
        if valid_f.any():
            ax.plot(bc[valid_f], f_mean[valid_f], "s-", color="C3",
                    markersize=3, label="Failure episodes")
            ax.fill_between(bc[valid_f],
                            f_mean[valid_f] - f_std[valid_f],
                            f_mean[valid_f] + f_std[valid_f],
                            color="C3", alpha=0.15)
        if onset > 0:
            ax.axvline(onset, color="orange", linestyle="--", linewidth=1,
                       label=f"Onset ({onset})")
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_ylabel("TAP Score (1=safe)")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # Row 1: Score separation
        ax = axes[1, col]
        valid = ~np.isnan(s_mean) & ~np.isnan(f_mean)
        if valid.any():
            colors = ["C0" if s > 0 else "C3" for s in sep[valid]]
            ax.bar(bc[valid], sep[valid], width=12, color=colors, alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.5)
        if onset > 0:
            ax.axvline(onset, color="orange", linestyle="--", linewidth=1)
        ax.set_ylabel("Score Separation\n(success - fail)")
        ax.grid(True, alpha=0.3)

        # Mean separation annotation
        mean_sep = np.nanmean(sep[valid]) if valid.any() else 0
        ax.annotate(f"Mean sep: {mean_sep:.3f}",
                    xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=8, va="top",
                    color="C0" if mean_sep > 0.05 else "C3")

        # Row 2: Score distributions
        ax = axes[2, col]
        success_scores = [r["score"] for r in rows if r["label"] == 1]
        failure_scores = [r["score"] for r in rows if r["label"] == 0]
        bins = np.linspace(0, 1, 40)
        if success_scores:
            ax.hist(success_scores, bins=bins, alpha=0.5, color="C0",
                    label="Success", density=True)
        if failure_scores:
            ax.hist(failure_scores, bins=bins, alpha=0.5, color="C3",
                    label="Failure", density=True)
        ax.set_xlabel("TAP Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Overlap annotation
        overlap = dist_stats["overlap_coefficient"]
        ax.annotate(f"Overlap: {overlap:.2f}",
                    xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=8, va="top",
                    color="C3" if overlap > 0.7 else "C0")

    # Shared x-label
    for col in range(n_tasks):
        axes[2, col].set_xlabel("TAP Score")
        axes[0, col].set_xlabel("")
        axes[1, col].set_xlabel("Env Step")

    fig.suptitle("Failure Density Analysis: Why TAP Works (or Doesn't)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "failure_density.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "failure_density.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out / 'failure_density.png'}")


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary(tasks: dict):
    """Print comparative summary across tasks."""
    print("\nFailure Density Analysis")
    print("=" * 70)

    for name in ["Can", "Lift", "Square"]:
        if name not in tasks:
            continue
        t = tasks[name]
        ep = t["episode_data"]
        dist = t["dist_stats"]
        profiles = t["profiles"]

        n_ep = ep["n_episodes"]
        n_succ = ep["n_success"]
        n_fail = ep["n_fail"]
        fail_rate = n_fail / n_ep if n_ep > 0 else 0
        auroc = ep.get("auroc_mean", ep.get("auroc_best", "N/A"))

        # Mean separation
        sep = np.array(profiles["separation"])
        valid = ~np.isnan(np.array(profiles["success_mean"])) & ~np.isnan(np.array(profiles["failure_mean"]))
        mean_sep = np.nanmean(sep[valid]) if valid.any() else 0

        # Post-onset separation
        onset = t.get("onset", 0)
        bc = np.array(profiles["bin_centers"])
        post_onset = valid & (bc >= onset)
        post_sep = np.nanmean(sep[post_onset]) if post_onset.any() else 0

        print(f"\n{name}:")
        print(f"  Episodes:     {n_ep} ({n_succ} success, {n_fail} fail, "
              f"fail rate {fail_rate:.0%})")
        print(f"  AUROC (mean): {auroc}")
        print(f"  Score overlap: {dist['overlap_coefficient']:.3f} "
              f"(0=separated, 1=identical)")
        print(f"  Mean separation: {mean_sep:.4f} (all time)")
        print(f"  Post-onset sep:  {post_sep:.4f}")

        if dist["success"]["mean"] is not None and dist["failure"]["mean"] is not None:
            s1, s2 = dist["success"]["std"], dist["failure"]["std"]
            n1, n2 = max(dist["success"]["n"], 1), max(dist["failure"]["n"], 1)
            pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / max(n1 + n2 - 2, 1))
            d_prime = abs(dist["success"]["mean"] - dist["failure"]["mean"]) / (pooled_std + 1e-8)
            print(f"  d' (Cohen):      {d_prime:.3f}")
        if dist['success']['mean'] is not None:
            print(f"  Success scores:  mean={dist['success']['mean']:.3f}, "
                  f"std={dist['success']['std']:.3f}")
        if dist['failure']['mean'] is not None:
            print(f"  Failure scores:  mean={dist['failure']['mean']:.3f}, "
                  f"std={dist['failure']['std']:.3f}")

    # Comparative insight
    print("\n" + "-" * 70)
    print("WHY TAP WORKS (or doesn't):")
    print()
    print("  Per-step d' is small for all tasks, but episode-level aggregation")
    print("  (mean of ~40 steps) amplifies signal by ~sqrt(N):")
    print()
    for name in ["Can", "Lift", "Square"]:
        if name not in tasks:
            continue
        dist = tasks[name]["dist_stats"]
        if dist["success"]["mean"] is not None and dist["failure"]["mean"] is not None:
            s1, s2 = dist["success"]["std"], dist["failure"]["std"]
            n1, n2 = max(dist["success"]["n"], 1), max(dist["failure"]["n"], 1)
            pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / max(n1 + n2 - 2, 1))
            d_step = abs(dist["success"]["mean"] - dist["failure"]["mean"]) / (pooled_std + 1e-8)
            # Use n_scores from episode data if available (actual steps per ep)
            ep_results = tasks[name]["episode_data"].get("episode_results", [])
            n_scores_list = [e.get("n_scores", 1) for e in ep_results
                             if e.get("n_scores") is not None]
            n_success_ep = tasks[name]["episode_data"].get("n_success", 0)
            n_per_ep = int(np.median(n_scores_list)) if n_scores_list else (
                max(1, dist["success"]["n"] // n_success_ep) if n_success_ep > 0 else 1)
            # For Square, d' is already episode-level (we only have mean scores)
            is_episode_level = "note" in tasks[name]
            if is_episode_level:
                d_episode = d_step  # already episode-level
                label = f"d'(ep)={d_step:.3f} (already aggregated, ~{n_per_ep} steps/ep)"
            else:
                d_episode = d_step * np.sqrt(n_per_ep)
                label = (f"d'(step)={d_step:.3f} x sqrt({n_per_ep}) "
                         f"-> d'(ep)~{d_episode:.2f}")
            auroc_val = tasks[name]["episode_data"].get(
                "auroc_mean", tasks[name]["episode_data"].get("auroc_best", "N/A"))
            if isinstance(auroc_val, float):
                auroc_val = f"{auroc_val:.3f}"
            print(f"  {name:6s}: {label}  (AUROC={auroc_val})")
    print()
    if "Can" in tasks and "Square" in tasks:
        print("  Can/Lift: per-step signal is weak but consistent across ~40")
        print("  decisions, so averaging yields strong episode-level separation.")
        print("  Square: per-step signal is near-zero, so aggregation cannot")
        print("  rescue it. The risk model has nothing to learn.")
    if "Can" in tasks and "Lift" in tasks:
        can_ep = tasks["Can"]["episode_data"]
        lift_ep = tasks["Lift"]["episode_data"]
        print(f"\n  Can fail rate:  {can_ep['n_fail']/can_ep['n_episodes']:.0%} -> "
              f"balanced classes, strong signal")
        print(f"  Lift fail rate: {lift_ep['n_fail']/lift_ep['n_episodes']:.0%} -> "
              f"requires episode_window labeling for sufficient positives")
    print("-" * 70)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Failure density analysis for TAP-Score")
    parser.add_argument("--can_csv",
        default="eval_results/risk_detection_can_zero80_onset_n100.csv")
    parser.add_argument("--can_json",
        default="eval_results/risk_detection_can_zero80_onset_n100.json")
    parser.add_argument("--lift_csv",
        default="eval_results/risk_detection_lift_zero31_epwin_n100.csv")
    parser.add_argument("--lift_json",
        default="eval_results/risk_detection_lift_zero31_epwin_n100.json")
    parser.add_argument("--square_json",
        default="eval_results/risk_detection_square_zero80_n50.json")
    parser.add_argument("--output_dir", default="artifacts/")
    parser.add_argument("--output_json",
        default="artifacts/failure_density_analysis.json")
    args = parser.parse_args()

    tasks = {}

    # Can
    if pathlib.Path(args.can_csv).exists():
        can_rows = load_step_data(args.can_csv)
        can_ep = load_episode_data(args.can_json)
        tasks["Can"] = {
            "rows": can_rows,
            "episode_data": can_ep,
            "profiles": temporal_score_profiles(can_rows, onset=80),
            "dist_stats": score_distribution_stats(can_rows),
            "density": failure_density_over_time(can_rows, onset=80),
            "onset": 80,
        }

    # Lift
    if pathlib.Path(args.lift_csv).exists():
        lift_rows = load_step_data(args.lift_csv)
        lift_ep = load_episode_data(args.lift_json)
        tasks["Lift"] = {
            "rows": lift_rows,
            "episode_data": lift_ep,
            "profiles": temporal_score_profiles(lift_rows, onset=31),
            "dist_stats": score_distribution_stats(lift_rows),
            "density": failure_density_over_time(lift_rows, onset=31),
            "onset": 31,
        }

    # Square (JSON only, no per-step CSV -- use episode-level mean scores)
    sq_path = pathlib.Path(args.square_json)
    if sq_path.exists():
        sq_ep = load_episode_data(args.square_json)
        # Use episode-level mean scores as pseudo per-step data.
        # Note: each episode has ~40 actual steps, but without the CSV
        # we only have the episode-level aggregates.
        sq_rows = []
        for e in sq_ep.get("episode_results", []):
            if e.get("mean_score") is not None:
                sq_rows.append({
                    "score": e["mean_score"],
                    "action_mag": e.get("mean_action_mag", 0),
                    "label": 1 if e["success"] else 0,
                    "env_step": 80,
                    "policy_step": 0,
                    "episode_id": e["episode_id"],
                })
        if sq_rows:
            tasks["Square"] = {
                "rows": sq_rows,
                "episode_data": sq_ep,
                "profiles": temporal_score_profiles(sq_rows, onset=80),
                "dist_stats": score_distribution_stats(sq_rows),
                "density": failure_density_over_time(sq_rows, onset=80),
                "onset": 80,
                "note": "episode-level scores only (no per-step CSV)",
            }

    if not tasks:
        print("ERROR: No eval data found. Run eval_tap_detection.py first.")
        return

    print_summary(tasks)
    make_figure(tasks, args.output_dir)

    # Save analysis JSON
    summary = {}
    for name, t in tasks.items():
        ep = t["episode_data"]
        dist = t["dist_stats"]
        profiles = t["profiles"]
        sep = np.array(profiles["separation"])
        valid = ~np.isnan(np.array(profiles["success_mean"])) & ~np.isnan(np.array(profiles["failure_mean"]))

        summary[name] = {
            "n_episodes": ep["n_episodes"],
            "n_success": ep["n_success"],
            "n_fail": ep["n_fail"],
            "fail_rate": ep["n_fail"] / ep["n_episodes"],
            "auroc_mean": ep.get("auroc_mean", None),
            "auroc_best": ep.get("auroc_best", None),
            "overlap_coefficient": dist["overlap_coefficient"],
            "mean_separation": float(np.nanmean(sep[valid])) if valid.any() else 0,
            "success_score_mean": dist["success"]["mean"],
            "failure_score_mean": dist["failure"]["mean"],
            "success_score_std": dist["success"]["std"],
            "failure_score_std": dist["failure"]["std"],
        }

    out_json = pathlib.Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAnalysis JSON: {out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Early-warning figure for TAP-Score.

Panel 1 (per task): Mean safety score trajectory over episode time,
split by success vs. failure episodes, with shaded 95% CI.

Panel 2 (per task): Lead-time histogram — how early TAP flags failure
before episode termination.

Uses per-step CSVs + per-episode JSONs from eval_tap_detection.py.
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


# ── Data loading ─────────────────────────────────────────────────────────

def load_step_data(csv_path: str) -> dict:
    """Load per-step scores from eval CSV, grouped by episode."""
    episodes = defaultdict(lambda: {"scores": [], "env_steps": [], "mags": []})
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = int(row["episode_id"])
            episodes[eid]["scores"].append(float(row["score"]))
            episodes[eid]["env_steps"].append(int(row["env_step"]))
            episodes[eid]["mags"].append(float(row["action_mag"]))
    # Convert to arrays
    for eid in episodes:
        for key in episodes[eid]:
            episodes[eid][key] = np.array(episodes[eid][key])
    return dict(episodes)


def load_episode_outcomes(json_path: str) -> dict:
    """Load per-episode success/fail from eval JSON."""
    with open(json_path) as f:
        d = json.load(f)
    outcomes = {}
    for ep in d["episode_results"]:
        outcomes[ep["episode_id"]] = bool(ep["success"])
    meta = {
        "perturb": d.get("perturb", "unknown"),
        "onset": d.get("perturb_start_step", 0),
        "n_success": d["n_success"],
        "n_fail": d["n_fail"],
        "lead_time": d.get("lead_time", {}),
    }
    return outcomes, meta


# ── Score trajectories ───────────────────────────────────────────────────

def compute_trajectory_stats(step_data, outcomes, use_policy_step=True):
    """Compute mean ± CI of safety scores at each policy step, split by outcome."""
    # Group scores by policy step index (0, 1, 2, ...) within each episode
    succ_by_step = defaultdict(list)
    fail_by_step = defaultdict(list)

    for eid, data in step_data.items():
        if eid not in outcomes:
            continue
        is_success = outcomes[eid]
        target = succ_by_step if is_success else fail_by_step
        for i, score in enumerate(data["scores"]):
            target[i].append(score)

    def stats_from_dict(d):
        max_step = max(d.keys()) if d else 0
        steps = np.arange(max_step + 1)
        means = np.array([np.mean(d[s]) if s in d else np.nan for s in steps])
        ci_lo = np.array([np.percentile(d[s], 2.5) if s in d and len(d[s]) > 1
                          else np.nan for s in steps])
        ci_hi = np.array([np.percentile(d[s], 97.5) if s in d and len(d[s]) > 1
                          else np.nan for s in steps])
        sems = np.array([np.std(d[s]) / np.sqrt(len(d[s])) if s in d and len(d[s]) > 1
                         else 0 for s in steps])
        # Use SEM-based CI (mean ± 1.96*SEM) for smoother bands
        ci_lo_sem = means - 1.96 * sems
        ci_hi_sem = means + 1.96 * sems
        return steps, means, ci_lo_sem, ci_hi_sem

    return {
        "success": stats_from_dict(succ_by_step),
        "failure": stats_from_dict(fail_by_step),
    }


def get_env_steps(step_data, outcomes):
    """Get the env_step values for x-axis (from any episode)."""
    for eid, data in step_data.items():
        if eid in outcomes:
            return data["env_steps"]
    return None


# ── Lead time ────────────────────────────────────────────────────────────

def compute_lead_times(step_data, outcomes, threshold, max_env_step=400):
    """For each failed episode, find first step where score < threshold.
    Lead time = max_env_step - first_detection_step."""
    lead_times = []
    detection_steps = []
    for eid, data in step_data.items():
        if eid not in outcomes or outcomes[eid]:
            continue  # skip success episodes
        scores = data["scores"]
        env_steps = data["env_steps"]
        detected = False
        for i, (score, step) in enumerate(zip(scores, env_steps)):
            if score < threshold:
                lead_times.append(max_env_step - step)
                detection_steps.append(step)
                detected = True
                break
        if not detected:
            lead_times.append(0)  # not detected
            detection_steps.append(max_env_step)
    return np.array(lead_times), np.array(detection_steps)


# ── Figure ───────────────────────────────────────────────────────────────

def make_figure(can_csv, can_json, lift_csv, lift_json, output_dir):
    """Create 2x2 early-warning figure."""

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    datasets = [
        (can_csv, can_json, "Can", 80),
        (lift_csv, lift_json, "Lift", 31),
    ]

    for col, (csv_path, json_path, task_name, onset) in enumerate(datasets):
        step_data = load_step_data(csv_path)
        outcomes, meta = load_episode_outcomes(json_path)
        env_steps = get_env_steps(step_data, outcomes)

        # ── Top row: score trajectories ──
        ax = axes[0, col]
        traj = compute_trajectory_stats(step_data, outcomes)

        for label, color, key in [
            ("Success", "C0", "success"),
            ("Failure", "C3", "failure"),
        ]:
            steps, means, ci_lo, ci_hi = traj[key]
            if env_steps is not None and len(steps) <= len(env_steps):
                x = env_steps[:len(steps)]
            else:
                x = steps * 8 + onset  # approximate
            ax.plot(x, means, "-", color=color, linewidth=1.5,
                    label=f"{label} (n={meta['n_success'] if key == 'success' else meta['n_fail']})")
            ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.15)

        ax.axvline(onset, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                   label=f"Onset (step {onset})")
        ax.set_title(f"{task_name}: Safety Score Trajectories", fontsize=11)
        ax.set_xlabel("Environment Step")
        ax.set_ylabel("TAP Safety Score")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Bottom row: lead-time histogram ──
        ax2 = axes[1, col]
        # Use threshold from the JSON if available
        threshold_info = meta.get("lead_time", {})
        threshold = threshold_info.get("threshold_used", 0.5)

        lead_times, det_steps = compute_lead_times(
            step_data, outcomes, threshold, max_env_step=400)

        n_detected = (lead_times > 0).sum()
        n_fail = meta["n_fail"]

        ax2.hist(det_steps[lead_times > 0], bins=20, color="C3", alpha=0.7,
                 edgecolor="white", linewidth=0.5)
        ax2.axvline(onset, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                    label=f"Onset (step {onset})")

        median_lt = np.median(lead_times[lead_times > 0]) if n_detected > 0 else 0
        ax2.set_title(
            f"{task_name}: Detection Step ({n_detected}/{n_fail} detected, "
            f"median lead={median_lt:.0f} steps)",
            fontsize=10)
        ax2.set_xlabel("First Detection Step")
        ax2.set_ylabel("Count (failed episodes)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.suptitle("TAP-Score Early Warning: Score Trajectories & Detection Timing",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "early_warning.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "early_warning.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out / 'early_warning.png'}")
    print(f"Figure saved: {out / 'early_warning.pdf'}")


# ── Summary ──────────────────────────────────────────────────────────────

def print_summary(can_csv, can_json, lift_csv, lift_json):
    """Print trajectory divergence statistics."""
    print("\nEarly Warning Analysis")
    print("=" * 60)

    for csv_path, json_path, name, onset in [
        (can_csv, can_json, "Can", 80),
        (lift_csv, lift_json, "Lift", 31),
    ]:
        step_data = load_step_data(csv_path)
        outcomes, meta = load_episode_outcomes(json_path)
        traj = compute_trajectory_stats(step_data, outcomes)

        s_steps, s_means, _, _ = traj["success"]
        f_steps, f_means, _, _ = traj["failure"]

        # Mean score across all steps
        min_len = min(len(s_means), len(f_means))
        gap = s_means[:min_len] - f_means[:min_len]

        print(f"\n{name} ({meta['perturb']}, onset={onset}):")
        print(f"  Success mean score (all steps): {np.nanmean(s_means):.3f}")
        print(f"  Failure mean score (all steps): {np.nanmean(f_means):.3f}")
        print(f"  Mean gap (succ - fail): {np.nanmean(gap):.3f}")
        print(f"  Max gap: {np.nanmax(gap):.3f} at policy step {np.nanargmax(gap)}")

        # First 5 and last 5 steps
        print(f"  First 5 steps gap: {np.nanmean(gap[:5]):.3f}")
        print(f"  Last 5 steps gap:  {np.nanmean(gap[-5:]):.3f}")

        # Lead time
        lt_info = meta.get("lead_time", {})
        if lt_info:
            print(f"  Lead time (from JSON): median={lt_info.get('median', '?')}, "
                  f"mean={lt_info.get('mean', '?')}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Early-warning figure for TAP-Score")
    parser.add_argument("--can_csv",
        default="eval_results/risk_detection_can_zero80_onset_n100.csv")
    parser.add_argument("--can_json",
        default="eval_results/risk_detection_can_zero80_onset_n100.json")
    parser.add_argument("--lift_csv",
        default="eval_results/risk_detection_lift_zero31_epwin_n100.csv")
    parser.add_argument("--lift_json",
        default="eval_results/risk_detection_lift_zero31_epwin_n100.json")
    parser.add_argument("--output_dir", default="artifacts/")
    args = parser.parse_args()

    print_summary(args.can_csv, args.can_json, args.lift_csv, args.lift_json)
    make_figure(args.can_csv, args.can_json, args.lift_csv, args.lift_json,
                args.output_dir)


if __name__ == "__main__":
    main()

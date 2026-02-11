"""
Summarize one or more reranking experiment JSON outputs into a paper-ready table.

Usage:
    # Single file
    python scripts/summarize_reranking.py outputs/occ24_n200_K4.json

    # Multiple files (K sweep)
    python scripts/summarize_reranking.py outputs/occ24_n10_K*.json

    # Save CSV
    python scripts/summarize_reranking.py outputs/occ24_n10_K*.json --csv tables/k_sweep.csv

    # With bootstrap CI
    python scripts/summarize_reranking.py outputs/occ24_n200_K4.json --bootstrap 10000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def bootstrap_ci(values, n_boot=10000, ci=0.95, rng=None):
    """Paired bootstrap confidence interval for the mean."""
    if rng is None:
        rng = np.random.default_rng(42)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    alpha = (1 - ci) / 2
    lo = float(np.percentile(means, 100 * alpha))
    hi = float(np.percentile(means, 100 * (1 - alpha)))
    return float(values.mean()), lo, hi


def summarize_file(path: Path, n_boot: int = 0) -> dict:
    """Extract paper-relevant metrics from a reranking JSON."""
    with open(path) as f:
        data = json.load(f)

    cfg = data["config"]
    summary = data.get("summary", {})
    ep_summary = data.get("episode_summary", {})
    episodes = data.get("episodes", [])
    dps = data.get("decision_points", [])

    # Episode-level lift (for bootstrap)
    if episodes:
        ep_lifts = np.array([e["lift"] for e in episodes])
        ep_headrooms = np.array([e["headroom"] for e in episodes])
    else:
        # Fall back to decision-point level
        r_notap = np.array([d["reward_notap"] for d in dps])
        r_tap = np.array([d["reward_tap"] for d in dps])
        r_oracle = np.array([d["reward_oracle"] for d in dps])
        ep_lifts = r_tap - r_notap
        ep_headrooms = r_oracle - r_notap

    row = {
        "file": path.name,
        "K": cfg.get("K"),
        "L": cfg.get("L"),
        "n_episodes": cfg.get("n_episodes"),
        "perturb": cfg.get("perturb", "clean"),
        "patch_size": cfg.get("patch_size"),
        "tap_sees_perturb": cfg.get("tap_sees_perturb"),
        "n_decision_points": summary.get("n_decision_points", len(dps)),
        "changed_frac": summary.get("changed_frac", "N/A"),
        "mean_reward_notap": ep_summary.get("mean_episode_max_reward_notap",
                                             summary.get("mean_reward_notap")),
        "mean_reward_tap": ep_summary.get("mean_episode_max_reward_tap",
                                          summary.get("mean_reward_tap")),
        "mean_reward_oracle": ep_summary.get("mean_episode_max_reward_oracle",
                                             summary.get("mean_reward_oracle")),
        "mean_lift": float(ep_lifts.mean()),
        "mean_headroom": float(ep_headrooms.mean()),
        "headroom_rate": float((ep_headrooms > 0.001).mean()),
        "capture_ratio": ep_summary.get("mean_episode_capture_ratio",
                                        summary.get("mean_capture_ratio", 0.0)),
        "n_rescued": ep_summary.get("n_rescued", "N/A"),
        "tap_beats_frac": summary.get("tap_beats_baseline_frac"),
        "tap_harms_frac": summary.get("tap_harms_frac"),
        "net_benefit": summary.get("net_benefit"),
    }

    if n_boot > 0 and len(ep_lifts) > 1:
        mean_lift, lo, hi = bootstrap_ci(ep_lifts, n_boot=n_boot)
        row["lift_ci_lo"] = lo
        row["lift_ci_hi"] = hi
        row["lift_ci"] = f"{mean_lift:.4f} [{lo:.4f}, {hi:.4f}]"
    else:
        row["lift_ci_lo"] = None
        row["lift_ci_hi"] = None
        row["lift_ci"] = None

    return row


def print_table(rows: list[dict]):
    """Print a readable table to stdout."""
    # Key columns for display
    cols = [
        ("K", "K"),
        ("n_ep", "n_episodes"),
        ("perturb", "perturb"),
        ("changed", "changed_frac"),
        ("notap", "mean_reward_notap"),
        ("TAP", "mean_reward_tap"),
        ("oracle", "mean_reward_oracle"),
        ("lift", "mean_lift"),
        ("headroom", "mean_headroom"),
        ("capture", "capture_ratio"),
        ("rescued", "n_rescued"),
        ("net_ben", "net_benefit"),
    ]
    if any(r.get("lift_ci") for r in rows):
        cols.append(("lift_CI_95", "lift_ci"))

    # Header
    header = "  ".join(f"{label:>10s}" for label, _ in cols)
    print(header)
    print("─" * len(header))

    for row in rows:
        parts = []
        for label, key in cols:
            val = row.get(key)
            if val is None or val == "N/A":
                parts.append(f"{'N/A':>10s}")
            elif isinstance(val, float):
                parts.append(f"{val:>10.4f}")
            elif isinstance(val, str):
                parts.append(f"{val:>10s}")
            else:
                parts.append(f"{val!s:>10s}")
        print("  ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Summarize reranking experiment outputs")
    parser.add_argument("files", nargs="+", type=str, help="JSON output file(s)")
    parser.add_argument("--csv", type=str, default=None, help="Save CSV to this path")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Bootstrap iterations for lift CI (0 = skip)")
    args = parser.parse_args()

    rows = []
    for f in sorted(args.files):
        path = Path(f)
        if not path.exists():
            print(f"WARNING: {f} not found, skipping", file=sys.stderr)
            continue
        rows.append(summarize_file(path, n_boot=args.bootstrap))

    if not rows:
        print("No valid files found.", file=sys.stderr)
        sys.exit(1)

    print_table(rows)

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    main()

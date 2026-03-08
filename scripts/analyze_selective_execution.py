#!/usr/bin/env python3
"""Selective execution analysis for TAP-Score.

Computes success-vs-coverage curves: sort episodes by score (descending),
execute only the top-k%, measure success rate.  Compares TAP risk score
against action-magnitude baseline and random baseline.

Outputs a 2-panel figure (Can + Lift) to artifacts/.
"""

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Data loading ─────────────────────────────────────────────────────────

def load_episode_data(json_path: str) -> dict:
    """Load per-episode results from eval JSON."""
    with open(json_path) as f:
        d = json.load(f)
    eps = d["episode_results"]
    return {
        "mean_score": np.array([e["mean_score"] for e in eps]),
        "min_score": np.array([e["min_score"] for e in eps]),
        "p10_score": np.array([e["p10_score"] for e in eps]),
        "mean_action_mag": np.array([e["mean_action_mag"] for e in eps]),
        "success": np.array([bool(e["success"]) for e in eps]),
        "n_episodes": len(eps),
        "n_success": d["n_success"],
        "n_fail": d["n_fail"],
        "perturb": d.get("perturb", "unknown"),
        "onset": d.get("perturb_start_step", "?"),
    }


# ── Selective execution curves ───────────────────────────────────────────

def compute_selective_curve(scores, successes, n_points=20):
    """Sort episodes by score descending, sweep coverage, return success rates."""
    n = len(scores)
    order = np.argsort(-scores)  # descending
    sorted_success = successes[order]

    coverages = np.linspace(1 / n, 1.0, n_points)
    success_rates = np.empty(n_points)
    for i, cov in enumerate(coverages):
        k = max(1, int(round(cov * n)))
        success_rates[i] = sorted_success[:k].mean()
    return coverages, success_rates


def bootstrap_ci(scores, successes, n_points=20, n_boot=1000, seed=42):
    """Bootstrap 95% CI for the selective execution curve."""
    rng = np.random.RandomState(seed)
    n = len(scores)
    curves = np.empty((n_boot, n_points))
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        _, sr = compute_selective_curve(scores[idx], successes[idx], n_points)
        curves[b] = sr
    ci_lo = np.percentile(curves, 2.5, axis=0)
    ci_hi = np.percentile(curves, 97.5, axis=0)
    return ci_lo, ci_hi


def auc_selective(coverages, success_rates):
    """Area under the selective execution curve (trapezoidal)."""
    return float(np.trapz(success_rates, coverages))


# ── Action magnitude direction ───────────────────────────────────────────

def orient_mag_scores(mag, successes):
    """Pick ascending or descending magnitude direction — whichever separates better."""
    # Try both: higher-mag-is-safer vs lower-mag-is-safer
    _, sr_desc = compute_selective_curve(mag, successes)
    _, sr_asc = compute_selective_curve(-mag, successes)
    auc_desc = sr_desc.mean()
    auc_asc = sr_asc.mean()
    if auc_asc > auc_desc:
        return -mag, "lower mag = safer"
    return mag, "higher mag = safer"


# ── Figure ───────────────────────────────────────────────────────────────

def make_figure(can, lift, output_dir, score_key="mean_score"):
    """Create 2-panel selective execution figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, data, title in [
        (axes[0], can, f"Can (onset={can['onset']})"),
        (axes[1], lift, f"Lift (onset={lift['onset']})"),
    ]:
        scores = data[score_key]
        successes = data["success"]
        mag = data["mean_action_mag"]
        baseline_sr = successes.mean()

        # TAP curve + CI
        cov, sr_tap = compute_selective_curve(scores, successes)
        ci_lo, ci_hi = bootstrap_ci(scores, successes)
        ax.plot(cov * 100, sr_tap, "o-", color="C0", markersize=4,
                label="TAP risk score", zorder=3)
        ax.fill_between(cov * 100, ci_lo, ci_hi, color="C0", alpha=0.15)

        # Magnitude baseline + CI
        mag_oriented, mag_dir = orient_mag_scores(mag, successes)
        cov_m, sr_mag = compute_selective_curve(mag_oriented, successes)
        ci_lo_m, ci_hi_m = bootstrap_ci(mag_oriented, successes)
        ax.plot(cov_m * 100, sr_mag, "s--", color="C1", markersize=4,
                label=f"Action magnitude", zorder=2)
        ax.fill_between(cov_m * 100, ci_lo_m, ci_hi_m, color="C1", alpha=0.12)

        # Random baseline
        ax.axhline(baseline_sr, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Random ({baseline_sr:.0%})", zorder=1)

        # Annotations
        idx_20 = np.argmin(np.abs(cov - 0.20))
        ax.annotate(f"{sr_tap[idx_20]:.0%}",
                    xy=(cov[idx_20] * 100, sr_tap[idx_20]),
                    xytext=(8, -12), textcoords="offset points",
                    fontsize=8, color="C0", fontweight="bold")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Coverage (%)")
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Success Rate")
    fig.suptitle("Selective Execution: TAP-Score vs Baselines", fontsize=12,
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "selective_execution.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "selective_execution.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out / 'selective_execution.png'}")
    print(f"Figure saved: {out / 'selective_execution.pdf'}")


# ── Summary table ────────────────────────────────────────────────────────

def print_summary(can, lift, score_key="mean_score"):
    """Print key numbers at standard coverage levels."""
    print("\nSelective Execution Analysis")
    print("=" * 60)

    for data, name in [(can, "Can"), (lift, "Lift")]:
        scores = data[score_key]
        successes = data["success"]
        mag = data["mean_action_mag"]
        baseline_sr = successes.mean()

        mag_oriented, mag_dir = orient_mag_scores(mag, successes)

        cov, sr_tap = compute_selective_curve(scores, successes)
        _, sr_mag = compute_selective_curve(mag_oriented, successes)

        auc_tap = auc_selective(cov, sr_tap)
        auc_mag = auc_selective(cov, sr_mag)
        auc_rand = baseline_sr  # constant line → AUC = baseline_sr * 1.0

        print(f"\n{name} ({data['perturb']}, onset={data['onset']}): "
              f"{data['n_episodes']} episodes "
              f"({data['n_success']} success, {data['n_fail']} fail)")
        print(f"  Mag baseline direction: {mag_dir}")
        print(f"  {'Coverage':>10s}  {'TAP':>7s}  {'Mag':>7s}  {'Random':>7s}")

        targets = [0.10, 0.20, 0.30, 0.50, 1.00]
        for t in targets:
            idx = np.argmin(np.abs(cov - t))
            print(f"  {cov[idx]:>9.0%}  {sr_tap[idx]:>7.1%}  "
                  f"{sr_mag[idx]:>7.1%}  {baseline_sr:>7.1%}")

        print(f"  AUC-SE: TAP={auc_tap:.3f}  Mag={auc_mag:.3f}  "
              f"Random={auc_rand:.3f}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Selective execution analysis for TAP-Score")
    parser.add_argument("--can",
        default="eval_results/risk_detection_can_zero80_onset_n100.json",
        help="Can eval JSON path")
    parser.add_argument("--lift",
        default="eval_results/risk_detection_lift_zero31_epwin_n100.json",
        help="Lift eval JSON path")
    parser.add_argument("--output_dir", default="artifacts/",
        help="Output directory for figures")
    parser.add_argument("--score_key", default="mean_score",
        choices=["mean_score", "min_score", "p10_score"],
        help="Which score to use for TAP curve")
    args = parser.parse_args()

    can = load_episode_data(args.can)
    lift = load_episode_data(args.lift)

    print_summary(can, lift, args.score_key)
    make_figure(can, lift, args.output_dir, args.score_key)


if __name__ == "__main__":
    main()

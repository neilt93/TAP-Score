"""
Compare Tier 0 baseline outputs (clean vs perturbed) from reranking_experiment.py.

Expects JSON files produced by scripts/reranking_experiment.py and reports:
  - per-episode mean endpoint (reward_notap) mean + 95% CI
  - paired delta (perturbed - clean) mean + 95% CI on shared seeds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def mean_ci95(values: np.ndarray) -> Dict[str, float]:
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


def load_seed_to_metric(path: Path) -> Dict[int, float]:
    data = json.loads(path.read_text())
    episodes = data.get("episodes", [])
    out: Dict[int, float] = {}
    for row in episodes:
        seed = int(row["seed"])
        # Tier 0 baseline metric: per-episode aggregation of decision-point endpoint.
        out[seed] = float(row["mean_reward_notap"])
    return out


def paired_arrays(clean: Dict[int, float], perturbed: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    shared = sorted(set(clean.keys()) & set(perturbed.keys()))
    clean_arr = np.array([clean[s] for s in shared], dtype=np.float64)
    pert_arr = np.array([perturbed[s] for s in shared], dtype=np.float64)
    return clean_arr, pert_arr, shared


def main():
    parser = argparse.ArgumentParser(description="Compare Tier 0 clean vs perturbed baselines")
    parser.add_argument("--clean", type=str, required=True, help="Path to clean baseline JSON")
    parser.add_argument("--perturbed", type=str, required=True, help="Path to perturbed baseline JSON")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    clean_path = Path(args.clean)
    pert_path = Path(args.perturbed)

    clean_map = load_seed_to_metric(clean_path)
    pert_map = load_seed_to_metric(pert_path)
    clean_arr, pert_arr, shared = paired_arrays(clean_map, pert_map)
    if clean_arr.size == 0:
        raise ValueError("No shared seeds between clean and perturbed outputs.")

    delta = pert_arr - clean_arr

    clean_stats = mean_ci95(clean_arr)
    pert_stats = mean_ci95(pert_arr)
    delta_stats = mean_ci95(delta)

    print("=" * 64)
    print("TIER 0 BASELINE COMPARISON")
    print("=" * 64)
    print(f"Shared seeds: {len(shared)}")
    print(f"Clean mean:      {clean_stats['mean']:.4f} [{clean_stats['ci95_low']:.4f}, {clean_stats['ci95_high']:.4f}]")
    print(f"Perturbed mean:  {pert_stats['mean']:.4f} [{pert_stats['ci95_low']:.4f}, {pert_stats['ci95_high']:.4f}]")
    print(f"Delta (pert-clean): {delta_stats['mean']:.4f} [{delta_stats['ci95_low']:.4f}, {delta_stats['ci95_high']:.4f}]")
    print("=" * 64)

    if args.output:
        out = {
            "clean_path": str(clean_path),
            "perturbed_path": str(pert_path),
            "n_shared_seeds": len(shared),
            "clean": clean_stats,
            "perturbed": pert_stats,
            "delta_perturbed_minus_clean": delta_stats,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"Saved comparison to {out_path}")


if __name__ == "__main__":
    main()


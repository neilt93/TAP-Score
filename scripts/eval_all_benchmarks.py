#!/usr/bin/env python3
"""
Evaluate TAP-Score across all benchmarks and create comparison table.

Usage:
    python scripts/eval_all_benchmarks.py
    python scripts/eval_all_benchmarks.py --benchmarks pusht lift
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tap.benchmarks import BENCHMARKS, get_benchmark_config


def run_eval(benchmark: str, checkpoint: str = None) -> dict:
    """Run evaluation for a single benchmark."""
    if checkpoint is None:
        checkpoint = f"checkpoints_contrastive/{benchmark}/contrastive_tap_best.pt"

    # Check if checkpoint exists
    checkpoint_path = PROJECT_ROOT / checkpoint
    if not checkpoint_path.exists():
        print(f"  [SKIP] Checkpoint not found: {checkpoint}")
        return None

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "eval_tap_final.py"),
        "--benchmark", benchmark,
        "--checkpoint", checkpoint,
    ]

    print(f"  Running: {' '.join(cmd[-4:])}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"  [ERROR] {result.stderr[:200]}")
            return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    # Load results
    results_file = PROJECT_ROOT / "eval_results" / f"{benchmark}_eval_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def print_comparison_table(results: dict):
    """Print comparison table across benchmarks."""
    print("\n" + "=" * 80)
    print("CROSS-BENCHMARK TAP-SCORE COMPARISON")
    print("=" * 80)

    print("\n| Benchmark   | Action Dim | AUROC | Prefix AUROC | TPR@1%FPR | TPR@5%FPR |")
    print("|-------------|------------|-------|--------------|-----------|-----------|")

    for benchmark, r in results.items():
        if r is None:
            print(f"| {benchmark:11} | {'N/A':^10} | {'N/A':^5} | {'N/A':^12} | {'N/A':^9} | {'N/A':^9} |")
        else:
            action_dim = r.get('action_dim', '?')
            auroc = r.get('tap_auroc', 0)
            prefix = r.get('prefix_auroc', 0) or 0
            tpr_1 = r.get('operating_points', {}).get('1%', {}).get('tpr', 0) * 100
            tpr_5 = r.get('operating_points', {}).get('5%', {}).get('tpr', 0) * 100
            print(f"| {benchmark:11} | {action_dim:^10} | {auroc:.3f} | {prefix:.3f}        | {tpr_1:6.1f}%   | {tpr_5:6.1f}%   |")

    print("\n" + "=" * 80)

    # Summary statistics
    valid_results = [r for r in results.values() if r is not None]
    if valid_results:
        avg_auroc = sum(r['tap_auroc'] for r in valid_results) / len(valid_results)
        print(f"\nSummary: {len(valid_results)}/{len(results)} benchmarks evaluated")
        print(f"Average AUROC: {avg_auroc:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TAP across all benchmarks")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Specific benchmarks to evaluate (default: all)")
    parser.add_argument("--output", type=str, default="eval_results/cross_benchmark_results.json",
                        help="Output file for combined results")
    args = parser.parse_args()

    benchmarks = args.benchmarks or list(BENCHMARKS.keys())

    print("=" * 80)
    print("TAP-Score Cross-Benchmark Evaluation")
    print("=" * 80)
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print("=" * 80)

    results = {}
    for benchmark in benchmarks:
        print(f"\n[{benchmark.upper()}]")
        config = get_benchmark_config(benchmark)
        print(f"  {config['name']} (action_dim={config['action_dim']})")

        r = run_eval(benchmark)
        results[benchmark] = r

        if r is not None:
            print(f"  AUROC: {r['tap_auroc']:.3f}")

    # Print comparison table
    print_comparison_table(results)

    # Save combined results
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

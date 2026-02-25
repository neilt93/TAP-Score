#!/usr/bin/env python3
"""
Preflight inspection of robomimic lowdim HDF5 datasets.

Run this BEFORE training DP or running the headroom audit to verify:
- HDF5 structure and demo count
- Obs key shapes and concatenated obs_dim
- Action shape and range
- Episode length statistics
- Consistency with TAP benchmark config expectations

Usage:
    python scripts/inspect_robomimic_lowdim.py [--task lift|can|all]
    python scripts/inspect_robomimic_lowdim.py --hdf5_path path/to/low_dim.hdf5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Expected configs (must match tap/benchmarks.py)
EXPECTED = {
    "lift": {
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "obs_dim": 19,
        "action_dim": 7,
        "hdf5_path": "data/robomimic/datasets/lift/ph/low_dim.hdf5",
    },
    "can": {
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "obs_dim": 23,
        "action_dim": 7,
        "hdf5_path": "data/robomimic/datasets/can/ph/low_dim.hdf5",
    },
}


def inspect_hdf5(hdf5_path: str, task_name: str | None = None):
    """Inspect a robomimic lowdim HDF5 file."""
    path = Path(hdf5_path)
    if not path.exists():
        print(f"  ERROR: File not found: {path}")
        return False

    print(f"  File: {path} ({path.stat().st_size / 1e6:.1f} MB)")

    ok = True
    with h5py.File(str(path), "r") as f:
        # Top-level keys
        print(f"  Top-level keys: {list(f.keys())}")

        if "data" not in f:
            print("  ERROR: No 'data' group found")
            return False

        demos = f["data"]
        demo_keys = sorted((k for k in demos.keys() if k.startswith("demo_")),
                           key=lambda k: int(k.split("_")[1]))
        print(f"  Demos: {len(demo_keys)}")

        if len(demo_keys) == 0:
            print("  ERROR: No demos found")
            return False

        # Inspect first demo
        first = demos[demo_keys[0]]
        print(f"\n  First demo ({demo_keys[0]}):")
        print(f"    Keys: {list(first.keys())}")

        # Actions
        if "actions" in first:
            actions = first["actions"]
            print(f"    actions: shape={actions.shape}, dtype={actions.dtype}")
            action_data = actions[:]
            print(f"    actions range: [{action_data.min():.4f}, {action_data.max():.4f}]")
        else:
            print("    ERROR: No 'actions' key")
            ok = False

        # Observations
        if "obs" in first:
            obs = first["obs"]
            print(f"    obs keys: {sorted(obs.keys())}")

            total_dim = 0
            for key in sorted(obs.keys()):
                shape = obs[key].shape
                print(f"      {key}: shape={shape}, dtype={obs[key].dtype}")
                if len(shape) == 2:
                    total_dim += shape[1]

            print(f"    Total obs dim (all keys): {total_dim}")

            # Check specific obs_keys if task is known
            if task_name and task_name in EXPECTED:
                expected = EXPECTED[task_name]
                concat_dim = 0
                for key in expected["obs_keys"]:
                    if key in obs:
                        concat_dim += obs[key].shape[-1]
                    else:
                        print(f"    WARNING: Expected obs key '{key}' not found!")
                        ok = False

                print(f"    Concat dim (expected keys only): {concat_dim}")
                if concat_dim != expected["obs_dim"]:
                    print(f"    MISMATCH: Expected obs_dim={expected['obs_dim']}, got {concat_dim}")
                    ok = False
                else:
                    print(f"    OK: obs_dim matches expected ({expected['obs_dim']})")

                if "actions" in first:
                    actual_action_dim = first["actions"].shape[1]
                    if actual_action_dim != expected["action_dim"]:
                        print(f"    MISMATCH: Expected action_dim={expected['action_dim']}, got {actual_action_dim}")
                        ok = False
                    else:
                        print(f"    OK: action_dim matches expected ({expected['action_dim']})")
        else:
            print("    ERROR: No 'obs' key")
            ok = False

        # Episode length stats
        lengths = []
        for dk in demo_keys:
            if "actions" in demos[dk]:
                lengths.append(demos[dk]["actions"].shape[0])
        if lengths:
            print(f"\n  Episode lengths:")
            print(f"    min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")
            print(f"    total timesteps: {sum(lengths)}")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Inspect robomimic lowdim HDF5 datasets")
    parser.add_argument("--task", default="all", choices=["lift", "can", "all"],
                        help="Which task to inspect")
    parser.add_argument("--hdf5_path", type=str, default=None,
                        help="Direct path to HDF5 file (overrides --task)")
    args = parser.parse_args()

    if args.hdf5_path:
        print(f"\n=== Inspecting: {args.hdf5_path} ===")
        ok = inspect_hdf5(args.hdf5_path)
        sys.exit(0 if ok else 1)

    tasks = ["lift", "can"] if args.task == "all" else [args.task]
    all_ok = True

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  Task: {task}")
        print(f"{'='*60}")
        hdf5_path = EXPECTED[task]["hdf5_path"]
        ok = inspect_hdf5(hdf5_path, task_name=task)
        if ok:
            print(f"\n  PASS: {task}")
        else:
            print(f"\n  FAIL: {task}")
            all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("All checks passed.")
    else:
        print("Some checks FAILED — review output above.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

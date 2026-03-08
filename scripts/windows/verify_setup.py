#!/usr/bin/env python3
"""
Cross-platform verification of TAP-Score + robomimic setup.
Checks all dependencies, data files, and checkpoints.

Usage:
    python scripts/windows/verify_setup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    status = "OK" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if condition:
        passed += 1
    else:
        failed += 1


def main():
    print("=" * 60)
    print("TAP-Score Setup Verification")
    print(f"Project root: {REPO_ROOT}")
    print("=" * 60)

    # ── 1. TAP import ──
    print("\n--- Core Imports ---")
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from tap.contrastive import ContrastiveTAPScore  # noqa: F401
        check("TAP import", True)
    except Exception as e:
        check("TAP import", False, str(e))

    # ── 2. PyTorch + CUDA ──
    try:
        import torch
        check("PyTorch", True, f"v{torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            dev_name = torch.cuda.get_device_name(0)
            check("CUDA", True, dev_name)
        else:
            check("CUDA", False, "not available")
    except Exception as e:
        check("PyTorch", False, str(e))

    # ── 3. robomimic ──
    try:
        import robomimic  # noqa: F401
        version = getattr(robomimic, "__version__", "unknown")
        check("robomimic", True, f"v{version}")
    except Exception as e:
        check("robomimic", False, str(e))

    # ── 4. MuJoCo ──
    try:
        import mujoco  # noqa: F401
        check("MuJoCo", True, f"v{mujoco.__version__}")
    except Exception as e:
        check("MuJoCo", False, str(e))

    # ── 5. Other key deps ──
    for mod in ["zarr", "h5py", "hydra", "omegaconf", "dill", "einops", "numba"]:
        try:
            __import__(mod)
            check(f"{mod}", True)
        except Exception as e:
            check(f"{mod}", False, str(e))

    # ── 6. Diffusion Policy ──
    print("\n--- Diffusion Policy ---")
    dp_root = REPO_ROOT / "baselines" / "diffusion_policy"
    check("DP repo cloned", dp_root.is_dir())

    if dp_root.is_dir():
        if str(dp_root) not in sys.path:
            sys.path.insert(0, str(dp_root))
        try:
            from diffusion_policy.policy.diffusion_unet_lowdim_policy import (  # noqa: F401
                DiffusionUnetLowdimPolicy,
            )
            check("DP import", True)
        except Exception as e:
            check("DP import", False, str(e))

    # ── 7. HDF5 datasets ──
    print("\n--- Robomimic Datasets ---")
    for task in ["lift", "can"]:
        hdf5_path = REPO_ROOT / "data" / "robomimic" / "datasets" / task / "ph" / "low_dim.hdf5"
        exists = hdf5_path.is_file()
        detail = ""
        if exists:
            try:
                import h5py

                with h5py.File(str(hdf5_path), "r") as f:
                    demos = [k for k in f["data"].keys() if k.startswith("demo_")]
                    detail = f"{len(demos)} demos, {hdf5_path.stat().st_size / 1e6:.0f} MB"
            except Exception as e:
                detail = str(e)
        else:
            detail = f"not found at {hdf5_path}"
        check(f"{task} HDF5 dataset", exists, detail)

    # ── 8. DP checkpoints ──
    print("\n--- DP Checkpoints ---")
    ckpt_dir = REPO_ROOT / "data" / "robomimic" / "checkpoints"
    for task, filename in [
        ("lift", "lift_ph_diffusion_policy_cnn.ckpt"),
        ("can", "can_ph_diffusion_policy_cnn.ckpt"),
    ]:
        ckpt_path = ckpt_dir / filename
        exists = ckpt_path.is_file()
        detail = ""
        if exists:
            size_mb = ckpt_path.stat().st_size / 1e6
            detail = f"{size_mb:.0f} MB"
        else:
            detail = f"not found at {ckpt_path}"
        check(f"{task} DP checkpoint", exists, detail)

    # ── 9. Push-T data (optional) ──
    print("\n--- Push-T Data (optional) ---")
    pusht_zarr = REPO_ROOT / "data" / "raw" / "pusht" / "pusht_cchi_v7_replay.zarr"
    if pusht_zarr.is_dir():
        try:
            import zarr

            z = zarr.open_group(str(pusht_zarr), "r")
            n_steps = z["data"]["action"].shape[0]
            check("Push-T zarr", True, f"{n_steps} timesteps")
        except Exception as e:
            check("Push-T zarr", False, str(e))
    else:
        check("Push-T zarr", False, "not found (optional — only needed for pusht benchmark)")

    # ── Summary ──
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("All checks passed!")
    else:
        print("Some checks failed. Review above and re-run setup if needed.")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

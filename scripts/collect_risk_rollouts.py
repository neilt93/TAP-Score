"""
Collect labeled rollout data for risk model training.

Runs DP policy under various perturbation regimes and saves per-episode NPZ files
with obs, action chunks, decision steps, and episode success labels.

Usage:
    python scripts/collect_risk_rollouts.py \
        --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
        --task can --n_episodes 200 --output_dir data/risk_rollouts/can/ \
        --regimes clean zero80 freeze80 dropout80
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DP_ROOT = REPO_ROOT / "baselines" / "diffusion_policy"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from scripts.robomimic_headroom_audit import (
    load_diffusion_policy,
    make_robomimic_env_factory,
    resolve_existing_path,
    cfg_get,
    obs_to_dict,
    build_obs_tensor,
    stack_history,
    step_chunk,
    trim_latency,
    _apply_obs_perturbation,
    _find_object_slice,
    extract_success,
    undo_transform_action,
    set_env_state,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

# Regime definitions: name -> (perturb_type, onset_can, onset_lift, onset_jitter, dropout_p)
REGIME_DEFS = {
    "clean":          ("none",                 0,   0,  0, 0.0),
    "zero80":         ("zero_object",          80,  31, 0, 0.0),
    "zero80_jitter":  ("zero_object",          80,  31, 8, 0.0),
    "freeze80":       ("freeze_object",        80,  31, 0, 0.0),
    "dropout80_p02":  ("intermittent_dropout", 80,  31, 0, 0.2),
    "dropout80_p04":  ("intermittent_dropout", 80,  31, 0, 0.4),
    "dropout80":      ("intermittent_dropout", 80,  31, 0, 0.3),
}


def get_onset(regime_name: str, task: str, seed: int = 0) -> int:
    """Get perturbation onset step for a regime and task, with optional jitter."""
    _, onset_can, onset_lift, jitter, _ = REGIME_DEFS[regime_name]
    base = onset_lift if task.lower() == "lift" else onset_can
    if jitter > 0:
        rng = np.random.RandomState(seed + 77777)
        base += rng.randint(-jitter, jitter + 1)
    return max(0, base)


def get_dropout_p(regime_name: str) -> float:
    """Get dropout probability for intermittent_dropout regimes."""
    return REGIME_DEFS[regime_name][4]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect risk rollout data")
    parser.add_argument("--dp_checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["can", "lift", "square"])
    parser.add_argument("--n_episodes", type=int, default=200,
                        help="Total episodes (split evenly across regimes)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--regimes", nargs="+",
                        default=["clean", "zero80", "zero80_jitter", "dropout80_p02", "dropout80_p04"],
                        choices=list(REGIME_DEFS.keys()))
    parser.add_argument("--kp", type=float, default=500.0)
    parser.add_argument("--max_env_steps", type=int, default=None)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if var not in os.environ:
            os.environ[var] = str(min(os.cpu_count() or 4, 4))

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Risk Rollout Collector")
    print("=" * 70)

    # ── Load DP policy ───────────────────────────────────────────────
    print(f"\nLoading DP policy from {args.dp_checkpoint} ...")
    dp_policy, cfg = load_diffusion_policy(args.dp_checkpoint, device=device)
    task_cfg = cfg.task

    task_name = str(cfg_get(task_cfg, "task_name", "unknown_task"))
    dataset_path_cfg = str(cfg_get(task_cfg, "dataset_path"))
    dataset_path = resolve_existing_path(dataset_path_cfg)

    abs_action = bool(cfg_get(task_cfg, "abs_action", False))
    rotation_transformer = (
        RotationTransformer("axis_angle", "rotation_6d") if abs_action else None
    )

    n_obs_steps = int(cfg_get(task_cfg.env_runner, "n_obs_steps"))
    n_action_steps = int(cfg_get(task_cfg.env_runner, "n_action_steps"))
    n_latency_steps = int(cfg_get(task_cfg.env_runner, "n_latency_steps", 0))
    cfg_max_steps = int(cfg_get(task_cfg.env_runner, "max_steps", 400))
    max_env_steps = args.max_env_steps if args.max_env_steps is not None else cfg_max_steps

    print(f"Task:            {task_name}")
    print(f"n_obs_steps:     {n_obs_steps}")
    print(f"n_action_steps:  {n_action_steps}")
    print(f"max_env_steps:   {max_env_steps}")
    print(f"abs_action:      {abs_action}")

    # ── Create env ───────────────────────────────────────────────────
    env_factory, env_info = make_robomimic_env_factory(
        cfg, dataset_path=dataset_path, kp=args.kp,
    )
    obs_keys = env_info.get("obs_keys", None)
    print(f"Obs keys:        {obs_keys}")

    # ── Load demo initial states ─────────────────────────────────────
    import h5py
    h5_file = h5py.File(str(dataset_path), "r")
    demo_keys = sorted(h5_file["data"].keys())
    print(f"Dataset demos:   {len(demo_keys)}")

    def sample_init_state(seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        demo = demo_keys[int(rng.randint(0, len(demo_keys)))]
        return np.array(h5_file["data"][demo]["states"][0], copy=True)

    # ── Build episode schedule ───────────────────────────────────────
    n_regimes = len(args.regimes)
    episodes_per_regime = args.n_episodes // n_regimes
    remainder = args.n_episodes % n_regimes

    schedule: List[Tuple[str, int]] = []  # (regime, seed)
    seed_counter = args.seed_offset
    for r_idx, regime in enumerate(args.regimes):
        n_this = episodes_per_regime + (1 if r_idx < remainder else 0)
        for _ in range(n_this):
            schedule.append((regime, seed_counter))
            seed_counter += 1

    print(f"\nSchedule: {len(schedule)} episodes across {args.regimes}")
    for regime in args.regimes:
        n = sum(1 for r, _ in schedule if r == regime)
        print(f"  {regime}: {n} episodes")

    # ── Checkpoint for resume ────────────────────────────────────────
    ckpt_path = output_dir / "_checkpoint.json"
    completed: set = set()
    if ckpt_path.exists():
        try:
            with open(ckpt_path) as f:
                completed = set(json.load(f).get("completed_seeds", []))
            print(f"Resumed: {len(completed)} episodes already done")
        except Exception as e:
            print(f"Warning: checkpoint load failed: {e}")

    # ── Create env once ──────────────────────────────────────────────
    env = env_factory()

    print(f"\nCollecting rollouts...")
    t0 = time.time()

    for ep_idx, (regime, seed) in enumerate(tqdm(schedule, desc="Rollouts")):
        if seed in completed:
            continue

        perturb_type = REGIME_DEFS[regime][0]
        onset = get_onset(regime, args.task, seed=seed)
        dropout_p = get_dropout_p(regime)

        try:
            if hasattr(env, "seed"):
                env.seed(seed)
            obs = env.reset()

            state0 = sample_init_state(seed)
            obs_dict = set_env_state(env, state0, obs_keys=obs_keys, task_name=task_name)

            # Track frozen object for freeze_object
            frozen_object = None
            if perturb_type == "freeze_object" and obs_keys is not None and onset == 0:
                if "obs" in obs_dict and obs_dict["obs"].ndim == 1:
                    obj_slice = _find_object_slice(obs_keys, task_name)
                    if obj_slice is not None:
                        s, e = obj_slice
                        frozen_object = obs_dict["obs"][s:e].copy()

            # Apply perturbation to initial obs if onset == 0
            if perturb_type != "none" and obs_keys is not None and 0 >= onset:
                if "obs" in obs_dict and obs_dict["obs"].ndim == 1:
                    perturb_rng = np.random.RandomState(seed + 9999)
                    obs_dict["obs"] = _apply_obs_perturbation(
                        obs_dict["obs"], obs_keys, perturb_type, task_name,
                        perturb_rng, 0.01, frozen_object,
                        dropout_p=dropout_p,
                    )

            obs_history: Dict[str, List[np.ndarray]] = defaultdict(list)
            for key, val in obs_dict.items():
                obs_history[key].append(val)

            # Collect obs and actions for this episode
            all_obs: List[np.ndarray] = []
            if "obs" in obs_dict:
                all_obs.append(obs_dict["obs"].copy())

            action_chunks_list: List[np.ndarray] = []
            decision_steps_list: List[int] = []

            episode_return = 0.0
            episode_success = False
            env_steps = 0
            done = False
            perturb_rng = np.random.RandomState(seed + 9999)

            while not done and env_steps < max_env_steps:
                # Build obs tensor for DP
                obs_tensor = build_obs_tensor(obs_history, n_obs_steps, device=device)

                with torch.no_grad():
                    action_dict = dp_policy.predict_action(obs_tensor)

                raw_chunk = action_dict["action"][0].detach().cpu().numpy()

                # Convert to axis_angle for storage
                if abs_action and rotation_transformer is not None:
                    full_chunk_7d = undo_transform_action(raw_chunk, rotation_transformer)
                else:
                    full_chunk_7d = raw_chunk

                # Record decision
                action_chunks_list.append(full_chunk_7d.astype(np.float32))
                decision_steps_list.append(env_steps)

                # Execute: trim latency, step
                exec_chunk = trim_latency(raw_chunk, n_latency_steps)

                r, done, succ, env_steps = step_chunk(
                    env, obs_history, exec_chunk, env_steps, max_env_steps,
                    abs_action, rotation_transformer,
                    obs_keys=obs_keys, task_name=task_name,
                    perturb_type=perturb_type, perturb_start_step=onset,
                    perturb_rng=perturb_rng, perturb_noise_std=0.01,
                    frozen_object=frozen_object,
                    dropout_p=dropout_p,
                )
                episode_return += r
                episode_success = episode_success or succ

                # Collect obs (post-perturbation, what DP saw)
                if "obs" in obs_history:
                    while len(all_obs) < len(obs_history["obs"]):
                        all_obs.append(obs_history["obs"][len(all_obs)].copy())

            if episode_return >= 1.0:
                episode_success = True

            # Save NPZ
            obs_seq = np.stack(all_obs, axis=0) if all_obs else np.zeros((0, 0))
            action_chunks_arr = np.stack(action_chunks_list, axis=0) if action_chunks_list else np.zeros((0, 0, 0))
            decision_steps_arr = np.array(decision_steps_list, dtype=np.int32)

            npz_path = output_dir / f"ep_{seed:05d}.npz"
            np.savez_compressed(
                str(npz_path),
                obs_seq=obs_seq,
                action_chunks=action_chunks_arr,
                decision_steps=decision_steps_arr,
                episode_success=np.array(episode_success),
                regime=np.array(regime),
                seed=np.array(seed, dtype=np.int32),
                onset=np.array(onset, dtype=np.int32),
            )

            status = "OK" if episode_success else "FAIL"
            tqdm.write(f"  ep={ep_idx:3d} seed={seed:3d} regime={regime:10s} "
                       f"{status:4s}  steps={env_steps}  decisions={len(decision_steps_list)}")

        except Exception as e:
            tqdm.write(f"  ep={ep_idx:3d} seed={seed:3d} regime={regime:10s} ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        completed.add(seed)

        # Checkpoint every 10 episodes
        if len(completed) % 10 == 0:
            try:
                with open(ckpt_path, "w") as f:
                    json.dump({"completed_seeds": sorted(completed)}, f)
            except Exception:
                pass

    # Final checkpoint
    try:
        with open(ckpt_path, "w") as f:
            json.dump({"completed_seeds": sorted(completed)}, f)
    except Exception:
        pass

    if hasattr(env, "close"):
        env.close()
    h5_file.close()

    elapsed = time.time() - t0
    print(f"\nDone: {len(completed)} episodes in {elapsed:.0f}s")
    print(f"Output: {output_dir}")

    # Summary stats
    npz_files = sorted(output_dir.glob("ep_*.npz"))
    n_success = 0
    n_fail = 0
    regime_counts: Dict[str, Dict[str, int]] = {}
    for f in npz_files:
        d = np.load(str(f), allow_pickle=True)
        r = str(d["regime"])
        s = bool(d["episode_success"])
        if r not in regime_counts:
            regime_counts[r] = {"success": 0, "fail": 0}
        if s:
            regime_counts[r]["success"] += 1
            n_success += 1
        else:
            regime_counts[r]["fail"] += 1
            n_fail += 1

    print(f"\nSummary: {n_success} success, {n_fail} fail")
    for r, counts in sorted(regime_counts.items()):
        total = counts["success"] + counts["fail"]
        print(f"  {r:12s}: {counts['success']}/{total} success "
              f"({100 * counts['success'] / max(total, 1):.0f}%)")


if __name__ == "__main__":
    main()

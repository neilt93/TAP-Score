"""
Intervention evaluation: TAP-Score as an active safety filter.

Two intervention modes:
1. **Resample**: When risk > threshold, generate K proposals from DP,
   score each with TAP, execute the safest one. Tests whether TAP can
   steer the policy toward safer actions at runtime.
2. **Early-stop + restart**: When cumulative risk over a sliding window
   exceeds threshold, terminate and restart the episode from the initial
   state. Budget of R restarts per trial. Tests early warning utility.

Unlike selective execution (which refuses entire episodes), intervention
improves end-to-end success rate by actually doing something about risk.

Usage:
    python scripts/eval_intervention.py \
        --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
        --tap_checkpoint checkpoints_contrastive/robomimic_can_lowdim/risk_tap_best.pt \
        --task can --mode resample --n_episodes 50 --K 4 \
        --output eval_results/intervention_can_resample_K4_n50.json

    python scripts/eval_intervention.py \
        --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
        --tap_checkpoint checkpoints_contrastive/robomimic_can_lowdim/risk_tap_best.pt \
        --task can --mode restart --n_episodes 50 --max_restarts 2 \
        --output eval_results/intervention_can_restart_R2_n50.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    build_obs_tensor,
    stack_history,
    step_chunk,
    trim_latency,
    _apply_obs_perturbation,
    _find_object_slice,
    undo_transform_action,
    set_env_state,
    PERTURB_TYPES,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from tap.risk import RiskTAPScore


def parse_args():
    parser = argparse.ArgumentParser(description="TAP-Score intervention evaluation")
    parser.add_argument("--dp_checkpoint", type=str, required=True)
    parser.add_argument("--tap_checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["can", "lift"])
    parser.add_argument("--mode", type=str, required=True,
                        choices=["resample", "restart", "baseline"],
                        help="Intervention mode")
    parser.add_argument("--n_episodes", type=int, default=50)

    # Resample mode params
    parser.add_argument("--K", type=int, default=4,
                        help="Number of DP proposals to sample (resample mode)")
    parser.add_argument("--risk_threshold", type=float, default=0.5,
                        help="Risk threshold to trigger resampling/restart")

    # Restart mode params
    parser.add_argument("--max_restarts", type=int, default=2,
                        help="Max restarts per trial (restart mode)")
    parser.add_argument("--risk_window", type=int, default=3,
                        help="Sliding window of policy steps for mean risk (restart mode)")

    # Perturbation
    parser.add_argument("--perturb", choices=PERTURB_TYPES, default="zero_object")
    parser.add_argument("--perturb_start_step", type=int, default=80)
    parser.add_argument("--perturb_noise_std", type=float, default=0.01)

    # Infra
    parser.add_argument("--kp", type=float, default=500.0)
    parser.add_argument("--max_env_steps", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def score_action(risk_model, obs_window, action, device, obs_only=False):
    """Score a single (obs, action) pair. Returns risk_prob in [0,1]."""
    obs_t = torch.from_numpy(obs_window).unsqueeze(0).to(device, dtype=torch.float32)
    act_t = (None if obs_only else
             torch.from_numpy(action).unsqueeze(0).to(device, dtype=torch.float32))
    with torch.no_grad():
        risk_prob = risk_model.predict_risk(obs_t, act_t)
    return float(risk_prob.item())


def run_episode(
    env, dp_policy, risk_model, state0,
    n_obs_steps, n_action_steps, n_latency_steps, max_env_steps,
    abs_action, rotation_transformer, obs_keys, task_name,
    perturb_type, perturb_start_step, perturb_noise_std,
    seed, device,
    tap_obs_window, tap_action_chunk, obs_only,
    # Intervention params
    mode="baseline", K=4, risk_threshold=0.5,
    max_restarts=2, risk_window=3,
):
    """Run one episode with optional intervention.

    Returns dict with episode stats including intervention metrics.
    """
    restarts_used = 0
    total_resamples = 0
    attempt = 0
    max_attempts = 1 + (max_restarts if mode == "restart" else 0)

    while attempt < max_attempts:
        attempt += 1

        # Reset env
        if hasattr(env, "seed"):
            env.seed(seed + attempt * 10000)
        obs = env.reset()
        obs_dict = set_env_state(env, state0, obs_keys=obs_keys, task_name=task_name)

        # Frozen object tracking
        frozen_object = None
        if perturb_type == "freeze_object" and obs_keys is not None and perturb_start_step == 0:
            if "obs" in obs_dict and obs_dict["obs"].ndim == 1:
                obj_slice = _find_object_slice(obs_keys, task_name)
                if obj_slice is not None:
                    s, e = obj_slice
                    frozen_object = obs_dict["obs"][s:e].copy()

        # Apply perturbation if onset == 0
        if perturb_type != "none" and obs_keys is not None and 0 >= perturb_start_step:
            if "obs" in obs_dict and obs_dict["obs"].ndim == 1:
                prng = np.random.RandomState(seed + 9999)
                obs_dict["obs"] = _apply_obs_perturbation(
                    obs_dict["obs"], obs_keys, perturb_type, task_name,
                    prng, perturb_noise_std, frozen_object,
                )

        obs_history = defaultdict(list)
        for key, val in obs_dict.items():
            obs_history[key].append(val)

        tap_obs_history = []
        if "obs" in obs_dict:
            tap_obs_history.append(obs_dict["obs"].copy())

        episode_return = 0.0
        episode_success = False
        env_steps = 0
        policy_step = 0
        done = False
        perturb_rng = np.random.RandomState(seed + 9999)
        recent_risks = []
        ep_resamples = 0
        restarted = False

        while not done and env_steps < max_env_steps:
            obs_tensor = build_obs_tensor(obs_history, n_obs_steps, device=device)

            # Get primary DP action
            with torch.no_grad():
                action_dict = dp_policy.predict_action(obs_tensor)
            raw_chunk = action_dict["action"][0].detach().cpu().numpy()

            if abs_action and rotation_transformer is not None:
                full_chunk = undo_transform_action(raw_chunk, rotation_transformer)
            else:
                full_chunk = raw_chunk

            # Prepare TAP action
            if full_chunk.shape[0] >= tap_action_chunk:
                tap_action = full_chunk[:tap_action_chunk].astype(np.float32)
            else:
                pad_len = tap_action_chunk - full_chunk.shape[0]
                pad = np.tile(full_chunk[-1:], (pad_len, 1))
                tap_action = np.concatenate([full_chunk, pad], axis=0).astype(np.float32)

            # Score if we have enough obs and are past onset
            risk_prob = None
            chosen_raw = raw_chunk
            if (env_steps >= perturb_start_step and
                    len(tap_obs_history) >= tap_obs_window):
                obs_win = stack_history(tap_obs_history, tap_obs_window)

                risk_prob = score_action(
                    risk_model, obs_win, tap_action, device, obs_only)

                # ── Resample intervention ────────────────────────────
                if mode == "resample" and risk_prob > risk_threshold:
                    best_risk = risk_prob
                    best_raw = raw_chunk
                    best_full = full_chunk

                    for k in range(K - 1):
                        with torch.no_grad():
                            alt_dict = dp_policy.predict_action(obs_tensor)
                        alt_raw = alt_dict["action"][0].detach().cpu().numpy()
                        if abs_action and rotation_transformer is not None:
                            alt_full = undo_transform_action(alt_raw, rotation_transformer)
                        else:
                            alt_full = alt_raw

                        if alt_full.shape[0] >= tap_action_chunk:
                            alt_tap = alt_full[:tap_action_chunk].astype(np.float32)
                        else:
                            plen = tap_action_chunk - alt_full.shape[0]
                            alt_tap = np.concatenate(
                                [alt_full, np.tile(alt_full[-1:], (plen, 1))],
                                axis=0).astype(np.float32)

                        alt_risk = score_action(
                            risk_model, obs_win, alt_tap, device, obs_only)

                        if alt_risk < best_risk:
                            best_risk = alt_risk
                            best_raw = alt_raw
                            best_full = alt_full

                    # Use the safest action
                    if best_risk < risk_prob:
                        ep_resamples += 1
                        raw_chunk = best_raw
                        full_chunk = best_full
                        chosen_raw = best_raw
                        risk_prob = best_risk

                # ── Restart intervention ─────────────────────────────
                if mode == "restart":
                    recent_risks.append(risk_prob)
                    if len(recent_risks) > risk_window:
                        recent_risks.pop(0)
                    mean_risk = np.mean(recent_risks)

                    if (mean_risk > risk_threshold and
                            restarts_used < max_restarts and
                            policy_step >= 2):
                        restarts_used += 1
                        restarted = True
                        break  # break inner loop, retry in outer

            # Execute action
            exec_chunk = trim_latency(chosen_raw, n_latency_steps)
            r, done, succ, env_steps = step_chunk(
                env, obs_history, exec_chunk,
                env_steps, max_env_steps,
                abs_action, rotation_transformer,
                obs_keys=obs_keys, task_name=task_name,
                perturb_type=perturb_type,
                perturb_start_step=perturb_start_step,
                perturb_rng=perturb_rng,
                perturb_noise_std=perturb_noise_std,
                frozen_object=frozen_object,
            )
            episode_return += r
            episode_success = episode_success or succ

            if "obs" in obs_history and len(obs_history["obs"]) > len(tap_obs_history):
                for i in range(len(tap_obs_history), len(obs_history["obs"])):
                    tap_obs_history.append(obs_history["obs"][i].copy())

            policy_step += 1

        if episode_return >= 1.0:
            episode_success = True

        # If not restarted, this attempt completed normally — done
        if not restarted:
            break

        total_resamples += ep_resamples
        # Reset for next attempt
        restarted = False

    total_resamples += ep_resamples

    return {
        "seed": seed,
        "success": episode_success,
        "return": float(episode_return),
        "restarts_used": restarts_used,
        "resamples": total_resamples,
        "attempts": attempt,
        "env_steps": env_steps,
    }


def main():
    args = parse_args()

    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if var not in os.environ:
            os.environ[var] = str(min(os.cpu_count() or 4, 4))

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    print("=" * 70)
    print(f"TAP-Score Intervention Evaluation — mode={args.mode}")
    print("=" * 70)

    # Load DP policy
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
    max_env_steps = args.max_env_steps or cfg_max_steps

    # Create env
    env_factory, env_info = make_robomimic_env_factory(
        cfg, dataset_path=dataset_path, kp=args.kp,
    )
    obs_keys = env_info.get("obs_keys", None)

    # Load demo initial states
    import h5py
    h5_file = h5py.File(str(dataset_path), "r")
    demo_keys = sorted(h5_file["data"].keys())

    def sample_init_state(seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        demo = demo_keys[int(rng.randint(0, len(demo_keys)))]
        return np.array(h5_file["data"][demo]["states"][0], copy=True)

    # Load risk model
    risk_ckpt = torch.load(args.tap_checkpoint, map_location=device, weights_only=False)
    risk_config = risk_ckpt["config"]
    risk_obs_only = risk_config.get("obs_only", False)
    risk_model = RiskTAPScore(
        obs_dim=risk_config["obs_dim"],
        action_dim=risk_config["action_dim"],
        obs_window=risk_config.get("obs_window", 2),
        action_chunk=risk_config.get("action_chunk", 8),
        hidden_dim=risk_config.get("hidden_dim", 128),
        obs_only=risk_obs_only,
    )
    risk_model.load_state_dict(risk_ckpt["model_state_dict"])
    risk_model.to(device).eval()

    tap_obs_window = risk_config.get("obs_window", 2)
    tap_action_chunk = risk_config.get("action_chunk", 8)

    print(f"Task:              {task_name}")
    print(f"Mode:              {args.mode}")
    print(f"Perturbation:      {args.perturb} (onset={args.perturb_start_step})")
    print(f"N episodes:        {args.n_episodes}")
    if args.mode == "resample":
        print(f"K (proposals):     {args.K}")
        print(f"Risk threshold:    {args.risk_threshold}")
    elif args.mode == "restart":
        print(f"Max restarts:      {args.max_restarts}")
        print(f"Risk window:       {args.risk_window}")
        print(f"Risk threshold:    {args.risk_threshold}")
    print("=" * 70)

    # Run episodes
    env = env_factory()
    seeds = list(range(args.seed_offset, args.seed_offset + args.n_episodes))
    results = []

    for ep_idx, seed in enumerate(tqdm(seeds, desc="Episodes")):
        try:
            state0 = sample_init_state(seed)
            ep_result = run_episode(
                env, dp_policy, risk_model, state0,
                n_obs_steps, n_action_steps, n_latency_steps, max_env_steps,
                abs_action, rotation_transformer, obs_keys, task_name,
                args.perturb, args.perturb_start_step, args.perturb_noise_std,
                seed, device,
                tap_obs_window, tap_action_chunk, risk_obs_only,
                mode=args.mode, K=args.K, risk_threshold=args.risk_threshold,
                max_restarts=args.max_restarts, risk_window=args.risk_window,
            )
            results.append(ep_result)

            status = "OK" if ep_result["success"] else "FAIL"
            extra = ""
            if args.mode == "resample" and ep_result["resamples"] > 0:
                extra = f"  resamples={ep_result['resamples']}"
            elif args.mode == "restart" and ep_result["restarts_used"] > 0:
                extra = f"  restarts={ep_result['restarts_used']}"
            tqdm.write(f"  ep={ep_idx:3d} seed={seed:3d} {status:4s}{extra}")

        except Exception as e:
            tqdm.write(f"  ep={ep_idx:3d} seed={seed:3d} ERROR: {e}")

    if hasattr(env, "close"):
        env.close()
    h5_file.close()

    # Compute summary
    n_success = sum(1 for r in results if r["success"])
    n_total = len(results)
    success_rate = n_success / n_total if n_total > 0 else 0

    total_restarts = sum(r["restarts_used"] for r in results)
    total_resamples = sum(r["resamples"] for r in results)

    summary = {
        "mode": args.mode,
        "task": args.task,
        "perturb": args.perturb,
        "perturb_start_step": args.perturb_start_step,
        "n_episodes": n_total,
        "n_success": n_success,
        "success_rate": success_rate,
        "risk_threshold": args.risk_threshold,
        "results": results,
    }

    if args.mode == "resample":
        summary["K"] = args.K
        summary["total_resamples"] = total_resamples
        summary["mean_resamples_per_episode"] = total_resamples / max(n_total, 1)
    elif args.mode == "restart":
        summary["max_restarts"] = args.max_restarts
        summary["risk_window"] = args.risk_window
        summary["total_restarts"] = total_restarts
        summary["mean_restarts_per_episode"] = total_restarts / max(n_total, 1)
        summary["episodes_with_restart"] = sum(
            1 for r in results if r["restarts_used"] > 0)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"INTERVENTION RESULTS — {args.mode}")
    print(f"{'=' * 70}")
    print(f"  Success rate:    {success_rate:.1%} ({n_success}/{n_total})")
    if args.mode == "resample":
        print(f"  K:               {args.K}")
        print(f"  Total resamples: {total_resamples}")
        print(f"  Mean per ep:     {total_resamples / max(n_total, 1):.1f}")
    elif args.mode == "restart":
        print(f"  Total restarts:  {total_restarts}")
        print(f"  Episodes w/ restart: {summary['episodes_with_restart']}")
    print(f"  Output:          {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

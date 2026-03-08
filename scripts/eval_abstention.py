"""
Phase 4: Evaluate abstention intervention with TAP-Score risk model.

When the risk model predicts P(fail) > threshold, the policy abstains
(skips one action chunk and re-observes). Compares success rate at
various thresholds vs baseline (no abstention).

Usage:
    python scripts/eval_abstention.py \
        --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
        --tap_checkpoint checkpoints_contrastive/robomimic_can_lowdim/risk_tap_best.pt \
        --task can --n_episodes 50 --output eval_results/abstention_can_zero80_n50.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

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
    parser = argparse.ArgumentParser(description="Abstention intervention evaluation")
    parser.add_argument("--dp_checkpoint", type=str, required=True)
    parser.add_argument("--tap_checkpoint", type=str, required=True,
                        help="Risk TAP-Score checkpoint (.pt)")
    parser.add_argument("--task", type=str, required=True, choices=["can", "lift"])
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--perturb", choices=PERTURB_TYPES, default="zero_object")
    parser.add_argument("--perturb_start_step", type=int, default=80)
    parser.add_argument("--perturb_noise_std", type=float, default=0.01)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        help="Risk thresholds to test (abstain if P(fail) > thresh)")
    parser.add_argument("--max_abstentions", type=int, default=5,
                        help="Max consecutive abstentions per episode")
    parser.add_argument("--kp", type=float, default=500.0)
    parser.add_argument("--max_env_steps", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip baseline (no-abstention) run")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Run only baseline (no thresholds)")
    parser.add_argument("--merge_from", nargs="*", default=None,
                        help="Merge results from multiple JSON files instead of running")
    return parser.parse_args()


def run_episode_with_abstention(
    env, dp_policy, risk_model, obs_history, tap_obs_history,
    n_obs_steps, n_action_steps, n_latency_steps, max_env_steps,
    abs_action, rotation_transformer, obs_keys, task_name,
    perturb_type, perturb_start_step, perturb_noise_std,
    frozen_object, perturb_rng, device,
    risk_threshold, max_abstentions,
    tap_obs_window, tap_action_chunk,
):
    """Run one episode with risk-based abstention.

    When risk > threshold, skip the action chunk (don't execute) and re-query
    the policy on the next loop. The env state doesn't change (abstention =
    "hold still"), but DP re-observes the same state and may propose different
    actions due to stochasticity.

    Returns dict with episode stats.
    """
    episode_return = 0.0
    episode_success = False
    env_steps = 0
    policy_step = 0
    done = False
    n_abstentions = 0
    consecutive_abstentions = 0
    scores = []

    while not done and env_steps < max_env_steps:
        obs_tensor = build_obs_tensor(obs_history, n_obs_steps, device=device)

        with torch.no_grad():
            action_dict = dp_policy.predict_action(obs_tensor)

        raw_chunk = action_dict["action"][0].detach().cpu().numpy()

        if abs_action and rotation_transformer is not None:
            full_chunk = undo_transform_action(raw_chunk, rotation_transformer)
        else:
            full_chunk = raw_chunk

        # Score with risk model
        should_abstain = False
        risk_prob = 0.0
        if (risk_threshold is not None
                and len(tap_obs_history) >= tap_obs_window
                and env_steps >= perturb_start_step):
            obs_window = stack_history(tap_obs_history, tap_obs_window)
            if full_chunk.shape[0] >= tap_action_chunk:
                tap_action = full_chunk[:tap_action_chunk].astype(np.float32)
            else:
                pad_len = tap_action_chunk - full_chunk.shape[0]
                pad = np.tile(full_chunk[-1:], (pad_len, 1))
                tap_action = np.concatenate([full_chunk, pad], axis=0).astype(np.float32)

            obs_t = torch.from_numpy(obs_window).unsqueeze(0).to(device, dtype=torch.float32)
            act_t = torch.from_numpy(tap_action).unsqueeze(0).to(device, dtype=torch.float32)
            with torch.no_grad():
                risk_prob = risk_model.predict_risk(obs_t, act_t).item()

            scores.append({
                "risk": risk_prob,
                "env_step": env_steps,
                "policy_step": policy_step,
            })

            if risk_prob > risk_threshold and consecutive_abstentions < max_abstentions:
                should_abstain = True
                n_abstentions += 1
                consecutive_abstentions += 1

        if should_abstain:
            # Don't execute — just advance policy step counter.
            # Env state stays the same. DP will re-observe on next iteration.
            policy_step += 1
            continue

        consecutive_abstentions = 0

        # Execute action in env
        exec_chunk = trim_latency(raw_chunk, n_latency_steps)
        r, done, succ, env_steps = step_chunk(
            env, obs_history, exec_chunk, env_steps, max_env_steps,
            abs_action, rotation_transformer,
            obs_keys=obs_keys, task_name=task_name,
            perturb_type=perturb_type, perturb_start_step=perturb_start_step,
            perturb_rng=perturb_rng, perturb_noise_std=perturb_noise_std,
            frozen_object=frozen_object,
        )
        episode_return += r
        episode_success = episode_success or succ

        # Update TAP obs history
        if "obs" in obs_history and len(obs_history["obs"]) > len(tap_obs_history):
            for i in range(len(tap_obs_history), len(obs_history["obs"])):
                tap_obs_history.append(obs_history["obs"][i].copy())

        policy_step += 1

    if episode_return >= 1.0:
        episode_success = True

    return {
        "success": episode_success,
        "return": float(episode_return),
        "n_abstentions": n_abstentions,
        "env_steps": env_steps,
        "policy_steps": policy_step,
        "scores": scores,
    }


def merge_results(files, output):
    """Merge multiple per-threshold JSON result files into one."""
    merged = {}
    meta = {}
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        if not meta:
            meta = {k: v for k, v in data.items() if k != "results"}
        merged.update(data.get("results", {}))

    # Compute improvement over baseline
    baseline = merged.get("baseline", {})
    baseline_rate = baseline.get("success_rate", 0)
    best_label, best_improvement = None, 0
    for label, d in merged.items():
        if label == "baseline":
            continue
        imp = d["success_rate"] - baseline_rate
        if imp > best_improvement:
            best_improvement = imp
            best_label = label

    meta["results"] = merged
    meta["baseline_success_rate"] = baseline_rate
    meta["best_threshold"] = best_label
    meta["best_improvement"] = best_improvement

    with open(output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Merged {len(files)} files → {output}")
    print(f"Baseline: {baseline_rate:.1%}, Best: {best_label} (+{best_improvement:.1%})")
    for label, d in merged.items():
        print(f"  {label:>12}: {d['n_success']}/{d['n_episodes']} ({d['success_rate']:.1%}), "
              f"abstentions={d['mean_abstentions']:.1f}")


def main():
    args = parse_args()

    if args.merge_from:
        merge_results(args.merge_from, args.output)
        return

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

    print("=" * 70)
    print("Abstention Intervention Evaluation")
    print("=" * 70)

    # Load DP
    dp_policy, cfg = load_diffusion_policy(args.dp_checkpoint, device=device)
    task_cfg = cfg.task
    task_name = str(cfg_get(task_cfg, "task_name", "unknown_task"))
    dataset_path = resolve_existing_path(str(cfg_get(task_cfg, "dataset_path")))
    abs_action = bool(cfg_get(task_cfg, "abs_action", False))
    rotation_transformer = (
        RotationTransformer("axis_angle", "rotation_6d") if abs_action else None
    )
    n_obs_steps = int(cfg_get(task_cfg.env_runner, "n_obs_steps"))
    n_action_steps = int(cfg_get(task_cfg.env_runner, "n_action_steps"))
    n_latency_steps = int(cfg_get(task_cfg.env_runner, "n_latency_steps", 0))
    cfg_max_steps = int(cfg_get(task_cfg.env_runner, "max_steps", 400))
    max_env_steps = args.max_env_steps if args.max_env_steps is not None else cfg_max_steps

    print(f"Task: {task_name}, max_steps={max_env_steps}, abs_action={abs_action}")

    # Create env
    env_factory, env_info = make_robomimic_env_factory(
        cfg, dataset_path=dataset_path, kp=args.kp,
    )
    obs_keys = env_info.get("obs_keys", None)

    # Load demo states
    import h5py
    h5_file = h5py.File(str(dataset_path), "r")
    demo_keys = sorted(h5_file["data"].keys())

    def sample_init_state(seed):
        rng = np.random.RandomState(seed)
        demo = demo_keys[int(rng.randint(0, len(demo_keys)))]
        return np.array(h5_file["data"][demo]["states"][0], copy=True)

    # Load risk model
    print(f"Loading risk model from {args.tap_checkpoint} ...")
    risk_ckpt = torch.load(args.tap_checkpoint, map_location=device, weights_only=False)
    risk_config = risk_ckpt["config"]
    risk_model = RiskTAPScore(
        obs_dim=risk_config["obs_dim"],
        action_dim=risk_config["action_dim"],
        obs_window=risk_config.get("obs_window", 2),
        action_chunk=risk_config.get("action_chunk", 8),
        hidden_dim=risk_config.get("hidden_dim", 128),
    )
    risk_model.load_state_dict(risk_ckpt["model_state_dict"])
    risk_model.to(device)
    risk_model.eval()

    tap_obs_window = risk_config.get("obs_window", 2)
    tap_action_chunk = risk_config.get("action_chunk", 8)
    print(f"Risk model: obs_window={tap_obs_window}, action_chunk={tap_action_chunk}, "
          f"fail_horizon={risk_config.get('fail_horizon', '?')}")

    # Thresholds to test (None = baseline with no abstention)
    if args.baseline_only:
        all_thresholds = [None]
    elif args.no_baseline:
        all_thresholds = args.thresholds
    else:
        all_thresholds = [None] + args.thresholds
    seeds = list(range(args.seed_offset, args.seed_offset + args.n_episodes))

    env = env_factory()

    results_by_threshold = {}

    for thresh in all_thresholds:
        thresh_label = "baseline" if thresh is None else f"thresh={thresh:.2f}"
        print(f"\n{'=' * 50}")
        print(f"Running: {thresh_label}")
        print(f"{'=' * 50}")

        ep_results = []

        for seed in tqdm(seeds, desc=thresh_label):
            try:
                if hasattr(env, "seed"):
                    env.seed(seed)
                obs = env.reset()
                state0 = sample_init_state(seed)
                obs_dict = set_env_state(env, state0, obs_keys=obs_keys, task_name=task_name)

                frozen_object = None
                if args.perturb == "freeze_object" and obs_keys is not None and args.perturb_start_step == 0:
                    if "obs" in obs_dict and obs_dict["obs"].ndim == 1:
                        obj_slice = _find_object_slice(obs_keys, task_name)
                        if obj_slice is not None:
                            s, e = obj_slice
                            frozen_object = obs_dict["obs"][s:e].copy()

                perturb_rng = np.random.RandomState(seed + 9999)
                if args.perturb != "none" and obs_keys is not None and 0 >= args.perturb_start_step:
                    if "obs" in obs_dict and obs_dict["obs"].ndim == 1:
                        obs_dict["obs"] = _apply_obs_perturbation(
                            obs_dict["obs"], obs_keys, args.perturb, task_name,
                            perturb_rng, args.perturb_noise_std, frozen_object,
                        )

                obs_history = defaultdict(list)
                for key, val in obs_dict.items():
                    obs_history[key].append(val)

                tap_obs_history = []
                if "obs" in obs_dict:
                    tap_obs_history.append(obs_dict["obs"].copy())

                result = run_episode_with_abstention(
                    env, dp_policy, risk_model, obs_history, tap_obs_history,
                    n_obs_steps, n_action_steps, n_latency_steps, max_env_steps,
                    abs_action, rotation_transformer, obs_keys, task_name,
                    args.perturb, args.perturb_start_step, args.perturb_noise_std,
                    frozen_object, perturb_rng, device,
                    risk_threshold=thresh,
                    max_abstentions=args.max_abstentions,
                    tap_obs_window=tap_obs_window,
                    tap_action_chunk=tap_action_chunk,
                )
                result["seed"] = seed
                ep_results.append(result)

                status = "OK" if result["success"] else "FAIL"
                tqdm.write(f"  seed={seed:3d} {status:4s} "
                           f"abstain={result['n_abstentions']} "
                           f"steps={result['env_steps']}")

            except Exception as e:
                tqdm.write(f"  seed={seed:3d} ERROR: {e}")
                continue

        n_success = sum(1 for r in ep_results if r["success"])
        n_total = len(ep_results)
        mean_abstentions = np.mean([r["n_abstentions"] for r in ep_results]) if ep_results else 0
        success_rate = n_success / max(n_total, 1)

        print(f"\n  {thresh_label}: {n_success}/{n_total} success ({success_rate:.1%}), "
              f"mean abstentions={mean_abstentions:.1f}")

        results_by_threshold[thresh_label] = {
            "threshold": thresh,
            "n_episodes": n_total,
            "n_success": n_success,
            "success_rate": success_rate,
            "mean_abstentions": float(mean_abstentions),
            "episodes": [{k: v for k, v in r.items() if k != "scores"} for r in ep_results],
        }

    if hasattr(env, "close"):
        env.close()
    h5_file.close()

    # Summary table
    print("\n" + "=" * 70)
    print("ABSTENTION RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Perturbation: {args.perturb} (onset={args.perturb_start_step})")
    print(f"  Max abstentions per episode: {args.max_abstentions}")
    print(f"\n  {'Threshold':>12}  {'Success':>8}  {'Rate':>7}  {'Abstentions':>12}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*7}  {'-'*12}")
    for label, data in results_by_threshold.items():
        print(f"  {label:>12}  {data['n_success']:>3}/{data['n_episodes']:<4}  "
              f"{data['success_rate']:>6.1%}  {data['mean_abstentions']:>11.1f}")

    # Compute improvement over baseline
    baseline = results_by_threshold.get("baseline", {})
    baseline_rate = baseline.get("success_rate", 0)
    best_label = None
    best_improvement = 0
    for label, data in results_by_threshold.items():
        if label == "baseline":
            continue
        improvement = data["success_rate"] - baseline_rate
        if improvement > best_improvement:
            best_improvement = improvement
            best_label = label

    if best_label:
        print(f"\n  Best improvement: {best_label} → +{best_improvement:.1%} over baseline")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "perturb": args.perturb,
            "perturb_start_step": args.perturb_start_step,
            "max_abstentions": args.max_abstentions,
            "thresholds_tested": args.thresholds,
            "results": results_by_threshold,
            "baseline_success_rate": baseline_rate,
            "best_threshold": best_label,
            "best_improvement": best_improvement,
        }, f, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()

"""
TAP-Score detection evaluation on robomimic Can.

Runs K=1 DP rollouts under observation perturbation (e.g. zero_object onset=80),
scores each policy step with TAP-Score, and computes AUROC + abstention metrics.

Usage:
    python scripts/eval_tap_detection.py \
        --dp_checkpoint data/robomimic/checkpoints/can_ph_diffusion_policy_cnn.ckpt \
        --n_episodes 50 --output eval_results/detection_can_zero80_n50.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

# ── Imports from TAP-Score evaluation ────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval_tap_final import load_model, score_action_logmargin, build_negative_pool
from tap.data_loaders import load_episodes
from tap.benchmarks import get_benchmark_config, get_data_path

# ── Imports from headroom audit (env setup, DP loading, obs utils) ──
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
    PERTURB_TYPES,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="TAP-Score detection evaluation on robomimic"
    )
    parser.add_argument("--dp_checkpoint", type=str, required=True,
                        help="Diffusion-policy checkpoint (.ckpt)")
    parser.add_argument("--tap_checkpoint", type=str,
                        default="checkpoints_contrastive/robomimic_can_lowdim/contrastive_tap_best.pt",
                        help="TAP-Score checkpoint")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--perturb", choices=PERTURB_TYPES, default="zero_object",
                        help="Observation perturbation type")
    parser.add_argument("--perturb_start_step", type=int, default=80,
                        help="Env step at which perturbation begins")
    parser.add_argument("--perturb_noise_std", type=float, default=0.01)
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path (e.g. eval_results/detection_can_zero80_n50.csv)")
    parser.add_argument("--kp", type=float, default=500.0,
                        help="OSC controller proportional gain")
    parser.add_argument("--m_negatives", type=int, default=15,
                        help="Number of negatives for TAP-Score log-margin scoring")
    parser.add_argument("--max_env_steps", type=int, default=None,
                        help="Override episode horizon (default: from DP config)")
    parser.add_argument("--dp_neg_pool", type=str, default=None,
                        help="NPZ cache of DP proposals to use as negative pool "
                             "(instead of expert actions). Ensures train/test "
                             "distribution match for DP-neg trained models.")
    parser.add_argument("--risk_model", action="store_true",
                        help="Use RiskTAPScore instead of contrastive model "
                             "(no negative pool needed)")
    parser.add_argument("--gate_threshold", type=float, default=None,
                        help="Risk threshold for online gating (0-1). "
                             "If per-step risk_prob > threshold, stop episode "
                             "and mark as abstained. Produces success-vs-coverage "
                             "metrics alongside detection metrics.")
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    # Make progress visible when stdout is redirected.
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
    print("TAP-Score Detection Evaluation")
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
    print(f"n_latency_steps: {n_latency_steps}")
    print(f"max_env_steps:   {max_env_steps}")
    print(f"abs_action:      {abs_action}")

    # ── Create env ───────────────────────────────────────────────────
    env_factory, env_info = make_robomimic_env_factory(
        cfg, dataset_path=dataset_path, kp=args.kp,
    )
    obs_keys = env_info.get("obs_keys", None)
    print(f"Env mode:        {env_info['mode']}")
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

    # ── Load TAP-Score model ─────────────────────────────────────────
    print(f"\nLoading TAP-Score from {args.tap_checkpoint} ...")
    use_risk_model = args.risk_model

    if use_risk_model:
        from tap.risk import RiskTAPScore
        risk_ckpt = torch.load(args.tap_checkpoint, map_location=device, weights_only=False)
        risk_config = risk_ckpt["config"]
        risk_obs_only = risk_config.get("obs_only", False)
        tap_model = RiskTAPScore(
            obs_dim=risk_config["obs_dim"],
            action_dim=risk_config["action_dim"],
            obs_window=risk_config.get("obs_window", 2),
            action_chunk=risk_config.get("action_chunk", 8),
            hidden_dim=risk_config.get("hidden_dim", 128),
            obs_only=risk_obs_only,
        )
        tap_model.load_state_dict(risk_ckpt["model_state_dict"])
        tap_model.to(device)
        tap_model.eval()
        tap_config = risk_config
        tap_action_chunk = risk_config.get("action_chunk", 8)
        tap_obs_window = risk_config.get("obs_window", 2)
        print(f"  Risk model loaded (fail_horizon={risk_config.get('fail_horizon', '?')}"
              f", obs_only={risk_obs_only})")
    else:
        tap_model, tap_config = load_model(args.tap_checkpoint, device)
        tap_action_chunk = tap_config.get("action_chunk", 16)
        tap_obs_window = tap_config.get("obs_window", 2)

    print(f"TAP action_chunk: {tap_action_chunk}")
    print(f"TAP obs_window:   {tap_obs_window}")
    print(f"Model type:       {'risk' if use_risk_model else 'contrastive'}")

    # ── Build negative pool ─────────────────────────────────────────
    if use_risk_model:
        # Risk model doesn't need a negative pool
        negative_pool = None
    elif args.dp_neg_pool is not None:
        # Use DP proposals as negative pool (matches training distribution).
        print(f"\nLoading DP-proposal negative pool from {args.dp_neg_pool} ...")
        cache = np.load(args.dp_neg_pool)
        dp_negs = cache['dp_negs']  # (N, K, H, action_dim)
        N_cache, K_cache = dp_negs.shape[:2]
        # Flatten to (N*K, H, action_dim), then subsample 2000.
        flat_negs = dp_negs.reshape(-1, dp_negs.shape[2], dp_negs.shape[3])
        pool_idx = np.random.choice(len(flat_negs), min(2000, len(flat_negs)), replace=False)
        negative_pool = flat_negs[pool_idx].astype(np.float32)
        # Truncate/pad to TAP action_chunk if needed.
        if negative_pool.shape[1] != tap_action_chunk:
            if negative_pool.shape[1] > tap_action_chunk:
                negative_pool = negative_pool[:, :tap_action_chunk]
            else:
                pad = np.tile(negative_pool[:, -1:], (1, tap_action_chunk - negative_pool.shape[1], 1))
                negative_pool = np.concatenate([negative_pool, pad], axis=1)
        print(f"Negative pool:   {negative_pool.shape} (from DP cache)")
    else:
        # Determine benchmark name from task_name
        benchmark_name = f"robomimic_{task_name.lower()}_lowdim"
        print(f"\nLoading expert episodes for negative pool (benchmark={benchmark_name}) ...")
        expert_data_path = str(get_data_path(benchmark_name))
        episodes = load_episodes(expert_data_path, benchmark_name)
        negative_pool = build_negative_pool(episodes, tap_action_chunk, n_samples=2000)
        print(f"Negative pool:   {negative_pool.shape}")

    # ── Perturbation config ──────────────────────────────────────────
    print(f"\nPerturbation:    {args.perturb}")
    print(f"Perturb onset:   {args.perturb_start_step}")
    print(f"M negatives:     {args.m_negatives}")
    print(f"N episodes:      {args.n_episodes}")
    print(f"Device:          {device}")
    print("=" * 70)

    # ── Rollout loop ─────────────────────────────────────────────────
    all_rows: List[Dict[str, Any]] = []  # CSV rows
    episode_results: List[Dict[str, Any]] = []

    seeds = list(range(args.seed_offset, args.seed_offset + args.n_episodes))

    # Create env ONCE and reuse via reset_to (avoids MuJoCo memory leak).
    env = env_factory()

    # Checkpoint path for crash recovery.
    ckpt_path = Path(args.output).with_suffix(".ckpt.json")

    # Resume from checkpoint if it exists.
    completed_seeds: set = set()
    if ckpt_path.exists():
        try:
            with open(ckpt_path) as f:
                ckpt = json.load(f)
            all_rows = ckpt.get("all_rows", [])
            episode_results = ckpt.get("episode_results", [])
            completed_seeds = {e["seed"] for e in episode_results}
            print(f"Resumed from checkpoint: {len(completed_seeds)} episodes done")
        except Exception as e:
            print(f"Warning: could not resume checkpoint: {e}")

    from scripts.robomimic_headroom_audit import set_env_state

    for ep_idx, seed in enumerate(tqdm(seeds, desc="Episodes")):
        if seed in completed_seeds:
            continue

        try:
            if hasattr(env, "seed"):
                env.seed(seed)
            obs = env.reset()

            # Reset to demo initial state
            state0 = sample_init_state(seed)
            obs_dict = set_env_state(env, state0, obs_keys=obs_keys, task_name=task_name)

            # Track frozen object for freeze_object perturbation
            frozen_object = None
            if args.perturb == "freeze_object" and obs_keys is not None and args.perturb_start_step == 0:
                if "obs" in obs_dict and obs_dict["obs"].ndim == 1:
                    obj_slice = _find_object_slice(obs_keys, task_name)
                    if obj_slice is not None:
                        s, e = obj_slice
                        frozen_object = obs_dict["obs"][s:e].copy()

            # Apply perturbation to initial obs if onset == 0
            if args.perturb != "none" and obs_keys is not None and 0 >= args.perturb_start_step:
                if "obs" in obs_dict and obs_dict["obs"].ndim == 1:
                    perturb_rng = np.random.RandomState(seed + 9999)
                    obs_dict["obs"] = _apply_obs_perturbation(
                        obs_dict["obs"], obs_keys, args.perturb, task_name,
                        perturb_rng, args.perturb_noise_std, frozen_object,
                    )

            obs_history: Dict[str, List[np.ndarray]] = defaultdict(list)
            for key, val in obs_dict.items():
                obs_history[key].append(val)

            # Also keep a separate flat-obs history for TAP-Score scoring
            tap_obs_history: List[np.ndarray] = []
            if "obs" in obs_dict:
                tap_obs_history.append(obs_dict["obs"].copy())

            episode_return = 0.0
            episode_success = False
            env_steps = 0
            policy_step = 0
            done = False
            gated = False
            gated_step = None
            scores_this_episode: List[Dict[str, Any]] = []
            perturb_rng = np.random.RandomState(seed + 9999)

            while not done and env_steps < max_env_steps:
                # Build obs tensor for DP policy
                obs_tensor = build_obs_tensor(obs_history, n_obs_steps, device=device)

                # Get DP prediction — full horizon (not just action steps)
                with torch.no_grad():
                    action_dict = dp_policy.predict_action(obs_tensor)

                # Full predicted chunk from DP (before trimming).
                # DP outputs rotation_6d format (10D) for abs_action checkpoints.
                raw_chunk = action_dict["action"][0].detach().cpu().numpy()  # (H, 10)

                # Convert to 7D (axis_angle) to match TAP-Score training data.
                if abs_action and rotation_transformer is not None:
                    full_chunk = undo_transform_action(raw_chunk, rotation_transformer)  # (H, 7)
                else:
                    full_chunk = raw_chunk

                # Execution chunk: trim latency, then take first n_action_steps
                # Note: step_chunk does its own undo_transform_action, so pass raw.
                exec_chunk = trim_latency(raw_chunk, n_latency_steps)

                # ── TAP-Score scoring ────────────────────────────────
                # Score the full chunk if we're past perturbation onset
                # and we have enough obs history
                if env_steps >= args.perturb_start_step and len(tap_obs_history) >= tap_obs_window:
                    # Build obs window for TAP: last tap_obs_window frames, shape (W, 23)
                    obs_window = stack_history(tap_obs_history, tap_obs_window)  # (W, obs_dim)

                    # Get the action chunk for TAP scoring.
                    # full_chunk is (H, 7). TAP expects (action_chunk, 7).
                    # If full_chunk has >= tap_action_chunk steps, take first tap_action_chunk.
                    # If shorter, pad by repeating last action.
                    if full_chunk.shape[0] >= tap_action_chunk:
                        tap_action = full_chunk[:tap_action_chunk].astype(np.float32)
                    else:
                        pad_len = tap_action_chunk - full_chunk.shape[0]
                        pad = np.tile(full_chunk[-1:], (pad_len, 1))
                        tap_action = np.concatenate([full_chunk, pad], axis=0).astype(np.float32)

                    if use_risk_model:
                        obs_t = torch.from_numpy(obs_window).unsqueeze(0).to(device, dtype=torch.float32)
                        act_t = (None if getattr(tap_model, 'obs_only', False) else
                                 torch.from_numpy(tap_action).unsqueeze(0).to(device, dtype=torch.float32))
                        with torch.no_grad():
                            risk_prob = tap_model.predict_risk(obs_t, act_t)
                        score = 1.0 - risk_prob.item()  # higher = safer
                    else:
                        score = score_action_logmargin(
                            tap_model, obs_window, tap_action,
                            negative_pool, device, args.m_negatives,
                        )

                    # Action magnitude baseline: mean L2 norm of the action chunk
                    action_mag = float(np.mean(np.linalg.norm(tap_action, axis=-1)))

                    scores_this_episode.append({
                        "score": score,
                        "action_mag": action_mag,
                        "env_step": env_steps,
                        "policy_step": policy_step,
                        "episode_id": ep_idx,
                    })

                    # ── Online gating: stop if risk exceeds threshold ─
                    if args.gate_threshold is not None:
                        # For risk model: score = 1 - risk_prob, so
                        # risk_prob > threshold ↔ score < 1 - threshold
                        gate_score_threshold = 1.0 - args.gate_threshold
                        if score < gate_score_threshold:
                            gated = True
                            gated_step = env_steps
                            break

                # ── Execute action in env ────────────────────────────
                r, done, succ, env_steps = step_chunk(
                    env,
                    obs_history,
                    exec_chunk,
                    env_steps,
                    max_env_steps,
                    abs_action,
                    rotation_transformer,
                    obs_keys=obs_keys,
                    task_name=task_name,
                    perturb_type=args.perturb,
                    perturb_start_step=args.perturb_start_step,
                    perturb_rng=perturb_rng,
                    perturb_noise_std=args.perturb_noise_std,
                    frozen_object=frozen_object,
                )
                episode_return += r
                episode_success = episode_success or succ

                # Update TAP obs history from the latest obs in obs_history
                if "obs" in obs_history and len(obs_history["obs"]) > len(tap_obs_history):
                    # Add all new obs frames
                    for i in range(len(tap_obs_history), len(obs_history["obs"])):
                        tap_obs_history.append(obs_history["obs"][i].copy())

                policy_step += 1

            if episode_return >= 1.0:
                episode_success = True

            # Backfill label and add to CSV rows
            label = 1 if episode_success else 0
            for row in scores_this_episode:
                row["label"] = label
                all_rows.append(row)

            sc_vals = [r["score"] for r in scores_this_episode] if scores_this_episode else []
            # Post-onset scores (only decisions after perturbation starts)
            post_onset = [r["score"] for r in scores_this_episode
                          if r["env_step"] >= args.perturb_start_step] if scores_this_episode else []
            episode_results.append({
                "episode_id": ep_idx,
                "seed": seed,
                "success": episode_success,
                "episode_success": episode_success,  # alias for cross-script compat
                "return": episode_return,
                "n_scores": len(scores_this_episode),
                "min_score": min(sc_vals) if sc_vals else None,
                "mean_score": np.mean(sc_vals) if sc_vals else None,
                "p10_score": float(np.percentile(sc_vals, 10)) if sc_vals else None,
                "mean_post_onset_score": float(np.mean(post_onset)) if post_onset else None,
                "min_post_onset_score": float(min(post_onset)) if post_onset else None,
                "mean_action_mag": np.mean([r["action_mag"] for r in scores_this_episode]) if scores_this_episode else None,
                "gated": gated,
                "gated_step": gated_step,
            })

            status = "GATE" if gated else ("OK" if episode_success else "FAIL")
            n_scores = len(scores_this_episode)
            min_sc = episode_results[-1]["min_score"]
            gate_info = f"  gated@{gated_step}" if gated else ""
            tqdm.write(f"  ep={ep_idx:3d} seed={seed:3d} {status:4s}  "
                        f"scores={n_scores}  min={min_sc:.3f}{gate_info}" if min_sc is not None else
                        f"  ep={ep_idx:3d} seed={seed:3d} {status:4s}  scores=0")

        except Exception as e:
            tqdm.write(f"  ep={ep_idx:3d} seed={seed:3d} ERROR: {e}")
            continue

        # Checkpoint every 5 episodes.
        if (ep_idx + 1) % 5 == 0 or ep_idx == len(seeds) - 1:
            try:
                with open(ckpt_path, "w") as f:
                    json.dump({
                        "all_rows": all_rows,
                        "episode_results": episode_results,
                    }, f)
            except Exception:
                pass

    # Close env after all episodes.
    if hasattr(env, "close"):
        env.close()

    # Clean up checkpoint after successful completion.
    if ckpt_path.exists():
        ckpt_path.unlink()

    # ── Write CSV ────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["score", "action_mag", "label", "env_step", "policy_step", "episode_id"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV written: {output_path}  ({len(all_rows)} rows)")

    # ── Compute metrics ──────────────────────────────────────────────
    n_success = sum(1 for e in episode_results if e["success"])
    n_fail = sum(1 for e in episode_results if not e["success"])
    print(f"\nEpisodes: {len(episode_results)} total, {n_success} success, {n_fail} fail")

    if n_success == 0 or n_fail == 0:
        print("WARNING: Need both success and failure episodes for AUROC. Skipping metrics.")
        summary = {
            "n_episodes": len(episode_results),
            "n_success": n_success,
            "n_fail": n_fail,
            "auroc": None,
            "abstention_curve": None,
            "lead_time": None,
            "perturb": args.perturb,
            "perturb_start_step": args.perturb_start_step,
        }
    else:
        from sklearn.metrics import roc_auc_score, average_precision_score

        # Episode-level AUROC using multiple aggregations
        ep_labels = []
        ep_min_scores = []
        ep_mean_scores = []
        ep_p10_scores = []
        ep_post_onset_scores = []
        ep_min_post_onset_scores = []
        for e in episode_results:
            if e["min_score"] is not None:
                ep_labels.append(1 if e["success"] else 0)
                ep_min_scores.append(e["min_score"])
                ep_mean_scores.append(e["mean_score"])
                ep_p10_scores.append(e["p10_score"])
                ep_post_onset_scores.append(e.get("mean_post_onset_score", e["mean_score"]))
                ep_min_post_onset_scores.append(e.get("min_post_onset_score", e["min_score"]))

        # Higher score = more compatible = more likely success
        auroc_min = roc_auc_score(ep_labels, ep_min_scores)
        auroc_mean = roc_auc_score(ep_labels, ep_mean_scores)
        auroc_p10 = roc_auc_score(ep_labels, ep_p10_scores)
        auroc_post_onset = roc_auc_score(ep_labels, ep_post_onset_scores)
        auroc_min_post_onset = roc_auc_score(ep_labels, ep_min_post_onset_scores)

        # AUPRC for each
        ep_fail_labels = [1 - l for l in ep_labels]
        auprc_min = average_precision_score(ep_fail_labels, [-s for s in ep_min_scores])
        auprc_mean = average_precision_score(ep_fail_labels, [-s for s in ep_mean_scores])
        auprc_p10 = average_precision_score(ep_fail_labels, [-s for s in ep_p10_scores])
        auprc_min_post_onset = average_precision_score(ep_fail_labels, [-s for s in ep_min_post_onset_scores])

        # Pick best aggregation for downstream analysis
        auroc_table = {
            "min": auroc_min, "mean": auroc_mean,
            "p10": auroc_p10, "post_onset": auroc_post_onset,
            "min_post_onset": auroc_min_post_onset,
        }
        best_agg = max(auroc_table, key=auroc_table.get)
        best_auroc = auroc_table[best_agg]
        best_scores = {"min": ep_min_scores, "mean": ep_mean_scores,
                       "p10": ep_p10_scores, "post_onset": ep_post_onset_scores,
                       "min_post_onset": ep_min_post_onset_scores}[best_agg]

        print(f"\nEpisode-level AUROC by aggregation:")
        for agg_name, agg_auroc in sorted(auroc_table.items(), key=lambda x: -x[1]):
            marker = " <-- best" if agg_name == best_agg else ""
            print(f"  {agg_name:>12}: {agg_auroc:.3f}{marker}")
        print(f"AUPRC (min): {auprc_min:.3f}, (mean): {auprc_mean:.3f}, (p10): {auprc_p10:.3f}, (min_post_onset): {auprc_min_post_onset:.3f}")

        # Use best aggregation as primary AUROC
        auroc = best_auroc

        # ── Magnitude baseline AUROC ──────────────────────────────────
        ep_mag_scores = [e["mean_action_mag"] for e in episode_results if e["mean_action_mag"] is not None]
        ep_labels_mag = [1 if e["success"] else 0 for e in episode_results if e["mean_action_mag"] is not None]
        if len(set(ep_labels_mag)) == 2:
            mag_auroc_raw = roc_auc_score(ep_labels_mag, ep_mag_scores)
            mag_auroc = max(mag_auroc_raw, 1 - mag_auroc_raw)
            print(f"Action magnitude baseline AUROC:  {mag_auroc:.3f}")
            print(f"TAP advantage over magnitude:    +{auroc - mag_auroc:.3f}")
        else:
            mag_auroc = None

        # Pre-compute success scores for threshold calculations
        success_best_scores = [s for s, l in zip(best_scores, ep_labels) if l == 1]

        # ── Bootstrap confidence intervals ─────────────────────────────
        n_boot = 1000
        boot_rng = np.random.RandomState(42)
        n_ep = len(ep_labels)
        ep_labels_arr = np.array(ep_labels)
        ep_min_arr = np.array(best_scores)  # use best aggregation

        boot_aurocs = []
        boot_succ_at_20 = []
        boot_lead_medians = []

        for _ in range(n_boot):
            idx = boot_rng.choice(n_ep, n_ep, replace=True)
            b_labels = ep_labels_arr[idx]
            b_scores = ep_min_arr[idx]

            # Need both classes for AUROC
            if len(set(b_labels)) < 2:
                continue

            b_auroc = roc_auc_score(b_labels, b_scores)
            boot_aurocs.append(b_auroc)

            # Success at 20% coverage
            paired = sorted(zip(b_scores, b_labels), key=lambda x: x[0])
            n_keep = max(1, int(len(paired) * 0.2))
            kept = paired[len(paired) - n_keep:]
            b_succ = sum(1 for _, l in kept if l == 1) / len(kept)
            boot_succ_at_20.append(b_succ)

        # Lead time bootstrap (resample failed episodes)
        fail_ep_ids = [e["episode_id"] for e in episode_results
                       if not e["success"] and e["min_score"] is not None]
        if success_best_scores:
            lt_threshold = np.median(success_best_scores)
            # Pre-compute lead time per failed episode
            fail_lead_map = {}
            for eid in fail_ep_ids:
                ep_rows = [r for r in all_rows if r["episode_id"] == eid]
                for row in ep_rows:
                    if row["score"] < lt_threshold:
                        fail_lead_map[eid] = max_env_steps - row["env_step"]
                        break
            all_fail_leads = [fail_lead_map[eid] for eid in fail_ep_ids if eid in fail_lead_map]
            if len(all_fail_leads) >= 2:
                for _ in range(n_boot):
                    idx = boot_rng.choice(len(all_fail_leads), len(all_fail_leads), replace=True)
                    boot_lead_medians.append(float(np.median([all_fail_leads[i] for i in idx])))

        boot_ci = {}
        if boot_aurocs:
            ci_lo, ci_hi = np.percentile(boot_aurocs, [2.5, 97.5])
            boot_ci["auroc_min"] = {"mean": float(np.mean(boot_aurocs)),
                                     "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}
            print(f"\n  Bootstrap 95% CI (n={n_boot}):")
            print(f"    AUROC (min):       {np.mean(boot_aurocs):.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        if boot_succ_at_20:
            ci_lo, ci_hi = np.percentile(boot_succ_at_20, [2.5, 97.5])
            boot_ci["success_at_20pct"] = {"mean": float(np.mean(boot_succ_at_20)),
                                            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}
            print(f"    Success@20%:       {np.mean(boot_succ_at_20):.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        if boot_lead_medians:
            ci_lo, ci_hi = np.percentile(boot_lead_medians, [2.5, 97.5])
            boot_ci["lead_time_median"] = {"mean": float(np.mean(boot_lead_medians)),
                                            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}
            print(f"    Lead time (med):   {np.mean(boot_lead_medians):.0f} [{ci_lo:.0f}, {ci_hi:.0f}]")

        # ── Abstention curve (episode-level) ─────────────────────────
        # Sort episodes by best aggregation score (ascending = worst first)
        scored_episodes = list(zip(best_scores, [l == 1 for l in ep_labels]))
        scored_episodes.sort(key=lambda x: x[0])

        n_scored = len(scored_episodes)
        abstention_curve = []
        coverage_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        print("\n  Abstention Curve (keep top-scoring episodes):")
        print(f"  {'Coverage':>10}  {'N_kept':>6}  {'SuccRate':>8}")
        for cov in coverage_thresholds:
            n_keep = max(1, int(n_scored * cov))
            # Keep the top n_keep episodes by score (highest scores)
            kept = scored_episodes[n_scored - n_keep:]
            succ_rate = sum(1 for _, s in kept if s) / len(kept)
            abstention_curve.append({
                "coverage": cov,
                "n_kept": n_keep,
                "success_rate": succ_rate,
            })
            print(f"  {cov:>9.0%}  {n_keep:>6}  {succ_rate:>7.1%}")

        # ── Lead time analysis ───────────────────────────────────────
        # For failed episodes: find first step where score drops below
        # the median success score threshold (using best aggregation)
        if success_best_scores:
            threshold = np.median(success_best_scores)
        else:
            threshold = 0.0

        lead_times = []
        for e in episode_results:
            if e["success"] or e["min_score"] is None:
                continue
            # Find first policy step where score < threshold
            ep_rows = [r for r in all_rows if r["episode_id"] == e["episode_id"]]
            for row in ep_rows:
                if row["score"] < threshold:
                    # Lead time = how many env steps before episode end
                    lead_env_steps = max_env_steps - row["env_step"]
                    lead_times.append(lead_env_steps)
                    break

        lead_time_stats = {}
        if lead_times:
            lead_time_stats = {
                "median": float(np.median(lead_times)),
                "q25": float(np.percentile(lead_times, 25)),
                "q75": float(np.percentile(lead_times, 75)),
                "mean": float(np.mean(lead_times)),
                "n_detected": len(lead_times),
                "n_fail_total": n_fail,
                "threshold_used": float(threshold),
            }
            print(f"\n  Lead time (env steps before episode end):")
            print(f"    Median: {lead_time_stats['median']:.0f}")
            print(f"    Q25-Q75: {lead_time_stats['q25']:.0f} - {lead_time_stats['q75']:.0f}")
            print(f"    Detected {len(lead_times)}/{n_fail} failed episodes")

        summary = {
            "n_episodes": len(episode_results),
            "n_success": n_success,
            "n_fail": n_fail,
            "best_aggregation": best_agg,
            "auroc_best": float(best_auroc),
            "auroc_min": float(auroc_min),
            "auroc_mean": float(auroc_mean),
            "auroc_p10": float(auroc_p10),
            "auroc_post_onset": float(auroc_post_onset),
            "auroc_min_post_onset": float(auroc_min_post_onset),
            "auprc_min": float(auprc_min),
            "auprc_mean": float(auprc_mean),
            "auprc_p10": float(auprc_p10),
            "auprc_min_post_onset": float(auprc_min_post_onset),
            "mag_baseline_auroc": float(mag_auroc) if mag_auroc is not None else None,
            "bootstrap_ci": boot_ci,
            "abstention_curve": abstention_curve,
            "lead_time": lead_time_stats,
            "perturb": args.perturb,
            "perturb_start_step": args.perturb_start_step,
            "m_negatives": args.m_negatives,
            "tap_checkpoint": args.tap_checkpoint,
            "dp_checkpoint": args.dp_checkpoint,
            "episode_results": [
                {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                 for k, v in e.items()}
                for e in episode_results
            ],
        }

    # ── Online gating summary ─────────────────────────────────────────
    if args.gate_threshold is not None:
        n_gated = sum(1 for e in episode_results if e.get("gated"))
        n_executed = len(episode_results) - n_gated
        n_exec_success = sum(1 for e in episode_results
                             if not e.get("gated") and e["success"])
        coverage = n_executed / len(episode_results) if episode_results else 0
        exec_sr = n_exec_success / n_executed if n_executed > 0 else 0
        baseline_sr = sum(1 for e in episode_results if e["success"]) / len(episode_results)

        summary["gating"] = {
            "threshold": args.gate_threshold,
            "n_gated": n_gated,
            "n_executed": n_executed,
            "coverage": coverage,
            "executed_success_rate": exec_sr,
            "baseline_success_rate": baseline_sr,
            "lift_pp": exec_sr - baseline_sr,
        }

        print(f"\n  Online Gating (threshold={args.gate_threshold:.2f}):")
        print(f"    Episodes gated:   {n_gated}/{len(episode_results)}")
        print(f"    Coverage:         {coverage:.1%}")
        print(f"    Success (exec):   {exec_sr:.1%}  (baseline: {baseline_sr:.1%})")
        print(f"    Lift:             +{exec_sr - baseline_sr:.1%}")

    # Write JSON summary alongside CSV
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON summary: {json_path}")

    # ── Final summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETECTION EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Perturbation:   {args.perturb} (onset={args.perturb_start_step})")
    print(f"  Episodes:       {n_success} success / {n_fail} fail")
    if summary.get("auroc_best") is not None:
        print(f"  Best agg:       {summary.get('best_aggregation', 'min')}")
        print(f"  AUROC (best):   {summary['auroc_best']:.3f}")
        print(f"  AUROC (min):    {summary['auroc_min']:.3f}")
        print(f"  AUROC (mean):   {summary['auroc_mean']:.3f}")
        print(f"  AUROC (p10):    {summary.get('auroc_p10', 0):.3f}")
        print(f"  AUROC (post-on):{summary.get('auroc_post_onset', 0):.3f}")
        print(f"  AUROC (min-po): {summary.get('auroc_min_post_onset', 0):.3f}")
        if summary.get("mag_baseline_auroc") is not None:
            print(f"  Mag baseline:   {summary['mag_baseline_auroc']:.3f}")
        if summary.get("bootstrap_ci"):
            ci = summary["bootstrap_ci"]
            if "auroc_min" in ci:
                a = ci["auroc_min"]
                print(f"  AUROC 95% CI:   [{a['ci_lo']:.3f}, {a['ci_hi']:.3f}]")
    print("=" * 70)

    h5_file.close()


if __name__ == "__main__":
    main()

"""
Robomimic oracle-headroom + diversity audit.

This script is the "fork in the road" check:
1) If oracle best-of-K has near-zero gain over K=1, ranking is not the bottleneck.
2) If oracle best-of-K has meaningful gain, reranking (e.g., TAP-Score) is worth pursuing.

At each decision point in an episode:
- sample K candidate action chunks from the same observation/history
- compute candidate spread (mean pairwise L2 over full chunk)
- roll out each candidate to episode end (true env return / success)
- compute oracle headroom (best - mean, best - candidate0)

Outputs a JSON report with:
- per-decision logs (optionally including full candidate chunks)
- per-episode summaries
- global pass/fail metrics against configurable thresholds
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Make bundled diffusion-policy importable without requiring external PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
DP_ROOT = REPO_ROOT / "baselines" / "diffusion_policy"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

try:
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
except ImportError as exc:
    raise ImportError(
        "robomimic is required for this script. Install robomimic and its MuJoCo/robosuite deps."
    ) from exc

from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from diffusion_policy.model.common.rotation_transformer import RotationTransformer


def cfg_get(cfg_obj: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg_obj, key):
        return getattr(cfg_obj, key)
    if isinstance(cfg_obj, dict) and key in cfg_obj:
        return cfg_obj[key]
    return default


def resolve_existing_path(path_value: str) -> Path:
    """Resolve config path values against common roots."""
    raw = Path(path_value).expanduser()
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                REPO_ROOT / raw,
                DP_ROOT / raw,
            ]
        )

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not resolve path '{path_value}'. Tried: {tried}")


def load_diffusion_policy(checkpoint_path: str, device: torch.device):
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        pickle_module=dill,
        weights_only=False,
    )
    cfg = payload["cfg"]
    target = getattr(cfg, "_target_", None)
    if target is None:
        ws_cfg = getattr(cfg, "workspace", None)
        if ws_cfg is not None:
            target = getattr(ws_cfg, "_target_", None)
    if target is None:
        raise ValueError("Could not find workspace _target_ in checkpoint cfg.")

    cls = hydra.utils.get_class(target)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()
    return policy, cfg


def infer_task_mode(task_cfg: Any) -> str:
    if hasattr(task_cfg, "shape_meta"):
        return "image"
    return "lowdim"


def infer_image_render_key(shape_meta: Dict[str, Any], env_runner_cfg: Any) -> str:
    render_key = cfg_get(env_runner_cfg, "render_obs_key", None)
    if render_key is not None:
        return str(render_key)

    for key in shape_meta["obs"].keys():
        if key.endswith("image"):
            return key
    raise ValueError("Could not infer image render key from shape_meta.")


def make_robomimic_env_factory(cfg: Any, dataset_path: Path):
    """Create a callable that builds fresh wrapped Robomimic envs."""
    task_cfg = cfg.task
    env_runner_cfg = task_cfg.env_runner
    mode = infer_task_mode(task_cfg)
    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))

    if mode == "lowdim":
        obs_keys = list(task_cfg.obs_keys)
        ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})

        def factory():
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=False,
                render_offscreen=False,
                use_image_obs=False,
            )
            wrapped = RobomimicLowdimWrapper(env=env, obs_keys=obs_keys)
            wrapped.reset()  # warm up internal state caches before reset_to
            return wrapped

        env_info = {"mode": "lowdim", "obs_keys": obs_keys}
        return factory, env_info

    shape_meta = OmegaConf.to_container(task_cfg.shape_meta, resolve=True)
    if not isinstance(shape_meta, dict):
        raise ValueError("shape_meta must resolve to a dictionary.")

    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(dict(modality_mapping))

    render_obs_key = infer_image_render_key(shape_meta, env_runner_cfg)

    def factory():
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=True,
        )
        # Keeps memory bounded in long audits.
        if hasattr(env, "env") and hasattr(env.env, "hard_reset"):
            env.env.hard_reset = False
        wrapped = RobomimicImageWrapper(
            env=env,
            shape_meta=shape_meta,
            render_obs_key=render_obs_key,
        )
        wrapped.reset()  # required before repeated reset_to calls
        return wrapped

    env_info = {
        "mode": "image",
        "shape_meta_obs_keys": list(shape_meta["obs"].keys()),
        "render_obs_key": render_obs_key,
    }
    return factory, env_info


def _ensure_float_obs(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype == np.uint8:
        out = out.astype(np.float32) / 255.0
    elif out.dtype != np.float32:
        out = out.astype(np.float32)

    # Robustness: convert HWC image to CHW if needed.
    if out.ndim == 3 and out.shape[-1] in (1, 3) and out.shape[0] not in (1, 3):
        out = np.transpose(out, (2, 0, 1))
    return out


def obs_to_dict(obs: Any) -> Dict[str, np.ndarray]:
    if isinstance(obs, dict):
        return {k: _ensure_float_obs(v) for k, v in obs.items()}
    return {"obs": _ensure_float_obs(obs)}


def stack_history(hist: List[np.ndarray], T: int) -> np.ndarray:
    if len(hist) < T:
        padded = [hist[0]] * (T - len(hist)) + list(hist)
    else:
        padded = list(hist[-T:])
    return np.stack(padded, axis=0)


def build_obs_tensor(
    obs_history: Dict[str, List[np.ndarray]],
    n_obs_steps: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    obs_tensor = {}
    for key, hist in obs_history.items():
        stacked = stack_history(hist, n_obs_steps)
        obs_tensor[key] = torch.from_numpy(stacked[None]).to(device=device, dtype=torch.float32)
    return obs_tensor


def clone_obs_history(obs_history: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    return {k: list(v) for k, v in obs_history.items()}


def get_env_state(env_wrapper) -> np.ndarray:
    state = env_wrapper.env.get_state()["states"]
    return np.array(state, copy=True)


def set_env_state(env_wrapper, state: np.ndarray) -> Dict[str, np.ndarray]:
    raw_obs = env_wrapper.env.reset_to({"states": np.array(state, copy=True)})
    try:
        obs = env_wrapper.get_observation(raw_obs)
    except TypeError:
        obs = env_wrapper.get_observation()
    return obs_to_dict(obs)


def mean_pairwise_l2(candidates: np.ndarray) -> float:
    """Mean pairwise L2 over full candidate chunks, candidates shape: (K, H, A)."""
    K = int(candidates.shape[0])
    if K < 2:
        return 0.0
    flat = candidates.reshape(K, -1).astype(np.float64)
    diff = flat[:, None, :] - flat[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(K, k=1)
    return float(dist[iu].mean())


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


def _any_true_like(value: Any) -> bool:
    if isinstance(value, dict):
        if "task" in value:
            return _any_true_like(value["task"])
        return any(_any_true_like(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_any_true_like(v) for v in value)

    arr = np.asarray(value)
    if arr.size == 0:
        return False
    if arr.dtype == np.bool_:
        return bool(arr.any())
    if np.issubdtype(arr.dtype, np.number):
        return bool((arr > 0).any())
    return bool(value)


def extract_success(info: Dict[str, Any]) -> bool:
    if not isinstance(info, dict):
        return False
    preferred_keys = ["is_success", "success", "task_success", "successes"]
    for key in preferred_keys:
        if key in info and _any_true_like(info[key]):
            return True
    for key, value in info.items():
        if "success" in str(key).lower() and _any_true_like(value):
            return True
    return False


def undo_transform_action(action: np.ndarray, rotation_transformer: RotationTransformer) -> np.ndarray:
    """Match diffusion-policy Robomimic runners for abs-action checkpoints."""
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        action = action.reshape(-1, 2, 10)

    d_rot = action.shape[-1] - 4
    pos = action[..., :3]
    rot = action[..., 3:3 + d_rot]
    gripper = action[..., [-1]]
    rot = rotation_transformer.inverse(rot)
    out = np.concatenate([pos, rot, gripper], axis=-1)

    if raw_shape[-1] == 20:
        out = out.reshape(*raw_shape[:-1], 14)
    return out


def trim_latency(action_chunk: np.ndarray, n_latency_steps: int) -> np.ndarray:
    if n_latency_steps <= 0:
        return action_chunk
    if action_chunk.shape[-2] <= n_latency_steps:
        raise ValueError(
            f"n_latency_steps={n_latency_steps} >= chunk horizon={action_chunk.shape[-2]}."
        )
    if action_chunk.ndim == 2:
        return action_chunk[n_latency_steps:, :]
    if action_chunk.ndim == 3:
        return action_chunk[:, n_latency_steps:, :]
    raise ValueError(f"Unsupported action chunk ndim={action_chunk.ndim}")


def maybe_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


@contextmanager
def preserve_torch_rng_state():
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def set_torch_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def step_chunk(
    env,
    obs_history: Dict[str, List[np.ndarray]],
    action_chunk: np.ndarray,
    env_steps: int,
    max_env_steps: int,
    abs_action: bool,
    rotation_transformer: RotationTransformer | None,
) -> Tuple[float, bool, bool, int]:
    if abs_action:
        if rotation_transformer is None:
            raise ValueError("rotation_transformer required when abs_action=True.")
        action_chunk = undo_transform_action(action_chunk, rotation_transformer)

    total_return = 0.0
    done = False
    success = False

    for act in action_chunk:
        obs, reward, step_done, info = env.step(act)
        obs_dict = obs_to_dict(obs)
        for key, val in obs_dict.items():
            obs_history[key].append(val)

        total_return += float(reward)
        success = success or extract_success(info)
        env_steps += 1
        if step_done or env_steps >= max_env_steps:
            done = True
            break

    return total_return, done, success, env_steps


def rollout_candidate_to_end(
    branch_env,
    dp_policy,
    saved_state: np.ndarray,
    saved_obs_history: Dict[str, List[np.ndarray]],
    candidate_chunk: np.ndarray,
    env_steps_at_decision: int,
    max_env_steps: int,
    n_obs_steps: int,
    n_latency_steps: int,
    abs_action: bool,
    rotation_transformer: RotationTransformer | None,
    device: torch.device,
    continuation_seed_base: int,
) -> Tuple[float, bool]:
    _ = set_env_state(branch_env, saved_state)
    obs_history = clone_obs_history(saved_obs_history)

    env_steps = int(env_steps_at_decision)
    total_return = 0.0
    success = False
    done = False

    # First chunk: the candidate being evaluated.
    r, done, succ, env_steps = step_chunk(
        branch_env,
        obs_history,
        candidate_chunk,
        env_steps,
        max_env_steps,
        abs_action,
        rotation_transformer,
    )
    total_return += r
    success = success or succ

    # Continue with K=1 policy until episode end for true return-to-go.
    continuation_idx = 0
    while not done and env_steps < max_env_steps:
        obs_tensor = build_obs_tensor(obs_history, n_obs_steps, device=device)
        set_torch_seed(int(continuation_seed_base + continuation_idx))
        with torch.no_grad():
            with maybe_autocast(device):
                action_dict = dp_policy.predict_action(obs_tensor)
        next_chunk = action_dict["action"][0].detach().cpu().numpy()
        next_chunk = trim_latency(next_chunk, n_latency_steps)

        r, done, succ, env_steps = step_chunk(
            branch_env,
            obs_history,
            next_chunk,
            env_steps,
            max_env_steps,
            abs_action,
            rotation_transformer,
        )
        total_return += r
        success = success or succ
        continuation_idx += 1

    return float(total_return), bool(success)


def evaluate_candidates(
    branch_envs: List[Any],
    dp_policy,
    saved_state: np.ndarray,
    saved_obs_history: Dict[str, List[np.ndarray]],
    candidates: np.ndarray,
    env_steps_at_decision: int,
    max_env_steps: int,
    n_obs_steps: int,
    n_latency_steps: int,
    abs_action: bool,
    rotation_transformer: RotationTransformer | None,
    device: torch.device,
    continuation_seed_base: int,
) -> Tuple[np.ndarray, np.ndarray]:
    K = int(candidates.shape[0])
    returns = np.zeros((K,), dtype=np.float64)
    successes = np.zeros((K,), dtype=np.bool_)

    # Preserve policy RNG so branch probing does not perturb the on-policy rollout stream.
    with preserve_torch_rng_state():
        for k in range(K):
            ret, succ = rollout_candidate_to_end(
                branch_env=branch_envs[k],
                dp_policy=dp_policy,
                saved_state=saved_state,
                saved_obs_history=saved_obs_history,
                candidate_chunk=candidates[k],
                env_steps_at_decision=env_steps_at_decision,
                max_env_steps=max_env_steps,
                n_obs_steps=n_obs_steps,
                n_latency_steps=n_latency_steps,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
                device=device,
                continuation_seed_base=continuation_seed_base,
            )
            returns[k] = ret
            successes[k] = succ

    return returns, successes


def run_episode(
    seed: int,
    choose_mode: str,
    log_decisions: bool,
    log_candidates: bool,
    env_factory,
    branch_envs: List[Any],
    dp_policy,
    K: int,
    n_obs_steps: int,
    n_latency_steps: int,
    decision_interval: int,
    skip_first: int,
    max_env_steps: int,
    abs_action: bool,
    rotation_transformer: RotationTransformer | None,
    device: torch.device,
    continuation_seed: int,
) -> Dict[str, Any]:
    env = env_factory()
    try:
        env.seed(seed)
        obs = env.reset()
        obs_dict = obs_to_dict(obs)

        obs_history = defaultdict(list)
        for key, val in obs_dict.items():
            obs_history[key].append(val)

        episode_return = 0.0
        episode_success = False
        env_steps = 0
        policy_step = 0
        done = False

        decision_points = []

        while not done and env_steps < max_env_steps:
            is_decision = (
                policy_step >= skip_first
                and (policy_step - skip_first) % decision_interval == 0
            )
            obs_tensor = build_obs_tensor(obs_history, n_obs_steps, device=device)

            if is_decision:
                saved_state = get_env_state(env)
                saved_obs_history = clone_obs_history(obs_history)

                # Sample K candidates from same observation/history by replicating batch item.
                obs_expanded = {k: v.repeat_interleave(K, dim=0) for k, v in obs_tensor.items()}
                with torch.no_grad():
                    with maybe_autocast(device):
                        action_dict = dp_policy.predict_action(obs_expanded)
                candidates = action_dict["action"].detach().cpu().numpy()
                candidates = trim_latency(candidates, n_latency_steps)

                continuation_seed_base = int(continuation_seed + seed * 100_000 + policy_step * 1_000)
                candidate_returns, candidate_success = evaluate_candidates(
                    branch_envs=branch_envs,
                    dp_policy=dp_policy,
                    saved_state=saved_state,
                    saved_obs_history=saved_obs_history,
                    candidates=candidates,
                    env_steps_at_decision=env_steps,
                    max_env_steps=max_env_steps,
                    n_obs_steps=n_obs_steps,
                    n_latency_steps=n_latency_steps,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                    device=device,
                    continuation_seed_base=continuation_seed_base,
                )

                spread = mean_pairwise_l2(candidates)
                best_idx = int(candidate_returns.argmax())
                best_ret = float(candidate_returns[best_idx])
                mean_ret = float(candidate_returns.mean())
                k0_ret = float(candidate_returns[0])

                if choose_mode == "oracle":
                    chosen_idx = best_idx
                else:
                    chosen_idx = 0

                # Execute chosen candidate on policy trajectory.
                chosen_chunk = candidates[chosen_idx]
                r, done, succ, env_steps = step_chunk(
                    env,
                    obs_history,
                    chosen_chunk,
                    env_steps,
                    max_env_steps,
                    abs_action,
                    rotation_transformer,
                )
                episode_return += r
                episode_success = episode_success or succ

                if log_decisions:
                    rec = {
                        "seed": int(seed),
                        "policy_step": int(policy_step),
                        "env_step": int(env_steps),
                        "K": int(K),
                        "candidate_spread_l2_mean_pairwise": float(spread),
                        "candidate_returns": candidate_returns.astype(float).tolist(),
                        "candidate_success": candidate_success.astype(bool).tolist(),
                        "candidate_logprobs": None,  # Not exposed by diffusion-policy inference API.
                        "oracle_index": int(best_idx),
                        "chosen_index": int(chosen_idx),
                        "oracle_headroom_best_minus_mean": float(best_ret - mean_ret),
                        "oracle_headroom_best_minus_k0": float(best_ret - k0_ret),
                    }
                    if log_candidates:
                        rec["candidates"] = candidates.astype(float).tolist()
                    decision_points.append(rec)
            else:
                with torch.no_grad():
                    with maybe_autocast(device):
                        action_dict = dp_policy.predict_action(obs_tensor)
                chunk = action_dict["action"][0].detach().cpu().numpy()
                chunk = trim_latency(chunk, n_latency_steps)

                r, done, succ, env_steps = step_chunk(
                    env,
                    obs_history,
                    chunk,
                    env_steps,
                    max_env_steps,
                    abs_action,
                    rotation_transformer,
                )
                episode_return += r
                episode_success = episode_success or succ

            policy_step += 1

        # In sparse-reward tasks success can exist without explicit info signal.
        # If max reward reaches 1.0, treat episode as success.
        if episode_return >= 1.0:
            episode_success = True

        ep_spreads = np.array([d["candidate_spread_l2_mean_pairwise"] for d in decision_points], dtype=np.float64)
        ep_h_best_mean = np.array([d["oracle_headroom_best_minus_mean"] for d in decision_points], dtype=np.float64)
        ep_h_best_k0 = np.array([d["oracle_headroom_best_minus_k0"] for d in decision_points], dtype=np.float64)
        ep_decision_gain = np.array(
            [float(d["candidate_success"][d["oracle_index"]]) - float(d["candidate_success"][0]) for d in decision_points],
            dtype=np.float64,
        )

        summary = {
            "seed": int(seed),
            "episode_return": float(episode_return),
            "episode_success": bool(episode_success),
            "n_decision_points": int(len(decision_points)),
            "mean_candidate_spread": float(ep_spreads.mean()) if ep_spreads.size else 0.0,
            "mean_oracle_headroom_best_minus_mean": float(ep_h_best_mean.mean()) if ep_h_best_mean.size else 0.0,
            "mean_oracle_headroom_best_minus_k0": float(ep_h_best_k0.mean()) if ep_h_best_k0.size else 0.0,
            "mean_decision_success_gain_oracle_minus_k0": float(ep_decision_gain.mean()) if ep_decision_gain.size else 0.0,
        }

        return {
            "summary": summary,
            "decision_points": decision_points,
        }
    finally:
        if hasattr(env, "close"):
            env.close()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robomimic oracle headroom + diversity audit")
    parser.add_argument("--dp_checkpoint", type=str, required=True, help="Diffusion-policy checkpoint")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--K", type=int, default=8, help="Candidates per decision point")
    parser.add_argument("--decision_interval", type=int, default=5)
    parser.add_argument("--skip_first", type=int, default=2)
    parser.add_argument(
        "--max_env_steps",
        type=int,
        default=None,
        help="Override episode horizon. Default: from checkpoint task cfg.",
    )
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument(
        "--eval_oracle_policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a second rollout per seed that executes oracle best-of-K at each decision.",
    )
    parser.add_argument(
        "--log_candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include full candidate chunks in decision logs (large output).",
    )
    parser.add_argument(
        "--headroom_fraction_threshold",
        type=float,
        default=0.01,
        help="Near-zero headroom threshold as fraction of observed return range.",
    )
    parser.add_argument(
        "--success_gain_threshold",
        type=float,
        default=0.02,
        help="Near-zero success-gain threshold in absolute points (e.g., 0.02 = 2pp).",
    )
    parser.add_argument(
        "--continuation_seed",
        type=int,
        default=12345,
        help="Base seed for common-random-number continuation during branch scoring.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Default: eval_results/robomimic_headroom_<task>_K<K>_n<N>.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main():
    args = make_parser().parse_args()
    if args.K < 1:
        raise ValueError("--K must be >= 1")
    if args.decision_interval < 1:
        raise ValueError("--decision_interval must be >= 1")
    if args.skip_first < 0:
        raise ValueError("--skip_first must be >= 0")

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    print("=" * 70)
    print("Robomimic Headroom Audit")
    print("=" * 70)
    print(f"Checkpoint:  {args.dp_checkpoint}")
    print(f"Episodes:    {args.n_episodes}")
    print(f"K:           {args.K}")
    print(f"DecisionInt: {args.decision_interval}")
    print(f"SkipFirst:   {args.skip_first}")
    print(f"Device:      {device}")
    print(f"Eval oracle policy: {args.eval_oracle_policy}")
    print("=" * 70)

    t0 = time.perf_counter()
    dp_policy, cfg = load_diffusion_policy(args.dp_checkpoint, device=device)
    task_cfg = cfg.task

    # Extract policy metadata for logging
    policy_class = type(dp_policy).__name__
    policy_target = str(cfg_get(cfg, "_target_", "unknown"))
    horizon = int(cfg_get(cfg, "horizon", -1))

    # Git commit hashes for reproducibility
    import subprocess as _sp
    def _git_hash(repo_dir):
        try:
            return _sp.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(repo_dir), stderr=_sp.DEVNULL,
            ).decode().strip()
        except Exception:
            return "unknown"
    tap_git_hash = _git_hash(REPO_ROOT)
    dp_git_hash = _git_hash(DP_ROOT)

    task_name = str(cfg_get(task_cfg, "task_name", "unknown_task"))
    dataset_path_cfg = str(cfg_get(task_cfg, "dataset_path"))
    dataset_path = resolve_existing_path(dataset_path_cfg)
    env_factory, env_info = make_robomimic_env_factory(cfg, dataset_path=dataset_path)

    n_obs_steps = int(cfg_get(task_cfg.env_runner, "n_obs_steps"))
    n_action_steps = int(cfg_get(task_cfg.env_runner, "n_action_steps"))
    n_latency_steps = int(cfg_get(task_cfg.env_runner, "n_latency_steps", 0))
    if n_latency_steps >= n_action_steps:
        raise ValueError(
            f"n_latency_steps ({n_latency_steps}) must be < n_action_steps ({n_action_steps})."
        )

    cfg_max_steps = int(cfg_get(task_cfg.env_runner, "max_steps", 400))
    max_env_steps = int(args.max_env_steps) if args.max_env_steps is not None else cfg_max_steps

    abs_action = bool(cfg_get(task_cfg, "abs_action", False))
    rotation_transformer = RotationTransformer("axis_angle", "rotation_6d") if abs_action else None

    print(f"PolicyClass: {policy_class}")
    print(f"PolicyTarget:{policy_target}")
    print(f"Horizon:     {horizon}")
    print(f"Task:        {task_name}")
    print(f"Data:        {dataset_path}")
    print(f"Mode:        {env_info['mode']}")
    print(f"ObsSteps:    {n_obs_steps}")
    print(f"ActionSteps: {n_action_steps}")
    print(f"Latency:     {n_latency_steps}")
    print(f"MaxEnvSteps: {max_env_steps}")
    print(f"AbsAction:   {abs_action}")
    print(f"TAP commit:  {tap_git_hash}")
    print(f"DP commit:   {dp_git_hash}")

    # Pre-create branch pool for candidate scoring.
    branch_envs = [env_factory() for _ in range(args.K)]

    seeds = list(range(args.seed_offset, args.seed_offset + args.n_episodes))
    baseline_episodes = []
    decision_points = []
    oracle_episode_summaries = []

    for seed in tqdm(seeds, desc="Baseline episodes"):
        result = run_episode(
            seed=seed,
            choose_mode="k0",
            log_decisions=True,
            log_candidates=args.log_candidates,
            env_factory=env_factory,
            branch_envs=branch_envs,
            dp_policy=dp_policy,
            K=args.K,
            n_obs_steps=n_obs_steps,
            n_latency_steps=n_latency_steps,
            decision_interval=args.decision_interval,
            skip_first=args.skip_first,
            max_env_steps=max_env_steps,
            abs_action=abs_action,
            rotation_transformer=rotation_transformer,
            device=device,
            continuation_seed=args.continuation_seed,
        )
        baseline_episodes.append(result["summary"])
        decision_points.extend(result["decision_points"])

    if args.eval_oracle_policy:
        for seed in tqdm(seeds, desc="Oracle episodes"):
            result = run_episode(
                seed=seed,
                choose_mode="oracle",
                log_decisions=False,
                log_candidates=False,
                env_factory=env_factory,
                branch_envs=branch_envs,
                dp_policy=dp_policy,
                K=args.K,
                n_obs_steps=n_obs_steps,
                n_latency_steps=n_latency_steps,
                decision_interval=args.decision_interval,
                skip_first=args.skip_first,
                max_env_steps=max_env_steps,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
                device=device,
                continuation_seed=args.continuation_seed,
            )
            oracle_episode_summaries.append(result["summary"])

    # Global aggregates
    spreads = np.array(
        [d["candidate_spread_l2_mean_pairwise"] for d in decision_points],
        dtype=np.float64,
    )
    headroom_best_mean = np.array(
        [d["oracle_headroom_best_minus_mean"] for d in decision_points],
        dtype=np.float64,
    )
    headroom_best_k0 = np.array(
        [d["oracle_headroom_best_minus_k0"] for d in decision_points],
        dtype=np.float64,
    )

    k0_decision_success = np.array(
        [float(d["candidate_success"][0]) for d in decision_points],
        dtype=np.float64,
    )
    oracle_decision_success = np.array(
        [float(d["candidate_success"][d["oracle_index"]]) for d in decision_points],
        dtype=np.float64,
    )
    decision_success_gain = oracle_decision_success - k0_decision_success

    all_candidate_returns = np.array(
        [r for d in decision_points for r in d["candidate_returns"]],
        dtype=np.float64,
    )
    if all_candidate_returns.size > 0:
        ret_min = float(all_candidate_returns.min())
        ret_max = float(all_candidate_returns.max())
        ret_range = ret_max - ret_min
    else:
        ret_min = float("nan")
        ret_max = float("nan")
        ret_range = 0.0

    headroom_abs_threshold = args.headroom_fraction_threshold * ret_range
    headroom_stats = mean_ci95(headroom_best_k0)
    decision_success_gain_stats = mean_ci95(decision_success_gain)

    baseline_ep_success = np.array(
        [float(ep["episode_success"]) for ep in baseline_episodes],
        dtype=np.float64,
    )
    baseline_ep_return = np.array(
        [float(ep["episode_return"]) for ep in baseline_episodes],
        dtype=np.float64,
    )
    baseline_success_stats = mean_ci95(baseline_ep_success)
    baseline_return_stats = mean_ci95(baseline_ep_return)

    oracle_success_stats = None
    oracle_return_stats = None
    episode_success_gain_stats = None
    success_gain_for_gate = decision_success_gain_stats
    success_gain_source = "decision_to_go"
    if args.eval_oracle_policy:
        oracle_ep_success = np.array(
            [float(ep["episode_success"]) for ep in oracle_episode_summaries],
            dtype=np.float64,
        )
        oracle_ep_return = np.array(
            [float(ep["episode_return"]) for ep in oracle_episode_summaries],
            dtype=np.float64,
        )
        oracle_success_stats = mean_ci95(oracle_ep_success)
        oracle_return_stats = mean_ci95(oracle_ep_return)

        # Paired by same seeds and count.
        ep_success_gain = oracle_ep_success - baseline_ep_success
        episode_success_gain_stats = mean_ci95(ep_success_gain)
        success_gain_for_gate = episode_success_gain_stats
        success_gain_source = "episode_policy"

    headroom_collapsed = (
        np.isfinite(headroom_stats["mean"])
        and headroom_stats["mean"] < headroom_abs_threshold
    )
    success_gain_small = (
        np.isfinite(success_gain_for_gate["ci95_high"])
        and success_gain_for_gate["ci95_high"] < args.success_gain_threshold
    )
    meaningful_headroom = (
        np.isfinite(headroom_stats["mean"])
        and headroom_stats["mean"] >= headroom_abs_threshold
        and np.isfinite(success_gain_for_gate["ci95_low"])
        and success_gain_for_gate["ci95_low"] >= args.success_gain_threshold
    )

    if meaningful_headroom:
        fork_decision = "meaningful_headroom"
        recommendation = "TAP-Score reranking is justified as an inference-time selector."
    elif headroom_collapsed and success_gain_small:
        fork_decision = "headroom_collapsed"
        recommendation = "Increase generator diversity first; ranking is not the bottleneck yet."
    else:
        fork_decision = "inconclusive"
        recommendation = "Signal is mixed. Run more episodes or tighten evaluation controls."

    elapsed = time.perf_counter() - t0

    report = {
        "meta": {
            "script": "scripts/robomimic_headroom_audit.py",
            "timestamp_unix": time.time(),
            "elapsed_sec": elapsed,
        },
        "config": {
            "dp_checkpoint": str(Path(args.dp_checkpoint).expanduser()),
            "policy_class": policy_class,
            "policy_target": policy_target,
            "horizon": horizon,
            "task_name": task_name,
            "dataset_path": str(dataset_path),
            "env_mode": env_info["mode"],
            "device": str(device),
            "tap_git_hash": tap_git_hash,
            "dp_git_hash": dp_git_hash,
            "n_episodes": int(args.n_episodes),
            "K": int(args.K),
            "n_obs_steps": int(n_obs_steps),
            "n_action_steps": int(n_action_steps),
            "n_latency_steps": int(n_latency_steps),
            "decision_interval": int(args.decision_interval),
            "skip_first": int(args.skip_first),
            "max_env_steps": int(max_env_steps),
            "seed_offset": int(args.seed_offset),
            "continuation_seed": int(args.continuation_seed),
            "eval_oracle_policy": bool(args.eval_oracle_policy),
            "log_candidates": bool(args.log_candidates),
            "thresholds": {
                "headroom_fraction_of_return_range": float(args.headroom_fraction_threshold),
                "success_gain_points": float(args.success_gain_threshold),
            },
        },
        "global_metrics": {
            "n_decision_points": int(len(decision_points)),
            "candidate_return_range": {
                "min": ret_min,
                "max": ret_max,
                "range": ret_range,
            },
            "candidate_spread_l2_mean_pairwise": mean_ci95(spreads),
            "oracle_headroom_best_minus_mean": mean_ci95(headroom_best_mean),
            "oracle_headroom_best_minus_k0": headroom_stats,
            "decision_success_k0": mean_ci95(k0_decision_success),
            "decision_success_oracle": mean_ci95(oracle_decision_success),
            "decision_success_gain_oracle_minus_k0": decision_success_gain_stats,
            "episode_success_k1": baseline_success_stats,
            "episode_return_k1": baseline_return_stats,
            "episode_success_oracle_policy": oracle_success_stats,
            "episode_return_oracle_policy": oracle_return_stats,
            "episode_success_gain_oracle_minus_k1": episode_success_gain_stats,
            "gate_metrics": {
                "headroom_threshold_abs": headroom_abs_threshold,
                "success_gain_threshold": float(args.success_gain_threshold),
                "success_gain_source": success_gain_source,
            },
            "fork_decision": fork_decision,
            "recommendation": recommendation,
        },
        "episodes": baseline_episodes,
        "oracle_policy_episodes": oracle_episode_summaries if args.eval_oracle_policy else None,
        "decision_points": decision_points,
    }

    output_path = Path(args.output) if args.output else Path(
        f"eval_results/robomimic_headroom_{task_name}_K{args.K}_n{args.n_episodes}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for env in branch_envs:
        if hasattr(env, "close"):
            env.close()

    print("\n" + "=" * 70)
    print("Audit Complete")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Decision points: {len(decision_points)}")
    print(f"Fork decision: {fork_decision}")
    print(f"Recommendation: {recommendation}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

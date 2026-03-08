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
from typing import Any, Dict, List, Optional, Tuple

import dill
import h5py
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


def _patch_abs_action_controller(env_wrapper, kp: float = 500.0):
    """Set robosuite 1.5.2 OSC controller to absolute input mode (world frame).

    robosuite 1.5.2 uses CompositeController with JSON config files
    (default_panda.json) that default to ``"input_type": "delta"`` and
    ``"input_ref_frame": "base"``.  The dict-based ``controller_configs``
    in ``env_meta`` is ignored.

    Patches applied:
    1. ``input_type = "absolute"`` — interpret actions as target poses, not deltas.
    2. ``input_ref_frame = "world"`` — interpret positions in world frame.
       Without this, robosuite 1.5.2 transforms through the robot base origin:
       ``desired = origin_pos + R(origin_ori) @ goal_pos``, which corrupts
       DP's world-frame absolute positions.
    3. Boost ``kp`` — robosuite 1.5.2's OSC controller converges more slowly
       than the old fork (cheng-chi/robosuite) used to train these checkpoints.
    """
    try:
        robosuite_env = env_wrapper.env.env
        for robot in robosuite_env.robots:
            for _part_name, controller in robot.part_controllers.items():
                if hasattr(controller, "input_type"):
                    controller.input_type = "absolute"
                if hasattr(controller, "input_ref_frame"):
                    controller.input_ref_frame = "world"
                if hasattr(controller, "kp"):
                    controller.kp = np.full(len(controller.kp), kp)
                    controller.kd = 2.0 * np.sqrt(controller.kp)
    except (AttributeError, KeyError):
        pass


def _wrap_reset_for_abs_action(wrapped, kp: float = 500.0):
    """Monkey-patch ``reset()`` so the absolute-mode patch survives resets.

    robosuite recreates controllers on every ``reset()`` call
    (``_load_controller()``), reverting our patch. This ensures the
    controller is re-patched every time.
    """
    original_reset = wrapped.reset

    def patched_reset(*a, **kw):
        result = original_reset(*a, **kw)
        _patch_abs_action_controller(wrapped, kp=kp)
        return result

    wrapped.reset = patched_reset


def make_robomimic_env_factory(
    cfg: Any, dataset_path: Path, abs_action_override: bool | None = None,
    kp: float = 500.0,
):
    """Create a callable that builds fresh wrapped Robomimic envs."""
    task_cfg = cfg.task
    env_runner_cfg = task_cfg.env_runner
    mode = infer_task_mode(task_cfg)
    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))

    abs_action = bool(cfg_get(task_cfg, "abs_action", False))
    if abs_action_override is not None:
        abs_action = bool(abs_action_override)

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
            if abs_action:
                _wrap_reset_for_abs_action(wrapped, kp=kp)
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
            render_offscreen=True,
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
        if abs_action:
            _wrap_reset_for_abs_action(wrapped, kp=kp)
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
    else:
        out = out.astype(np.float32)
        # If wrapper gives float images in 0..255, normalize.
        if out.ndim == 3:
            mx = float(out.max()) if out.size else 0.0
            if mx > 1.5 and mx <= 255.0:
                out = out / 255.0

    # Robustness: convert HWC image to CHW if needed.
    if out.ndim == 3 and out.shape[-1] in (1, 3) and out.shape[0] not in (1, 3):
        out = np.transpose(out, (2, 0, 1))
    return out


# ── Task-aware observation layout ─────────────────────────────────────
#
# Non-object key dims are consistent across Lift/Can/etc.
_ROBOT_KEY_DIMS = {
    "robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2,
}

# Task-specific object observation layout.
#
# robosuite 1.5.2 differs from the old cheng-chi/robosuite fork in two ways:
#
#   1) **Sign flip** (Lift): gripper_to_cube_pos was eef-cube in old fork,
#      now cube-eef in 1.5.2.  Fix: negate the relative-position indices.
#
#   2) **Field reorder** (Can): old fork produced
#        [abs_pos(3), abs_quat(4), rel_pos(3), rel_quat(4)]
#      but 1.5.2 produces
#        [rel_pos(3), rel_quat(4), abs_pos(3), abs_quat(4)]
#      Fix: swap the two halves back to old-fork order.
#
# Fields:
#   object_dim       – total dimensionality of the "object" obs key
#   sign_flip_slice  – indices to negate (sign flip), or None
#   reorder          – permutation list mapping 1.5.2 → old-fork order, or None
#
# Lift (10D): [cube_pos(3), cube_quat(4), gripper_to_cube_pos(3)]
#   → sign-flip at [7:10]
# Can  (14D): 1.5.2 gives [rel_pos(3), rel_quat(4), abs_pos(3), abs_quat(4)]
#   → old fork was  [abs_pos(3), abs_quat(4), rel_pos(3), rel_quat(4)]
#   → reorder: take indices [7..13, 0..6]
_TASK_OBJECT_INFO: Dict[str, Dict[str, Any]] = {
    "lift": {
        "object_dim": 10,
        "sign_flip_slice": slice(7, 10),
        "reorder": None,
    },
    "can": {
        "object_dim": 14,
        "sign_flip_slice": None,
        "reorder": list(range(7, 14)) + list(range(0, 7)),  # swap halves
    },
    "square": {
        "object_dim": 14,
        "sign_flip_slice": None,
        "reorder": list(range(7, 14)) + list(range(0, 7)),  # same layout as Can
    },
}


def _get_object_dim(task_name: str) -> int:
    """Return object obs dimensionality for *task_name*."""
    info = _TASK_OBJECT_INFO.get(task_name.lower())
    if info is None:
        raise ValueError(
            f"Unknown task '{task_name}' — add it to _TASK_OBJECT_INFO. "
            f"Known tasks: {list(_TASK_OBJECT_INFO)}"
        )
    return info["object_dim"]


def _obs_key_dims(task_name: str) -> Dict[str, int]:
    """Build full obs-key → dim mapping for *task_name*."""
    return {**_ROBOT_KEY_DIMS, "object": _get_object_dim(task_name)}


def _fix_object_obs_compat(
    obs_flat: np.ndarray, obs_keys: List[str], task_name: str,
) -> np.ndarray:
    """Fix ``object`` obs for robosuite 1.5.2 compatibility.

    Applies task-specific transforms (sign flip and/or field reorder) so that
    the observation matches the layout the DP checkpoint was trained on.
    See ``_TASK_OBJECT_INFO`` for per-task details.
    """
    info = _TASK_OBJECT_INFO.get(task_name.lower())
    if info is None:
        return obs_flat  # unknown task — don't touch

    dims = _obs_key_dims(task_name)
    idx = 0
    for key in obs_keys:
        if key == "object":
            obs_flat = obs_flat.copy()
            obj_dim = info["object_dim"]
            # Reorder fields (e.g. Can: swap absolute/relative halves).
            reorder = info.get("reorder")
            if reorder is not None:
                obs_flat[idx : idx + obj_dim] = obs_flat[idx : idx + obj_dim][reorder]
            # Sign-flip relative position (e.g. Lift: negate gripper_to_cube_pos).
            sf = info.get("sign_flip_slice")
            if sf is not None:
                obs_flat[idx + sf.start : idx + sf.stop] *= -1.0
            break
        idx += dims.get(key, 0)
    return obs_flat


# ── Observation perturbation (simulated occlusion / sensor failure) ───

PERTURB_TYPES = ("none", "zero_object", "noise_object", "freeze_object", "intermittent_dropout")


def _find_object_slice(
    obs_keys: List[str], task_name: str,
) -> tuple[int, int] | None:
    """Return (start, end) indices of 'object' block in the flat obs vector."""
    dims = _obs_key_dims(task_name)
    idx = 0
    for key in obs_keys:
        if key == "object":
            return idx, idx + dims["object"]
        idx += dims.get(key, 0)
    return None


def _apply_obs_perturbation(
    obs_flat: np.ndarray,
    obs_keys: List[str],
    perturb_type: str,
    task_name: str,
    perturb_rng: np.random.RandomState | None = None,
    noise_std: float = 0.01,
    frozen_object: np.ndarray | None = None,
    dropout_p: float = 0.3,
) -> np.ndarray:
    """Apply observation perturbation to simulate sensor failures / occlusion.

    Perturbation types:
      - zero_object:   zero all object dims (total occlusion — can't see object)
      - noise_object:  add Gaussian noise to object dims (noisy perception)
      - freeze_object: replace object dims with captured values (tracker lost / frozen)
    """
    if perturb_type == "none":
        return obs_flat

    obj_slice = _find_object_slice(obs_keys, task_name)
    if obj_slice is None:
        return obs_flat

    s, e = obj_slice
    obs_flat = obs_flat.copy()

    if perturb_type == "zero_object":
        obs_flat[s:e] = 0.0
    elif perturb_type == "noise_object":
        if perturb_rng is None:
            perturb_rng = np.random.RandomState()
        obs_flat[s:e] += perturb_rng.randn(e - s) * noise_std
    elif perturb_type == "freeze_object":
        if frozen_object is not None:
            obs_flat[s:e] = frozen_object
    elif perturb_type == "intermittent_dropout":
        if perturb_rng is not None and perturb_rng.rand() < dropout_p:
            obs_flat[s:e] = 0.0
    return obs_flat


def obs_to_dict(
    obs: Any,
    obs_keys: List[str] | None = None,
    task_name: str | None = None,
) -> Dict[str, np.ndarray]:
    if isinstance(obs, dict):
        return {k: _ensure_float_obs(v) for k, v in obs.items()}
    out = _ensure_float_obs(obs)
    if obs_keys is not None and task_name is not None and out.ndim == 1:
        out = _fix_object_obs_compat(out, obs_keys, task_name)
    return {"obs": out}


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


def set_env_state(
    env_wrapper, state: np.ndarray,
    obs_keys: List[str] | None = None,
    task_name: str | None = None,
) -> Dict[str, np.ndarray]:
    raw_obs = env_wrapper.env.reset_to({"states": np.array(state, copy=True)})
    # reset_to doesn't call _reset_internal, so robosuite's timestep/done
    # accumulate across reuses and eventually crash with "terminated episode".
    try:
        robosuite_env = env_wrapper.env.env
        robosuite_env.timestep = 0
        robosuite_env.done = False
    except AttributeError:
        pass
    try:
        obs = env_wrapper.get_observation(raw_obs)
    except TypeError:
        obs = env_wrapper.get_observation()
    return obs_to_dict(obs, obs_keys=obs_keys, task_name=task_name)


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
    # DP's runner uses fp32 for inference. fp16 autocast can degrade diffusion
    # sampling quality (DDPM scheduler uses small alpha values). Use fp32.
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
    obs_keys: List[str] | None = None,
    task_name: str | None = None,
    perturb_type: str = "none",
    perturb_start_step: int = 0,
    perturb_rng: np.random.RandomState | None = None,
    perturb_noise_std: float = 0.01,
    frozen_object: np.ndarray | None = None,
    dropout_p: float = 0.3,
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
        obs_dict = obs_to_dict(obs, obs_keys=obs_keys, task_name=task_name)
        # Apply perturbation (simulated occlusion) to obs before policy sees it.
        if perturb_type != "none" and obs_keys is not None and task_name is not None and env_steps >= perturb_start_step:
            if "obs" in obs_dict and obs_dict["obs"].ndim == 1:
                # Capture frozen_object at the moment perturbation starts.
                if perturb_type == "freeze_object" and frozen_object is None:
                    obj_slice = _find_object_slice(obs_keys, task_name)
                    if obj_slice is not None:
                        s, e = obj_slice
                        frozen_object = obs_dict["obs"][s:e].copy()
                obs_dict["obs"] = _apply_obs_perturbation(
                    obs_dict["obs"], obs_keys, perturb_type, task_name,
                    perturb_rng, perturb_noise_std, frozen_object,
                    dropout_p=dropout_p,
                )
        for key, val in obs_dict.items():
            obs_history[key].append(val)

        total_return += float(reward)
        success = success or extract_success(info) or float(reward) >= 1.0
        env_steps += 1
        if step_done or env_steps >= max_env_steps:
            done = True
            break

    return total_return, done, success, env_steps


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
    branch_horizon: int | None = None,
    obs_keys: List[str] | None = None,
    task_name: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate K candidate branches using batched lockstep rollout.

    Instead of K separate threads each doing batch_size=1 policy inference,
    this steps all K branches in lockstep and does a single batched
    predict_action(batch_size=K) per continuation step.  This gives ~K×
    speedup on GPU (batched inference) and avoids thread overhead entirely.

    If *branch_horizon* is set, each branch rolls out at most that many
    env steps instead of running to episode end.  This makes each decision
    O(K * branch_horizon) instead of O(K * remaining_horizon).
    """
    K = int(candidates.shape[0])
    branch_max_steps = max_env_steps
    if branch_horizon is not None:
        branch_max_steps = min(max_env_steps, env_steps_at_decision + branch_horizon)

    returns = np.zeros((K,), dtype=np.float64)
    successes = np.zeros((K,), dtype=np.bool_)
    active = np.ones((K,), dtype=np.bool_)
    env_steps_arr = np.full((K,), env_steps_at_decision, dtype=np.int64)

    # Per-branch obs histories
    branch_obs_histories: List[Dict[str, List[np.ndarray]]] = []

    # Restore state and execute candidate chunk in each branch.
    for k in range(K):
        _ = set_env_state(branch_envs[k], saved_state, obs_keys=obs_keys, task_name=task_name)
        obs_hist_k = clone_obs_history(saved_obs_history)
        r, done, succ, es = step_chunk(
            branch_envs[k],
            obs_hist_k,
            candidates[k],
            int(env_steps_arr[k]),
            branch_max_steps,
            abs_action,
            rotation_transformer,
            obs_keys=obs_keys,
            task_name=task_name,
        )
        returns[k] += r
        successes[k] = successes[k] or succ
        env_steps_arr[k] = es
        if done or es >= branch_max_steps:
            active[k] = False
        branch_obs_histories.append(obs_hist_k)

    # Lockstep continuation: batch all active branches into one GPU call.
    with preserve_torch_rng_state():
        continuation_idx = 0
        while active.any():
            active_indices = np.where(active)[0]

            # Build batched obs tensor for all active branches.
            obs_batch: Dict[str, List[torch.Tensor]] = {}
            for k_idx in active_indices:
                obs_t = build_obs_tensor(branch_obs_histories[k_idx], n_obs_steps, device=device)
                for key, val in obs_t.items():
                    obs_batch.setdefault(key, []).append(val)
            obs_batched = {key: torch.cat(tensors, dim=0) for key, tensors in obs_batch.items()}

            set_torch_seed(int(continuation_seed_base + continuation_idx))
            with maybe_autocast(device):
                action_dict = dp_policy.predict_action(obs_batched)
            all_chunks = action_dict["action"].detach().cpu().numpy()  # (n_active, H, A)
            all_chunks = trim_latency(all_chunks, n_latency_steps)

            # Step each active branch with its action chunk.
            for i, k_idx in enumerate(active_indices):
                chunk_k = all_chunks[i]
                r, done, succ, es = step_chunk(
                    branch_envs[k_idx],
                    branch_obs_histories[k_idx],
                    chunk_k,
                    int(env_steps_arr[k_idx]),
                    branch_max_steps,
                    abs_action,
                    rotation_transformer,
                    obs_keys=obs_keys,
                    task_name=task_name,
                )
                returns[k_idx] += r
                successes[k_idx] = successes[k_idx] or succ
                env_steps_arr[k_idx] = es
                if done or es >= branch_max_steps:
                    active[k_idx] = False

            continuation_idx += 1

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
    branch_horizon: int | None = None,
    init_state_fn=None,
    obs_keys: List[str] | None = None,
    task_name: str | None = None,
    perturb_type: str = "none",
    perturb_start_step: int = 0,
    perturb_rng: np.random.RandomState | None = None,
    perturb_noise_std: float = 0.01,
) -> Dict[str, Any]:
    env = env_factory()
    try:
        if hasattr(env, "seed"):
            env.seed(seed)
        elif hasattr(env, "set_seed"):
            env.set_seed(seed)
        obs = env.reset()
        if init_state_fn is not None:
            state0 = init_state_fn(seed)
            obs_dict = set_env_state(env, state0, obs_keys=obs_keys, task_name=task_name)
        else:
            obs_dict = obs_to_dict(obs, obs_keys=obs_keys, task_name=task_name)

        # For freeze_object: capture at perturb_start_step (deferred to step_chunk
        # if perturb_start_step > 0).  Only capture here for start_step == 0.
        frozen_object = None
        if perturb_type == "freeze_object" and obs_keys is not None and task_name is not None and perturb_start_step == 0:
            if "obs" in obs_dict and obs_dict["obs"].ndim == 1:
                obj_slice = _find_object_slice(obs_keys, task_name)
                if obj_slice is not None:
                    s, e = obj_slice
                    frozen_object = obs_dict["obs"][s:e].copy()

        # Apply perturbation to initial obs if perturb_start_step == 0.
        if perturb_type != "none" and obs_keys is not None and task_name is not None and 0 >= perturb_start_step:
            if "obs" in obs_dict and obs_dict["obs"].ndim == 1:
                obs_dict["obs"] = _apply_obs_perturbation(
                    obs_dict["obs"], obs_keys, perturb_type, task_name,
                    perturb_rng, perturb_noise_std, frozen_object,
                )

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
                    branch_horizon=branch_horizon,
                    obs_keys=obs_keys,
                    task_name=task_name,
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
                    obs_keys=obs_keys,
                    task_name=task_name,
                    perturb_type=perturb_type,
                    perturb_start_step=perturb_start_step,
                    perturb_rng=perturb_rng,
                    perturb_noise_std=perturb_noise_std,
                    frozen_object=frozen_object,
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
                    obs_keys=obs_keys,
                    task_name=task_name,
                    perturb_type=perturb_type,
                    perturb_start_step=perturb_start_step,
                    perturb_rng=perturb_rng,
                    perturb_noise_std=perturb_noise_std,
                    frozen_object=frozen_object,
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
        "--reset_to_dataset_init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset each episode to a demo initial state from the dataset (robomimic-style eval).",
    )
    parser.add_argument(
        "--action_mode",
        choices=["auto", "abs", "delta"],
        default="auto",
        help="Override action mode. 'auto' reads from checkpoint config.",
    )
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
        "--branch_horizon",
        type=int,
        default=None,
        help="Max env steps per branch rollout (bounded lookahead). "
             "Default: roll out to episode end. 50-100 is sufficient for fork decisions.",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=500.0,
        help="OSC controller proportional gain (robosuite 1.5.2 default is 150, "
             "boosted to compensate for dynamics mismatch with old fork).",
    )
    parser.add_argument(
        "--perturb",
        choices=PERTURB_TYPES,
        default="none",
        help="Observation perturbation type (simulated occlusion). "
             "zero_object: zero all object dims. noise_object: add Gaussian noise. "
             "freeze_object: freeze object obs at initial values.",
    )
    parser.add_argument(
        "--perturb_start_step",
        type=int,
        default=0,
        help="Env step at which perturbation begins (0 = from start, >0 = mid-episode occlusion).",
    )
    parser.add_argument(
        "--perturb_noise_std",
        type=float,
        default=0.01,
        help="Noise std for noise_object perturbation.",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from checkpoint file. Skips already-completed episodes.",
    )
    return parser


# ── Checkpoint helpers ──────────────────────────────────────────────

def _checkpoint_path(output_path: Path) -> Path:
    """Derive checkpoint path from output path: foo.json -> foo.ckpt.json"""
    return output_path.with_suffix(".ckpt.json")


def save_checkpoint(
    ckpt_path: Path,
    phase: str,
    baseline_episodes: List[Dict],
    decision_points: List[Dict],
    oracle_episode_summaries: List[Dict],
    completed_baseline_seeds: List[int],
    completed_oracle_seeds: List[int],
    args_dict: Dict[str, Any],
):
    ckpt = {
        "phase": phase,
        "completed_baseline_seeds": completed_baseline_seeds,
        "completed_oracle_seeds": completed_oracle_seeds,
        "baseline_episodes": baseline_episodes,
        "decision_points": decision_points,
        "oracle_episode_summaries": oracle_episode_summaries,
        "args": args_dict,
    }
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt, f)
    tmp.replace(ckpt_path)  # atomic on same filesystem


def load_checkpoint(ckpt_path: Path, args_dict: Dict[str, Any]) -> Optional[Dict]:
    if not ckpt_path.is_file():
        return None
    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    # Validate key args match
    saved = ckpt.get("args", {})
    for key in ("dp_checkpoint", "K", "decision_interval", "skip_first",
                "n_episodes", "seed_offset", "continuation_seed",
                "kp", "action_mode", "reset_to_dataset_init", "branch_horizon",
                "perturb", "perturb_start_step", "perturb_noise_std"):
        if str(saved.get(key)) != str(args_dict.get(key)):
            print(f"WARNING: checkpoint arg '{key}' mismatch: "
                  f"saved={saved.get(key)} vs current={args_dict.get(key)}")
            print("Ignoring checkpoint — starting fresh.")
            return None
    return ckpt


def main():
    args = make_parser().parse_args()

    # Make progress visible when stdout is redirected (background tasks, log files).
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    if args.K < 1:
        raise ValueError("--K must be >= 1")
    if args.decision_interval < 1:
        raise ValueError("--decision_interval must be >= 1")
    if args.skip_first < 0:
        raise ValueError("--skip_first must be >= 0")

    import os
    # Cap CPU thread oversubscription — MuJoCo + PyTorch can spawn too many.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if var not in os.environ:
            os.environ[var] = str(min(os.cpu_count() or 4, 4))

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
    print(f"BranchHorz:  {args.branch_horizon or 'full'}")
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

    # Resolve action mode before creating envs (controller patches depend on it).
    abs_action_cfg = bool(cfg_get(task_cfg, "abs_action", False))
    if args.action_mode == "abs":
        abs_action = True
    elif args.action_mode == "delta":
        abs_action = False
    else:
        abs_action = abs_action_cfg
    rotation_transformer = RotationTransformer("axis_angle", "rotation_6d") if abs_action else None

    env_factory, env_info = make_robomimic_env_factory(
        cfg, dataset_path=dataset_path,
        abs_action_override=abs_action if args.action_mode != "auto" else None,
        kp=args.kp,
    )

    n_obs_steps = int(cfg_get(task_cfg.env_runner, "n_obs_steps"))
    n_action_steps = int(cfg_get(task_cfg.env_runner, "n_action_steps"))
    n_latency_steps = int(cfg_get(task_cfg.env_runner, "n_latency_steps", 0))
    if n_latency_steps >= n_action_steps:
        raise ValueError(
            f"n_latency_steps ({n_latency_steps}) must be < n_action_steps ({n_action_steps})."
        )

    cfg_max_steps = int(cfg_get(task_cfg.env_runner, "max_steps", 400))
    max_env_steps = int(args.max_env_steps) if args.max_env_steps is not None else cfg_max_steps

    # Load dataset initial states for robomimic-style evaluation.
    demo_keys: List[str] = []
    h5_file = None
    init_state_fn = None
    if args.reset_to_dataset_init:
        h5_file = h5py.File(str(dataset_path), "r")
        demo_keys = sorted(h5_file["data"].keys())
        if not demo_keys:
            raise RuntimeError(f"No demos found in dataset: {dataset_path}")

        def sample_init_state(seed: int) -> np.ndarray:
            rng = np.random.RandomState(seed)
            demo = demo_keys[int(rng.randint(0, len(demo_keys)))]
            return np.array(h5_file["data"][demo]["states"][0], copy=True)

        init_state_fn = sample_init_state
        print(f"DatasetInit: {len(demo_keys)} demos available")

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
    print(f"AbsAction:   {abs_action} (cfg={abs_action_cfg}, override={args.action_mode})")

    # Extract obs_keys for gripper_to_cube sign fix (robosuite 1.5.2 compat).
    lowdim_obs_keys = env_info.get("obs_keys", None)
    print(f"Kp:          {args.kp}")
    print(f"ResetToData: {args.reset_to_dataset_init}")
    if args.perturb != "none":
        print(f"Perturb:     {args.perturb} (start_step={args.perturb_start_step})")
        if args.perturb == "noise_object":
            print(f"NoiseStd:    {args.perturb_noise_std}")
    print(f"TAP commit:  {tap_git_hash}")
    print(f"DP commit:   {dp_git_hash}")

    # Pre-create branch pool for candidate scoring.
    branch_envs = [env_factory() for _ in range(args.K)]

    # Verify absolute-action controller patch is active.
    if abs_action:
        try:
            ctrl = branch_envs[0].env.env.robots[0].part_controllers["right"]
            print(f"Controller:  {type(ctrl).__name__}  input_type={ctrl.input_type}  ref_frame={ctrl.input_ref_frame}")
            assert ctrl.input_type == "absolute", (
                f"FATAL: controller input_type is '{ctrl.input_type}', expected 'absolute'. "
                "The abs_action patch did not apply."
            )
            assert ctrl.input_ref_frame == "world", (
                f"FATAL: controller input_ref_frame is '{ctrl.input_ref_frame}', expected 'world'. "
                "The abs_action patch did not apply."
            )
        except (AttributeError, KeyError) as exc:
            print(f"WARNING: Could not verify controller input_type: {exc}")

    seeds = list(range(args.seed_offset, args.seed_offset + args.n_episodes))

    # ── Checkpoint / resume logic ──
    output_path = Path(args.output) if args.output else Path(
        f"eval_results/robomimic_headroom_{task_name}_K{args.K}_n{args.n_episodes}.json"
    )
    ckpt_path = _checkpoint_path(output_path)
    args_dict = {
        "dp_checkpoint": args.dp_checkpoint,
        "K": args.K,
        "decision_interval": args.decision_interval,
        "skip_first": args.skip_first,
        "n_episodes": args.n_episodes,
        "seed_offset": args.seed_offset,
        "continuation_seed": args.continuation_seed,
        "kp": args.kp,
        "perturb": args.perturb,
        "perturb_start_step": args.perturb_start_step,
        "perturb_noise_std": args.perturb_noise_std,
        "action_mode": args.action_mode,
        "reset_to_dataset_init": args.reset_to_dataset_init,
        "branch_horizon": args.branch_horizon,
    }

    baseline_episodes = []
    decision_points = []
    oracle_episode_summaries = []
    completed_baseline_seeds: List[int] = []
    completed_oracle_seeds: List[int] = []
    start_phase = "baseline"

    if args.resume:
        ckpt = load_checkpoint(ckpt_path, args_dict)
        if ckpt is not None:
            baseline_episodes = ckpt["baseline_episodes"]
            decision_points = ckpt["decision_points"]
            oracle_episode_summaries = ckpt["oracle_episode_summaries"]
            completed_baseline_seeds = ckpt["completed_baseline_seeds"]
            completed_oracle_seeds = ckpt["completed_oracle_seeds"]
            start_phase = ckpt["phase"]
            n_bl = len(completed_baseline_seeds)
            n_or = len(completed_oracle_seeds)
            print(f"Resumed from checkpoint: {n_bl} baseline, {n_or} oracle episodes done.")

    remaining_baseline = [s for s in seeds if s not in set(completed_baseline_seeds)]
    remaining_oracle = [s for s in seeds if s not in set(completed_oracle_seeds)]

    if start_phase == "baseline" and remaining_baseline:
        total_bl = len(seeds)
        done_bl = total_bl - len(remaining_baseline)
        with torch.inference_mode():
            for seed in tqdm(remaining_baseline, desc="Baseline episodes",
                             initial=done_bl, total=total_bl):
                ep_t0 = time.perf_counter()
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
                    branch_horizon=args.branch_horizon,
                    init_state_fn=init_state_fn,
                    obs_keys=lowdim_obs_keys,
                    task_name=task_name,
                    perturb_type=args.perturb,
                    perturb_start_step=args.perturb_start_step,
                    perturb_rng=np.random.RandomState(seed) if args.perturb == "noise_object" else None,
                    perturb_noise_std=args.perturb_noise_std,
                )
                ep_dt = time.perf_counter() - ep_t0
                n_dp = result["summary"]["n_decision_points"]
                print(f"  [seed={seed}] {ep_dt:.1f}s  dps={n_dp}  ret={result['summary']['episode_return']:.3f}", flush=True)
                baseline_episodes.append(result["summary"])
                decision_points.extend(result["decision_points"])
                completed_baseline_seeds.append(seed)
                save_checkpoint(ckpt_path, "baseline", baseline_episodes,
                                decision_points, oracle_episode_summaries,
                                completed_baseline_seeds, completed_oracle_seeds, args_dict)
    elif remaining_baseline:
        # start_phase is "oracle" but baseline not finished — shouldn't happen
        pass

    if args.eval_oracle_policy and remaining_oracle:
        total_or = len(seeds)
        done_or = total_or - len(remaining_oracle)
        with torch.inference_mode():
            for seed in tqdm(remaining_oracle, desc="Oracle episodes",
                             initial=done_or, total=total_or):
                ep_t0 = time.perf_counter()
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
                    branch_horizon=args.branch_horizon,
                    init_state_fn=init_state_fn,
                    obs_keys=lowdim_obs_keys,
                    task_name=task_name,
                    perturb_type=args.perturb,
                    perturb_start_step=args.perturb_start_step,
                    perturb_rng=np.random.RandomState(seed) if args.perturb == "noise_object" else None,
                    perturb_noise_std=args.perturb_noise_std,
                )
                ep_dt = time.perf_counter() - ep_t0
                print(f"  [seed={seed}] {ep_dt:.1f}s  ret={result['summary']['episode_return']:.3f}", flush=True)
                oracle_episode_summaries.append(result["summary"])
                completed_oracle_seeds.append(seed)
                save_checkpoint(ckpt_path, "oracle", baseline_episodes,
                                decision_points, oracle_episode_summaries,
                                completed_baseline_seeds, completed_oracle_seeds, args_dict)

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
            "abs_action": bool(abs_action),
            "kp": float(args.kp),
            "reset_to_dataset_init": bool(args.reset_to_dataset_init),
            "branch_horizon": args.branch_horizon,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Clean up checkpoint now that final output is written.
    if ckpt_path.is_file():
        ckpt_path.unlink()
        print(f"Removed checkpoint: {ckpt_path}")

    for env in branch_envs:
        if hasattr(env, "close"):
            env.close()

    if h5_file is not None:
        h5_file.close()

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

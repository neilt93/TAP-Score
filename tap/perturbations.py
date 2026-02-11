"""
Image Perturbations for OOD Evaluation

These are applied at test time only, NOT used as training labels.
"""

import numpy as np
import torch


def sample_patch_xy(H: int, W: int, patch_h: int, patch_w: int,
                    rng: np.random.Generator) -> tuple[int, int]:
    """Sample top-left (x, y) so a patch_h x patch_w patch fits fully in image."""
    max_y = max(0, H - patch_h)
    max_x = max(0, W - patch_w)
    y = int(rng.integers(0, max_y + 1))  # inclusive of max_y
    x = int(rng.integers(0, max_x + 1))  # inclusive of max_x
    return x, y


def apply_brightness(obs, factor=1.5):
    """
    Scale pixel values by factor.

    Args:
        obs: (T, C, H, W) or (B, T, C, H, W) in [0, 1]
        factor: Brightness multiplier (< 1 = darker, > 1 = brighter)

    Returns:
        Perturbed observation, clipped to [0, 1]
    """
    return np.clip(obs * factor, 0, 1)


def apply_gaussian_noise(obs, std=0.1, rng=None):
    """
    Add Gaussian noise to observation.

    Args:
        obs: (T, C, H, W) or (B, T, C, H, W) in [0, 1]
        std: Standard deviation of noise
        rng: Optional np.random.Generator for reproducibility.
             If None, uses a fresh default_rng() instance.

    Returns:
        Perturbed observation, clipped to [0, 1]
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.standard_normal(obs.shape, dtype=obs.dtype) * std
    return np.clip(obs + noise, 0, 1)


def apply_occlusion(
    obs,
    patch_size=20,
    rng=None,
    xy=None,
    *,
    # realism knobs
    aspect_ratio_range=(1.0, 1.0),     # (min_ar, max_ar), width/height
    fill_mode="black",                 # "black" | "mean" | "random" | "noise"
    alpha=1.0,                         # 1.0 = fully opaque, <1 blends with original
    temporal="static",                 # "static" | "drift" | "per_frame"
    max_drift=2,                       # pixels per frame for "drift"
):
    """
    Apply an occlusion patch with configurable realism.

    Args:
        obs: (T, C, H, W) or (B, T, C, H, W) in [0, 1]
        patch_size: Base size of occlusion patch (height). Width derived from aspect_ratio.
        rng: np.random.Generator. Falls back to a fresh default_rng if None.
        xy: Optional (x, y) to fix patch position. Clamped to valid range.
        aspect_ratio_range: (min, max) for width/height ratio. Sampled uniformly.
        fill_mode: What to fill the patch with:
            "black" — zeros (classic),
            "mean" — per-channel mean of obs,
            "random" — uniform random color,
            "noise" — per-pixel uniform noise.
        alpha: Opacity. 1.0 = fully opaque, <1.0 blends fill with original pixels.
        temporal: How the patch varies across T frames:
            "static" — same position all frames,
            "drift" — small random walk per frame,
            "per_frame" — independent random position each frame.
        max_drift: Max pixel shift per frame when temporal="drift".

    Returns:
        Perturbed observation (copy).
    """
    obs = obs.copy()

    if rng is None:
        rng = np.random.default_rng()

    # Get H, W and batch flag
    if obs.ndim == 4:
        T, C, H, W = obs.shape
        batch = False
    elif obs.ndim == 5:
        B, T, C, H, W = obs.shape
        batch = True
    else:
        raise ValueError(f"apply_occlusion expected 4D or 5D obs, got shape {obs.shape}")

    # Compute patch dimensions from base size + aspect ratio
    patch_size = max(1, int(patch_size))

    min_ar, max_ar = aspect_ratio_range
    if min_ar <= 0 or max_ar <= 0 or min_ar > max_ar:
        raise ValueError("aspect_ratio_range must be positive and min <= max")

    ar = float(rng.uniform(min_ar, max_ar))
    patch_h = min(patch_size, H)
    patch_w = min(max(1, int(round(patch_size * ar))), W)

    # Build fill tile (C, patch_h, patch_w)
    def _make_fill():
        if fill_mode == "black":
            return np.zeros((C, patch_h, patch_w), dtype=obs.dtype)
        elif fill_mode == "mean":
            if batch:
                mean_c = obs.mean(axis=(0, 1, 3, 4))  # (C,)
            else:
                mean_c = obs.mean(axis=(0, 2, 3))      # (C,)
            return (mean_c[:, None, None] * np.ones((C, patch_h, patch_w), dtype=obs.dtype))
        elif fill_mode == "random":
            color = rng.uniform(0.0, 1.0, size=(C,)).astype(obs.dtype)
            return (color[:, None, None] * np.ones((C, patch_h, patch_w), dtype=obs.dtype))
        elif fill_mode == "noise":
            return rng.uniform(0.0, 1.0, size=(C, patch_h, patch_w)).astype(obs.dtype)
        else:
            raise ValueError(f"Unknown fill_mode: {fill_mode}")

    fill = _make_fill()
    alpha_c = float(np.clip(alpha, 0.0, 1.0))

    # Base position
    if xy is not None:
        x0 = int(np.clip(int(xy[0]), 0, max(0, W - patch_w)))
        y0 = int(np.clip(int(xy[1]), 0, max(0, H - patch_h)))
    else:
        x0, y0 = sample_patch_xy(H, W, patch_h, patch_w, rng)

    def _apply_at(x, y, frame_slice):
        region = frame_slice[..., :, y:y + patch_h, x:x + patch_w]
        if alpha_c >= 1.0:
            frame_slice[..., :, y:y + patch_h, x:x + patch_w] = fill
        else:
            frame_slice[..., :, y:y + patch_h, x:x + patch_w] = (
                (1 - alpha_c) * region + alpha_c * fill
            )

    if temporal == "static":
        _apply_at(x0, y0, obs)

    elif temporal == "per_frame":
        for t in range(T):
            x, y = sample_patch_xy(H, W, patch_h, patch_w, rng)
            if batch:
                _apply_at(x, y, obs[:, t:t + 1])
            else:
                _apply_at(x, y, obs[t:t + 1])

    elif temporal == "drift":
        x, y = x0, y0
        for t in range(T):
            dx = int(rng.integers(-max_drift, max_drift + 1))
            dy = int(rng.integers(-max_drift, max_drift + 1))
            x = int(np.clip(x + dx, 0, max(0, W - patch_w)))
            y = int(np.clip(y + dy, 0, max(0, H - patch_h)))
            if batch:
                _apply_at(x, y, obs[:, t:t + 1])
            else:
                _apply_at(x, y, obs[t:t + 1])
    else:
        raise ValueError(f"Unknown temporal mode: {temporal}")

    return obs


def apply_translation(obs, shift_x=5, shift_y=5):
    """
    Translate observation (crop and pad).

    Args:
        obs: (T, C, H, W) or (B, T, C, H, W) in [0, 1]
        shift_x: Horizontal shift (positive = right)
        shift_y: Vertical shift (positive = down)

    Returns:
        Perturbed observation
    """
    obs = obs.copy()

    # Get spatial dimensions
    if obs.ndim == 4:
        T, C, H, W = obs.shape
        result = np.zeros_like(obs)

        # Compute valid regions
        src_y1 = max(0, -shift_y)
        src_y2 = min(H, H - shift_y)
        src_x1 = max(0, -shift_x)
        src_x2 = min(W, W - shift_x)

        dst_y1 = max(0, shift_y)
        dst_y2 = min(H, H + shift_y)
        dst_x1 = max(0, shift_x)
        dst_x2 = min(W, W + shift_x)

        result[:, :, dst_y1:dst_y2, dst_x1:dst_x2] = obs[:, :, src_y1:src_y2, src_x1:src_x2]
        return result
    else:
        B, T, C, H, W = obs.shape
        result = np.zeros_like(obs)

        src_y1 = max(0, -shift_y)
        src_y2 = min(H, H - shift_y)
        src_x1 = max(0, -shift_x)
        src_x2 = min(W, W - shift_x)

        dst_y1 = max(0, shift_y)
        dst_y2 = min(H, H + shift_y)
        dst_x1 = max(0, shift_x)
        dst_x2 = min(W, W + shift_x)

        result[:, :, :, dst_y1:dst_y2, dst_x1:dst_x2] = obs[:, :, :, src_y1:src_y2, src_x1:src_x2]
        return result


class Perturbation:
    """Wrapper class for perturbations."""

    def __init__(self, name, fn, **kwargs):
        self.name = name
        self.fn = fn
        self.kwargs = kwargs

    def __call__(self, obs):
        return self.fn(obs, **self.kwargs)

    def __repr__(self):
        return f"Perturbation({self.name}, {self.kwargs})"


# Pre-defined perturbations
PERTURBATIONS = {
    'brightness_low': Perturbation('brightness_low', apply_brightness, factor=0.5),
    'brightness_high': Perturbation('brightness_high', apply_brightness, factor=1.5),
    'noise_low': Perturbation('noise_low', apply_gaussian_noise, std=0.05),
    'noise_high': Perturbation('noise_high', apply_gaussian_noise, std=0.15),
    'occlusion_small': Perturbation('occlusion_small', apply_occlusion, patch_size=15),
    'occlusion_large': Perturbation('occlusion_large', apply_occlusion, patch_size=25),
    'translation': Perturbation('translation', apply_translation, shift_x=8, shift_y=8),
}


def get_perturbation(name):
    """Get perturbation by name."""
    if name not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation: {name}. Available: {list(PERTURBATIONS.keys())}")
    return PERTURBATIONS[name]

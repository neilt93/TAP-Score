"""
Image Perturbations for OOD Evaluation

These are applied at test time only, NOT used as training labels.
"""

import numpy as np
import torch


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


def apply_gaussian_noise(obs, std=0.1):
    """
    Add Gaussian noise to observation.

    Args:
        obs: (T, C, H, W) or (B, T, C, H, W) in [0, 1]
        std: Standard deviation of noise

    Returns:
        Perturbed observation, clipped to [0, 1]
    """
    noise = np.random.randn(*obs.shape).astype(obs.dtype) * std
    return np.clip(obs + noise, 0, 1)


def apply_occlusion(obs, patch_size=20):
    """
    Add random black square occlusion.

    Args:
        obs: (T, C, H, W) or (B, T, C, H, W) in [0, 1]
        patch_size: Size of occlusion patch

    Returns:
        Perturbed observation
    """
    obs = obs.copy()

    # Get spatial dimensions
    if obs.ndim == 4:
        _, _, H, W = obs.shape
    else:
        _, _, _, H, W = obs.shape

    # Random position
    y = np.random.randint(0, H - patch_size)
    x = np.random.randint(0, W - patch_size)

    # Apply occlusion
    if obs.ndim == 4:
        obs[:, :, y:y+patch_size, x:x+patch_size] = 0
    else:
        obs[:, :, :, y:y+patch_size, x:x+patch_size] = 0

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

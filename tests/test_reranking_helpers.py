"""Tests for reranking experiment helper functions.

Covers: perturbation logic, shared-perturbation invariant, RNG seeding,
stack_history, _ensure_float_chw, compute_progress, and analysis math.

Run:  python -m pytest tests/test_reranking_helpers.py -v
"""
import math
import sys
import os
import types
from unittest import mock

import numpy as np
import pytest

# ── Mock heavy dependencies so we can import the pure helpers ──
# These modules are only on RunPod (GPU machine), not needed for unit tests.
_MOCK_MODULES = [
    "dill", "hydra", "hydra.utils",
    "torch", "torch.nn", "torch.nn.functional", "torch.amp",
    "torch.backends", "torch.backends.cudnn", "torch.backends.cuda",
    "diffusion_policy", "diffusion_policy.env",
    "diffusion_policy.env.pusht", "diffusion_policy.env.pusht.pusht_image_env",
    "tap.env_wrapper", "tap.contrastive",
    "tqdm",
]

for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Now we can import the pure numpy/math functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import only the functions we actually test (no torch needed)
import importlib
_spec = importlib.util.spec_from_file_location(
    "reranking_experiment",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "reranking_experiment.py"),
)
_mod = types.ModuleType("reranking_experiment")

# Execute only the function definitions we need (they're pure numpy/math)
# by reading the source and extracting them
_src_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "reranking_experiment.py")
with open(_src_path) as f:
    _source = f.read()

# Import sample_patch_xy directly from tap/perturbations.py (avoid tap/__init__.py side effects)
_perturb_path = os.path.join(os.path.dirname(__file__), "..", "tap", "perturbations.py")
_perturb_ns = {"np": np, "__builtins__": __builtins__}
with open(_perturb_path) as f:
    _perturb_source = f.read()
_sample_start = _perturb_source.find("def sample_patch_xy(")
_sample_end = _perturb_source.find("\ndef ", _sample_start + 1)
if _sample_end == -1:
    _sample_end = len(_perturb_source)
exec(_perturb_source[_sample_start:_sample_end], _perturb_ns)
sample_patch_xy = _perturb_ns["sample_patch_xy"]

# Extract the pure functions by exec-ing them with numpy/math in scope
_ns = {"np": np, "math": math, "sample_patch_xy": sample_patch_xy, "__builtins__": __builtins__}
for func_name in ["perturb_single_obs", "stack_history", "_ensure_float_chw", "compute_progress"]:
    # Find the function in source
    start = _source.find(f"\ndef {func_name}(")
    if start == -1:
        start = _source.find(f"def {func_name}(")
    # Find the next function definition or class at the same indent level
    next_def = _source.find("\ndef ", start + 1)
    if next_def == -1:
        next_def = len(_source)
    func_source = _source[start:next_def]
    exec(func_source, _ns)

perturb_single_obs = _ns["perturb_single_obs"]
stack_history = _ns["stack_history"]
_ensure_float_chw = _ns["_ensure_float_chw"]
compute_progress = _ns["compute_progress"]


# ── fixtures ──────────────────────────────────────────────────

@pytest.fixture
def fake_obs():
    """(T=2, C=3, H=96, W=96) float32 observation window, values in [0, 1]."""
    rng = np.random.default_rng(0)
    return rng.random((2, 3, 96, 96), dtype=np.float32)


@pytest.fixture
def white_obs():
    """All-ones observation for easy visual inspection of occlusion."""
    return np.ones((2, 3, 96, 96), dtype=np.float32)


# ── perturb_single_obs ───────────────────────────────────────

class TestPerturbSingleObs:

    def test_prob_zero_never_perturbs(self, fake_obs):
        rng = np.random.default_rng(99)
        out = perturb_single_obs(fake_obs, "occlusion", rng, prob=0.0)
        assert out is fake_obs

    def test_prob_one_always_perturbs(self, white_obs):
        rng = np.random.default_rng(0)
        out = perturb_single_obs(white_obs, "occlusion", rng, patch_size=24, prob=1.0)
        assert out.min() == 0.0
        assert out is not white_obs

    def test_occlusion_zeros_a_patch(self, white_obs):
        rng = np.random.default_rng(7)
        out = perturb_single_obs(white_obs, "occlusion", rng, patch_size=24, prob=1.0)
        n_zeroed = (out == 0.0).sum()
        expected = 24 * 24 * 2 * 3  # patch_size^2 * T * C
        assert n_zeroed == expected

    def test_occlusion_patch_same_across_timesteps(self, white_obs):
        rng = np.random.default_rng(7)
        out = perturb_single_obs(white_obs, "occlusion", rng, patch_size=24, prob=1.0)
        mask_t0 = out[0] == 0.0
        mask_t1 = out[1] == 0.0
        np.testing.assert_array_equal(mask_t0, mask_t1)

    def test_noise_stays_in_range(self, fake_obs):
        rng = np.random.default_rng(0)
        out = perturb_single_obs(fake_obs, "noise", rng, prob=1.0)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_noise_changes_pixels(self, fake_obs):
        rng = np.random.default_rng(0)
        out = perturb_single_obs(fake_obs, "noise", rng, prob=1.0)
        assert not np.array_equal(out, fake_obs)

    def test_does_not_mutate_input(self, fake_obs):
        original = fake_obs.copy()
        rng = np.random.default_rng(0)
        perturb_single_obs(fake_obs, "occlusion", rng, prob=1.0)
        np.testing.assert_array_equal(fake_obs, original)

    def test_both_applies_occlusion_and_noise(self, white_obs):
        rng = np.random.default_rng(0)
        out = perturb_single_obs(white_obs, "both", rng, patch_size=16, prob=1.0)
        assert (out == 0.0).any()
        non_zero_mask = out > 0.0
        assert not np.all(out[non_zero_mask] == 1.0)

    def test_patch_size_respected(self, white_obs):
        for ps in [8, 16, 32, 48]:
            rng = np.random.default_rng(0)
            out = perturb_single_obs(white_obs, "occlusion", rng, patch_size=ps, prob=1.0)
            n_zeroed_per_frame = (out[0] == 0.0).sum()
            expected = ps * ps * 3
            assert n_zeroed_per_frame == expected, f"patch_size={ps}"


# ── xy override (episode mode) ────────────────────────────────

class TestXYOverride:
    """When xy is provided, patch goes to that exact position."""

    def test_xy_places_patch_at_exact_position(self):
        obs = np.ones((2, 3, 96, 96), dtype=np.float32)
        rng = np.random.default_rng(0)
        out = perturb_single_obs(obs, "occlusion", rng, patch_size=10, prob=1.0, xy=(5, 10))
        # Patch at x=5, y=10, size=10 → zeros at [10:20, 5:15]
        assert out[0, 0, 10, 5] == 0.0
        assert out[0, 0, 19, 14] == 0.0
        # Just outside should be 1
        assert out[0, 0, 9, 5] == 1.0
        assert out[0, 0, 10, 4] == 1.0

    def test_xy_overrides_rng_position(self):
        """Different RNG seeds but same xy → same patch position."""
        obs = np.ones((2, 3, 96, 96), dtype=np.float32)
        results = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            out = perturb_single_obs(obs, "occlusion", rng, patch_size=16, prob=1.0, xy=(20, 30))
            results.append((out[0, 0] == 0.0))
        for r in results[1:]:
            np.testing.assert_array_equal(results[0], r)

    def test_xy_none_samples_from_rng(self):
        """xy=None (default) falls back to RNG-based sampling."""
        obs = np.ones((2, 3, 96, 96), dtype=np.float32)
        rng = np.random.default_rng(42)
        out = perturb_single_obs(obs, "occlusion", rng, patch_size=24, prob=1.0, xy=None)
        assert (out == 0.0).any()


class TestSamplePatchXY:

    def test_returns_valid_coords(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            x, y = sample_patch_xy(96, 96, 24, 24, rng)
            assert 0 <= x <= 96 - 24
            assert 0 <= y <= 96 - 24

    def test_deterministic(self):
        r1 = sample_patch_xy(96, 96, 24, 24, np.random.default_rng(42))
        r2 = sample_patch_xy(96, 96, 24, 24, np.random.default_rng(42))
        assert r1 == r2

    def test_varies_with_seed(self):
        coords = set()
        for seed in range(20):
            coords.add(sample_patch_xy(96, 96, 24, 24, np.random.default_rng(seed)))
        assert len(coords) > 1


# ── shared perturbation invariant (THE critical fix) ──────────

class TestSharedPerturbation:
    """All K branches must see the SAME coin flip and patch position."""

    def test_same_seed_same_coin_flip(self):
        obs = np.ones((2, 3, 96, 96), dtype=np.float32)
        results = []
        for _ in range(8):  # simulate K=8 branches
            rng = np.random.default_rng(42)
            out = perturb_single_obs(obs, "occlusion", rng, prob=0.5)
            results.append(out is obs)  # True if not perturbed
        assert len(set(results)) == 1, "Branches disagree on perturb-or-not!"

    def test_same_seed_same_patch_position(self):
        imgs = [np.random.default_rng(i).random((2, 3, 96, 96), dtype=np.float32)
                for i in range(4)]
        patches = []
        for img in imgs:
            rng = np.random.default_rng(42)
            out = perturb_single_obs(img, "occlusion", rng, patch_size=24, prob=1.0)
            mask = (out[0, 0] == 0.0)
            patches.append(mask)
        for i, p in enumerate(patches[1:], 1):
            np.testing.assert_array_equal(
                patches[0], p,
                err_msg=f"Branch 0 vs branch {i} have different patch positions"
            )

    def test_different_seed_different_outcome(self):
        obs = np.ones((2, 3, 96, 96), dtype=np.float32)
        masks = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            out = perturb_single_obs(obs, "occlusion", rng, patch_size=24, prob=1.0)
            masks.append((out[0, 0] == 0.0))
        unique = len(set(m.tobytes() for m in masks))
        assert unique > 1


# ── RNG seeding for decision points ──────────────────────────

class TestRNGSeeding:

    def test_same_seed_step_reproduces(self):
        obs = np.ones((2, 3, 96, 96), dtype=np.float32)
        base_seed = 42
        seed, step = 5, 7
        results = []
        for _ in range(3):
            rng = np.random.default_rng(base_seed + seed * 10000 + step)
            out = perturb_single_obs(obs, "occlusion", rng, prob=1.0)
            results.append(out.copy())
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[1], results[2])

    def test_different_steps_differ(self):
        obs = np.ones((2, 3, 96, 96), dtype=np.float32)
        base_seed, seed = 42, 3
        outputs = []
        for step in range(5):
            rng = np.random.default_rng(base_seed + seed * 10000 + step)
            out = perturb_single_obs(obs, "occlusion", rng, prob=1.0)
            outputs.append(out.copy())
        diffs = sum(not np.array_equal(outputs[0], outputs[i]) for i in range(1, 5))
        assert diffs >= 1

    def test_decision_and_continuation_seeds_dont_collide(self):
        """For practical param ranges, decision and continuation seeds never collide."""
        base = 42
        # skip_first=2, max ~25 policy steps, 20 episodes
        practical_decision = {base + s * 10000 + p
                              for s in range(20) for p in range(2, 25)}
        practical_cont = {base + s * 10000 + p * 100
                          for s in range(20) for p in range(2, 25)}
        assert len(practical_decision & practical_cont) == 0


# ── stack_history ─────────────────────────────────────────────

class TestStackHistory:

    def test_exact_length(self):
        hist = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        out = stack_history(hist, T=3)
        np.testing.assert_array_equal(out, [[1.0], [2.0], [3.0]])

    def test_longer_than_T_takes_last(self):
        hist = [np.array([i]) for i in range(10)]
        out = stack_history(hist, T=3)
        np.testing.assert_array_equal(out, [[7], [8], [9]])

    def test_shorter_than_T_pads_first(self):
        hist = [np.array([5.0])]
        out = stack_history(hist, T=3)
        np.testing.assert_array_equal(out, [[5.0], [5.0], [5.0]])

    def test_two_of_three(self):
        hist = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        out = stack_history(hist, T=3)
        expected = [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]
        np.testing.assert_array_equal(out, expected)

    def test_T_equals_one(self):
        hist = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        out = stack_history(hist, T=1)
        np.testing.assert_array_equal(out, [[3.0]])


# ── _ensure_float_chw ────────────────────────────────────────

class TestEnsureFloatCHW:

    def test_uint8_to_float(self):
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        img[0, 0, 0] = 255
        out = _ensure_float_chw(img)
        assert out.dtype == np.float32
        assert out.max() == pytest.approx(1.0)

    def test_hwc_to_chw(self):
        img = np.zeros((96, 96, 3), dtype=np.float32)
        out = _ensure_float_chw(img)
        assert out.shape == (3, 96, 96)

    def test_already_chw_noop(self):
        img = np.zeros((3, 96, 96), dtype=np.float32)
        out = _ensure_float_chw(img)
        assert out.shape == (3, 96, 96)

    def test_already_float_no_rescale(self):
        img = np.full((96, 96, 3), 0.5, dtype=np.float32)
        out = _ensure_float_chw(img)
        assert out.max() == pytest.approx(0.5)


# ── compute_progress ─────────────────────────────────────────

class TestComputeProgress:

    def test_at_goal(self):
        pose = np.array([100.0, 200.0, 1.5])
        assert compute_progress(pose, pose) == pytest.approx(1.0)

    def test_far_away(self):
        pose = np.array([0.0, 0.0, 0.0])
        goal = np.array([362.0, 0.0, math.pi])
        assert compute_progress(pose, goal) == pytest.approx(0.0)

    def test_symmetric(self):
        a = np.array([50.0, 50.0, 0.5])
        b = np.array([100.0, 100.0, 1.0])
        assert compute_progress(a, b) == pytest.approx(compute_progress(b, a))

    def test_between_zero_and_one(self):
        pose = np.array([50.0, 50.0, 0.5])
        goal = np.array([200.0, 200.0, 2.0])
        assert 0.0 <= compute_progress(pose, goal) <= 1.0


# ── analysis math ─────────────────────────────────────────────

class TestAnalysisMath:
    """Test the capture ratio / harms counting logic from the analysis section."""

    @staticmethod
    def _compute_metrics(r_notap, r_tap, r_oracle):
        r_notap = np.array(r_notap, dtype=float)
        r_tap = np.array(r_tap, dtype=float)
        r_oracle = np.array(r_oracle, dtype=float)

        headroom = r_oracle - r_notap
        tap_gain = r_tap - r_notap

        has_headroom = headroom > 0.001
        if has_headroom.sum() > 0:
            capture_ratio = tap_gain[has_headroom] / headroom[has_headroom]
            capture_ratio = np.clip(capture_ratio, 0, 1)
            mean_capture = float(capture_ratio.mean())
        else:
            mean_capture = 0.0

        tap_beats = int((r_tap > r_notap + 0.001).sum())
        tap_harms = int((r_tap < r_notap - 0.001).sum())
        return {
            "headroom": headroom,
            "tap_gain": tap_gain,
            "mean_capture": mean_capture,
            "tap_beats": tap_beats,
            "tap_harms": tap_harms,
            "net_benefit": tap_beats - tap_harms,
        }

    def test_perfect_tap(self):
        m = self._compute_metrics([0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
        assert m["mean_capture"] == pytest.approx(1.0)
        assert m["tap_beats"] == 3
        assert m["tap_harms"] == 0

    def test_useless_tap(self):
        m = self._compute_metrics([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.5, 0.6, 0.7])
        assert m["mean_capture"] == pytest.approx(0.0)
        assert m["tap_beats"] == 0
        assert m["tap_harms"] == 0

    def test_harmful_tap_clipped(self):
        m = self._compute_metrics([0.5], [0.1], [0.8])
        assert m["mean_capture"] == pytest.approx(0.0)  # clipped to 0
        assert m["tap_gain"].item() == pytest.approx(-0.4)  # honest
        assert m["tap_harms"] == 1
        assert m["net_benefit"] == -1

    def test_no_headroom(self):
        m = self._compute_metrics([0.5, 0.5], [0.5, 0.5], [0.5, 0.5])
        assert m["mean_capture"] == 0.0
        assert m["net_benefit"] == 0

    def test_mixed_helps_and_harms(self):
        m = self._compute_metrics(
            [0.3, 0.3, 0.3, 0.3],
            [0.5, 0.5, 0.1, 0.1],
            [0.7, 0.7, 0.7, 0.7],
        )
        assert m["tap_beats"] == 2
        assert m["tap_harms"] == 2
        assert m["net_benefit"] == 0

    def test_half_capture(self):
        m = self._compute_metrics([0.0], [0.5], [1.0])
        assert m["mean_capture"] == pytest.approx(0.5)

"""Tests for tap/perturbations.py — verify rng, xy, fill_mode, alpha, temporal.

Run:  python -m pytest tests/test_perturbations.py -v
"""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tap.perturbations import apply_occlusion, apply_gaussian_noise, sample_patch_xy


@pytest.fixture
def white_4d():
    """(T=2, C=3, H=96, W=96) all-ones."""
    return np.ones((2, 3, 96, 96), dtype=np.float32)


@pytest.fixture
def white_5d():
    """(B=4, T=2, C=3, H=96, W=96) all-ones."""
    return np.ones((4, 2, 3, 96, 96), dtype=np.float32)


@pytest.fixture
def rand_4d():
    """Random (T=2, C=3, H=96, W=96) for mean-fill tests."""
    return np.random.default_rng(0).random((2, 3, 96, 96), dtype=np.float32)


# ── sample_patch_xy ─────────────────────────────────────────

class TestSamplePatchXY:

    def test_returns_valid_coords(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            x, y = sample_patch_xy(96, 96, 24, 24, rng)
            assert 0 <= x <= 96 - 24
            assert 0 <= y <= 96 - 24

    def test_inclusive_upper_bound(self):
        """Off-by-one fix: max position H-patch_h must be reachable."""
        # With H=25, patch_h=24, the only valid y values are 0 and 1
        found_max = False
        rng = np.random.default_rng(0)
        for _ in range(500):
            _, y = sample_patch_xy(25, 25, 24, 24, rng)
            if y == 1:
                found_max = True
                break
        assert found_max, "y=1 (H-patch_h) was never sampled — off-by-one bug"

    def test_deterministic(self):
        r1 = sample_patch_xy(96, 96, 24, 24, np.random.default_rng(42))
        r2 = sample_patch_xy(96, 96, 24, 24, np.random.default_rng(42))
        assert r1 == r2

    def test_varies_with_seed(self):
        coords = set()
        for seed in range(20):
            coords.add(sample_patch_xy(96, 96, 24, 24, np.random.default_rng(seed)))
        assert len(coords) > 1

    def test_rectangular_patch(self):
        rng = np.random.default_rng(0)
        x, y = sample_patch_xy(96, 96, 30, 10, rng)
        assert 0 <= y <= 96 - 30
        assert 0 <= x <= 96 - 10

    def test_patch_equals_image_returns_zero(self):
        """When patch fills the whole image, (0, 0) is the only option."""
        rng = np.random.default_rng(0)
        x, y = sample_patch_xy(96, 96, 96, 96, rng)
        assert (x, y) == (0, 0)

    def test_patch_larger_than_image_returns_zero(self):
        """When patch > image, clamped to (0, 0)."""
        rng = np.random.default_rng(0)
        x, y = sample_patch_xy(96, 96, 200, 200, rng)
        assert (x, y) == (0, 0)


# ── apply_occlusion: basic (black fill, static) ────────────

class TestApplyOcclusion:

    def test_basic_black(self, white_4d):
        out = apply_occlusion(white_4d, patch_size=20, fill_mode="black")
        assert (out == 0).any()
        assert out is not white_4d  # returns a copy

    def test_rng_deterministic(self, white_4d):
        r1 = apply_occlusion(white_4d, patch_size=20, rng=np.random.default_rng(42))
        r2 = apply_occlusion(white_4d, patch_size=20, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_rng_different_seeds(self, white_4d):
        r1 = apply_occlusion(white_4d, patch_size=20, rng=np.random.default_rng(0))
        r2 = apply_occlusion(white_4d, patch_size=20, rng=np.random.default_rng(1))
        assert not np.array_equal(r1, r2)

    def test_xy_explicit(self, white_4d):
        out = apply_occlusion(white_4d, patch_size=10, xy=(5, 10), fill_mode="black")
        assert out[0, 0, 10, 5] == 0.0
        assert out[0, 0, 19, 14] == 0.0  # corner of 10x10 patch at (5, 10)
        assert out[0, 0, 9, 5] == 1.0    # outside
        assert out[0, 0, 10, 4] == 1.0

    def test_5d(self, white_5d):
        out = apply_occlusion(white_5d, patch_size=20, rng=np.random.default_rng(0),
                              fill_mode="black")
        assert (out == 0).any()
        assert out.shape == white_5d.shape

    def test_no_crash_patch_larger_than_image(self, white_4d):
        """patch_size > H should not crash, just clamp."""
        out = apply_occlusion(white_4d, patch_size=200, rng=np.random.default_rng(0),
                              fill_mode="black")
        # Entire image zeroed
        assert (out == 0).all()

    def test_xy_clamped_when_out_of_bounds(self, white_4d):
        """xy far outside image gets clamped, no crash."""
        out = apply_occlusion(white_4d, patch_size=20, xy=(999, 999), fill_mode="black")
        assert (out == 0).any()
        assert out.shape == white_4d.shape


# ── apply_occlusion: fill modes ────────────────────────────

class TestFillModes:

    def test_mean_fill_on_constant_image(self, white_4d):
        """On all-ones image, mean fill = ones → no visible change."""
        out = apply_occlusion(white_4d, patch_size=20, rng=np.random.default_rng(0),
                              fill_mode="mean")
        np.testing.assert_allclose(out, white_4d, atol=1e-6)

    def test_mean_fill_on_random_image(self, rand_4d):
        """Mean fill should produce pixels near the channel means."""
        out = apply_occlusion(rand_4d, patch_size=40, rng=np.random.default_rng(0),
                              fill_mode="mean")
        # Something changed
        assert not np.array_equal(out, rand_4d)

    def test_random_fill_uses_uniform_color(self, white_4d):
        out = apply_occlusion(white_4d, patch_size=20, rng=np.random.default_rng(0),
                              fill_mode="random")
        # Occluded region should have a single color per channel
        # (not per-pixel variation like noise)
        # Find the occluded region by comparing with original
        diff_mask = (out[0, 0] != white_4d[0, 0])
        if diff_mask.any():
            occluded_vals = out[0, 0, diff_mask]
            assert len(set(occluded_vals.tolist())) == 1  # single value per channel

    def test_noise_fill_varies_per_pixel(self, white_4d):
        out = apply_occlusion(white_4d, patch_size=30, rng=np.random.default_rng(0),
                              fill_mode="noise")
        diff_mask = (out[0, 0] != white_4d[0, 0])
        if diff_mask.any():
            occluded_vals = out[0, 0, diff_mask]
            assert len(set(occluded_vals.tolist())) > 1  # multiple distinct values

    def test_invalid_fill_mode_raises(self, white_4d):
        with pytest.raises(ValueError, match="Unknown fill_mode"):
            apply_occlusion(white_4d, fill_mode="sparkle")


# ── apply_occlusion: alpha blending ────────────────────────

class TestAlphaBlending:

    def test_alpha_zero_is_noop(self, white_4d):
        """alpha=0 → no change (fully transparent occluder)."""
        out = apply_occlusion(white_4d, patch_size=20, rng=np.random.default_rng(0),
                              fill_mode="black", alpha=0.0)
        np.testing.assert_array_equal(out, white_4d)

    def test_alpha_half_blends(self, white_4d):
        """alpha=0.5, black fill on ones → occluded region should be ~0.5."""
        out = apply_occlusion(white_4d, patch_size=20, xy=(10, 10),
                              fill_mode="black", alpha=0.5)
        # Occluded pixel should be 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert out[0, 0, 10, 10] == pytest.approx(0.5)

    def test_alpha_one_fully_opaque(self, white_4d):
        """alpha=1.0, black fill → zeros."""
        out = apply_occlusion(white_4d, patch_size=20, xy=(10, 10),
                              fill_mode="black", alpha=1.0)
        assert out[0, 0, 10, 10] == 0.0


# ── apply_occlusion: temporal modes ────────────────────────

class TestTemporalModes:

    def test_static_same_across_frames(self, white_4d):
        out = apply_occlusion(white_4d, patch_size=20, rng=np.random.default_rng(0),
                              fill_mode="black", temporal="static")
        mask_t0 = (out[0] == 0.0)
        mask_t1 = (out[1] == 0.0)
        np.testing.assert_array_equal(mask_t0, mask_t1)

    def test_per_frame_differs_across_frames(self, white_4d):
        out = apply_occlusion(white_4d, patch_size=20, rng=np.random.default_rng(0),
                              fill_mode="black", temporal="per_frame")
        mask_t0 = (out[0] == 0.0)
        mask_t1 = (out[1] == 0.0)
        # Very likely different positions
        assert not np.array_equal(mask_t0, mask_t1)

    def test_drift_changes_slightly(self, white_4d):
        out = apply_occlusion(white_4d, patch_size=20, rng=np.random.default_rng(0),
                              fill_mode="black", temporal="drift", max_drift=5)
        mask_t0 = (out[0] == 0.0)
        mask_t1 = (out[1] == 0.0)
        # Drift is small, so masks overlap but may differ
        overlap = (mask_t0 & mask_t1).sum()
        total = mask_t0.sum()
        # Most pixels should overlap (drift is only a few pixels)
        assert overlap > total * 0.5

    def test_invalid_temporal_raises(self, white_4d):
        with pytest.raises(ValueError, match="Unknown temporal"):
            apply_occlusion(white_4d, temporal="teleport")

    def test_static_5d(self, white_5d):
        out = apply_occlusion(white_5d, patch_size=20, rng=np.random.default_rng(0),
                              fill_mode="black", temporal="static")
        # All batch items and frames should have same occlusion
        mask_b0_t0 = (out[0, 0] == 0.0)
        mask_b1_t1 = (out[1, 1] == 0.0)
        np.testing.assert_array_equal(mask_b0_t0, mask_b1_t1)


# ── apply_occlusion: aspect ratio ──────────────────────────

class TestAspectRatio:

    def test_wide_patch(self, white_4d):
        """aspect_ratio > 1 → wider patch."""
        out = apply_occlusion(white_4d, patch_size=20,
                              rng=np.random.default_rng(0),
                              fill_mode="black",
                              aspect_ratio_range=(2.0, 2.0))
        # patch_h=20, patch_w=40
        n_zeroed = (out[0, 0] == 0.0).sum()
        assert n_zeroed == 20 * 40

    def test_tall_patch(self, white_4d):
        """aspect_ratio < 1 → narrower patch."""
        out = apply_occlusion(white_4d, patch_size=20,
                              rng=np.random.default_rng(0),
                              fill_mode="black",
                              aspect_ratio_range=(0.5, 0.5))
        # patch_h=20, patch_w=round(20*0.5)=10
        n_zeroed = (out[0, 0] == 0.0).sum()
        assert n_zeroed == 20 * 10

    def test_invalid_aspect_ratio_raises(self, white_4d):
        with pytest.raises(ValueError, match="aspect_ratio_range"):
            apply_occlusion(white_4d, aspect_ratio_range=(2.0, 1.0))  # min > max
        with pytest.raises(ValueError, match="aspect_ratio_range"):
            apply_occlusion(white_4d, aspect_ratio_range=(-1.0, 1.0))  # negative


# ── apply_gaussian_noise (unchanged API) ───────────────────

class TestApplyGaussianNoise:

    def test_basic(self, white_4d):
        out = apply_gaussian_noise(white_4d, std=0.1)
        assert not np.array_equal(out, white_4d)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_rng_deterministic(self, white_4d):
        r1 = apply_gaussian_noise(white_4d, std=0.1, rng=np.random.default_rng(42))
        r2 = apply_gaussian_noise(white_4d, std=0.1, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_rng_different_seeds(self, white_4d):
        r1 = apply_gaussian_noise(white_4d, std=0.1, rng=np.random.default_rng(0))
        r2 = apply_gaussian_noise(white_4d, std=0.1, rng=np.random.default_rng(1))
        assert not np.array_equal(r1, r2)

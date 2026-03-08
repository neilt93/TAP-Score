"""
State restore sanity test for PushT.

Tests the invariant that matters for counterfactual branching:
  Two fresh envs restored to the same saved state and stepped with
  the same actions produce identical outcomes.

pymunk's collision solver keeps hidden arbiter state (cached impulses
for warm-starting) that cannot be captured from Python.  Restoring into
a *fresh* env avoids inheriting stale solver caches, so all branches
start from an identical clean state.

Usage:
    python scripts/test_state_restore.py
"""
from __future__ import annotations

import numpy as np
import sys

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from tap.env_wrapper import PushTStateWrapper


def make_env():
    return PushTStateWrapper(PushTImageEnv(legacy=False, render_size=96))


def states_match(s1: dict, s2: dict, atol: float = 1e-10) -> bool:
    ok = True
    for key in ["agent_position", "agent_velocity", "block_position", "block_velocity"]:
        if not np.allclose(s1[key], s2[key], atol=atol):
            print(f"  MISMATCH {key}: {s1[key]} vs {s2[key]}, diff={np.abs(s1[key]-s2[key])}")
            ok = False
    for key in ["block_angle", "block_angular_velocity"]:
        if abs(s1[key] - s2[key]) > atol:
            print(f"  MISMATCH {key}: {s1[key]} vs {s2[key]}, diff={abs(s1[key]-s2[key])}")
            ok = False
    return ok


def test_deterministic_restore(seed: int = 42, N: int = 10, M: int = 8):
    """Two fresh envs restored to the same state must evolve identically."""
    # Build up some physics history to get a non-trivial save point
    env_src = make_env()
    env_src.seed(seed)
    env_src.reset()

    rng = np.random.default_rng(seed + 1000)
    for _ in range(N):
        action = rng.uniform(50, 450, size=2).astype(np.float64)
        env_src.step(action)

    saved = env_src.get_state()

    # Generate M actions to replay on both branches
    actions = [rng.uniform(50, 450, size=2).astype(np.float64) for _ in range(M)]

    # Branch A: fresh env
    env_a = make_env()
    env_a.seed(seed)
    env_a.reset()
    env_a.set_state(saved)
    for action in actions:
        env_a.step(action)
    end_A = env_a.get_state()

    # Branch B: another fresh env
    env_b = make_env()
    env_b.seed(seed)
    env_b.reset()
    env_b.set_state(saved)
    for action in actions:
        env_b.step(action)
    end_B = env_b.get_state()

    return states_match(end_A, end_B)


def test_branch_divergence(seed: int = 42, N: int = 10, M: int = 8):
    """Two different action sequences from the same saved state must diverge."""
    env_src = make_env()
    env_src.seed(seed)
    env_src.reset()

    rng = np.random.default_rng(seed + 2000)
    for _ in range(N):
        action = rng.uniform(50, 450, size=2).astype(np.float64)
        env_src.step(action)

    saved = env_src.get_state()

    # Branch A
    env_a = make_env()
    env_a.seed(seed)
    env_a.reset()
    env_a.set_state(saved)
    for _ in range(M):
        action = rng.uniform(50, 450, size=2).astype(np.float64)
        env_a.step(action)
    end_A = env_a.get_state()

    # Branch B (different actions)
    env_b = make_env()
    env_b.seed(seed)
    env_b.reset()
    env_b.set_state(saved)
    rng_b = np.random.default_rng(seed + 3000)
    for _ in range(M):
        action = rng_b.uniform(50, 450, size=2).astype(np.float64)
        env_b.step(action)
    end_B = env_b.get_state()

    diverged = not states_match(end_A, end_B, atol=0.01)
    return diverged


def test_multiple_seeds(n_seeds: int = 10, N: int = 10, M: int = 8):
    """Run deterministic restore across many seeds."""
    passed = 0
    for s in range(n_seeds):
        ok = test_deterministic_restore(seed=s, N=N, M=M)
        if ok:
            passed += 1
        else:
            print(f"  FAILED seed={s}")
    return passed, n_seeds


def main():
    print("=" * 60)
    print("PushT State Restore Sanity Tests (fresh-env pool)")
    print("=" * 60)

    print("\n[1/3] Deterministic restore (two fresh envs, same state, same actions)...")
    ok = test_deterministic_restore()
    print(f"  {'PASS' if ok else 'FAIL'}")

    print("\n[2/3] Branch divergence (same start, different actions must differ)...")
    div = test_branch_divergence()
    print(f"  {'PASS' if div else 'FAIL — branches did not diverge'}")

    print("\n[3/3] Multi-seed deterministic restore (10 seeds)...")
    passed, total = test_multiple_seeds()
    print(f"  {passed}/{total} passed")

    all_ok = ok and div and (passed == total)
    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

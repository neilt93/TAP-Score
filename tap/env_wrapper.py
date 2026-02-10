"""
PushT environment wrapper with full state save/restore.

Adds get_state() and set_state() to PushTImageEnv, capturing all
physics state (positions + velocities) needed for counterfactual branching.

Note: pymunk's collision solver maintains hidden arbiter state (cached
impulses for warm-starting) that cannot be saved/restored via the Python
API.  For deterministic counterfactual branching, always restore into a
*fresh* env instance so the solver starts from a clean cache.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class PushTStateWrapper:
    """Wraps PushTImageEnv with state save/restore for counterfactual branching.

    Usage:
        env = PushTImageEnv(legacy=False, render_size=96)
        env = PushTStateWrapper(env)

        obs = env.reset()
        ...
        state = env.get_state()
        ...  # do stuff
        env.set_state(state)
        # env is now back to the saved state
    """

    def __init__(self, env):
        self.env = env

    def get_state(self) -> Dict[str, Any]:
        """Snapshot the full physics state (positions + velocities)."""
        e = self.env
        return {
            "agent_position": np.array(e.agent.position, dtype=np.float64),
            "agent_velocity": np.array(e.agent.velocity, dtype=np.float64),
            "block_position": np.array(e.block.position, dtype=np.float64),
            "block_angle": float(e.block.angle),
            "block_velocity": np.array(e.block.velocity, dtype=np.float64),
            "block_angular_velocity": float(e.block.angular_velocity),
            "n_contact_points": int(e.n_contact_points),
            "latest_action": np.array(e.latest_action, dtype=np.float64) if e.latest_action is not None else None,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore a previously saved state. No physics step is taken."""
        e = self.env
        e.agent.position = tuple(state["agent_position"])
        e.agent.velocity = tuple(state["agent_velocity"])
        # Set angle before position (non-legacy mode) to avoid CoG shift
        e.block.angle = state["block_angle"]
        e.block.position = tuple(state["block_position"])
        e.block.velocity = tuple(state["block_velocity"])
        e.block.angular_velocity = state["block_angular_velocity"]
        e.n_contact_points = state["n_contact_points"]
        e.latest_action = state["latest_action"]

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __getattr__(self, name):
        return getattr(self.env, name)

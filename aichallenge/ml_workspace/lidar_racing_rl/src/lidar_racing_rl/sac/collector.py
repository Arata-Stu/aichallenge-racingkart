"""Ego-only transition assembly at the simulator information boundary."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from lidar_racing_rl.sac.replay import TransitionBatch


def transition_from_step(
    observation: jax.Array,
    normalized_action: jax.Array,
    step_result: Any,
) -> TransitionBatch:
    """Build a replay batch from a vectorized LiDAR-only environment step.

    ``step_result.observation`` may already be the next episode's reset
    observation.  Replay must instead use ``transition_next_observation``,
    which is the final observation associated with the reward and done flags.
    """

    batch_size = observation.shape[0]
    if normalized_action.shape != (batch_size, 2):
        raise ValueError("normalized_action must have shape [num_envs, 2]")
    if step_result.transition_next_observation.shape != observation.shape:
        raise ValueError("transition next observation shape changed across the step")
    for name in ("reward", "terminated", "truncated"):
        if getattr(step_result, name).shape != (batch_size,):
            raise ValueError(f"step_result.{name} must have shape [num_envs]")
    return TransitionBatch(
        observation=observation.astype(jnp.float32),
        action=normalized_action.astype(jnp.float32),
        reward=step_result.reward.astype(jnp.float32),
        terminated=step_result.terminated.astype(jnp.bool_),
        truncated=step_result.truncated.astype(jnp.bool_),
        next_observation=step_result.transition_next_observation.astype(jnp.float32),
    )


__all__ = ["transition_from_step"]

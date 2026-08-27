"""Reward terms for the ego-only LiDAR racing task.

The functions in this module consume simulator ground truth only for reward
calculation.  Their outputs are scalars and none of their inputs may be joined
to the Actor or Critic observation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def wrapped_progress_delta(
    previous_s: jax.Array, current_s: jax.Array, track_length: jax.Array
) -> jax.Array:
    """Return signed closed-track progress in ``[-length/2, length/2)``."""

    half = 0.5 * track_length
    return jnp.mod(current_s - previous_s + half, track_length) - half


def step1_reward(
    previous_s: jax.Array,
    current_s: jax.Array,
    previous_action: jax.Array,
    action: jax.Array,
    *,
    track_length: jax.Array,
    collision: jax.Array,
    off_track: jax.Array,
    reversing: jax.Array,
    progress_weight: float,
    collision_weight: float,
    off_track_weight: float,
    smoothness_weight: float,
    reverse_weight: float,
) -> jax.Array:
    """Compute the Step-1 reward for one ego transition."""

    progress = wrapped_progress_delta(previous_s, current_s, track_length)
    action_delta = jnp.sum(jnp.square(action - previous_action))
    progress = jnp.where(jnp.isfinite(progress), progress, 0.0)
    action_delta = jnp.where(jnp.isfinite(action_delta), action_delta, 0.0)
    return (
        progress_weight * progress
        - collision_weight * collision.astype(jnp.float32)
        - off_track_weight * off_track.astype(jnp.float32)
        - smoothness_weight * action_delta
        - reverse_weight * reversing.astype(jnp.float32)
    )


def step2_reward(
    base_reward: jax.Array,
    *,
    relative_progress: jax.Array,
    pass_events: jax.Array,
    unsafe_contact: jax.Array,
    stalled_behind_vehicle: jax.Array,
    relative_progress_weight: float,
    pass_weight: float,
    unsafe_contact_weight: float,
    stalled_behind_weight: float,
) -> jax.Array:
    """Add Step-2 opponent terms to one Ego transition reward.

    ``pass_events`` has shape ``[opponents]``; every other signal is scalar.
    The caller derives all signals from simulator ground truth at the reward
    boundary.  Only the resulting scalar may flow into replay.
    """

    finite_relative_progress = jnp.where(
        jnp.isfinite(relative_progress), relative_progress, 0.0
    )
    pass_count = jnp.sum(jnp.asarray(pass_events, dtype=jnp.float32))
    return (
        base_reward
        + relative_progress_weight * finite_relative_progress
        + pass_weight * pass_count
        - unsafe_contact_weight * unsafe_contact.astype(jnp.float32)
        - stalled_behind_weight * stalled_behind_vehicle.astype(jnp.float32)
    )


__all__ = ["step1_reward", "step2_reward", "wrapped_progress_delta"]

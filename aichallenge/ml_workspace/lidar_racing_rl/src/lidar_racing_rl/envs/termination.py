"""Separated termination and time-limit truncation semantics."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def update_episode_progress(
    episode_progress: jax.Array,
    progress_delta: jax.Array,
    *,
    track_length: jax.Array,
    max_num_laps: int,
) -> tuple[jax.Array, jax.Array]:
    """Accumulate net forward distance and detect a full lap from reset.

    The simulator lap counter is tied to the map start line.  Because this
    wrapper randomizes the reset anchor, crossing that line is not equivalent
    to completing a lap from the episode's own starting point.
    """

    finite_delta = jnp.where(jnp.isfinite(progress_delta), progress_delta, 0.0)
    next_progress = episode_progress + finite_delta
    race_complete = next_progress >= (jnp.asarray(max_num_laps) * track_length)
    return next_progress, race_complete


def ego_done_flags(
    *,
    collision: jax.Array,
    off_track: jax.Array,
    race_complete: jax.Array,
    unrecoverable: jax.Array,
    step_count: jax.Array,
    max_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(terminated, truncated)`` for the ego vehicle.

    A time limit is truncation only, so downstream SAC target calculation can
    retain bootstrap on that transition.
    """

    terminated = collision | off_track | race_complete | unrecoverable
    truncated = (~terminated) & (step_count >= jnp.asarray(max_steps))
    return terminated, truncated

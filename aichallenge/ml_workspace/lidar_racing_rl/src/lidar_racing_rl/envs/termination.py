"""Separated termination and time-limit truncation semantics."""

from __future__ import annotations

import jax
import jax.numpy as jnp


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

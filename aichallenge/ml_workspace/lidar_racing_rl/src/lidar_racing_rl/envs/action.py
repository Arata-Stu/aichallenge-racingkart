"""Conversion between normalized policy actions and simulator commands."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def scale_normalized_action(
    action: jax.Array,
    *,
    max_steering_angle: float,
    min_acceleration: float,
    max_acceleration: float,
) -> jax.Array:
    """Map ``[..., 2]`` actions from ``[-1, 1]`` to physical commands."""

    clipped = jnp.clip(action, -1.0, 1.0)
    steering = clipped[..., 0] * max_steering_angle
    acceleration = min_acceleration + 0.5 * (clipped[..., 1] + 1.0) * (
        max_acceleration - min_acceleration
    )
    return jnp.stack((steering, acceleration), axis=-1)


def normalize_physical_action(
    action: jax.Array,
    *,
    max_steering_angle: float,
    min_acceleration: float,
    max_acceleration: float,
) -> jax.Array:
    """Map physical ``[..., 2]`` commands back to the policy's ``[-1, 1]`` space."""

    steering = action[..., 0] / max_steering_angle
    acceleration = 2.0 * (
        (action[..., 1] - min_acceleration)
        / (max_acceleration - min_acceleration)
    ) - 1.0
    return jnp.clip(jnp.stack((steering, acceleration), axis=-1), -1.0, 1.0)


__all__ = ["normalize_physical_action", "scale_normalized_action"]

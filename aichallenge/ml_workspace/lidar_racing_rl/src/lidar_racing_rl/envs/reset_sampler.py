"""JAX reset sampling for one ego vehicle and three fixed NPCs."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp


def sample_four_vehicle_frenet(
    key: jax.Array,
    track_length: jax.Array,
    *,
    longitudinal_spacing: float,
    lateral_jitter: float,
    heading_jitter: float,
) -> jax.Array:
    """Sample non-overlapping Frenet poses with shape ``[4, 3]``.

    Ground truth is used only to initialize the simulator.  Vehicle order is
    randomized by rotating a fixed, safely-spaced grid around the closed track.
    """

    if not math.isfinite(longitudinal_spacing) or longitudinal_spacing <= 0.0:
        raise ValueError("longitudinal_spacing must be positive")
    if (
        not math.isfinite(lateral_jitter)
        or not math.isfinite(heading_jitter)
        or lateral_jitter < 0.0
        or heading_jitter < 0.0
    ):
        raise ValueError("reset jitter bounds cannot be negative")

    anchor_key, lateral_key, heading_key = jax.random.split(key, 3)
    anchor = jax.random.uniform(anchor_key, minval=0.0, maxval=track_length)
    offsets = longitudinal_spacing * jnp.arange(4, dtype=jnp.float32)
    s = jnp.mod(anchor + offsets, track_length)
    lateral = jax.random.uniform(
        lateral_key,
        shape=(4,),
        minval=-lateral_jitter,
        maxval=lateral_jitter,
    )
    heading = jax.random.uniform(
        heading_key,
        shape=(4,),
        minval=-heading_jitter,
        maxval=heading_jitter,
    )
    return jnp.stack((s, lateral, heading), axis=-1)


def replace_with_frenet_reset(
    base_state: Any,
    track: Any,
    winding_point: jax.Array,
    frenet_states: jax.Array,
) -> Any:
    """Replace an upstream F1TENTH state with externally sampled poses.

    ``track`` and ``base_state`` are deliberately duck-typed to keep the parent
    project independent of private upstream classes.  This is intended to run
    inside a method where the environment object is static under ``jax.jit``.
    """

    if frenet_states.shape != (4, 3):
        raise ValueError("frenet_states must have shape [4, 3]")

    poses = track.vmap_frenet_to_cartesian_jax(frenet_states)
    cartesian = jnp.zeros_like(base_state.cartesian_states)
    cartesian = cartesian.at[:, jnp.array([0, 1, 4])].set(poses)
    zeros_agents = jnp.zeros((4,), dtype=cartesian.dtype)

    return base_state.replace(
        rewards=zeros_agents,
        done=jnp.zeros((4,), dtype=bool),
        step=0,
        cartesian_states=cartesian,
        last_cartesian_states=cartesian,
        frenet_states=frenet_states,
        last_frenet_states=frenet_states,
        num_laps=jnp.zeros((4,), dtype=jnp.int32),
        collisions=jnp.zeros((4,), dtype=bool),
        scans=jnp.zeros_like(base_state.scans),
        prev_winding_vector=cartesian[:, 0:2] - winding_point,
        accumulated_angles=zeros_agents,
        last_accumulated_angles=zeros_agents,
    )

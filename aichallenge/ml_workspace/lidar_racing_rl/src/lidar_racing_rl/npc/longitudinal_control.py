"""GT-only longitudinal safety controller for fixed NPC vehicles."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def limit_speed_for_leading_vehicle(
    npc_states: jax.Array,
    all_vehicle_states: jax.Array,
    npc_indices: jax.Array,
    waypoint_target_speed: jax.Array,
    safe_distance: jax.Array,
    *,
    distance_gain: float,
    lateral_gate: float,
    minimum_speed: float = 0.0,
) -> jax.Array:
    """Limit each NPC target speed when another vehicle is directly ahead.

    This function uses simulator ground truth solely for NPC control, which is
    allowed by the project information boundary.  Inputs are shaped
    ``npc_states=[npcs,state_dim]``, ``all_vehicle_states=[vehicles,state_dim]``,
    and all remaining one-dimensional arguments are ``[npcs]``.
    """

    if (
        npc_states.ndim != 2
        or all_vehicle_states.ndim != 2
        or npc_states.shape[1] < 5
        or all_vehicle_states.shape[1] < 5
    ):
        raise ValueError("vehicle states must have shape [vehicles, state_dim>=5]")
    npc_count = npc_states.shape[0]
    if npc_indices.shape != (npc_count,):
        raise ValueError("npc_indices must have shape [npcs]")
    if waypoint_target_speed.shape != (npc_count,):
        raise ValueError("waypoint_target_speed must have shape [npcs]")
    if safe_distance.shape != (npc_count,):
        raise ValueError("safe_distance must have shape [npcs]")
    if not jnp.issubdtype(npc_indices.dtype, jnp.integer):
        raise ValueError("npc_indices must contain integers")
    if distance_gain < 0.0 or lateral_gate <= 0.0:
        raise ValueError("distance_gain must be non-negative and lateral_gate positive")

    relative_world = (
        all_vehicle_states[None, :, 0:2] - npc_states[:, None, 0:2]
    )
    yaw = npc_states[:, 4:5]
    cos_yaw = jnp.cos(yaw)
    sin_yaw = jnp.sin(yaw)
    longitudinal = (
        relative_world[..., 0] * cos_yaw + relative_world[..., 1] * sin_yaw
    )
    lateral = (
        -relative_world[..., 0] * sin_yaw + relative_world[..., 1] * cos_yaw
    )

    vehicle_indices = jnp.arange(all_vehicle_states.shape[0])[None, :]
    is_self = vehicle_indices == npc_indices[:, None]
    is_candidate = (
        (~is_self) & (longitudinal > 0.0) & (jnp.abs(lateral) < lateral_gate)
    )
    candidate_distance = jnp.where(is_candidate, longitudinal, jnp.inf)
    lead_index = jnp.argmin(candidate_distance, axis=1)
    lead_distance = jnp.min(candidate_distance, axis=1)
    has_leader = jnp.isfinite(lead_distance)

    lead_speed = all_vehicle_states[lead_index, 3]
    following_speed = lead_speed + distance_gain * (lead_distance - safe_distance)
    limited = jnp.where(
        has_leader,
        jnp.minimum(waypoint_target_speed, following_speed),
        waypoint_target_speed,
    )
    return jnp.maximum(limited, minimum_speed)

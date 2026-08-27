"""Vectorized Pure Pursuit controller for fixed NPC vehicles.

Ground-truth poses are intentionally accepted here.  NPC control is one of the
explicitly permitted uses of simulator ground truth; this module must never be
called while constructing the SAC Actor or Critic observation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def ordered_braking_target_speeds(
    cartesian_states: jax.Array,
    vehicle_waypoints: jax.Array,
    speed_multiplier: jax.Array,
    *,
    maximum_deceleration: float,
) -> jax.Array:
    """Return speeds that can brake for every upcoming ordered speed limit."""

    vehicle_count = cartesian_states.shape[0]
    if vehicle_waypoints.ndim != 3 or vehicle_waypoints.shape[0] != vehicle_count:
        raise ValueError("vehicle_waypoints must have shape [vehicles, waypoints, 3+]")
    if speed_multiplier.shape != (vehicle_count,):
        raise ValueError("speed_multiplier must have shape [vehicles]")
    if maximum_deceleration <= 0.0:
        raise ValueError("maximum_deceleration must be positive")

    offsets = vehicle_waypoints[..., 0:2] - cartesian_states[:, None, 0:2]
    distances = jnp.linalg.norm(offsets, axis=-1)
    nearest_index = jnp.argmin(distances, axis=1)
    waypoint_count = vehicle_waypoints.shape[1]
    ordered_indices = (
        nearest_index[:, None] + jnp.arange(waypoint_count)[None, :]
    ) % waypoint_count
    ordered_xy = jnp.take_along_axis(
        vehicle_waypoints[..., 0:2],
        ordered_indices[..., None],
        axis=1,
    )
    ordered_speed = jnp.take_along_axis(
        vehicle_waypoints[..., 2],
        ordered_indices,
        axis=1,
    ) * speed_multiplier[:, None]
    first_distance = jnp.linalg.norm(
        ordered_xy[:, 0] - cartesian_states[:, 0:2], axis=-1
    )
    segment_lengths = jnp.linalg.norm(jnp.diff(ordered_xy, axis=1), axis=-1)
    path_distances = jnp.concatenate(
        (
            first_distance[:, None],
            first_distance[:, None] + jnp.cumsum(segment_lengths, axis=1),
        ),
        axis=1,
    )
    feasible_speed = jnp.sqrt(
        jnp.maximum(
            jnp.square(ordered_speed)
            + 2.0 * maximum_deceleration * path_distances,
            0.0,
        )
    )
    return jnp.min(feasible_speed, axis=1)


def pure_pursuit_actions(
    cartesian_states: jax.Array,
    waypoints: jax.Array,
    lookahead: jax.Array,
    speed_multiplier: jax.Array,
    *,
    wheelbase: float,
    control_dt: float,
    steering_min: float,
    steering_max: float,
    acceleration_min: float,
    acceleration_max: float,
) -> jax.Array:
    """Return steering-angle and acceleration commands for a vehicle batch.

    Parameters
    ----------
    cartesian_states:
        ``[vehicles, state_dim]`` F1TENTH states.  Indices ``0, 1, 2, 3,
        4`` are interpreted as x, y, steering angle, velocity, and yaw.
    waypoints:
        Shared ``[waypoints, 3+]`` or per-vehicle
        ``[vehicles, waypoints, 3+]`` rows containing x, y, and target speed.
    lookahead, speed_multiplier:
        Per-vehicle arrays with shape ``[vehicles]``.

    Returns
    -------
    jax.Array
        ``[vehicles, 2]`` commands ordered as steering angle, acceleration.

    Notes
    -----
    Waypoint and vehicle axes are handled by broadcasting.  There is no Python
    loop over vehicles or waypoints, so the function can be nested under
    ``jax.vmap`` for the environment axis.
    """

    if cartesian_states.ndim != 2 or cartesian_states.shape[1] < 5:
        raise ValueError("cartesian_states must have shape [vehicles, state_dim>=5]")
    vehicle_count = cartesian_states.shape[0]
    lookahead = jnp.asarray(lookahead)
    speed_multiplier = jnp.asarray(speed_multiplier)
    if lookahead.shape != (vehicle_count,):
        raise ValueError("lookahead must have shape [vehicles]")
    if speed_multiplier.shape != (vehicle_count,):
        raise ValueError("speed_multiplier must have shape [vehicles]")
    if wheelbase <= 0.0 or control_dt <= 0.0:
        raise ValueError("wheelbase and control_dt must be positive")
    if steering_min >= steering_max:
        raise ValueError("steering bounds must be ordered")
    if acceleration_min >= acceleration_max:
        raise ValueError("acceleration bounds must be ordered")
    if waypoints.ndim == 2:
        if waypoints.shape[0] < 1 or waypoints.shape[1] < 3:
            raise ValueError("waypoints must have shape [waypoints>=1, 3+]")
        vehicle_waypoints = jnp.broadcast_to(
            waypoints[jnp.newaxis, ...],
            (cartesian_states.shape[0], *waypoints.shape),
        )
    elif waypoints.ndim == 3:
        if (
            waypoints.shape[0] != cartesian_states.shape[0]
            or waypoints.shape[1] < 1
            or waypoints.shape[2] < 3
        ):
            raise ValueError(
                "per-vehicle waypoints must have shape "
                "[vehicles, waypoints>=1, 3+]"
            )
        vehicle_waypoints = waypoints
    else:
        raise ValueError("waypoints must have shape [waypoints, 3+] or [vehicles, ...]")

    positions = cartesian_states[:, None, 0:2]
    offsets = vehicle_waypoints[..., 0:2] - positions
    yaw = cartesian_states[:, 4:5]
    cos_yaw = jnp.cos(yaw)
    sin_yaw = jnp.sin(yaw)

    local_y = -offsets[..., 0] * sin_yaw + offsets[..., 1] * cos_yaw
    distances = jnp.linalg.norm(offsets, axis=-1)

    # Preserve the configured closed-loop waypoint order. Selecting from every
    # geometrically forward point can jump to a nearby but topologically
    # unrelated track segment. Start at the nearest waypoint and walk only in
    # the reference-line direction until the lookahead radius is reached.
    nearest_index = jnp.argmin(distances, axis=1)
    waypoint_count = vehicle_waypoints.shape[1]
    ordered_indices = (
        nearest_index[:, None] + jnp.arange(waypoint_count)[None, :]
    ) % waypoint_count
    ordered_distances = jnp.take_along_axis(distances, ordered_indices, axis=1)
    reached_lookahead = ordered_distances >= lookahead[:, None]
    first_reached_offset = jnp.argmax(reached_lookahead, axis=1)
    fallback_offset = jnp.full_like(first_reached_offset, waypoint_count - 1)
    target_offset = jnp.where(
        jnp.any(reached_lookahead, axis=1),
        first_reached_offset,
        fallback_offset,
    )
    target_index = jnp.take_along_axis(
        ordered_indices,
        target_offset[:, None],
        axis=1,
    )[:, 0]

    target_y = jnp.take_along_axis(local_y, target_index[:, None], axis=1)[:, 0]
    target_distance = jnp.take_along_axis(
        distances, target_index[:, None], axis=1
    )[:, 0]
    target_distance = jnp.maximum(target_distance, 1.0e-3)

    curvature = 2.0 * target_y / jnp.square(target_distance)
    steering = jnp.arctan(wheelbase * curvature)
    steering = jnp.clip(steering, steering_min, steering_max)

    waypoint_speed = ordered_braking_target_speeds(
        cartesian_states,
        vehicle_waypoints,
        speed_multiplier,
        maximum_deceleration=abs(acceleration_min),
    )
    acceleration = (waypoint_speed - cartesian_states[:, 3]) / control_dt
    acceleration = jnp.clip(acceleration, acceleration_min, acceleration_max)

    return jnp.stack((steering, acceleration), axis=-1)


__all__ = ["ordered_braking_target_speeds", "pure_pursuit_actions"]

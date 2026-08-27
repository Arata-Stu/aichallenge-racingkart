"""Composed JAX controller for the three fixed opponent vehicles.

Simulator ground truth is intentionally consumed here only for NPC control.
Neither the inputs nor the controller state are part of the SAC Actor or Critic
observation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from lidar_racing_rl.npc.longitudinal_control import (
    limit_speed_for_leading_vehicle,
)
from lidar_racing_rl.npc.pure_pursuit import (
    ordered_braking_target_speeds,
    pure_pursuit_actions,
)
from lidar_racing_rl.npc.randomization import (
    NpcEpisodeParameters,
    apply_braking_event,
    offset_waypoint_lines,
    select_delayed_actions,
)


@struct.dataclass
class NpcControllerState:
    """Per-environment command history.

    ``action_history`` has shape ``[npcs, max_delay + 1, 2]`` and is ordered
    newest first.  Keeping this state outside the racing environment makes the
    information boundary explicit: it belongs to the fixed NPC policy, not the
    learned Ego observation.
    """

    action_history: jax.Array


def initialize_npc_controller_state(
    *,
    npc_count: int,
    max_control_delay_steps: int,
) -> NpcControllerState:
    """Return a zero-command history for one environment.

    Both arguments are static shape parameters.  Batch this function at reset
    time, or broadcast the returned pytree, for an environment axis.
    """

    if npc_count < 1:
        raise ValueError("npc_count must be positive")
    if max_control_delay_steps < 0:
        raise ValueError("max_control_delay_steps cannot be negative")
    return NpcControllerState(
        action_history=jnp.zeros(
            (npc_count, max_control_delay_steps + 1, 2),
            dtype=jnp.float32,
        )
    )


def _validate_shapes(
    all_vehicle_states: jax.Array,
    base_waypoints: jax.Array,
    npc_indices: jax.Array,
    parameters: NpcEpisodeParameters,
    controller_state: NpcControllerState,
) -> None:
    if all_vehicle_states.ndim != 2 or all_vehicle_states.shape[1] < 5:
        raise ValueError("all_vehicle_states must have shape [vehicles, state_dim>=5]")
    if base_waypoints.ndim != 2 or base_waypoints.shape[0] < 1:
        raise ValueError("base_waypoints must have shape [waypoints>=1, 3+]")
    if base_waypoints.shape[1] < 3:
        raise ValueError("base_waypoints must have shape [waypoints>=1, 3+]")
    if npc_indices.ndim != 1:
        raise ValueError("npc_indices must have shape [npcs]")

    npc_count = npc_indices.shape[0]
    for field_name in NpcEpisodeParameters.__dataclass_fields__:
        field_value = getattr(parameters, field_name)
        if field_value.shape != (npc_count,):
            raise ValueError(f"parameters.{field_name} must have shape [npcs]")
    if not jnp.issubdtype(npc_indices.dtype, jnp.integer):
        raise ValueError("npc_indices must contain integers")
    if not jnp.issubdtype(parameters.control_delay_steps.dtype, jnp.integer):
        raise ValueError("parameters.control_delay_steps must contain integers")
    expected_history_prefix = (npc_count,)
    history = controller_state.action_history
    if (
        history.ndim != 3
        or history.shape[0:1] != expected_history_prefix
        or history.shape[1] < 1
        or history.shape[2] != 2
    ):
        raise ValueError(
            "controller_state.action_history must have shape [npcs, history>=1, 2]"
        )


def npc_controller_step(
    all_vehicle_states: jax.Array,
    base_waypoints: jax.Array,
    npc_indices: jax.Array,
    parameters: NpcEpisodeParameters,
    controller_state: NpcControllerState,
    step: jax.Array,
    *,
    wheelbase: float,
    control_dt: float,
    steering_min: float,
    steering_max: float,
    acceleration_min: float,
    acceleration_max: float,
    distance_gain: float,
    lateral_gate: float,
    minimum_speed: float = 0.0,
) -> tuple[jax.Array, NpcControllerState]:
    """Compose all fixed-NPC behavior for one environment.

    Parameters use these shapes: ``all_vehicle_states=[vehicles,state_dim]``,
    ``base_waypoints=[waypoints,3+]``, ``npc_indices=[npcs]``, every field of
    ``parameters=[npcs]``, and command history ``[npcs,history,2]``.  The result
    is physical ``[npcs,2]`` steering-angle/acceleration commands plus the next
    history state.  Environment batching is supported with ``jax.vmap``.

    The NPC and waypoint axes are processed with broadcasting only.  The order
    is waypoint-line and episode parameter randomization, Pure Pursuit, GT-only
    safe following, scripted braking, then per-NPC command delay.
    """

    _validate_shapes(
        all_vehicle_states,
        base_waypoints,
        npc_indices,
        parameters,
        controller_state,
    )
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")
    if steering_min >= steering_max:
        raise ValueError("steering bounds must be ordered")
    if acceleration_min >= acceleration_max:
        raise ValueError("acceleration bounds must be ordered")
    if distance_gain < 0.0 or lateral_gate <= 0.0:
        raise ValueError("following gains and gates must be non-negative")

    npc_states = all_vehicle_states[npc_indices]
    waypoint_lines = offset_waypoint_lines(
        base_waypoints,
        parameters.lateral_offset,
    )
    nominal_actions = pure_pursuit_actions(
        npc_states,
        waypoint_lines,
        parameters.lookahead,
        parameters.speed_multiplier,
        wheelbase=wheelbase,
        control_dt=control_dt,
        steering_min=steering_min,
        steering_max=steering_max,
        acceleration_min=acceleration_min,
        acceleration_max=acceleration_max,
    )

    waypoint_target_speed = ordered_braking_target_speeds(
        npc_states,
        waypoint_lines,
        parameters.speed_multiplier,
        maximum_deceleration=abs(acceleration_min),
    )
    safe_target_speed = limit_speed_for_leading_vehicle(
        npc_states,
        all_vehicle_states,
        npc_indices,
        waypoint_target_speed,
        parameters.safe_distance,
        distance_gain=distance_gain,
        lateral_gate=lateral_gate,
        minimum_speed=minimum_speed,
    )

    steering = jnp.clip(
        nominal_actions[:, 0] * parameters.steering_gain,
        steering_min,
        steering_max,
    )
    acceleration = (
        (safe_target_speed - npc_states[:, 3])
        / control_dt
        * parameters.acceleration_gain
    )
    acceleration = jnp.clip(acceleration, acceleration_min, acceleration_max)
    acceleration = apply_braking_event(acceleration, step, parameters)
    acceleration = jnp.clip(acceleration, acceleration_min, acceleration_max)
    undelayed_actions = jnp.stack((steering, acceleration), axis=-1)

    next_history = jnp.concatenate(
        (
            undelayed_actions[:, None, :],
            controller_state.action_history[:, :-1, :],
        ),
        axis=1,
    )
    delayed_actions = select_delayed_actions(
        next_history,
        parameters.control_delay_steps,
    )
    return delayed_actions, NpcControllerState(action_history=next_history)


__all__ = [
    "NpcControllerState",
    "initialize_npc_controller_state",
    "npc_controller_step",
]

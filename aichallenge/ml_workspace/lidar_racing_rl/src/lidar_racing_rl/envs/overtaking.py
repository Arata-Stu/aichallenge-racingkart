"""GT-only relative-progress and pass-event helpers for Step 2.

The arrays produced here are reward/evaluation signals.  They must never be
concatenated to the LiDAR observation stored in replay.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import struct

from lidar_racing_rl.envs.reward import wrapped_progress_delta


@struct.dataclass
class OvertakingState:
    """Per-opponent pass hysteresis state for one environment."""

    armed_from_behind: jax.Array
    consecutive_ahead_steps: jax.Array
    cooldown_steps_remaining: jax.Array


def signed_opponent_gaps(
    ego_s: jax.Array,
    opponent_s: jax.Array,
    track_length: jax.Array,
) -> jax.Array:
    """Return signed Ego-minus-opponent gaps on a closed track.

    Negative means Ego is behind the opponent and positive means Ego is ahead.
    """

    return wrapped_progress_delta(opponent_s, ego_s, track_length)


def initialize_overtaking_state(
    ego_s: jax.Array,
    opponent_s: jax.Array,
    track_length: jax.Array,
    *,
    behind_distance: float,
) -> OvertakingState:
    """Arm only opponents that start at least ``behind_distance`` ahead."""

    if not math.isfinite(behind_distance) or behind_distance < 0.0:
        raise ValueError("behind_distance must be finite and non-negative")
    opponents = jnp.asarray(opponent_s)
    if opponents.ndim != 1:
        raise ValueError("opponent_s must have shape [opponents]")
    gaps = signed_opponent_gaps(ego_s, opponents, track_length)
    return OvertakingState(
        armed_from_behind=gaps <= -behind_distance,
        consecutive_ahead_steps=jnp.zeros(opponents.shape, dtype=jnp.int32),
        cooldown_steps_remaining=jnp.zeros(opponents.shape, dtype=jnp.int32),
    )


def update_overtaking_state(
    state: OvertakingState,
    ego_s: jax.Array,
    opponent_s: jax.Array,
    track_length: jax.Array,
    *,
    behind_distance: float,
    ahead_distance: float,
    hold_steps: int,
    cooldown_steps: int,
) -> tuple[OvertakingState, jax.Array, jax.Array]:
    """Advance pass hysteresis without an opponent-axis Python loop.

    A pass is emitted once an armed Ego remains at least ``ahead_distance`` in
    front for ``hold_steps`` consecutive transitions.  The same opponent can
    only re-arm after cooldown and after Ego is genuinely behind again.
    """

    for name, value in (
        ("behind_distance", behind_distance),
        ("ahead_distance", ahead_distance),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if isinstance(hold_steps, bool) or hold_steps < 1:
        raise ValueError("hold_steps must be a positive integer")
    if isinstance(cooldown_steps, bool) or cooldown_steps < 0:
        raise ValueError("cooldown_steps must be a non-negative integer")

    opponents = jnp.asarray(opponent_s)
    expected_shape = opponents.shape
    if opponents.ndim != 1:
        raise ValueError("opponent_s must have shape [opponents]")
    for name, value in (
        ("armed_from_behind", state.armed_from_behind),
        ("consecutive_ahead_steps", state.consecutive_ahead_steps),
        ("cooldown_steps_remaining", state.cooldown_steps_remaining),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must match opponent_s shape")

    gaps = signed_opponent_gaps(ego_s, opponents, track_length)
    remaining_cooldown = jnp.maximum(state.cooldown_steps_remaining - 1, 0)
    can_rearm = remaining_cooldown == 0
    armed = state.armed_from_behind | (
        can_rearm & (gaps <= -behind_distance)
    )
    ahead = armed & can_rearm & (gaps >= ahead_distance)
    ahead_steps = jnp.where(
        ahead,
        state.consecutive_ahead_steps + 1,
        jnp.zeros_like(state.consecutive_ahead_steps),
    )
    pass_events = ahead_steps >= hold_steps
    next_state = OvertakingState(
        armed_from_behind=jnp.where(pass_events, False, armed),
        consecutive_ahead_steps=jnp.where(pass_events, 0, ahead_steps),
        cooldown_steps_remaining=jnp.where(
            pass_events,
            jnp.asarray(cooldown_steps, dtype=jnp.int32),
            remaining_cooldown,
        ),
    )
    return next_state, pass_events, gaps


def nearest_opponent_relative_progress(
    previous_ego_s: jax.Array,
    current_ego_s: jax.Array,
    previous_opponent_s: jax.Array,
    current_opponent_s: jax.Array,
    track_length: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return Ego progress minus the previously nearest forward opponent.

    The second result is the current forward arc gap to the nearest opponent.
    Empty opponent arrays return two zeros so Step 1 uses the same JIT shape
    contract without fabricating an opponent.
    """

    previous = jnp.asarray(previous_opponent_s)
    current = jnp.asarray(current_opponent_s)
    if previous.ndim != 1 or current.shape != previous.shape:
        raise ValueError("opponent progress arrays must have the same [opponents] shape")
    if previous.shape[0] == 0:
        zero = jnp.asarray(0.0, dtype=jnp.result_type(previous_ego_s, jnp.float32))
        return zero, zero

    forward_gaps = jnp.mod(previous - previous_ego_s, track_length)
    nearest_index = jnp.argmin(forward_gaps)
    ego_delta = wrapped_progress_delta(previous_ego_s, current_ego_s, track_length)
    opponent_delta = wrapped_progress_delta(
        previous[nearest_index],
        current[nearest_index],
        track_length,
    )
    current_forward_gap = jnp.mod(
        current[nearest_index] - current_ego_s,
        track_length,
    )
    return ego_delta - opponent_delta, current_forward_gap


def minimum_opponent_distance(
    ego_xy: jax.Array,
    opponent_xy: jax.Array,
    *,
    no_opponent_value: float = 0.0,
) -> jax.Array:
    """Return the nearest center distance without an opponent-axis loop."""

    ego = jnp.asarray(ego_xy)
    opponents = jnp.asarray(opponent_xy)
    if ego.shape != (2,) or opponents.ndim != 2 or opponents.shape[1] != 2:
        raise ValueError("expected ego_xy=[2] and opponent_xy=[opponents, 2]")
    if opponents.shape[0] == 0:
        return jnp.asarray(no_opponent_value, dtype=jnp.result_type(ego, jnp.float32))
    return jnp.min(jnp.linalg.norm(opponents - ego, axis=-1))


def ego_opponent_obb_overlaps(
    cartesian_states: jax.Array,
    *,
    vehicle_length: float,
    vehicle_width: float,
) -> jax.Array:
    """Return exact planar OBB overlaps between Ego and every opponent.

    F1TENTH cartesian state uses x/y at indices 0/1 and yaw at index 4. The
    separating-axis test is broadcast across opponents; no vehicle loop is
    introduced. Touching boxes count as contact.
    """

    raw_states = jnp.asarray(cartesian_states)
    dtype = jnp.result_type(
        raw_states,
        vehicle_length,
        vehicle_width,
        jnp.float32,
    )
    states = jnp.asarray(raw_states, dtype=dtype)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 5:
        raise ValueError("cartesian_states must have shape [agents>=1, state_dim>=5]")
    if (
        not math.isfinite(vehicle_length)
        or not math.isfinite(vehicle_width)
        or vehicle_length <= 0.0
        or vehicle_width <= 0.0
    ):
        raise ValueError("vehicle dimensions must be finite and positive")

    yaws = states[:, 4]
    cos_yaw = jnp.cos(yaws)
    sin_yaw = jnp.sin(yaws)
    axes = jnp.stack(
        (
            jnp.stack((cos_yaw, sin_yaw), axis=-1),
            jnp.stack((-sin_yaw, cos_yaw), axis=-1),
        ),
        axis=-1,
    )
    ego_axes = axes[0]
    opponent_axes = axes[1:]
    rotation = jnp.einsum("ki,nkj->nij", ego_axes, opponent_axes)
    absolute_rotation = jnp.abs(rotation) + jnp.asarray(1.0e-7, states.dtype)
    center_delta_world = states[1:, 0:2] - states[0, 0:2]
    center_delta_ego = jnp.einsum("ki,nk->ni", ego_axes, center_delta_world)
    half_extents = jnp.asarray(
        (0.5 * vehicle_length, 0.5 * vehicle_width),
        dtype=states.dtype,
    )

    separated_on_ego_axes = jnp.abs(center_delta_ego) > (
        half_extents
        + jnp.einsum("nij,j->ni", absolute_rotation, half_extents)
    )
    center_delta_opponent = jnp.einsum(
        "ni,nij->nj",
        center_delta_ego,
        rotation,
    )
    separated_on_opponent_axes = jnp.abs(center_delta_opponent) > (
        half_extents
        + jnp.einsum("i,nij->nj", half_extents, absolute_rotation)
    )
    overlaps = ~(
        jnp.any(separated_on_ego_axes, axis=-1)
        | jnp.any(separated_on_opponent_axes, axis=-1)
    )
    finite_pose = jnp.all(
        jnp.isfinite(states[:, jnp.asarray([0, 1, 4])]),
        axis=-1,
    )
    return overlaps & finite_pose[0] & finite_pose[1:]


__all__ = [
    "OvertakingState",
    "ego_opponent_obb_overlaps",
    "initialize_overtaking_state",
    "minimum_opponent_distance",
    "nearest_opponent_relative_progress",
    "signed_opponent_gaps",
    "update_overtaking_state",
]

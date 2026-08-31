"""Integration contract for the composed fixed-NPC controller."""

import jax
import jax.numpy as jnp
import numpy as np

from lidar_racing_rl.npc.controller import (
    initialize_npc_controller_state,
    npc_controller_step,
)
from lidar_racing_rl.npc.randomization import NpcEpisodeParameters


def _parameters(
    *,
    delays: jax.Array | None = None,
    braking_enabled: jax.Array | None = None,
) -> NpcEpisodeParameters:
    return NpcEpisodeParameters(
        speed_multiplier=jnp.array([0.7, 0.9, 1.1]),
        lateral_offset=jnp.array([-0.25, 0.0, 0.25]),
        lookahead=jnp.array([2.0, 2.5, 3.0]),
        steering_gain=jnp.array([0.9, 1.0, 1.1]),
        acceleration_gain=jnp.array([0.8, 1.0, 1.2]),
        safe_distance=jnp.array([5.0, 4.0, 3.0]),
        control_delay_steps=(
            jnp.array([0, 1, 2]) if delays is None else delays
        ),
        braking_enabled=(
            jnp.array([False, True, False])
            if braking_enabled is None
            else braking_enabled
        ),
        braking_start_step=jnp.array([10, 10, 10]),
        braking_duration_steps=jnp.array([5, 5, 5]),
        braking_acceleration=jnp.array([-2.0, -2.0, -2.0]),
    )


def _waypoints() -> jax.Array:
    return jnp.array(
        [
            [-2.0, 0.0, 4.0],
            [0.0, 0.0, 4.0],
            [2.0, 0.0, 4.0],
            [4.0, 0.0, 4.0],
            [6.0, 0.0, 4.0],
        ],
        dtype=jnp.float32,
    )


def _step(
    all_states: jax.Array,
    parameters: NpcEpisodeParameters,
    state: object,
    step: jax.Array,
) -> tuple[jax.Array, object]:
    return npc_controller_step(
        all_states,
        _waypoints(),
        jnp.array([1, 2, 3]),
        parameters,
        state,
        step,
        wheelbase=1.087,
        control_dt=0.1,
        steering_min=-0.5,
        steering_max=0.5,
        acceleration_min=-3.2,
        acceleration_max=3.2,
        distance_gain=0.5,
        lateral_gate=1.5,
    )


def test_composed_controller_applies_following_braking_and_delay() -> None:
    # Ego is a slow leader four metres in front of NPC 1.  The other NPCs are
    # on separate lateral lines so they are not following that Ego vehicle.
    all_states = jnp.array(
        [
            [4.0, -0.25, 0.0, 1.0, 0.0],
            [0.0, -0.25, 0.0, 2.0, 0.0],
            [0.0, 4.0, 0.0, 2.0, 0.0],
            [0.0, -4.0, 0.0, 2.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    parameters = _parameters()
    initial_state = initialize_npc_controller_state(
        npc_count=3,
        max_control_delay_steps=2,
    )

    actions, next_state = jax.jit(_step)(
        all_states,
        parameters,
        initial_state,
        jnp.asarray(12),
    )

    assert actions.shape == (3, 2)
    assert next_state.action_history.shape == (3, 3, 2)
    # Delay zero exposes NPC 1's current safe-following deceleration.
    assert float(actions[0, 1]) < 0.0
    # NPCs with one/two-step delay start with the zero command history.
    np.testing.assert_allclose(actions[1:], 0.0, atol=1.0e-6)
    # Braking is present in the undelayed history before NPC 2's delay.
    np.testing.assert_allclose(next_state.action_history[1, 0, 1], -2.0)


def test_controller_is_vmap_compatible_and_parameters_change_commands() -> None:
    all_states = jnp.array(
        [
            [10.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, -0.25, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.25, 0.0, 1.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    parameters = _parameters(
        delays=jnp.zeros((3,), dtype=jnp.int32),
        braking_enabled=jnp.zeros((3,), dtype=bool),
    )
    state = initialize_npc_controller_state(
        npc_count=3,
        max_control_delay_steps=2,
    )
    lower_gain_parameters = parameters.replace(
        acceleration_gain=parameters.acceleration_gain * 0.1,
    )
    batched_states = jnp.stack((all_states, all_states))
    batched_parameters = jax.tree.map(
        lambda first, second: jnp.stack((first, second)),
        parameters,
        lower_gain_parameters,
    )
    batched_controller_state = jax.tree.map(
        lambda value: jnp.stack((value, value)),
        state,
    )

    actions, next_state = jax.jit(jax.vmap(_step))(
        batched_states,
        batched_parameters,
        batched_controller_state,
        jnp.array([0, 0]),
    )

    assert actions.shape == (2, 3, 2)
    assert next_state.action_history.shape == (2, 3, 3, 2)
    assert bool(jnp.all(jnp.isfinite(actions)))
    assert not bool(jnp.allclose(actions[0, :, 1], actions[1, :, 1]))


def test_delay_history_uses_each_npc_configured_step() -> None:
    all_states = jnp.array(
        [
            [10.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, -0.25, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.25, 0.0, 1.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    parameters = _parameters(braking_enabled=jnp.zeros((3,), dtype=bool))
    initial_state = initialize_npc_controller_state(
        npc_count=3,
        max_control_delay_steps=2,
    )

    first_actions, first_state = _step(
        all_states,
        parameters,
        initial_state,
        jnp.asarray(0),
    )
    second_actions, _ = _step(
        all_states,
        parameters,
        first_state,
        jnp.asarray(1),
    )

    assert not bool(jnp.allclose(first_actions[0], 0.0))
    np.testing.assert_allclose(first_actions[1:], 0.0, atol=1.0e-6)
    assert not bool(jnp.allclose(second_actions[1], 0.0))
    np.testing.assert_allclose(second_actions[2], 0.0, atol=1.0e-6)


def test_final_acceleration_cannot_drive_an_npc_through_zero_speed() -> None:
    all_states = jnp.array(
        [
            [0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.0],
            [0.0, 4.0, 0.0, 0.1, 0.0],
            [0.0, -4.0, 0.0, 0.1, 0.0],
        ],
        dtype=jnp.float32,
    )
    parameters = _parameters(
        delays=jnp.zeros((3,), dtype=jnp.int32),
        braking_enabled=jnp.ones((3,), dtype=bool),
    )
    state = initialize_npc_controller_state(
        npc_count=3,
        max_control_delay_steps=0,
    )

    actions, _ = _step(all_states, parameters, state, jnp.asarray(12))

    assert bool(jnp.all(actions[:, 1] >= -1.0))

"""Tests for the vectorized fixed-NPC controllers."""

import jax
import jax.numpy as jnp
import numpy as np

from lidar_racing_rl.npc.longitudinal_control import (
    limit_speed_for_leading_vehicle,
)
from lidar_racing_rl.npc.pure_pursuit import pure_pursuit_actions


def test_pure_pursuit_is_vmap_compatible() -> None:
    states = jnp.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.2, 0.0, 1.0, 0.0],
            [0.0, -0.2, 0.0, 1.0, 0.0],
        ]
    )
    waypoints = jnp.array([[1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [3.0, 0.0, 2.0]])

    def controller(batch_states: jax.Array) -> jax.Array:
        return pure_pursuit_actions(
            batch_states,
            waypoints,
            lookahead=jnp.full((3,), 2.0),
            speed_multiplier=jnp.ones((3,)),
            wheelbase=1.087,
            control_dt=0.1,
            steering_min=-0.5,
            steering_max=0.5,
            acceleration_min=-3.2,
            acceleration_max=3.2,
        )

    result = jax.vmap(controller)(jnp.stack((states, states)))
    assert result.shape == (2, 3, 2)
    assert bool(jnp.all(jnp.isfinite(result)))


def test_pure_pursuit_accepts_distinct_waypoint_line_per_npc() -> None:
    states = jnp.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0, 1.0, 0.0],
            [0.0, -0.5, 0.0, 1.0, 0.0],
        ]
    )
    base = jnp.array([[1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [3.0, 0.0, 2.0]])
    lines = jnp.stack(
        (
            base,
            base.at[:, 1].set(0.5),
            base.at[:, 1].set(-0.5),
        )
    )

    actions = pure_pursuit_actions(
        states,
        lines,
        lookahead=jnp.full((3,), 2.0),
        speed_multiplier=jnp.ones((3,)),
        wheelbase=1.087,
        control_dt=0.1,
        steering_min=-0.5,
        steering_max=0.5,
        acceleration_min=-3.2,
        acceleration_max=3.2,
    )

    assert actions.shape == (3, 2)
    np.testing.assert_allclose(actions[:, 0], 0.0, atol=1.0e-6)


def test_following_controller_slows_for_leader() -> None:
    all_states = jnp.array(
        [
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3.0, 0.0],
            [4.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 4.0, 0.0, 2.0, 0.0],
        ]
    )
    target = limit_speed_for_leading_vehicle(
        npc_states=all_states[jnp.array([1, 3])],
        all_vehicle_states=all_states,
        npc_indices=jnp.array([1, 3]),
        waypoint_target_speed=jnp.array([5.0, 5.0]),
        safe_distance=jnp.array([5.0, 5.0]),
        distance_gain=0.5,
        lateral_gate=1.5,
    )

    np.testing.assert_allclose(target, jnp.array([0.5, 5.0]), rtol=1.0e-6)

"""Tests for normalized action scaling."""

import jax.numpy as jnp
import numpy as np

from lidar_racing_rl.envs.action import (
    clamp_nonreversing_acceleration,
    normalize_physical_action,
    scale_normalized_action,
)


def test_action_scaling_endpoints() -> None:
    actions = jnp.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    actual = scale_normalized_action(
        actions,
        max_steering_angle=0.5,
        min_acceleration=-3.2,
        max_acceleration=3.2,
    )
    expected = jnp.array([[-0.5, -3.2], [0.0, 0.0], [0.5, 3.2]])
    np.testing.assert_allclose(actual, expected, atol=1.0e-6)


def test_physical_action_round_trip() -> None:
    normalized = jnp.array([[-1.0, -1.0], [-0.25, 0.4], [1.0, 1.0]])
    physical = scale_normalized_action(
        normalized,
        max_steering_angle=0.5,
        min_acceleration=-3.2,
        max_acceleration=3.2,
    )
    recovered = normalize_physical_action(
        physical,
        max_steering_angle=0.5,
        min_acceleration=-3.2,
        max_acceleration=3.2,
    )

    np.testing.assert_allclose(recovered, normalized, atol=1.0e-6)


def test_braking_is_clamped_before_crossing_zero_speed() -> None:
    action = jnp.asarray([0.2, -9.51])
    clamped = clamp_nonreversing_acceleration(
        action,
        jnp.asarray(0.1),
        control_dt=0.05,
    )
    np.testing.assert_allclose(clamped, jnp.asarray([0.2, -2.0]))

    stopped = clamp_nonreversing_acceleration(
        action,
        jnp.asarray(0.0),
        control_dt=0.05,
    )
    np.testing.assert_allclose(stopped, jnp.asarray([0.2, 0.0]))

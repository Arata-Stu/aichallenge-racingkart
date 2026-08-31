"""Tests for vectorized fixed-opponent diversity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lidar_racing_rl.npc.randomization import (
    NpcRandomizationBounds,
    apply_braking_event,
    offset_waypoint_lines,
    sample_npc_episode_parameters,
    select_delayed_actions,
)


BOUNDS = NpcRandomizationBounds(
    speed_multiplier=(0.65, 1.05),
    lateral_offset=(-0.5, 0.5),
    lookahead=(1.5, 4.0),
    steering_gain=(0.9, 1.1),
    acceleration_gain=(0.8, 1.2),
    safe_distance=(3.0, 6.0),
    control_delay_steps=(0, 2),
    braking_probability=1.0,
    braking_start_step=(10, 10),
    braking_duration_steps=(5, 5),
    braking_acceleration=-2.0,
)


def test_episode_parameters_are_independent_arrays_and_jittable() -> None:
    parameters = jax.jit(
        lambda key: sample_npc_episode_parameters(key, npc_count=3, bounds=BOUNDS)
    )(jax.random.key(4))

    assert parameters.speed_multiplier.shape == (3,)
    assert parameters.lateral_offset.shape == (3,)
    assert parameters.control_delay_steps.shape == (3,)
    assert bool(jnp.all(parameters.speed_multiplier >= 0.65))
    assert bool(jnp.all(parameters.speed_multiplier <= 1.05))
    assert bool(jnp.all(jnp.diff(parameters.speed_multiplier) >= 0.0))


def test_reset_spacing_includes_safe_distance_and_vehicle_length() -> None:
    BOUNDS.validate_reset_spacing(8.0, 0.58)

    with pytest.raises(ValueError, match="safe-following distance plus vehicle"):
        BOUNDS.validate_reset_spacing(6.5, 0.58)


def test_waypoint_offsets_create_separate_parallel_lines() -> None:
    waypoints = jnp.array(
        [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [3.0, 0.0, 2.0]]
    )
    lines = jax.jit(offset_waypoint_lines)(waypoints, jnp.array([-0.5, 0.0, 0.5]))

    assert lines.shape == (3, 4, 3)
    np.testing.assert_allclose(lines[1], waypoints, atol=1.0e-6)
    assert not bool(jnp.allclose(lines[0, :, :2], lines[2, :, :2]))


def test_braking_and_control_delay_are_selected_without_vehicle_loop() -> None:
    parameters = sample_npc_episode_parameters(
        jax.random.key(8), npc_count=3, bounds=BOUNDS
    )
    acceleration = apply_braking_event(jnp.ones((3,)), jnp.asarray(12), parameters)
    np.testing.assert_allclose(acceleration, -2.0)

    history = jnp.arange(18, dtype=jnp.float32).reshape(3, 3, 2)
    selected = select_delayed_actions(history, jnp.array([0, 1, 2]))
    np.testing.assert_allclose(
        selected,
        jnp.stack((history[0, 0], history[1, 1], history[2, 2])),
    )


def test_config_rejects_fractional_control_delay_instead_of_truncating() -> None:
    config = {
        "npc": {
            "randomization": {
                "speed_multiplier": {"min": 0.65, "max": 1.05},
                "lateral_offset": {"min": -0.5, "max": 0.5},
                "lookahead": {"min": 1.5, "max": 4.0},
                "steering_gain": {"min": 0.9, "max": 1.1},
                "acceleration_gain": {"min": 0.8, "max": 1.2},
                "control_delay_steps": {"min": 0.5, "max": 2},
                "braking_event": {
                    "probability": 0.15,
                    "start_step": {"min": 100, "max": 1200},
                    "duration_steps": {"min": 10, "max": 60},
                    "acceleration": -2.0,
                },
            },
            "longitudinal_controller": {
                "safe_following_distance": {"min": 3.0, "max": 6.0}
            },
        }
    }

    with pytest.raises(ValueError, match="must contain integers"):
        NpcRandomizationBounds.from_config(config)

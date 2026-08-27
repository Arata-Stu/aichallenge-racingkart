"""Tests for GT-only relative progress and pass hysteresis."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from lidar_racing_rl.envs.overtaking import (
    ego_opponent_obb_overlaps,
    initialize_overtaking_state,
    minimum_opponent_distance,
    nearest_opponent_relative_progress,
    update_overtaking_state,
)


def test_pass_requires_behind_then_sustained_ahead_and_cooldown() -> None:
    state = initialize_overtaking_state(
        jnp.asarray(0.0),
        jnp.asarray([4.0]),
        jnp.asarray(100.0),
        behind_distance=0.5,
    )

    state, first, _ = update_overtaking_state(
        state,
        jnp.asarray(5.0),
        jnp.asarray([4.0]),
        jnp.asarray(100.0),
        behind_distance=0.5,
        ahead_distance=1.0,
        hold_steps=2,
        cooldown_steps=3,
    )
    state, second, _ = update_overtaking_state(
        state,
        jnp.asarray(6.0),
        jnp.asarray([4.0]),
        jnp.asarray(100.0),
        behind_distance=0.5,
        ahead_distance=1.0,
        hold_steps=2,
        cooldown_steps=3,
    )
    assert not bool(first[0])
    assert bool(second[0])

    state, duplicate, _ = update_overtaking_state(
        state,
        jnp.asarray(7.0),
        jnp.asarray([4.0]),
        jnp.asarray(100.0),
        behind_distance=0.5,
        ahead_distance=1.0,
        hold_steps=2,
        cooldown_steps=3,
    )
    assert not bool(duplicate[0])


def test_closed_track_relative_progress_uses_nearest_forward_opponent() -> None:
    relative, gap = nearest_opponent_relative_progress(
        jnp.asarray(99.0),
        jnp.asarray(1.0),
        jnp.asarray([2.0, 20.0]),
        jnp.asarray([3.0, 22.0]),
        jnp.asarray(100.0),
    )
    np.testing.assert_allclose(relative, 1.0)
    np.testing.assert_allclose(gap, 2.0)


def test_helpers_are_jittable_and_handle_no_opponents() -> None:
    relative, gap = jax.jit(nearest_opponent_relative_progress)(
        jnp.asarray(1.0),
        jnp.asarray(2.0),
        jnp.empty((0,)),
        jnp.empty((0,)),
        jnp.asarray(100.0),
    )
    np.testing.assert_allclose(relative, 0.0)
    np.testing.assert_allclose(gap, 0.0)
    np.testing.assert_allclose(
        minimum_opponent_distance(
            jnp.asarray([0.0, 0.0]),
            jnp.asarray([[3.0, 4.0], [1.0, 0.0]]),
        ),
        1.0,
    )


def test_obb_contact_classifies_opponent_collisions_without_vehicle_loop() -> None:
    states = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.9, 0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.4, 0.0, 0.0, 0.5 * jnp.pi],
        ]
    )
    overlaps = jax.jit(
        lambda value: ego_opponent_obb_overlaps(
            value,
            vehicle_length=2.0,
            vehicle_width=1.45,
        )
    )(states)

    np.testing.assert_array_equal(overlaps, jnp.asarray([True, False, True]))


def test_obb_contact_handles_single_vehicle() -> None:
    overlaps = ego_opponent_obb_overlaps(
        jnp.zeros((1, 5)),
        vehicle_length=2.0,
        vehicle_width=1.45,
    )
    assert overlaps.shape == (0,)


def test_obb_contact_casts_integer_states_and_rejects_nonfinite_poses() -> None:
    integer_states = jnp.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    integer_overlaps = ego_opponent_obb_overlaps(
        integer_states,
        vehicle_length=2.0,
        vehicle_width=1.45,
    )
    np.testing.assert_array_equal(integer_overlaps, jnp.asarray([True]))

    nonfinite_states = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [jnp.nan, 0.0, 0.0, 0.0, 0.0],
            [1.9, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    nonfinite_overlaps = ego_opponent_obb_overlaps(
        nonfinite_states,
        vehicle_length=2.0,
        vehicle_width=1.45,
    )
    np.testing.assert_array_equal(
        nonfinite_overlaps,
        jnp.asarray([False, True]),
    )

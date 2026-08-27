"""Tests for loop-free dynamic vehicle LiDAR scan composition."""

import jax
import jax.numpy as jnp
import pytest

from lidar_racing_rl.envs.observation import canonicalize_scan
from lidar_racing_rl.geometry.dynamic_scan import (
    combine_static_and_dynamic_scan,
    dynamic_lidar_scan,
    dynamic_vehicle_scan,
    pairwise_dynamic_scan,
)


MAX_RANGE = 30.0
VEHICLE_DIMENSIONS = jnp.array([2.0, 1.0])


def test_self_vehicle_is_masked_from_scan() -> None:
    poses = jnp.array([[0.0, 0.0, 0.0]])
    beam_angles = jnp.array([-0.5, 0.0, 0.5])

    pairwise = pairwise_dynamic_scan(
        poses, VEHICLE_DIMENSIONS, beam_angles, MAX_RANGE
    )
    dynamic = dynamic_vehicle_scan(
        poses, VEHICLE_DIMENSIONS, beam_angles, MAX_RANGE
    )

    assert pairwise.shape == (1, 1, 3)
    assert dynamic.shape == (1, 3)
    assert bool(jnp.allclose(pairwise, jnp.full((1, 1, 3), MAX_RANGE)))
    assert bool(jnp.allclose(dynamic, jnp.full((1, 3), MAX_RANGE)))


def test_front_vehicle_is_visible_at_its_near_face() -> None:
    poses = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    dynamic = dynamic_vehicle_scan(
        poses, VEHICLE_DIMENSIONS, jnp.array([0.0]), MAX_RANGE
    )

    assert dynamic.shape == (2, 1)
    assert float(dynamic[0, 0]) == pytest.approx(4.0)
    assert float(dynamic[1, 0]) == pytest.approx(MAX_RANGE)


def test_nearest_of_two_overlapping_targets_is_used() -> None:
    poses = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ]
    )
    dynamic = dynamic_vehicle_scan(
        poses, VEHICLE_DIMENSIONS, jnp.array([0.0]), MAX_RANGE
    )

    assert float(dynamic[0, 0]) == pytest.approx(4.0)


def test_static_wall_occludes_vehicle_behind_it() -> None:
    poses = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    static_scan = jnp.array([[3.0], [MAX_RANGE]])
    combined = dynamic_lidar_scan(
        static_scan,
        poses,
        VEHICLE_DIMENSIONS,
        jnp.array([0.0]),
        MAX_RANGE,
    )

    assert float(combined[0, 0]) == pytest.approx(3.0)


def test_vehicle_in_front_of_wall_replaces_static_range() -> None:
    poses = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    static_scan = jnp.array([[10.0], [MAX_RANGE]])
    combined = dynamic_lidar_scan(
        static_scan,
        poses,
        VEHICLE_DIMENSIONS,
        jnp.array([0.0]),
        MAX_RANGE,
    )

    assert float(combined[0, 0]) == pytest.approx(4.0)


def test_target_outside_sampled_beam_has_no_effect() -> None:
    poses = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ]
    )
    dynamic = dynamic_vehicle_scan(
        poses, VEHICLE_DIMENSIONS, jnp.array([0.0]), MAX_RANGE
    )

    assert float(dynamic[0, 0]) == pytest.approx(MAX_RANGE)


def test_invalid_ranges_are_sanitized_before_minimum() -> None:
    static_scan = jnp.array([[jnp.nan, jnp.inf, -1.0, 8.0]])
    dynamic_scan = jnp.array([[5.0, 6.0, 7.0, jnp.nan]])
    combined = combine_static_and_dynamic_scan(
        static_scan, dynamic_scan, MAX_RANGE
    )

    assert bool(jnp.all(jnp.isfinite(combined)))
    assert bool(jnp.allclose(combined, jnp.array([[5.0, 6.0, 7.0, 8.0]])))


def test_invalid_static_return_stays_invalid_when_dynamic_ray_misses() -> None:
    static_scan = jnp.full((1, 360), jnp.inf)
    dynamic_scan = jnp.full((1, 360), MAX_RANGE)

    combined = combine_static_and_dynamic_scan(
        static_scan,
        dynamic_scan,
        MAX_RANGE,
    )
    canonical = canonicalize_scan(
        combined,
        range_min=0.001,
        range_max=MAX_RANGE,
    )

    assert bool(jnp.all(jnp.isfinite(combined)))
    assert bool(jnp.all(combined == -1.0))
    assert bool(jnp.all(canonical[:, 0, :] == 1.0))
    assert bool(jnp.all(canonical[:, 1, :] == 0.0))


def test_environment_vmap_matches_individual_geometry() -> None:
    poses = jnp.array(
        [
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        ]
    )
    static_scan = jnp.full((2, 2, 1), MAX_RANGE)
    mapped_scan = jax.jit(
        jax.vmap(dynamic_lidar_scan, in_axes=(0, 0, None, None, None))
    )(
        static_scan,
        poses,
        VEHICLE_DIMENSIONS,
        jnp.array([0.0]),
        MAX_RANGE,
    )
    individual_scan = jnp.stack(
        (
            dynamic_lidar_scan(
                static_scan[0],
                poses[0],
                VEHICLE_DIMENSIONS,
                jnp.array([0.0]),
                MAX_RANGE,
            ),
            dynamic_lidar_scan(
                static_scan[1],
                poses[1],
                VEHICLE_DIMENSIONS,
                jnp.array([0.0]),
                MAX_RANGE,
            ),
        )
    )

    assert mapped_scan.shape == (2, 2, 1)
    assert bool(jnp.allclose(mapped_scan, individual_scan))
    assert float(mapped_scan[0, 0, 0]) == pytest.approx(4.0)
    assert float(mapped_scan[1, 0, 0]) == pytest.approx(6.0)

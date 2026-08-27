"""Tests for vectorized ray--OBB slab intersection geometry."""

import jax
import jax.numpy as jnp
import pytest

from lidar_racing_rl.geometry.ray_obb import ray_obb_distance


MAX_RANGE = 30.0


def test_axis_aligned_box_returns_front_face_distance() -> None:
    distance = ray_obb_distance(
        ray_origin=jnp.array([0.0, 0.0]),
        ray_direction=jnp.array([1.0, 0.0]),
        obb_center=jnp.array([5.0, 0.0]),
        obb_yaw=jnp.array(0.0),
        obb_half_extents=jnp.array([1.0, 0.5]),
        max_range=MAX_RANGE,
    )

    assert float(distance) == pytest.approx(4.0)


def test_ninety_degree_rotation_uses_rotated_half_extent() -> None:
    distance = ray_obb_distance(
        ray_origin=jnp.array([0.0, 0.0]),
        ray_direction=jnp.array([1.0, 0.0]),
        obb_center=jnp.array([5.0, 0.0]),
        obb_yaw=jnp.array(jnp.pi / 2.0),
        obb_half_extents=jnp.array([1.0, 0.5]),
        max_range=MAX_RANGE,
    )

    assert float(distance) == pytest.approx(4.5)


def test_parallel_tangent_ray_is_finite() -> None:
    distance = ray_obb_distance(
        ray_origin=jnp.array([0.0, 0.5]),
        ray_direction=jnp.array([1.0, 0.0]),
        obb_center=jnp.array([5.0, 0.0]),
        obb_yaw=jnp.array(0.0),
        obb_half_extents=jnp.array([1.0, 0.5]),
        max_range=MAX_RANGE,
    )

    assert bool(jnp.isfinite(distance))
    assert float(distance) == pytest.approx(4.0)


def test_parallel_ray_outside_slab_is_a_miss() -> None:
    distance = ray_obb_distance(
        ray_origin=jnp.array([0.0, 1.0]),
        ray_direction=jnp.array([1.0, 0.0]),
        obb_center=jnp.array([5.0, 0.0]),
        obb_yaw=jnp.array(0.0),
        obb_half_extents=jnp.array([1.0, 0.5]),
        max_range=MAX_RANGE,
    )

    assert float(distance) == pytest.approx(MAX_RANGE)


def test_intersection_behind_origin_is_a_miss() -> None:
    distance = ray_obb_distance(
        ray_origin=jnp.array([0.0, 0.0]),
        ray_direction=jnp.array([1.0, 0.0]),
        obb_center=jnp.array([-5.0, 0.0]),
        obb_yaw=jnp.array(0.0),
        obb_half_extents=jnp.array([1.0, 0.5]),
        max_range=MAX_RANGE,
    )

    assert float(distance) == pytest.approx(MAX_RANGE)


def test_invalid_or_zero_direction_returns_finite_max_range() -> None:
    invalid_distance = ray_obb_distance(
        ray_origin=jnp.array([jnp.nan, 0.0]),
        ray_direction=jnp.array([1.0, 0.0]),
        obb_center=jnp.array([5.0, 0.0]),
        obb_yaw=jnp.array(0.0),
        obb_half_extents=jnp.array([1.0, 0.5]),
        max_range=MAX_RANGE,
    )
    zero_direction_distance = ray_obb_distance(
        ray_origin=jnp.array([0.0, 0.0]),
        ray_direction=jnp.array([0.0, 0.0]),
        obb_center=jnp.array([5.0, 0.0]),
        obb_yaw=jnp.array(0.0),
        obb_half_extents=jnp.array([1.0, 0.5]),
        max_range=MAX_RANGE,
    )

    assert float(invalid_distance) == pytest.approx(MAX_RANGE)
    assert float(zero_direction_distance) == pytest.approx(MAX_RANGE)


def test_jit_and_vmap_match_expected_distances() -> None:
    origins = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    directions = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    centers = jnp.array([[5.0, 0.0], [0.0, 7.0]])
    yaws = jnp.array([0.0, 0.0])
    half_extents = jnp.array([[1.0, 0.5], [1.0, 0.5]])
    mapped = jax.jit(
        jax.vmap(ray_obb_distance, in_axes=(0, 0, 0, 0, 0, None))
    )(
        origins,
        directions,
        centers,
        yaws,
        half_extents,
        MAX_RANGE,
    )

    assert mapped.shape == (2,)
    assert bool(jnp.allclose(mapped, jnp.array([4.0, 6.5])))

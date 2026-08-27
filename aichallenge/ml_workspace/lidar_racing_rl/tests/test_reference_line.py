"""Reference-line contracts for F1TENTH training and AWSIM transfer."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from lidar_racing_rl.npc.reference_line import (
    build_reference_waypoints,
    validate_centerline_clearance,
)


def _simulator() -> SimpleNamespace:
    centerline = SimpleNamespace(
        xs=jnp.asarray([0.0, 1.0, 2.0]),
        ys=jnp.asarray([0.0, 0.5, 0.0]),
    )
    return SimpleNamespace(
        track=SimpleNamespace(
            centerline=centerline,
            left_widths=jnp.asarray([1.1, 1.1, 1.1]),
            right_widths=jnp.asarray([1.1, 1.1, 1.1]),
        )
    )


def test_centerline_gets_explicit_base_target_speed() -> None:
    waypoints = build_reference_waypoints(
        _simulator(),
        reference_line="centerline",
        base_target_speed=8.0,
    )

    assert waypoints.shape == (3, 3)
    assert bool(jnp.all(waypoints[:, 2] == 8.0))


def test_unsafe_f1tenth_raceline_is_not_silently_selected() -> None:
    with pytest.raises(ValueError, match="reference_line=centerline"):
        build_reference_waypoints(
            _simulator(),
            reference_line="raceline",
            base_target_speed=8.0,
        )


def test_curvature_speed_profile_is_fast_on_straights_and_slow_in_corners() -> None:
    simulator = _simulator()
    simulator.track.centerline.s = jnp.asarray([0.0, 1.0, 2.0])
    curvatures = {0.0: 0.0, 1.0: 1.0 / 3.0, 2.0: 1.0}
    simulator.track.centerline.calc_curvature = lambda value: curvatures[value]

    waypoints = build_reference_waypoints(
        simulator,
        reference_line="centerline",
        base_target_speed=8.0,
        minimum_corner_speed=3.0,
        maximum_lateral_acceleration=3.0,
    )

    assert waypoints[:, 2].tolist() == pytest.approx([8.0, 3.0, 3.0])


def test_full_size_kart_offset_is_checked_against_track_width() -> None:
    clearances = validate_centerline_clearance(
        _simulator(),
        vehicle_width=1.45,
        lateral_offset_min=-0.2,
        lateral_offset_max=0.2,
    )
    assert clearances == pytest.approx((0.375, 0.375))

    with pytest.raises(ValueError, match="left-side"):
        validate_centerline_clearance(
            _simulator(),
            vehicle_width=1.45,
            lateral_offset_min=-0.2,
            lateral_offset_max=0.5,
        )

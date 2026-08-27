import pytest

from lidar_racing_controller.safety import (
    CommandRateLimiter,
    ControlTarget,
    UnsafeControlError,
    safe_stop_target,
    scale_normalized_action,
    scan_timed_out,
)


def test_action_scaling_matches_sac_contract() -> None:
    target = scale_normalized_action(
        (0.5, 0.0),
        steering_max_abs=1.0,
        acceleration_min=-3.2,
        acceleration_max=3.2,
    )
    assert target.steering_angle == pytest.approx(0.5)
    assert target.acceleration == pytest.approx(0.0)


def test_non_finite_action_is_rejected() -> None:
    with pytest.raises(UnsafeControlError, match="NaN or Inf"):
        scale_normalized_action(
            (float("nan"), 0.0),
            steering_max_abs=1.0,
            acceleration_min=-3.2,
            acceleration_max=3.2,
        )


def test_safe_stop_must_not_accelerate() -> None:
    assert safe_stop_target(acceleration=-1.0) == ControlTarget(0.0, -1.0)
    with pytest.raises(UnsafeControlError):
        safe_stop_target(acceleration=0.1)


def test_scan_timeout_uses_explicit_monotonic_times() -> None:
    assert scan_timed_out(
        now_monotonic=1.0,
        last_scan_monotonic=None,
        timeout_seconds=0.25,
    )
    assert not scan_timed_out(
        now_monotonic=1.2,
        last_scan_monotonic=1.0,
        timeout_seconds=0.25,
    )
    assert scan_timed_out(
        now_monotonic=1.3,
        last_scan_monotonic=1.0,
        timeout_seconds=0.25,
    )


def test_rate_limiter_bounds_each_control_axis() -> None:
    limiter = CommandRateLimiter(steering_rate=1.0, acceleration_rate=2.0)
    limiter.reset(ControlTarget(0.0, 0.0))

    limited = limiter.step(ControlTarget(1.0, 2.0), elapsed_seconds=0.5)

    assert limited.steering_angle == pytest.approx(0.5)
    assert limited.acceleration == pytest.approx(1.0)

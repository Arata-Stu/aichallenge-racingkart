"""Pure control scaling and fail-safe helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


class UnsafeControlError(ValueError):
    """Raised when a policy output or control configuration is unsafe."""


@dataclass(frozen=True)
class ControlTarget:
    """Physical steering angle and longitudinal acceleration target."""

    steering_angle: float
    acceleration: float


def scale_normalized_action(
    action: Sequence[float],
    *,
    steering_max_abs: float,
    acceleration_min: float,
    acceleration_max: float,
) -> ControlTarget:
    """Scale a finite normalized SAC action from ``[-1, 1]`` to physical units."""
    if len(action) != 2:
        raise UnsafeControlError(f"expected two policy outputs, got {len(action)}")
    if not math.isfinite(steering_max_abs) or steering_max_abs <= 0.0:
        raise UnsafeControlError("steering_max_abs must be finite and positive")
    if (
        not math.isfinite(acceleration_min)
        or not math.isfinite(acceleration_max)
        or acceleration_max <= acceleration_min
    ):
        raise UnsafeControlError("acceleration bounds must be finite and increasing")

    steering_action = float(action[0])
    acceleration_action = float(action[1])
    if not math.isfinite(steering_action) or not math.isfinite(acceleration_action):
        raise UnsafeControlError("policy output contains NaN or Inf")

    steering_action = min(1.0, max(-1.0, steering_action))
    acceleration_action = min(1.0, max(-1.0, acceleration_action))
    acceleration = acceleration_min + 0.5 * (acceleration_action + 1.0) * (
        acceleration_max - acceleration_min
    )
    return ControlTarget(
        steering_angle=steering_action * steering_max_abs,
        acceleration=acceleration,
    )


def safe_stop_target(*, acceleration: float) -> ControlTarget:
    """Create a centered-steering stop/brake target."""
    if not math.isfinite(acceleration) or acceleration > 0.0:
        raise UnsafeControlError("safe-stop acceleration must be finite and non-positive")
    return ControlTarget(steering_angle=0.0, acceleration=float(acceleration))


def scan_timed_out(
    *,
    now_monotonic: float,
    last_scan_monotonic: float | None,
    timeout_seconds: float,
) -> bool:
    """Return true when no scan has arrived within the configured steady-time window."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")
    if last_scan_monotonic is None:
        return True
    return now_monotonic - last_scan_monotonic > timeout_seconds


class CommandRateLimiter:
    """Bound steering and acceleration slew rates between control ticks."""

    def __init__(self, *, steering_rate: float, acceleration_rate: float):
        if not math.isfinite(steering_rate) or steering_rate <= 0.0:
            raise ValueError("steering_rate must be finite and positive")
        if not math.isfinite(acceleration_rate) or acceleration_rate <= 0.0:
            raise ValueError("acceleration_rate must be finite and positive")
        self.steering_rate = steering_rate
        self.acceleration_rate = acceleration_rate
        self._current: ControlTarget | None = None

    def reset(self, target: ControlTarget | None = None) -> None:
        """Reset limiter history, optionally to a known safe command."""
        self._current = target

    @staticmethod
    def _approach(current: float, target: float, maximum_delta: float) -> float:
        return current + min(max(target - current, -maximum_delta), maximum_delta)

    def step(self, target: ControlTarget, *, elapsed_seconds: float) -> ControlTarget:
        """Move toward a target without exceeding the configured rates."""
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if self._current is None:
            self._current = target
            return target

        limited = ControlTarget(
            steering_angle=self._approach(
                self._current.steering_angle,
                target.steering_angle,
                self.steering_rate * elapsed_seconds,
            ),
            acceleration=self._approach(
                self._current.acceleration,
                target.acceleration,
                self.acceleration_rate * elapsed_seconds,
            ),
        )
        self._current = limited
        return limited

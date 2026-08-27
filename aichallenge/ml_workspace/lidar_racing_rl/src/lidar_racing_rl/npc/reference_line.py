"""Configured, vehicle-safe waypoint reference lines for teachers and NPCs."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np


def validate_centerline_clearance(
    simulator: Any,
    *,
    vehicle_width: float,
    lateral_offset_min: float,
    lateral_offset_max: float,
) -> tuple[float, float]:
    """Validate configured offsets against sampled left/right track widths.

    Returns the minimum available center offset toward ``(right, left)`` after
    subtracting half the vehicle width. Heading-error clearance still requires
    rollout validation, so this is a necessary geometric gate rather than a
    claim that every controller trajectory is safe.
    """

    values = (vehicle_width, lateral_offset_min, lateral_offset_max)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("vehicle width and lateral offsets must be finite")
    if vehicle_width <= 0.0 or lateral_offset_min > lateral_offset_max:
        raise ValueError("vehicle width must be positive and offsets ordered")
    try:
        left_widths = np.asarray(simulator.track.left_widths, dtype=float)
        right_widths = np.asarray(simulator.track.right_widths, dtype=float)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("track boundary widths are required for offset validation") from error
    if (
        left_widths.ndim != 1
        or right_widths.shape != left_widths.shape
        or left_widths.size == 0
        or not np.all(np.isfinite(left_widths))
        or not np.all(np.isfinite(right_widths))
    ):
        raise RuntimeError("track boundary widths must be matching finite arrays")
    half_width = 0.5 * vehicle_width
    right_clearance = float(np.min(right_widths)) - half_width
    left_clearance = float(np.min(left_widths)) - half_width
    if right_clearance < 0.0 or left_clearance < 0.0:
        raise ValueError("configured vehicle is wider than the track centerline envelope")
    tolerance = 1.0e-9
    if lateral_offset_min < -right_clearance - tolerance:
        raise ValueError("negative NPC lateral offset exceeds right-side kart clearance")
    if lateral_offset_max > left_clearance + tolerance:
        raise ValueError("positive NPC lateral offset exceeds left-side kart clearance")
    return right_clearance, left_clearance


def build_reference_waypoints(
    simulator: Any,
    *,
    reference_line: str,
    base_target_speed: float,
) -> jnp.ndarray:
    """Build ``[waypoints, x/y/speed]`` from the configured track line.

    The initial AI Challenge kart contract deliberately supports only the
    track centerline. F1TENTH's optimized raceline approaches the 2.2 m-wide
    Spielberg boundary closely enough that a 1.45 m-wide kart starts outside
    the usable footprint. Per-NPC diversity is added later by the bounded
    lateral-offset transform.
    """

    if reference_line != "centerline":
        raise ValueError("initial full-size kart contract requires reference_line=centerline")
    if not math.isfinite(base_target_speed) or base_target_speed <= 0.0:
        raise ValueError("base_target_speed must be finite and positive")
    try:
        centerline = simulator.track.centerline
        xs = jnp.asarray(centerline.xs, dtype=jnp.float32)
        ys = jnp.asarray(centerline.ys, dtype=jnp.float32)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("simulator must expose track.centerline.xs/ys") from error
    if xs.ndim != 1 or ys.shape != xs.shape or xs.shape[0] < 2:
        raise RuntimeError("track centerline must contain matching x/y waypoint arrays")
    speeds = jnp.full(xs.shape, base_target_speed, dtype=jnp.float32)
    return jnp.stack((xs, ys, speeds), axis=-1)


__all__ = ["build_reference_waypoints", "validate_centerline_clearance"]

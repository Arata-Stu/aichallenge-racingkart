"""Dependency-free runtime metric helpers."""

from __future__ import annotations

from collections.abc import Iterable
import math


def percentile(samples: Iterable[float], percent: float) -> float:
    """Return a linearly interpolated percentile for finite samples."""
    percent = float(percent)
    if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
        raise ValueError("percent must be finite and in [0, 100]")

    ordered = sorted(float(sample) for sample in samples)
    if not ordered:
        raise ValueError("at least one sample is required")
    if not all(math.isfinite(sample) for sample in ordered):
        raise ValueError("samples must be finite")
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percent / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )

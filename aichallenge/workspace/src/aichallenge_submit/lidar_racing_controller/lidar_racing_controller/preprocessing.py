"""Canonical LiDAR preprocessing shared with the deployment contract."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


class ScanValidationError(ValueError):
    """Raised when a LaserScan cannot satisfy the canonical observation contract."""


@dataclass(frozen=True)
class CanonicalScan:
    """A two-channel canonical scan and its usable-beam ratio."""

    values: np.ndarray
    valid_ratio: float


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ScanValidationError(f"{name} must be finite, got {value!r}")
    return value


def canonicalize_laserscan(
    ranges: Iterable[float],
    *,
    range_min: float,
    range_max: float,
    angle_min: float,
    angle_max: float,
    angle_increment: float,
    expected_raw_beams: int = 1080,
    canonical_beams: int = 360,
    expected_range_min: float | None = None,
    expected_range_max: float | None = 30.0,
    expected_angle_min: float = -3.0 * math.pi / 4.0,
    expected_angle_max: float = 3.0 * math.pi / 4.0,
    canonical_range_max: float | None = None,
    canonical_angle_min: float | None = None,
    canonical_angle_max: float | None = None,
    metadata_tolerance: float = 0.01,
) -> CanonicalScan:
    """Validate and convert a LaserScan to ``[range, validity, beam]``.

    Invalid samples are represented as ``range=1.0`` and ``validity=0.0``.
    Raw rays are assigned to canonical angular bins and minimum-pooled using
    valid samples only. Canonical bins outside the raw field of view remain
    invalid, which permits a narrower AWSIM scan to feed a legacy wider-FOV
    Actor without inventing observations.
    """
    if expected_raw_beams <= 0:
        raise ValueError("expected_raw_beams must be positive")
    if canonical_beams < 2:
        raise ValueError("canonical_beams must be at least two")
    if metadata_tolerance < 0.0:
        raise ValueError("metadata_tolerance must be non-negative")

    scan = np.asarray(ranges, dtype=np.float32)
    if scan.ndim != 1 or scan.shape[0] != expected_raw_beams:
        raise ScanValidationError(
            f"expected {expected_raw_beams} ranges, got shape {scan.shape}"
        )

    range_min = _require_finite("range_min", range_min)
    range_max = _require_finite("range_max", range_max)
    angle_min = _require_finite("angle_min", angle_min)
    angle_max = _require_finite("angle_max", angle_max)
    angle_increment = _require_finite("angle_increment", angle_increment)
    expected_angle_min = _require_finite("expected_angle_min", expected_angle_min)
    expected_angle_max = _require_finite("expected_angle_max", expected_angle_max)
    canonical_range_max = _require_finite(
        "canonical_range_max",
        range_max if canonical_range_max is None else canonical_range_max,
    )
    canonical_angle_min = _require_finite(
        "canonical_angle_min",
        expected_angle_min if canonical_angle_min is None else canonical_angle_min,
    )
    canonical_angle_max = _require_finite(
        "canonical_angle_max",
        expected_angle_max if canonical_angle_max is None else canonical_angle_max,
    )

    if range_min < 0.0 or range_max <= range_min:
        raise ScanValidationError(
            f"invalid range bounds: range_min={range_min}, range_max={range_max}"
        )
    if angle_increment <= 0.0 or angle_max <= angle_min:
        raise ScanValidationError(
            "angle_increment must be positive and angle_max must exceed angle_min"
        )
    if expected_angle_max <= expected_angle_min:
        raise ValueError("expected_angle_max must exceed expected_angle_min")
    if canonical_range_max <= 0.0:
        raise ValueError("canonical_range_max must be positive")
    if canonical_angle_max <= canonical_angle_min:
        raise ValueError("canonical_angle_max must exceed canonical_angle_min")

    if expected_range_min is not None:
        expected_range_min = _require_finite("expected_range_min", expected_range_min)
        if not math.isclose(
            range_min,
            expected_range_min,
            rel_tol=0.0,
            abs_tol=metadata_tolerance,
        ):
            raise ScanValidationError(
                f"range_min mismatch: expected {expected_range_min}, got {range_min}"
            )

    if expected_range_max is not None:
        expected_range_max = _require_finite("expected_range_max", expected_range_max)
        if not math.isclose(
            range_max,
            expected_range_max,
            rel_tol=0.0,
            abs_tol=metadata_tolerance,
        ):
            raise ScanValidationError(
                f"range_max mismatch: expected {expected_range_max}, got {range_max}"
            )

    expected_angles = (
        ("angle_min", angle_min, expected_angle_min),
        ("angle_max", angle_max, expected_angle_max),
    )
    for name, actual, expected in expected_angles:
        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=metadata_tolerance,
        ):
            raise ScanValidationError(
                f"{name} mismatch: expected {expected}, got {actual}"
            )

    predicted_angle_max = angle_min + (expected_raw_beams - 1) * angle_increment
    angle_tolerance = max(metadata_tolerance, abs(angle_increment) * 1.5)
    if not math.isclose(
        predicted_angle_max,
        angle_max,
        rel_tol=0.0,
        abs_tol=angle_tolerance,
    ):
        raise ScanValidationError(
            "LaserScan angle metadata is inconsistent with the range count: "
            f"predicted angle_max={predicted_angle_max}, message angle_max={angle_max}"
        )

    same_field_of_view = math.isclose(
        angle_min,
        canonical_angle_min,
        rel_tol=0.0,
        abs_tol=metadata_tolerance,
    ) and math.isclose(
        angle_max,
        canonical_angle_max,
        rel_tol=0.0,
        abs_tol=metadata_tolerance,
    )
    if expected_raw_beams % canonical_beams == 0 and same_field_of_view:
        pooling_width = expected_raw_beams // canonical_beams
        grouped = scan.reshape(canonical_beams, pooling_width)
        valid = np.isfinite(grouped) & (grouped >= range_min) & (grouped <= range_max)
        pooled_valid = np.any(valid, axis=1)
        pooled_range = np.min(
            np.where(valid, np.minimum(grouped, canonical_range_max), canonical_range_max),
            axis=1,
        )
        normalized_range = np.clip(
            pooled_range / canonical_range_max,
            0.0,
            1.0,
        ).astype(np.float32)
        validity = pooled_valid.astype(np.float32)
        canonical = np.stack((normalized_range, validity), axis=0)
        return CanonicalScan(values=canonical, valid_ratio=float(np.mean(validity)))

    raw_angles = angle_min + np.arange(
        expected_raw_beams,
        dtype=np.float64,
    ) * angle_increment
    canonical_increment = (canonical_angle_max - canonical_angle_min) / (
        canonical_beams - 1
    )
    first_bin_edge = canonical_angle_min - 0.5 * canonical_increment
    bin_indices = np.floor(
        (raw_angles - first_bin_edge) / canonical_increment
    ).astype(np.int64)
    in_canonical_fov = (bin_indices >= 0) & (bin_indices < canonical_beams)
    valid_samples = (
        in_canonical_fov
        & np.isfinite(scan)
        & (scan >= range_min)
        & (scan <= range_max)
    )

    pooled_range = np.full(canonical_beams, canonical_range_max, dtype=np.float32)
    pooled_valid = np.zeros(canonical_beams, dtype=bool)
    valid_bins = bin_indices[valid_samples]
    np.minimum.at(
        pooled_range,
        valid_bins,
        np.minimum(scan[valid_samples], canonical_range_max),
    )
    np.logical_or.at(pooled_valid, valid_bins, True)
    normalized_range = np.clip(
        pooled_range / canonical_range_max,
        0.0,
        1.0,
    ).astype(np.float32)
    validity = pooled_valid.astype(np.float32)

    canonical = np.stack((normalized_range, validity), axis=0)
    return CanonicalScan(values=canonical, valid_ratio=float(np.mean(validity)))


class FrameStack:
    """Maintain chronological canonical scans and expose Actor input shape."""

    def __init__(self, *, frame_count: int = 4, channels: int = 2, beams: int = 360):
        if frame_count <= 0 or channels <= 0 or beams <= 0:
            raise ValueError("frame_count, channels, and beams must be positive")
        self.frame_count = frame_count
        self.channels = channels
        self.beams = beams
        self._frames: deque[np.ndarray] = deque(maxlen=frame_count)

    def reset(self) -> None:
        """Discard all history, for example after a sensor fault or episode reset."""
        self._frames.clear()

    def append(self, scan: np.ndarray) -> None:
        """Append a canonical scan, repeating the first frame to fill history."""
        value = np.asarray(scan, dtype=np.float32)
        expected_shape = (self.channels, self.beams)
        if value.shape != expected_shape:
            raise ValueError(f"expected scan shape {expected_shape}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("canonical scan must contain only finite values")

        if not self._frames:
            for _ in range(self.frame_count):
                self._frames.append(value.copy())
            return
        self._frames.append(value.copy())

    def stacked(self) -> np.ndarray:
        """Return ``[frame, channel, beam]`` without exposing internal storage."""
        if len(self._frames) != self.frame_count:
            raise RuntimeError("frame history is not initialized")
        return np.stack(tuple(self._frames), axis=0)

    def actor_input(self) -> np.ndarray:
        """Return chronological frames flattened to ``[frame * channel, beam]``."""
        return self.stacked().reshape(self.frame_count * self.channels, self.beams)

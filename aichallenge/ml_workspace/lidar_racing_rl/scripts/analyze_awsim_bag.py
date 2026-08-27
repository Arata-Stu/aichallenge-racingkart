#!/usr/bin/env python3
"""Analyze AWSIM ``sensor_msgs/msg/LaserScan`` data from a ROS 2 bag.

Run this script in the existing AI Challenge development container. It imports
ROS 2 bag APIs lazily so ``--help`` and static checks do not require the JAX
training environment or any additional training dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TOPIC = "/sensing/lidar/scan"
EXPECTED_MESSAGE_TYPE = "sensor_msgs/msg/LaserScan"
SCHEMA_VERSION = 1
NANOSECONDS_PER_SECOND = 1_000_000_000
ANGLE_METADATA_ABSOLUTE_TOLERANCE = 1.0e-12
MAX_RECORDED_HOLD_CANDIDATES = 100
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "assets" / "calibration"


class AnalysisError(RuntimeError):
    """Raised for actionable bag or analysis errors."""


@dataclass
class NumericSeries:
    """Finite scalar samples retained for descriptive statistics."""

    values: list[float] = field(default_factory=list)
    nonfinite_count: int = 0

    def add(self, value: float) -> None:
        value = float(value)
        if math.isfinite(value):
            self.values.append(value)
        else:
            self.nonfinite_count += 1

    def summary(self) -> dict[str, Any]:
        return summarize_values(self.values, nonfinite_count=self.nonfinite_count)


@dataclass
class BeamStatistics:
    """Validity counts for one beam index across all frames."""

    total: int = 0
    invalid: int = 0
    nan: int = 0
    positive_inf: int = 0
    negative_inf: int = 0
    reference_angle: float | None = None
    max_reference_angle_delta: float = 0.0


@dataclass
class DistanceBinStatistics:
    """Invalid counts assigned to one proxy-distance bin."""

    total: int = 0
    invalid: int = 0


class LidarStatisticsAccumulator:
    """Streaming LaserScan statistics without retaining complete frames."""

    def __init__(
        self,
        *,
        topic: str,
        source_path: Path,
        storage_id: str,
        message_type: str,
        distance_bin_width: float,
        frame_hold_atol: float,
        far_jump_threshold: float | None,
    ) -> None:
        self.topic = topic
        self.source_path = source_path
        self.storage_id = storage_id
        self.message_type = message_type
        self.distance_bin_width = distance_bin_width
        self.frame_hold_atol = frame_hold_atol
        self.far_jump_threshold = far_jump_threshold

        self.frame_count = 0
        self.total_points = 0
        self.valid_points = 0
        self.nan_points = 0
        self.positive_inf_points = 0
        self.negative_inf_points = 0
        self.below_range_min_points = 0
        self.above_range_max_points = 0
        self.invalid_metadata_frames = 0
        self.empty_scan_frames = 0

        self.beam_count_histogram: Counter[int] = Counter()
        self.frame_id_histogram: Counter[str] = Counter()
        self.angle_min = NumericSeries()
        self.angle_max = NumericSeries()
        self.angle_increment = NumericSeries()
        self.range_min = NumericSeries()
        self.range_max = NumericSeries()
        self.scan_time = NumericSeries()
        self.time_increment = NumericSeries()
        self.angle_span_error = NumericSeries()

        self.bag_periods = NumericSeries()
        self.header_periods = NumericSeries()
        self.nonpositive_bag_period_count = 0
        self.nonpositive_header_period_count = 0
        self.header_stamp_frame_count = 0
        self.first_bag_timestamp_ns: int | None = None
        self.last_bag_timestamp_ns: int | None = None
        self.first_header_timestamp_ns: int | None = None
        self.last_header_timestamp_ns: int | None = None
        self._previous_bag_timestamp_ns: int | None = None
        self._previous_header_timestamp_ns: int | None = None

        self.invalid_run_width_histogram: Counter[int] = Counter()
        self.maximum_invalid_run_per_frame = NumericSeries()

        self.beam_statistics: list[BeamStatistics] = []
        self._previous_valid_ranges: list[float | None] = []
        self._previous_frame_valid_ranges: list[float | None] = []
        self.previous_range_reset_count = 0
        self.distance_bins: dict[int, DistanceBinStatistics] = {}
        self.distance_proxy_unassigned_invalid_count = 0

        self.frame_hold_comparable_pairs = 0
        self.frame_hold_candidate_count = 0
        self.frame_hold_current_streak = 0
        self.frame_hold_longest_streak = 0
        self.frame_hold_candidates: list[dict[str, int]] = []
        self._previous_ranges: tuple[float, ...] | None = None
        self._previous_geometry: tuple[int, float, float, float, float, float] | None = None

        self.far_jump_comparable_transitions = 0
        self.far_jump_candidate_count = 0

    def add(self, message: Any, bag_timestamp_ns: int) -> None:
        """Consume one LaserScan-like ROS message."""
        ranges = tuple(float(value) for value in message.ranges)
        beam_count = len(ranges)
        angle_min = float(message.angle_min)
        angle_max = float(message.angle_max)
        angle_increment = float(message.angle_increment)
        range_min = float(message.range_min)
        range_max = float(message.range_max)
        scan_time = float(message.scan_time)
        time_increment = float(message.time_increment)
        header_timestamp_ns = _header_timestamp_ns(message)
        frame_id = str(getattr(getattr(message, "header", None), "frame_id", ""))

        self.frame_count += 1
        self.total_points += beam_count
        self.beam_count_histogram[beam_count] += 1
        self.frame_id_histogram[frame_id] += 1
        if beam_count == 0:
            self.empty_scan_frames += 1

        self.angle_min.add(angle_min)
        self.angle_max.add(angle_max)
        self.angle_increment.add(angle_increment)
        self.range_min.add(range_min)
        self.range_max.add(range_max)
        self.scan_time.add(scan_time)
        self.time_increment.add(time_increment)
        if beam_count > 0 and all(
            math.isfinite(value) for value in (angle_min, angle_max, angle_increment)
        ):
            expected_angle_max = angle_min + (beam_count - 1) * angle_increment
            self.angle_span_error.add(angle_max - expected_angle_max)
        else:
            self.angle_span_error.add(math.nan)

        bounds_are_valid = (
            math.isfinite(range_min)
            and math.isfinite(range_max)
            and 0.0 <= range_min < range_max
        )
        if not bounds_are_valid:
            self.invalid_metadata_frames += 1

        self._add_timestamps(bag_timestamp_ns, header_timestamp_ns)
        geometry = (
            beam_count,
            angle_min,
            angle_max,
            angle_increment,
            range_min,
            range_max,
        )
        self._ensure_beam_capacity(beam_count)
        if (
            self._previous_geometry is not None
            and self._previous_geometry[0] == beam_count
            and not _geometry_matches(self._previous_geometry, geometry)
        ):
            self._previous_valid_ranges = [None] * beam_count
            self._previous_frame_valid_ranges = [None] * beam_count
            self.previous_range_reset_count += 1

        invalid_mask: list[bool] = []
        current_frame_valid_ranges: list[float | None] = [None] * beam_count
        for beam_index, value in enumerate(ranges):
            is_nan = math.isnan(value)
            is_positive_inf = math.isinf(value) and value > 0.0
            is_negative_inf = math.isinf(value) and value < 0.0
            is_finite = math.isfinite(value)
            below_minimum = is_finite and bounds_are_valid and value < range_min
            above_maximum = is_finite and bounds_are_valid and value > range_max
            is_valid = (
                bounds_are_valid
                and is_finite
                and range_min <= value <= range_max
            )
            is_invalid = not is_valid
            invalid_mask.append(is_invalid)

            self.valid_points += int(is_valid)
            self.nan_points += int(is_nan)
            self.positive_inf_points += int(is_positive_inf)
            self.negative_inf_points += int(is_negative_inf)
            self.below_range_min_points += int(below_minimum)
            self.above_range_max_points += int(above_maximum)

            beam = self.beam_statistics[beam_index]
            beam.total += 1
            beam.invalid += int(is_invalid)
            beam.nan += int(is_nan)
            beam.positive_inf += int(is_positive_inf)
            beam.negative_inf += int(is_negative_inf)
            angle = angle_min + beam_index * angle_increment
            if math.isfinite(angle):
                if beam.reference_angle is None:
                    beam.reference_angle = angle
                else:
                    beam.max_reference_angle_delta = max(
                        beam.max_reference_angle_delta,
                        abs(angle - beam.reference_angle),
                    )

            previous_valid_range = self._previous_valid_ranges[beam_index]
            proxy_distance = value if is_valid else previous_valid_range
            if proxy_distance is not None and math.isfinite(proxy_distance):
                bin_index = math.floor(proxy_distance / self.distance_bin_width)
                distance_bin = self.distance_bins.setdefault(
                    bin_index, DistanceBinStatistics()
                )
                distance_bin.total += 1
                distance_bin.invalid += int(is_invalid)
            elif is_invalid:
                self.distance_proxy_unassigned_invalid_count += 1

            if is_valid:
                previous_frame_range = self._previous_frame_valid_ranges[beam_index]
                if previous_frame_range is not None and self.far_jump_threshold is not None:
                    self.far_jump_comparable_transitions += 1
                    if value - previous_frame_range >= self.far_jump_threshold:
                        self.far_jump_candidate_count += 1
                self._previous_valid_ranges[beam_index] = value
                current_frame_valid_ranges[beam_index] = value

        self._add_invalid_runs(invalid_mask)
        self._add_frame_hold_candidate(
            ranges,
            geometry,
            bag_timestamp_ns,
            header_timestamp_ns,
        )
        self._previous_frame_valid_ranges = current_frame_valid_ranges
        self._previous_ranges = ranges
        self._previous_geometry = geometry

    def _add_timestamps(
        self,
        bag_timestamp_ns: int,
        header_timestamp_ns: int | None,
    ) -> None:
        if self.first_bag_timestamp_ns is None:
            self.first_bag_timestamp_ns = bag_timestamp_ns
        self.last_bag_timestamp_ns = bag_timestamp_ns
        if self._previous_bag_timestamp_ns is not None:
            delta = bag_timestamp_ns - self._previous_bag_timestamp_ns
            if delta > 0:
                self.bag_periods.add(delta / NANOSECONDS_PER_SECOND)
            else:
                self.nonpositive_bag_period_count += 1
        self._previous_bag_timestamp_ns = bag_timestamp_ns

        if header_timestamp_ns is None:
            return
        self.header_stamp_frame_count += 1
        if self.first_header_timestamp_ns is None:
            self.first_header_timestamp_ns = header_timestamp_ns
        self.last_header_timestamp_ns = header_timestamp_ns
        if self._previous_header_timestamp_ns is not None:
            delta = header_timestamp_ns - self._previous_header_timestamp_ns
            if delta > 0:
                self.header_periods.add(delta / NANOSECONDS_PER_SECOND)
            else:
                self.nonpositive_header_period_count += 1
        self._previous_header_timestamp_ns = header_timestamp_ns

    def _ensure_beam_capacity(self, beam_count: int) -> None:
        if len(self._previous_valid_ranges) != beam_count:
            if self._previous_valid_ranges:
                self.previous_range_reset_count += 1
            self._previous_valid_ranges = [None] * beam_count
            self._previous_frame_valid_ranges = [None] * beam_count
        missing_beam_statistics = beam_count - len(self.beam_statistics)
        if missing_beam_statistics > 0:
            self.beam_statistics.extend(
                BeamStatistics() for _ in range(missing_beam_statistics)
            )

    def _add_invalid_runs(self, invalid_mask: Sequence[bool]) -> None:
        current_width = 0
        maximum_width = 0
        for is_invalid in invalid_mask:
            if is_invalid:
                current_width += 1
                maximum_width = max(maximum_width, current_width)
            elif current_width:
                self._record_invalid_run(current_width)
                current_width = 0
        if current_width:
            self._record_invalid_run(current_width)
        self.maximum_invalid_run_per_frame.add(maximum_width)

    def _record_invalid_run(self, width: int) -> None:
        self.invalid_run_width_histogram[width] += 1

    def _add_frame_hold_candidate(
        self,
        ranges: tuple[float, ...],
        geometry: tuple[int, float, float, float, float, float],
        bag_timestamp_ns: int,
        header_timestamp_ns: int | None,
    ) -> None:
        if self._previous_ranges is None or self._previous_geometry is None:
            return
        if not _geometry_matches(self._previous_geometry, geometry):
            self.frame_hold_current_streak = 0
            return
        self.frame_hold_comparable_pairs += 1
        is_candidate = _ranges_match(
            self._previous_ranges,
            ranges,
            absolute_tolerance=self.frame_hold_atol,
        )
        if not is_candidate:
            self.frame_hold_current_streak = 0
            return

        self.frame_hold_candidate_count += 1
        self.frame_hold_current_streak += 1
        self.frame_hold_longest_streak = max(
            self.frame_hold_longest_streak,
            self.frame_hold_current_streak,
        )
        if len(self.frame_hold_candidates) < MAX_RECORDED_HOLD_CANDIDATES:
            candidate: dict[str, int] = {
                "frame_index": self.frame_count - 1,
                "bag_timestamp_ns": bag_timestamp_ns,
            }
            if header_timestamp_ns is not None:
                candidate["header_timestamp_ns"] = header_timestamp_ns
            self.frame_hold_candidates.append(candidate)

    def build_report(self, *, max_frames: int | None) -> dict[str, Any]:
        """Build the JSON-serializable report after streaming is complete."""
        if self.frame_count == 0:
            raise AnalysisError(f"No messages were found on {self.topic}")

        invalid_points = self.total_points - self.valid_points
        inf_points = self.positive_inf_points + self.negative_inf_points
        timing = {
            "bag_timestamp": _timing_report(
                self.bag_periods,
                self.first_bag_timestamp_ns,
                self.last_bag_timestamp_ns,
                self.nonpositive_bag_period_count,
            ),
            "header_timestamp": _timing_report(
                self.header_periods,
                self.first_header_timestamp_ns,
                self.last_header_timestamp_ns,
                self.nonpositive_header_period_count,
            ),
            "frames_with_header_timestamp": self.header_stamp_frame_count,
        }
        warnings = self._warnings()
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "bag_path": str(self.source_path),
                "storage_id": self.storage_id,
                "topic": self.topic,
                "message_type": self.message_type,
                "frame_count": self.frame_count,
                "max_frames_limit": max_frames,
            },
            "scan_geometry": {
                "beam_count": _histogram_report(self.beam_count_histogram),
                "frame_id_histogram": dict(sorted(self.frame_id_histogram.items())),
                "angle_min_radians": self.angle_min.summary(),
                "angle_max_radians": self.angle_max.summary(),
                "angle_increment_radians": self.angle_increment.summary(),
                "range_min": self.range_min.summary(),
                "range_max": self.range_max.summary(),
                "scan_time_seconds": self.scan_time.summary(),
                "time_increment_seconds": self.time_increment.summary(),
                "angle_span_error_radians": self.angle_span_error.summary(),
                "invalid_range_metadata_frames": self.invalid_metadata_frames,
                "empty_scan_frames": self.empty_scan_frames,
            },
            "timing": timing,
            "validity": {
                "total_points": self.total_points,
                "valid_points": self.valid_points,
                "invalid_points": invalid_points,
                "valid_rate": safe_rate(self.valid_points, self.total_points),
                "invalid_rate": safe_rate(invalid_points, self.total_points),
                "nan_points": self.nan_points,
                "nan_rate": safe_rate(self.nan_points, self.total_points),
                "inf_points": inf_points,
                "inf_rate": safe_rate(inf_points, self.total_points),
                "positive_inf_points": self.positive_inf_points,
                "positive_inf_rate": safe_rate(
                    self.positive_inf_points, self.total_points
                ),
                "negative_inf_points": self.negative_inf_points,
                "negative_inf_rate": safe_rate(
                    self.negative_inf_points, self.total_points
                ),
                "below_range_min_points": self.below_range_min_points,
                "above_range_max_points": self.above_range_max_points,
            },
            "consecutive_invalid_runs": {
                "boundary_policy": "scan endpoints are not joined",
                "run_count": sum(self.invalid_run_width_histogram.values()),
                "width_summary_beams": summarize_histogram(
                    self.invalid_run_width_histogram
                ),
                "maximum_width_per_frame_beams": (
                    self.maximum_invalid_run_per_frame.summary()
                ),
                "width_histogram": {
                    str(width): count
                    for width, count in sorted(self.invalid_run_width_histogram.items())
                },
            },
            "distance_conditioned_invalid_rate": self._distance_report(),
            "angle_conditioned_invalid_rate": self._angle_report(),
            "frame_hold_candidates": {
                "classification": "candidate_not_confirmed",
                "comparison": (
                    "all ranges equal; NaN and signed infinity compare by category"
                ),
                "absolute_tolerance": self.frame_hold_atol,
                "comparable_frame_pairs": self.frame_hold_comparable_pairs,
                "candidate_count": self.frame_hold_candidate_count,
                "candidate_rate": safe_rate(
                    self.frame_hold_candidate_count,
                    self.frame_hold_comparable_pairs,
                ),
                "longest_consecutive_candidate_run": self.frame_hold_longest_streak,
                "recorded_candidates": self.frame_hold_candidates,
                "recorded_candidate_limit": MAX_RECORDED_HOLD_CANDIDATES,
            },
            "wall_leak_candidates": self._wall_leak_report(),
            "warnings": warnings,
        }

    def _distance_report(self) -> dict[str, Any]:
        bins = []
        for index, statistics in sorted(self.distance_bins.items()):
            lower = index * self.distance_bin_width
            bins.append(
                {
                    "lower_inclusive": lower,
                    "upper_exclusive": lower + self.distance_bin_width,
                    "assigned_points": statistics.total,
                    "invalid_points": statistics.invalid,
                    "invalid_rate": safe_rate(statistics.invalid, statistics.total),
                }
            )
        return {
            "classification": "proxy_not_ground_truth",
            "method": (
                "valid samples use their current range; invalid samples use the "
                "last valid range at the same beam index"
            ),
            "units": "LaserScan range units (normally meters)",
            "bin_width": self.distance_bin_width,
            "bins": bins,
            "unassigned_invalid_points": self.distance_proxy_unassigned_invalid_count,
            "history_reset_count_due_to_scan_geometry_change": (
                self.previous_range_reset_count
            ),
        }

    def _angle_report(self) -> dict[str, Any]:
        beams = []
        for index, statistics in enumerate(self.beam_statistics):
            beams.append(
                {
                    "beam_index": index,
                    "reference_angle_radians": statistics.reference_angle,
                    "max_reference_angle_delta_radians": (
                        statistics.max_reference_angle_delta
                    ),
                    "observations": statistics.total,
                    "invalid_points": statistics.invalid,
                    "invalid_rate": safe_rate(statistics.invalid, statistics.total),
                    "nan_points": statistics.nan,
                    "positive_inf_points": statistics.positive_inf,
                    "negative_inf_points": statistics.negative_inf,
                }
            )
        worst_beams = sorted(
            beams,
            key=lambda item: (
                item["invalid_rate"] if item["invalid_rate"] is not None else -1.0,
                item["observations"],
            ),
            reverse=True,
        )[:20]
        metadata_changed_count = sum(
            statistics.max_reference_angle_delta > ANGLE_METADATA_ABSOLUTE_TOLERANCE
            for statistics in self.beam_statistics
        )
        return {
            "angle_reference": "first finite angle observed for each beam index",
            "beam_index_metadata_changed_count": metadata_changed_count,
            "beams": beams,
            "worst_20_beams": worst_beams,
        }

    def _wall_leak_report(self) -> dict[str, Any]:
        heuristic: dict[str, Any]
        if self.far_jump_threshold is None:
            heuristic = {
                "enabled": False,
                "reason": "--far-jump-threshold was not provided",
                "candidate_count": None,
                "candidate_rate": None,
            }
        else:
            heuristic = {
                "enabled": True,
                "classification": "temporal_far_jump_only_not_wall_leak",
                "comparison": "consecutive frames where both samples are valid",
                "threshold": self.far_jump_threshold,
                "units": "LaserScan range units (normally meters)",
                "comparable_valid_transitions": self.far_jump_comparable_transitions,
                "candidate_count": self.far_jump_candidate_count,
                "candidate_rate": safe_rate(
                    self.far_jump_candidate_count,
                    self.far_jump_comparable_transitions,
                ),
            }
        return {
            "classification": "not_determinable_without_gt_or_reference_wall_distance",
            "confirmed_count": None,
            "confirmed_rate": None,
            "explanation": (
                "LaserScan alone cannot distinguish a wall leak from genuinely open "
                "space. A map/GT ray-cast distance aligned to every frame is required."
            ),
            "temporal_far_jump_heuristic": heuristic,
        }

    def _warnings(self) -> list[str]:
        warnings = [
            (
                "Wall-leak frequency is not asserted because no aligned GT/reference "
                "wall distance is available."
            )
        ]
        if len(self.beam_count_histogram) > 1:
            warnings.append("Beam count changed during the recording.")
        if self.invalid_metadata_frames:
            warnings.append("Some frames had invalid range_min/range_max metadata.")
        if self.header_stamp_frame_count < self.frame_count:
            warnings.append("Some frames had no positive ROS header timestamp.")
        if any(
            statistics.max_reference_angle_delta > ANGLE_METADATA_ABSOLUTE_TOLERANCE
            for statistics in self.beam_statistics
        ):
            warnings.append("Beam-index angle metadata changed during the recording.")
        if self.frame_hold_candidate_count:
            warnings.append(
                "Identical consecutive scans are frame-hold candidates, not proof of "
                "a sensor or transport fault."
            )
        return warnings


def summarize_values(
    values: Sequence[float],
    *,
    nonfinite_count: int = 0,
) -> dict[str, Any]:
    """Summarize finite values with population standard deviation."""
    if not values:
        return {
            "count": 0,
            "nonfinite_count": nonfinite_count,
            "min": None,
            "max": None,
            "mean": None,
            "stddev": None,
            "p50": None,
            "p95": None,
        }
    ordered = sorted(float(value) for value in values)
    mean = math.fsum(ordered) / len(ordered)
    variance = math.fsum((value - mean) ** 2 for value in ordered) / len(ordered)
    return {
        "count": len(ordered),
        "nonfinite_count": nonfinite_count,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean,
        "stddev": math.sqrt(variance),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
    }


def summarize_histogram(histogram: Counter[int]) -> dict[str, Any]:
    """Summarize an integer histogram without expanding every observation."""
    count = sum(histogram.values())
    if count == 0:
        return {
            "count": 0,
            "nonfinite_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "stddev": None,
            "p50": None,
            "p95": None,
        }
    ordered_items = sorted(histogram.items())
    mean = math.fsum(value * frequency for value, frequency in ordered_items) / count
    variance = (
        math.fsum(
            ((value - mean) ** 2) * frequency
            for value, frequency in ordered_items
        )
        / count
    )
    return {
        "count": count,
        "nonfinite_count": 0,
        "min": ordered_items[0][0],
        "max": ordered_items[-1][0],
        "mean": mean,
        "stddev": math.sqrt(variance),
        "p50": histogram_percentile(ordered_items, count, 0.50),
        "p95": histogram_percentile(ordered_items, count, 0.95),
    }


def histogram_percentile(
    ordered_items: Sequence[tuple[int, int]],
    count: int,
    quantile: float,
) -> float:
    """Return the nearest-rank value for a compact integer histogram."""
    rank = max(1, math.ceil(quantile * count))
    cumulative = 0
    for value, frequency in ordered_items:
        cumulative += frequency
        if cumulative >= rank:
            return float(value)
    return float(ordered_items[-1][0])


def percentile(ordered_values: Sequence[float], quantile: float) -> float | None:
    """Return a linearly interpolated quantile from an already sorted sequence."""
    if not ordered_values:
        return None
    if len(ordered_values) == 1:
        return float(ordered_values[0])
    position = (len(ordered_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered_values[lower_index])
    weight = position - lower_index
    return float(
        ordered_values[lower_index] * (1.0 - weight)
        + ordered_values[upper_index] * weight
    )


def safe_rate(numerator: int, denominator: int) -> float | None:
    """Return a ratio or ``None`` when no denominator exists."""
    if denominator == 0:
        return None
    return numerator / denominator


def _histogram_report(histogram: Counter[int]) -> dict[str, Any]:
    total = sum(histogram.values())
    mode = None
    if histogram:
        mode = min(
            histogram,
            key=lambda value: (-histogram[value], value),
        )
    return {
        "stable": len(histogram) <= 1,
        "mode": mode,
        "histogram": {
            str(value): count for value, count in sorted(histogram.items())
        },
        "observations": total,
    }


def _timing_report(
    periods: NumericSeries,
    first_timestamp_ns: int | None,
    last_timestamp_ns: int | None,
    nonpositive_period_count: int,
) -> dict[str, Any]:
    period_summary = periods.summary()
    median_period = period_summary["p50"]
    frequency = None
    if median_period is not None and median_period > 0.0:
        frequency = 1.0 / median_period
    duration = None
    if first_timestamp_ns is not None and last_timestamp_ns is not None:
        duration = (last_timestamp_ns - first_timestamp_ns) / NANOSECONDS_PER_SECOND
    return {
        "first_timestamp_ns": first_timestamp_ns,
        "last_timestamp_ns": last_timestamp_ns,
        "duration_seconds": duration,
        "period_seconds": period_summary,
        "median_frequency_hz": frequency,
        "nonpositive_period_count": nonpositive_period_count,
    }


def _header_timestamp_ns(message: Any) -> int | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    timestamp_ns = int(stamp.sec) * NANOSECONDS_PER_SECOND + int(stamp.nanosec)
    return timestamp_ns if timestamp_ns > 0 else None


def _geometry_matches(
    first: tuple[int, float, float, float, float, float],
    second: tuple[int, float, float, float, float, float],
) -> bool:
    if first[0] != second[0]:
        return False
    if not all(math.isfinite(value) for value in (*first[1:], *second[1:])):
        return False
    return all(
        _finite_values_close(
            left,
            right,
            absolute_tolerance=ANGLE_METADATA_ABSOLUTE_TOLERANCE,
        )
        for left, right in zip(first[1:], second[1:], strict=True)
    )


def _ranges_match(
    first: Sequence[float],
    second: Sequence[float],
    *,
    absolute_tolerance: float,
) -> bool:
    if len(first) != len(second):
        return False
    return all(
        _finite_values_close(left, right, absolute_tolerance=absolute_tolerance)
        for left, right in zip(first, second, strict=True)
    )


def _finite_values_close(
    first: float,
    second: float,
    *,
    absolute_tolerance: float,
) -> bool:
    if math.isnan(first) or math.isnan(second):
        return math.isnan(first) and math.isnan(second)
    if math.isinf(first) or math.isinf(second):
        return first == second
    return abs(first - second) <= absolute_tolerance


def _import_rosbag_tools() -> tuple[Any, Any, Any]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise AnalysisError(
            "ROS 2 Python bag APIs are unavailable. Run this script in the "
            "AI Challenge dev container, after sourcing the ROS 2 environment."
        ) from exc
    return rosbag2_py, deserialize_message, get_message


def detect_storage_id(path: Path, requested_storage_id: str) -> str:
    """Resolve ``mcap`` or ``sqlite3`` from CLI, metadata, or file suffix."""
    if requested_storage_id != "auto":
        return requested_storage_id

    metadata_path = path / "metadata.yaml" if path.is_dir() else path.parent / "metadata.yaml"
    if metadata_path.is_file():
        metadata_text = metadata_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r"^\s*storage_identifier:\s*['\"]?([^'\"\s#]+)",
            metadata_text,
            flags=re.MULTILINE,
        )
        if match:
            return match.group(1)

    lowercase_name = path.name.lower()
    if lowercase_name.endswith((".mcap", ".mcap.zstd")):
        return "mcap"
    if lowercase_name.endswith(".db3"):
        return "sqlite3"
    if path.is_dir():
        has_mcap = any(path.glob("*.mcap")) or any(path.glob("*.mcap.zstd"))
        has_sqlite = any(path.glob("*.db3"))
        if has_mcap and not has_sqlite:
            return "mcap"
        if has_sqlite and not has_mcap:
            return "sqlite3"
    raise AnalysisError(
        "Could not infer rosbag2 storage. Pass --storage-id mcap or --storage-id sqlite3."
    )


def iter_laser_scans(
    bag_path: Path,
    *,
    storage_id: str,
    topic: str,
) -> tuple[str, Iterator[tuple[Any, int]]]:
    """Open a rosbag2 reader and return the topic type plus a streaming iterator."""
    rosbag2_py, deserialize_message, get_message = _import_rosbag_tools()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path),
        storage_id=storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as exc:
        raise AnalysisError(
            f"Failed to open rosbag2 path {bag_path} with storage_id={storage_id}: {exc}"
        ) from exc

    topic_types = {
        topic_metadata.name: topic_metadata.type
        for topic_metadata in reader.get_all_topics_and_types()
    }
    if topic not in topic_types:
        available = ", ".join(sorted(topic_types)) or "<none>"
        raise AnalysisError(
            f"Topic {topic!r} is not present in the bag. Available topics: {available}"
        )
    message_type = topic_types[topic]
    if message_type != EXPECTED_MESSAGE_TYPE:
        raise AnalysisError(
            f"Topic {topic!r} has type {message_type!r}; expected {EXPECTED_MESSAGE_TYPE!r}"
        )
    message_class = get_message(message_type)

    def messages() -> Iterator[tuple[Any, int]]:
        while reader.has_next():
            topic_name, serialized_data, bag_timestamp_ns = reader.read_next()
            if topic_name != topic:
                continue
            try:
                message = deserialize_message(serialized_data, message_class)
            except Exception as exc:
                raise AnalysisError(
                    f"Failed to deserialize {topic} at bag timestamp {bag_timestamp_ns}: {exc}"
                ) from exc
            yield message, int(bag_timestamp_ns)

    return message_type, messages()


def analyze_bag(args: argparse.Namespace) -> dict[str, Any]:
    """Read the selected LaserScan stream and return a report dictionary."""
    bag_path = args.bag.expanduser().resolve()
    if not bag_path.exists():
        raise AnalysisError(f"Bag path does not exist: {bag_path}")
    storage_id = detect_storage_id(bag_path, args.storage_id)
    message_type, messages = iter_laser_scans(
        bag_path,
        storage_id=storage_id,
        topic=args.topic,
    )
    accumulator = LidarStatisticsAccumulator(
        topic=args.topic,
        source_path=bag_path,
        storage_id=storage_id,
        message_type=message_type,
        distance_bin_width=args.distance_bin_width,
        frame_hold_atol=args.frame_hold_atol,
        far_jump_threshold=args.far_jump_threshold,
    )
    for message, bag_timestamp_ns in messages:
        accumulator.add(message, bag_timestamp_ns)
        if args.max_frames is not None and accumulator.frame_count >= args.max_frames:
            break
    return accumulator.build_report(max_frames=args.max_frames)


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact, reviewable Markdown companion to the JSON report."""
    source = report["source"]
    geometry = report["scan_geometry"]
    validity = report["validity"]
    timing = report["timing"]
    invalid_runs = report["consecutive_invalid_runs"]
    hold = report["frame_hold_candidates"]
    wall_leak = report["wall_leak_candidates"]
    lines = [
        "# AWSIM LiDAR statistics",
        "",
        "## Source",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Bag: `{source['bag_path']}`",
        f"- Storage: `{source['storage_id']}`",
        f"- Topic: `{source['topic']}`",
        f"- Type: `{source['message_type']}`",
        f"- Frames: `{source['frame_count']}`",
        "",
        "## Scan metadata",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Beam-count mode | {_format_value(geometry['beam_count']['mode'])} |",
        f"| Beam count stable | `{geometry['beam_count']['stable']}` |",
        f"| angle_min mean [rad] | {_format_summary_mean(geometry['angle_min_radians'])} |",
        f"| angle_max mean [rad] | {_format_summary_mean(geometry['angle_max_radians'])} |",
        (
            "| angle_increment mean [rad] | "
            f"{_format_summary_mean(geometry['angle_increment_radians'])} |"
        ),
        f"| range_min mean | {_format_summary_mean(geometry['range_min'])} |",
        f"| range_max mean | {_format_summary_mean(geometry['range_max'])} |",
        f"| scan_time mean [s] | {_format_summary_mean(geometry['scan_time_seconds'])} |",
        "",
        "Beam-count histogram: "
        f"`{json.dumps(geometry['beam_count']['histogram'], ensure_ascii=False)}`",
        "",
        "## Publish timing",
        "",
        "| Clock | Period p50 [s] | Period p95 [s] | Median frequency [Hz] |",
        "| --- | ---: | ---: | ---: |",
        _timing_markdown_row("Bag timestamp", timing["bag_timestamp"]),
        _timing_markdown_row("Header timestamp", timing["header_timestamp"]),
        "",
        "## Validity",
        "",
        "| Metric | Count | Rate |",
        "| --- | ---: | ---: |",
        _count_rate_row("Valid", validity["valid_points"], validity["valid_rate"]),
        _count_rate_row("Invalid", validity["invalid_points"], validity["invalid_rate"]),
        _count_rate_row("NaN", validity["nan_points"], validity["nan_rate"]),
        _count_rate_row("Inf (total)", validity["inf_points"], validity["inf_rate"]),
        _count_rate_row(
            "+Inf", validity["positive_inf_points"], validity["positive_inf_rate"]
        ),
        _count_rate_row(
            "-Inf", validity["negative_inf_points"], validity["negative_inf_rate"]
        ),
        (
            f"| Finite below range_min | {validity['below_range_min_points']} | - |"
        ),
        (
            f"| Finite above range_max | {validity['above_range_max_points']} | - |"
        ),
        "",
        "## Consecutive invalid sectors",
        "",
        f"- Run count: `{invalid_runs['run_count']}`",
        (
            "- Maximum run width: "
            f"`{_format_value(invalid_runs['width_summary_beams']['max'])}` beams"
        ),
        (
            "- Per-frame maximum p95: "
            f"`{_format_value(invalid_runs['maximum_width_per_frame_beams']['p95'])}` beams"
        ),
        f"- Boundary policy: {invalid_runs['boundary_policy']}",
        "",
        "## Distance-conditioned invalid rate",
        "",
        (
            "> This is a proxy, not ground truth. Invalid samples are assigned the "
            "last valid range observed at the same beam index."
        ),
        "",
        "| Distance bin | Assigned | Invalid | Invalid rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    distance_bins = report["distance_conditioned_invalid_rate"]["bins"]
    if distance_bins:
        for distance_bin in distance_bins:
            lines.append(
                "| "
                f"[{_format_value(distance_bin['lower_inclusive'])}, "
                f"{_format_value(distance_bin['upper_exclusive'])}) | "
                f"{distance_bin['assigned_points']} | "
                f"{distance_bin['invalid_points']} | "
                f"{_format_rate(distance_bin['invalid_rate'])} |"
            )
    else:
        lines.append("| No assignable samples | 0 | 0 | - |")

    lines.extend(
        [
            "",
            "## Worst angle/index invalid rates",
            "",
            "| Beam | Reference angle [rad] | Observations | Invalid rate |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for beam in report["angle_conditioned_invalid_rate"]["worst_20_beams"]:
        lines.append(
            f"| {beam['beam_index']} | "
            f"{_format_value(beam['reference_angle_radians'])} | "
            f"{beam['observations']} | {_format_rate(beam['invalid_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Frame-hold candidates",
            "",
            (
                f"- Comparable pairs: `{hold['comparable_frame_pairs']}`; "
                f"candidates: `{hold['candidate_count']}` "
                f"({_format_rate(hold['candidate_rate'])})"
            ),
            (
                "- Longest consecutive candidate run: "
                f"`{hold['longest_consecutive_candidate_run']}`"
            ),
            f"- Absolute comparison tolerance: `{hold['absolute_tolerance']}`",
            "- Identical frames are candidates only; they do not prove a transport fault.",
            "",
            "## Wall-leak assessment",
            "",
            f"- Classification: `{wall_leak['classification']}`",
            f"- Confirmed count/rate: `{wall_leak['confirmed_count']}` / "
            f"`{wall_leak['confirmed_rate']}`",
            f"- {wall_leak['explanation']}",
            (
                "- Temporal far-jump heuristic: `"
                f"{wall_leak['temporal_far_jump_heuristic']}`"
            ),
            "",
            "## Warnings",
            "",
        ]
    )
    warnings = report["warnings"]
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def _format_summary_mean(summary: dict[str, Any]) -> str:
    return _format_value(summary["mean"])


def _format_rate(rate: float | None) -> str:
    if rate is None:
        return "-"
    return f"{rate:.6%}"


def _timing_markdown_row(label: str, timing: dict[str, Any]) -> str:
    periods = timing["period_seconds"]
    return (
        f"| {label} | {_format_value(periods['p50'])} | "
        f"{_format_value(periods['p95'])} | "
        f"{_format_value(timing['median_frequency_hz'])} |"
    )


def _count_rate_row(label: str, count: int, rate: float | None) -> str:
    return f"| {label} | {count} | {_format_rate(rate)} |"


def write_outputs(report: dict[str, Any], output_directory: Path) -> tuple[Path, Path]:
    """Atomically write the canonical JSON and Markdown calibration reports."""
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "lidar_statistics.json"
    markdown_path = output_directory / "lidar_statistics.md"
    json_text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"
    markdown_text = render_markdown(report)
    _atomic_write_text(json_path, json_text)
    _atomic_write_text(markdown_path, markdown_text)
    return json_path, markdown_path


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze /sensing/lidar/scan from an MCAP or SQLite rosbag2 and "
            "write lidar_statistics.json/.md."
        )
    )
    parser.add_argument("bag", type=Path, help="Rosbag2 directory or .mcap/.db3 path")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument(
        "--storage-id",
        choices=("auto", "mcap", "sqlite3"),
        default="auto",
        help="Storage plugin; auto reads metadata.yaml or file suffix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--distance-bin-width",
        type=float,
        default=1.0,
        help="Width of proxy-distance bins in LaserScan range units",
    )
    parser.add_argument(
        "--frame-hold-atol",
        type=float,
        default=0.0,
        help="Absolute range tolerance for frame-hold candidates (default: exact)",
    )
    parser.add_argument(
        "--far-jump-threshold",
        type=float,
        default=None,
        help=(
            "Optional positive temporal range-jump threshold. Results remain a "
            "heuristic and are never labeled confirmed wall leaks."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional positive frame limit for a quick analysis",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not math.isfinite(args.distance_bin_width) or args.distance_bin_width <= 0.0:
        raise AnalysisError("--distance-bin-width must be finite and positive")
    if not math.isfinite(args.frame_hold_atol) or args.frame_hold_atol < 0.0:
        raise AnalysisError("--frame-hold-atol must be finite and non-negative")
    if args.far_jump_threshold is not None and (
        not math.isfinite(args.far_jump_threshold) or args.far_jump_threshold <= 0.0
    ):
        raise AnalysisError("--far-jump-threshold must be finite and positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise AnalysisError("--max-frames must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        validate_arguments(args)
        report = analyze_bag(args)
        json_path, markdown_path = write_outputs(report, args.output_dir.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

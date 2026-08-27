#!/usr/bin/env python3
"""Identify an AWSIM vehicle-response model from explicitly labelled bag intervals.

The ROS 2 imports are lazy.  ``--help`` and source-level checks therefore work
without the AI Challenge development image.  Identification is fail-closed:
missing topics, invalid timestamps, missing experiment windows, and insufficient
excitation produce ``null`` values with reasons instead of guessed defaults.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
NANOSECONDS_PER_SECOND = 1_000_000_000
FOPDT_TEN_PERCENT = 0.10
FOPDT_TIME_CONSTANT_PERCENT = 1.0 - math.exp(-1.0)
FOPDT_TEN_PERCENT_TIME_CONSTANTS = -math.log(1.0 - FOPDT_TEN_PERCENT)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "calibration" / "awsim_vehicle_model.yaml"

DEFAULT_CONTROL_TOPIC = "/control/command/control_cmd"
DEFAULT_STEERING_TOPIC = "/vehicle/status/steering_status"
DEFAULT_VELOCITY_TOPIC = "/vehicle/status/velocity_status"
DEFAULT_ODOMETRY_TOPIC = "/localization/kinematic_state"
DEFAULT_ACCELERATION_TOPIC = "/localization/acceleration"

CONTROL_MESSAGE_TYPE = "autoware_auto_control_msgs/msg/AckermannControlCommand"
STEERING_MESSAGE_TYPE = "autoware_auto_vehicle_msgs/msg/SteeringReport"
VELOCITY_MESSAGE_TYPE = "autoware_auto_vehicle_msgs/msg/VelocityReport"
ODOMETRY_MESSAGE_TYPE = "nav_msgs/msg/Odometry"
ACCELERATION_MESSAGE_TYPE = "geometry_msgs/msg/AccelWithCovarianceStamped"

PARAMETERS = (
    "steering_gain",
    "steering_time_constant",
    "steering_delay",
    "acceleration_gain",
    "acceleration_delay",
    "effective_wheelbase",
    "velocity_drag",
)
PARAMETER_UNITS = {
    "steering_gain": "dimensionless",
    "steering_time_constant": "seconds",
    "steering_delay": "seconds",
    "acceleration_gain": "dimensionless",
    "acceleration_delay": "seconds",
    "effective_wheelbase": "meters",
    "velocity_drag": "per_second",
}
PARAMETER_KINDS = {
    "steering_gain": "steering_step",
    "steering_time_constant": "steering_step",
    "steering_delay": "steering_step",
    "acceleration_gain": "acceleration_step",
    "acceleration_delay": "acceleration_step",
    "effective_wheelbase": "constant_speed_turn",
    "velocity_drag": "coast",
}
EXPERIMENT_KINDS = (
    "steering_step",
    "steering_sine_sweep",
    "acceleration_step",
    "coast",
    "constant_speed_turn",
)


class AnalysisError(RuntimeError):
    """Raised for invalid input that prevents creation of a truthful report."""


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently accepting the final value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"Experiment manifest contains duplicate key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class ScalarSample:
    """One finite scalar at an absolute ROS or bag timestamp."""

    timestamp_ns: int
    value: float


@dataclass(frozen=True)
class MotionSample:
    """Longitudinal speed and yaw rate sharing one source timestamp."""

    timestamp_ns: int
    speed: float
    yaw_rate: float


@dataclass(frozen=True)
class Experiment:
    """One operator-labelled experiment interval in analysis-relative seconds."""

    identifier: str
    kind: str
    start_seconds: float
    end_seconds: float
    settings: dict[str, float | int]
    notes: str | None = None

    def as_report(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "kind": self.kind,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "settings": self.settings,
            "notes": self.notes,
        }


@dataclass
class TopicReport:
    """Observed contract and extraction diagnostics for one logical topic role."""

    topic: str
    expected_type: str
    actual_type: str | None = None
    status: str = "unchecked"
    message_count: int = 0
    deserialization_error_count: int = 0
    extraction_error_count: int = 0
    missing_timestamp_count: int = 0
    nonfinite_field_counts: Counter[str] = field(default_factory=Counter)
    timestamp_field_counts: Counter[str] = field(default_factory=Counter)
    first_error: str | None = None

    def as_report(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "expected_type": self.expected_type,
            "actual_type": self.actual_type,
            "status": self.status,
            "message_count": self.message_count,
            "deserialization_error_count": self.deserialization_error_count,
            "extraction_error_count": self.extraction_error_count,
            "missing_timestamp_count": self.missing_timestamp_count,
            "nonfinite_field_counts": dict(sorted(self.nonfinite_field_counts.items())),
            "timestamp_field_counts": dict(sorted(self.timestamp_field_counts.items())),
            "first_error": self.first_error,
        }


@dataclass
class BagSeries:
    """Compact scalar series retained while raw bag messages are streamed."""

    steering_command: list[ScalarSample] = field(default_factory=list)
    acceleration_command: list[ScalarSample] = field(default_factory=list)
    steering_response: list[ScalarSample] = field(default_factory=list)
    velocity_motion: list[MotionSample] = field(default_factory=list)
    odometry_motion: list[MotionSample] = field(default_factory=list)
    localization_acceleration: list[ScalarSample] = field(default_factory=list)
    nonmonotonic_counts: Counter[str] = field(default_factory=Counter)


@dataclass
class BagData:
    """Streaming extraction result and the selected analysis clock origin."""

    storage_id: str
    bag_start_timestamp_ns: int
    bag_end_timestamp_ns: int
    analysis_origin_timestamp_ns: int
    timestamp_source: str
    topics: dict[str, TopicReport]
    series: BagSeries


@dataclass(frozen=True)
class Candidate:
    """One parameter candidate produced by one valid experiment interval."""

    experiment_id: str
    value: float
    evidence: dict[str, Any]

    def as_report(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "value": self.value,
            "evidence": self.evidence,
        }


def _finite_number(
    mapping: dict[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{key} must be a finite number, not {value!r}")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise AnalysisError(f"{key} must be representable as a finite float") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"{key} must be finite")
    if minimum is not None:
        invalid = result <= minimum if strict_minimum else result < minimum
        if invalid:
            relation = "greater than" if strict_minimum else "at least"
            raise AnalysisError(f"{key} must be {relation} {minimum}")
    return result


def _positive_integer(mapping: dict[str, Any], key: str, *, minimum: int) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnalysisError(f"{key} must be an integer >= {minimum}, not {value!r}")
    return value


def load_experiments(
    path: Path,
    *,
    timestamp_source: str,
    acceleration_response: str,
) -> list[Experiment]:
    """Load and strictly validate the operator-labelled JSON experiment manifest."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except OSError as exc:
        raise AnalysisError(f"Could not read experiment manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Experiment manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AnalysisError("Experiment manifest root must be an object")
    manifest_schema = payload.get("schema_version")
    if (
        isinstance(manifest_schema, bool)
        or not isinstance(manifest_schema, int)
        or manifest_schema != SCHEMA_VERSION
    ):
        raise AnalysisError(f"Experiment manifest schema_version must be {SCHEMA_VERSION}")
    expected_reference = (
        "selected_message_stamp_seconds"
        if timestamp_source == "message"
        else "bag_start_seconds"
    )
    if payload.get("time_reference") != expected_reference:
        raise AnalysisError(
            "Experiment manifest time_reference must be "
            f"{expected_reference!r} for --timestamp-source={timestamp_source}"
        )
    raw_experiments = payload.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise AnalysisError("Experiment manifest experiments must be a non-empty list")

    experiments: list[Experiment] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(raw_experiments):
        prefix = f"experiments[{index}]"
        if not isinstance(raw, dict):
            raise AnalysisError(f"{prefix} must be an object")
        identifier = raw.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise AnalysisError(f"{prefix}.id must be a non-empty string")
        if identifier in identifiers:
            raise AnalysisError(f"duplicate experiment id {identifier!r}")
        identifiers.add(identifier)
        kind = raw.get("kind")
        if kind not in EXPERIMENT_KINDS:
            raise AnalysisError(
                f"{prefix}.kind must be one of {', '.join(EXPERIMENT_KINDS)}"
            )
        start = _finite_number(raw, "start_seconds", minimum=0.0)
        end = _finite_number(raw, "end_seconds", minimum=start, strict_minimum=True)
        settings: dict[str, float | int] = {}

        if kind in ("steering_step", "acceleration_step"):
            input_start = _finite_number(
                raw, "input_start_seconds", minimum=start, strict_minimum=True
            )
            steady_start = _finite_number(
                raw, "steady_start_seconds", minimum=input_start, strict_minimum=True
            )
            if steady_start >= end:
                raise AnalysisError(f"{prefix}.steady_start_seconds must be < end_seconds")
            settings.update(
                input_start_seconds=input_start,
                steady_start_seconds=steady_start,
                minimum_command_step=_finite_number(
                    raw, "minimum_command_step", minimum=0.0, strict_minimum=True
                ),
                minimum_response_step=_finite_number(
                    raw, "minimum_response_step", minimum=0.0, strict_minimum=True
                ),
                minimum_samples_per_window=_positive_integer(
                    raw, "minimum_samples_per_window", minimum=2
                ),
            )
            if kind == "acceleration_step" and acceleration_response == "velocity_derivative":
                settings["maximum_derivative_gap_seconds"] = _finite_number(
                    raw,
                    "maximum_derivative_gap_seconds",
                    minimum=0.0,
                    strict_minimum=True,
                )
        elif kind == "steering_sine_sweep":
            settings.update(
                minimum_samples=_positive_integer(raw, "minimum_samples", minimum=3),
                minimum_command_peak_to_peak=_finite_number(
                    raw,
                    "minimum_command_peak_to_peak",
                    minimum=0.0,
                    strict_minimum=True,
                ),
                minimum_response_peak_to_peak=_finite_number(
                    raw,
                    "minimum_response_peak_to_peak",
                    minimum=0.0,
                    strict_minimum=True,
                ),
            )
        elif kind == "coast":
            minimum_r_squared = _finite_number(
                raw, "minimum_r_squared", minimum=0.0
            )
            if minimum_r_squared > 1.0:
                raise AnalysisError("minimum_r_squared must be <= 1.0")
            settings.update(
                max_abs_command_acceleration=_finite_number(
                    raw, "max_abs_command_acceleration", minimum=0.0
                ),
                minimum_speed_mps=_finite_number(
                    raw, "minimum_speed_mps", minimum=0.0, strict_minimum=True
                ),
                minimum_samples=_positive_integer(raw, "minimum_samples", minimum=3),
                minimum_r_squared=minimum_r_squared,
            )
        elif kind == "constant_speed_turn":
            settings.update(
                minimum_speed_mps=_finite_number(
                    raw, "minimum_speed_mps", minimum=0.0, strict_minimum=True
                ),
                minimum_abs_steering_radians=_finite_number(
                    raw,
                    "minimum_abs_steering_radians",
                    minimum=0.0,
                    strict_minimum=True,
                ),
                minimum_abs_yaw_rate_radians_per_second=_finite_number(
                    raw,
                    "minimum_abs_yaw_rate_radians_per_second",
                    minimum=0.0,
                    strict_minimum=True,
                ),
                max_alignment_gap_seconds=_finite_number(
                    raw,
                    "max_alignment_gap_seconds",
                    minimum=0.0,
                    strict_minimum=True,
                ),
                minimum_samples=_positive_integer(raw, "minimum_samples", minimum=3),
            )
        notes = raw.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise AnalysisError(f"{prefix}.notes must be a string or null")
        experiments.append(Experiment(identifier, kind, start, end, settings, notes))
    return experiments


def detect_storage_id(path: Path, requested: str) -> str:
    """Resolve rosbag2 storage from CLI, metadata, or an unambiguous suffix."""
    if requested != "auto":
        return requested
    metadata = path / "metadata.yaml" if path.is_dir() else path.parent / "metadata.yaml"
    if metadata.is_file():
        text = metadata.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r"^\s*storage_identifier:\s*['\"]?([^'\"\s#]+)",
            text,
            flags=re.MULTILINE,
        )
        if match:
            return match.group(1)
    lower_name = path.name.lower()
    if lower_name.endswith((".mcap", ".mcap.zstd")):
        return "mcap"
    if lower_name.endswith(".db3"):
        return "sqlite3"
    if path.is_dir():
        has_mcap = any(path.glob("*.mcap")) or any(path.glob("*.mcap.zstd"))
        has_sqlite = any(path.glob("*.db3"))
        if has_mcap != has_sqlite:
            return "mcap" if has_mcap else "sqlite3"
    raise AnalysisError(
        "Could not infer rosbag2 storage; pass --storage-id mcap or sqlite3"
    )


def _import_rosbag_tools() -> tuple[Any, Any, Any]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise AnalysisError(
            "ROS 2 Python bag APIs are unavailable. Run in the AI Challenge dev "
            "container after sourcing the workspace."
        ) from exc
    return rosbag2_py, deserialize_message, get_message


def _stamp_ns(stamp: Any) -> int | None:
    if stamp is None:
        return None
    try:
        seconds = stamp.sec
        nanoseconds = stamp.nanosec
    except AttributeError:
        return None
    if (
        isinstance(seconds, bool)
        or isinstance(nanoseconds, bool)
        or not isinstance(seconds, int)
        or not isinstance(nanoseconds, int)
        or seconds < 0
        or nanoseconds < 0
        or nanoseconds >= NANOSECONDS_PER_SECOND
    ):
        return None
    result = seconds * NANOSECONDS_PER_SECOND + nanoseconds
    return result if result > 0 else None


def _message_timestamp_ns(
    message: Any,
    *,
    role: str,
    channel: str,
) -> tuple[int | None, str | None]:
    candidates: list[tuple[str, Any]] = []
    if role == "control":
        nested = getattr(message, channel, None)
        candidates.append((f"{channel}.stamp", getattr(nested, "stamp", None)))
        candidates.append(("stamp", getattr(message, "stamp", None)))
    elif role in ("steering", "velocity"):
        candidates.append(("stamp", getattr(message, "stamp", None)))
        header = getattr(message, "header", None)
        candidates.append(("header.stamp", getattr(header, "stamp", None)))
    else:
        header = getattr(message, "header", None)
        candidates.append(("header.stamp", getattr(header, "stamp", None)))
        candidates.append(("stamp", getattr(message, "stamp", None)))
    for name, stamp in candidates:
        timestamp_ns = _stamp_ns(stamp)
        if timestamp_ns is not None:
            return timestamp_ns, name
    return None, None


def _selected_timestamp_ns(
    message: Any,
    *,
    role: str,
    channel: str,
    bag_timestamp_ns: int,
    timestamp_source: str,
    report: TopicReport,
) -> int | None:
    if timestamp_source == "bag":
        report.timestamp_field_counts["bag_timestamp"] += 1
        return bag_timestamp_ns
    timestamp_ns, field_name = _message_timestamp_ns(
        message,
        role=role,
        channel=channel,
    )
    if timestamp_ns is None:
        report.missing_timestamp_count += 1
        return None
    report.timestamp_field_counts[field_name or "unknown"] += 1
    return timestamp_ns


def _append_scalar(
    samples: list[ScalarSample],
    *,
    timestamp_ns: int | None,
    value: Any,
    field_name: str,
    series_name: str,
    report: TopicReport,
    store: BagSeries,
) -> None:
    if timestamp_ns is None:
        return
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError(f"{field_name} is not numeric: {exc}") from exc
    if not math.isfinite(number):
        report.nonfinite_field_counts[field_name] += 1
        return
    if samples and timestamp_ns < samples[-1].timestamp_ns:
        store.nonmonotonic_counts[series_name] += 1
    samples.append(ScalarSample(timestamp_ns, number))


def _append_motion(
    samples: list[MotionSample],
    *,
    timestamp_ns: int | None,
    speed: Any,
    yaw_rate: Any,
    speed_field: str,
    yaw_rate_field: str,
    series_name: str,
    report: TopicReport,
    store: BagSeries,
) -> None:
    if timestamp_ns is None:
        return
    speed_number = float(speed)
    yaw_rate_number = float(yaw_rate)
    speed_is_finite = math.isfinite(speed_number)
    yaw_rate_is_finite = math.isfinite(yaw_rate_number)
    if not speed_is_finite:
        report.nonfinite_field_counts[speed_field] += 1
    if not yaw_rate_is_finite:
        report.nonfinite_field_counts[yaw_rate_field] += 1
    # Acceleration and coast identification need speed but not yaw rate.  Keep a
    # finite speed sample even when yaw rate is unavailable; the wheelbase
    # estimator rejects that sample separately instead of coupling the channels.
    if not speed_is_finite:
        return
    if samples and timestamp_ns < samples[-1].timestamp_ns:
        store.nonmonotonic_counts[series_name] += 1
    samples.append(MotionSample(timestamp_ns, speed_number, yaw_rate_number))


def _extract_message(
    role: str,
    message: Any,
    bag_timestamp_ns: int,
    *,
    timestamp_source: str,
    report: TopicReport,
    store: BagSeries,
) -> None:
    if role == "control":
        lateral = message.lateral
        longitudinal = message.longitudinal
        steering_timestamp = _selected_timestamp_ns(
            message,
            role=role,
            channel="lateral",
            bag_timestamp_ns=bag_timestamp_ns,
            timestamp_source=timestamp_source,
            report=report,
        )
        acceleration_timestamp = _selected_timestamp_ns(
            message,
            role=role,
            channel="longitudinal",
            bag_timestamp_ns=bag_timestamp_ns,
            timestamp_source=timestamp_source,
            report=report,
        )
        _append_scalar(
            store.steering_command,
            timestamp_ns=steering_timestamp,
            value=lateral.steering_tire_angle,
            field_name="lateral.steering_tire_angle",
            series_name="steering_command",
            report=report,
            store=store,
        )
        _append_scalar(
            store.acceleration_command,
            timestamp_ns=acceleration_timestamp,
            value=longitudinal.acceleration,
            field_name="longitudinal.acceleration",
            series_name="acceleration_command",
            report=report,
            store=store,
        )
    elif role == "steering":
        timestamp = _selected_timestamp_ns(
            message,
            role=role,
            channel="",
            bag_timestamp_ns=bag_timestamp_ns,
            timestamp_source=timestamp_source,
            report=report,
        )
        _append_scalar(
            store.steering_response,
            timestamp_ns=timestamp,
            value=message.steering_tire_angle,
            field_name="steering_tire_angle",
            series_name="steering_response",
            report=report,
            store=store,
        )
    elif role == "velocity":
        timestamp = _selected_timestamp_ns(
            message,
            role=role,
            channel="",
            bag_timestamp_ns=bag_timestamp_ns,
            timestamp_source=timestamp_source,
            report=report,
        )
        _append_motion(
            store.velocity_motion,
            timestamp_ns=timestamp,
            speed=message.longitudinal_velocity,
            yaw_rate=message.heading_rate,
            speed_field="longitudinal_velocity",
            yaw_rate_field="heading_rate",
            series_name="velocity_motion",
            report=report,
            store=store,
        )
    elif role == "odometry":
        timestamp = _selected_timestamp_ns(
            message,
            role=role,
            channel="",
            bag_timestamp_ns=bag_timestamp_ns,
            timestamp_source=timestamp_source,
            report=report,
        )
        twist = message.twist.twist
        _append_motion(
            store.odometry_motion,
            timestamp_ns=timestamp,
            speed=twist.linear.x,
            yaw_rate=twist.angular.z,
            speed_field="twist.twist.linear.x",
            yaw_rate_field="twist.twist.angular.z",
            series_name="odometry_motion",
            report=report,
            store=store,
        )
    elif role == "acceleration":
        timestamp = _selected_timestamp_ns(
            message,
            role=role,
            channel="",
            bag_timestamp_ns=bag_timestamp_ns,
            timestamp_source=timestamp_source,
            report=report,
        )
        _append_scalar(
            store.localization_acceleration,
            timestamp_ns=timestamp,
            value=message.accel.accel.linear.x,
            field_name="accel.accel.linear.x",
            series_name="localization_acceleration",
            report=report,
            store=store,
        )


def stream_bag(
    bag_path: Path,
    *,
    storage_id: str,
    timestamp_source: str,
    topic_specs: dict[str, tuple[str, str]],
) -> BagData:
    """Stream the bag once and retain only finite scalar response channels."""
    rosbag2_py, deserialize_message, get_message = _import_rosbag_tools()
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            ),
        )
    except Exception as exc:
        raise AnalysisError(
            f"Failed to open {bag_path} with storage_id={storage_id}: {exc}"
        ) from exc

    available_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    topics: dict[str, TopicReport] = {}
    message_classes: dict[str, Any] = {}
    topic_to_role: dict[str, str] = {}
    for role, (topic, expected_type) in topic_specs.items():
        report = TopicReport(
            topic=topic,
            expected_type=expected_type,
            actual_type=available_types.get(topic),
        )
        topics[role] = report
        if report.actual_type is None:
            report.status = "missing"
        elif report.actual_type != expected_type:
            report.status = "type_mismatch"
        else:
            report.status = "available"
            try:
                message_classes[role] = get_message(expected_type)
            except Exception as exc:
                report.status = "message_class_unavailable"
                report.first_error = str(exc)
                continue
            topic_to_role[topic] = role

    store = BagSeries()
    bag_start_ns: int | None = None
    bag_end_ns: int | None = None
    while reader.has_next():
        topic_name, serialized, raw_timestamp_ns = reader.read_next()
        bag_timestamp_ns = int(raw_timestamp_ns)
        if bag_start_ns is None:
            bag_start_ns = bag_timestamp_ns
        bag_end_ns = bag_timestamp_ns
        role = topic_to_role.get(topic_name)
        if role is None:
            continue
        report = topics[role]
        report.message_count += 1
        try:
            message = deserialize_message(serialized, message_classes[role])
        except Exception as exc:
            report.deserialization_error_count += 1
            report.first_error = report.first_error or str(exc)
            continue
        try:
            _extract_message(
                role,
                message,
                bag_timestamp_ns,
                timestamp_source=timestamp_source,
                report=report,
                store=store,
            )
        except (AnalysisError, AttributeError, TypeError, ValueError, OverflowError) as exc:
            report.extraction_error_count += 1
            report.first_error = report.first_error or str(exc)

    if bag_start_ns is None or bag_end_ns is None:
        raise AnalysisError("The bag contains no messages")
    for report in topics.values():
        if report.status != "available":
            continue
        if report.message_count == 0:
            report.status = "no_messages"
        elif report.deserialization_error_count or report.extraction_error_count:
            report.status = "extraction_failed"

    all_timestamps = [
        sample.timestamp_ns
        for samples in (
            store.steering_command,
            store.acceleration_command,
            store.steering_response,
            store.localization_acceleration,
        )
        for sample in samples
    ]
    all_timestamps.extend(sample.timestamp_ns for sample in store.velocity_motion)
    all_timestamps.extend(sample.timestamp_ns for sample in store.odometry_motion)
    if timestamp_source == "message":
        if not all_timestamps:
            origin_ns = bag_start_ns
        else:
            origin_ns = min(all_timestamps)
    else:
        origin_ns = bag_start_ns
    return BagData(
        storage_id=storage_id,
        bag_start_timestamp_ns=bag_start_ns,
        bag_end_timestamp_ns=bag_end_ns,
        analysis_origin_timestamp_ns=origin_ns,
        timestamp_source=timestamp_source,
        topics=topics,
        series=store,
    )


def _seconds(timestamp_ns: int, origin_ns: int) -> float:
    return (timestamp_ns - origin_ns) / NANOSECONDS_PER_SECOND


def _scalar_window(
    samples: Sequence[ScalarSample],
    experiment: Experiment,
    origin_ns: int,
    *,
    start: float | None = None,
    end: float | None = None,
    include_end: bool = True,
) -> list[tuple[float, float]]:
    lower = experiment.start_seconds if start is None else start
    upper = experiment.end_seconds if end is None else end
    result = []
    for sample in samples:
        time_seconds = _seconds(sample.timestamp_ns, origin_ns)
        if time_seconds < lower:
            continue
        if time_seconds > upper or (not include_end and time_seconds == upper):
            continue
        result.append((time_seconds, sample.value))
    return result


def _motion_window(
    samples: Sequence[MotionSample],
    experiment: Experiment,
    origin_ns: int,
) -> list[tuple[float, float, float]]:
    return [
        (_seconds(sample.timestamp_ns, origin_ns), sample.speed, sample.yaw_rate)
        for sample in samples
        if experiment.start_seconds
        <= _seconds(sample.timestamp_ns, origin_ns)
        <= experiment.end_seconds
    ]


def _series_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "median": None, "stddev": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _crossing_time(
    samples: Sequence[tuple[float, float]],
    *,
    baseline: float,
    delta: float,
    level: float,
    not_before: float,
) -> float | None:
    if delta == 0.0:
        return None
    for left, right in zip(samples, samples[1:], strict=False):
        if right[0] < not_before or right[0] <= left[0]:
            continue
        left_level = (left[1] - baseline) / delta
        right_level = (right[1] - baseline) / delta
        if left_level <= level <= right_level and right_level > left_level:
            fraction = (level - left_level) / (right_level - left_level)
            crossing = left[0] + fraction * (right[0] - left[0])
            return crossing if crossing >= not_before else None
    return None


def _step_candidates(
    experiment: Experiment,
    command: Sequence[ScalarSample],
    response: Sequence[ScalarSample],
    *,
    origin_ns: int,
    gain_parameter: str,
    delay_parameter: str,
    time_constant_parameter: str | None,
) -> dict[str, Candidate | str]:
    input_start = float(experiment.settings["input_start_seconds"])
    steady_start = float(experiment.settings["steady_start_seconds"])
    minimum_samples = int(experiment.settings["minimum_samples_per_window"])
    baseline_command = _scalar_window(
        command,
        experiment,
        origin_ns,
        end=input_start,
        include_end=False,
    )
    baseline_response = _scalar_window(
        response,
        experiment,
        origin_ns,
        end=input_start,
        include_end=False,
    )
    steady_command = _scalar_window(
        command,
        experiment,
        origin_ns,
        start=steady_start,
    )
    steady_response = _scalar_window(
        response,
        experiment,
        origin_ns,
        start=steady_start,
    )
    outputs = [gain_parameter, delay_parameter]
    if time_constant_parameter is not None:
        outputs.append(time_constant_parameter)
    for label, window in (
        ("baseline command", baseline_command),
        ("baseline response", baseline_response),
        ("steady command", steady_command),
        ("steady response", steady_response),
    ):
        if len(window) < minimum_samples:
            reason = f"{label} has {len(window)} samples; requires {minimum_samples}"
            return {parameter: reason for parameter in outputs}

    command_baseline = statistics.median(value for _, value in baseline_command)
    command_steady = statistics.median(value for _, value in steady_command)
    response_baseline = statistics.median(value for _, value in baseline_response)
    response_steady = statistics.median(value for _, value in steady_response)
    command_delta = command_steady - command_baseline
    response_delta = response_steady - response_baseline
    if abs(command_delta) < float(experiment.settings["minimum_command_step"]):
        reason = "measured command step is below minimum_command_step"
        return {parameter: reason for parameter in outputs}
    if abs(response_delta) < float(experiment.settings["minimum_response_step"]):
        reason = "measured response step is below minimum_response_step"
        return {parameter: reason for parameter in outputs}
    gain = response_delta / command_delta
    if not math.isfinite(gain) or gain <= 0.0:
        reason = f"step gain is not finite and positive: {gain!r}"
        return {parameter: reason for parameter in outputs}

    evidence = {
        "command_baseline": command_baseline,
        "command_steady": command_steady,
        "response_baseline": response_baseline,
        "response_steady": response_steady,
        "command_delta": command_delta,
        "response_delta": response_delta,
        "baseline_command": _series_summary([value for _, value in baseline_command]),
        "steady_command": _series_summary([value for _, value in steady_command]),
        "baseline_response": _series_summary([value for _, value in baseline_response]),
        "steady_response": _series_summary([value for _, value in steady_response]),
    }
    result: dict[str, Candidate | str] = {
        gain_parameter: Candidate(experiment.identifier, gain, evidence)
    }
    all_command = _scalar_window(command, experiment, origin_ns)
    all_response = _scalar_window(response, experiment, origin_ns)
    command_crossing = _crossing_time(
        all_command,
        baseline=command_baseline,
        delta=command_delta,
        level=0.5,
        not_before=input_start,
    )
    if command_crossing is None:
        dynamic_reason = "command 50% crossing was not observed after input_start_seconds"
    else:
        response_ten = _crossing_time(
            all_response,
            baseline=response_baseline,
            delta=response_delta,
            level=FOPDT_TEN_PERCENT,
            not_before=command_crossing,
        )
        response_tau = _crossing_time(
            all_response,
            baseline=response_baseline,
            delta=response_delta,
            level=FOPDT_TIME_CONSTANT_PERCENT,
            not_before=command_crossing,
        )
        if response_ten is None or response_tau is None or response_tau <= response_ten:
            dynamic_reason = "response 10% and 63.2% crossings were not observed in order"
        else:
            time_constant = (response_tau - response_ten) / (
                1.0 - FOPDT_TEN_PERCENT_TIME_CONSTANTS
            )
            dead_time_absolute = (
                response_ten
                - FOPDT_TEN_PERCENT_TIME_CONSTANTS * time_constant
            )
            delay = dead_time_absolute - command_crossing
            if not math.isfinite(time_constant) or time_constant <= 0.0:
                dynamic_reason = "derived first-order time constant is not positive"
            elif not math.isfinite(delay) or delay < 0.0:
                dynamic_reason = (
                    "derived delay is negative; response or timestamp quality is insufficient"
                )
            else:
                dynamics_evidence = {
                    **evidence,
                    "command_50_percent_time_seconds": command_crossing,
                    "response_10_percent_time_seconds": response_ten,
                    "response_63_2_percent_time_seconds": response_tau,
                    "derived_time_constant_seconds": time_constant,
                    "method": "FOPDT 10%/63.2% crossing interpolation",
                }
                result[delay_parameter] = Candidate(
                    experiment.identifier, delay, dynamics_evidence
                )
                if time_constant_parameter is not None:
                    result[time_constant_parameter] = Candidate(
                        experiment.identifier, time_constant, dynamics_evidence
                    )
                dynamic_reason = ""
    if dynamic_reason:
        result[delay_parameter] = dynamic_reason
        if time_constant_parameter is not None:
            result[time_constant_parameter] = dynamic_reason
    return result


def _derive_acceleration(
    speed: Sequence[MotionSample],
    experiment: Experiment,
    *,
    origin_ns: int,
) -> list[ScalarSample]:
    """Differentiate only inside one labelled interval with a nonuniform stencil."""
    interval_samples = [
        sample
        for sample in speed
        if experiment.start_seconds
        <= _seconds(sample.timestamp_ns, origin_ns)
        <= experiment.end_seconds
    ]
    maximum_gap = float(experiment.settings["maximum_derivative_gap_seconds"])
    result: list[ScalarSample] = []
    for previous, current, following in zip(
        interval_samples,
        interval_samples[1:],
        interval_samples[2:],
        strict=False,
    ):
        previous_gap = (
            current.timestamp_ns - previous.timestamp_ns
        ) / NANOSECONDS_PER_SECOND
        following_gap = (
            following.timestamp_ns - current.timestamp_ns
        ) / NANOSECONDS_PER_SECOND
        if (
            previous_gap <= 0.0
            or following_gap <= 0.0
            or previous_gap > maximum_gap
            or following_gap > maximum_gap
        ):
            continue
        total_gap = previous_gap + following_gap
        acceleration = (
            -following_gap / (previous_gap * total_gap) * previous.speed
            + (following_gap - previous_gap)
            / (previous_gap * following_gap)
            * current.speed
            + previous_gap / (following_gap * total_gap) * following.speed
        )
        if math.isfinite(acceleration):
            result.append(ScalarSample(current.timestamp_ns, acceleration))
    return result


def _interpolate_scalar(
    samples: Sequence[tuple[float, float]],
    times: Sequence[float],
    target: float,
    *,
    max_gap: float,
) -> float | None:
    index = bisect.bisect_left(times, target)
    if index < len(times) and times[index] == target:
        return samples[index][1]
    if index == 0 or index == len(times):
        return None
    left_time, left_value = samples[index - 1]
    right_time, right_value = samples[index]
    if target - left_time > max_gap or right_time - target > max_gap:
        return None
    if right_time <= left_time:
        return None
    weight = (target - left_time) / (right_time - left_time)
    return left_value * (1.0 - weight) + right_value * weight


def _wheelbase_candidate(
    experiment: Experiment,
    steering: Sequence[ScalarSample],
    motion: Sequence[MotionSample],
    *,
    origin_ns: int,
) -> Candidate | str:
    steering_window = _scalar_window(steering, experiment, origin_ns)
    motion_window = _motion_window(motion, experiment, origin_ns)
    minimum_samples = int(experiment.settings["minimum_samples"])
    if len(steering_window) < 2:
        return "fewer than two steering-status samples are available for interpolation"
    if len(motion_window) < minimum_samples:
        return f"motion window has {len(motion_window)} samples; requires {minimum_samples}"
    steering_times = [time_value for time_value, _ in steering_window]
    wheelbases: list[float] = []
    rejected = Counter()
    for time_value, speed, yaw_rate in motion_window:
        actual_steering = _interpolate_scalar(
            steering_window,
            steering_times,
            time_value,
            max_gap=float(experiment.settings["max_alignment_gap_seconds"]),
        )
        if actual_steering is None:
            rejected["alignment_gap"] += 1
            continue
        if speed < float(experiment.settings["minimum_speed_mps"]):
            rejected["speed"] += 1
            continue
        if abs(actual_steering) < float(
            experiment.settings["minimum_abs_steering_radians"]
        ):
            rejected["steering"] += 1
            continue
        if not math.isfinite(yaw_rate):
            rejected["nonfinite_yaw_rate"] += 1
            continue
        if abs(yaw_rate) < float(
            experiment.settings["minimum_abs_yaw_rate_radians_per_second"]
        ):
            rejected["yaw_rate"] += 1
            continue
        if actual_steering * yaw_rate <= 0.0:
            rejected["sign_mismatch"] += 1
            continue
        wheelbase = speed * math.tan(actual_steering) / yaw_rate
        if not math.isfinite(wheelbase) or wheelbase <= 0.0:
            rejected["nonpositive_or_nonfinite"] += 1
            continue
        wheelbases.append(wheelbase)
    if len(wheelbases) < minimum_samples:
        return (
            f"only {len(wheelbases)} valid aligned turn samples; requires "
            f"{minimum_samples}; rejected={dict(rejected)}"
        )
    return Candidate(
        experiment.identifier,
        statistics.median(wheelbases),
        {
            "method": "median(v * tan(actual_steering) / yaw_rate)",
            "valid_samples": len(wheelbases),
            "rejected_samples": dict(sorted(rejected.items())),
            "sample_values_meters": _series_summary(wheelbases),
        },
    )


def _linear_regression(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float] | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    denominator = math.fsum((value - mean_x) ** 2 for value in x)
    if denominator <= 0.0:
        return None
    slope = math.fsum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x, y, strict=True)
    ) / denominator
    intercept = mean_y - slope * mean_x
    residual = math.fsum(
        (y_value - (intercept + slope * x_value)) ** 2
        for x_value, y_value in zip(x, y, strict=True)
    )
    total = math.fsum((value - mean_y) ** 2 for value in y)
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return slope, intercept, r_squared


def _drag_candidate(
    experiment: Experiment,
    acceleration_command: Sequence[ScalarSample],
    motion: Sequence[MotionSample],
    *,
    origin_ns: int,
) -> Candidate | str:
    command_window = _scalar_window(acceleration_command, experiment, origin_ns)
    motion_window = _motion_window(motion, experiment, origin_ns)
    minimum_samples = int(experiment.settings["minimum_samples"])
    if len(command_window) < minimum_samples:
        return (
            f"coast command window has {len(command_window)} samples; "
            f"requires {minimum_samples}"
        )
    maximum_command = max(abs(value) for _, value in command_window)
    allowed_command = float(experiment.settings["max_abs_command_acceleration"])
    if maximum_command > allowed_command:
        return (
            f"coast command magnitude {maximum_command} exceeds explicitly supplied "
            f"limit {allowed_command}"
        )
    filtered = [
        (time_value, speed)
        for time_value, speed, _ in motion_window
        if speed >= float(experiment.settings["minimum_speed_mps"])
    ]
    if len(filtered) < minimum_samples:
        return f"coast speed window has {len(filtered)} usable samples; requires {minimum_samples}"
    start_time = filtered[0][0]
    times = [time_value - start_time for time_value, _ in filtered]
    log_speeds = [math.log(speed) for _, speed in filtered]
    fit = _linear_regression(times, log_speeds)
    if fit is None:
        return "log-speed regression is rank deficient"
    slope, intercept, r_squared = fit
    drag = -slope
    if not math.isfinite(drag) or drag <= 0.0:
        return "coast speed did not exhibit positive exponential drag"
    if r_squared < float(experiment.settings["minimum_r_squared"]):
        return (
            f"coast log-speed r_squared {r_squared} is below explicitly supplied "
            f"minimum {experiment.settings['minimum_r_squared']}"
        )
    return Candidate(
        experiment.identifier,
        drag,
        {
            "method": "OLS log(v) = intercept - velocity_drag * time",
            "valid_samples": len(filtered),
            "maximum_abs_command_acceleration": maximum_command,
            "intercept": intercept,
            "r_squared": r_squared,
            "speed_mps": _series_summary([speed for _, speed in filtered]),
        },
    )


def _topic_problem(data: BagData, role: str) -> str | None:
    report = data.topics[role]
    if report.status != "available":
        return f"{report.topic} is {report.status}"
    return None


def _series_problem(data: BagData, series_name: str) -> str | None:
    count = data.series.nonmonotonic_counts[series_name]
    if count:
        return f"{series_name} has {count} backwards timestamp transition(s)"
    return None


def _aggregate_parameter(
    parameter: str,
    candidates: Sequence[Candidate],
    failures: Sequence[dict[str, str]],
    experiments: Sequence[Experiment],
) -> dict[str, Any]:
    required_kind = PARAMETER_KINDS[parameter]
    if not candidates:
        if not any(experiment.kind == required_kind for experiment in experiments):
            reason = f"no {required_kind} experiment interval was provided"
        elif failures:
            reason = "; ".join(
                f"{item['experiment_id']}: {item['reason']}" for item in failures
            )
        else:
            reason = "no valid candidate was produced"
        return {
            "status": "unavailable",
            "value": None,
            "unit": PARAMETER_UNITS[parameter],
            "aggregation": None,
            "reason": reason,
            "candidates": [],
            "failed_experiments": list(failures),
        }
    values = [candidate.value for candidate in candidates]
    return {
        "status": "estimated",
        "value": statistics.median(values),
        "unit": PARAMETER_UNITS[parameter],
        "aggregation": "median of valid labelled intervals",
        "reason": None,
        "candidate_summary": _series_summary(values),
        "candidates": [candidate.as_report() for candidate in candidates],
        "failed_experiments": list(failures),
    }


def identify(
    data: BagData,
    experiments: Sequence[Experiment],
    *,
    motion_source: str,
    acceleration_response: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run only estimators whose labelled intervals and measured channels are valid."""
    candidates: dict[str, list[Candidate]] = {parameter: [] for parameter in PARAMETERS}
    failures: dict[str, list[dict[str, str]]] = {parameter: [] for parameter in PARAMETERS}
    diagnostics: list[dict[str, Any]] = []

    motion_role = "velocity" if motion_source == "velocity_status" else "odometry"
    motion_series_name = (
        "velocity_motion" if motion_source == "velocity_status" else "odometry_motion"
    )
    motion = getattr(data.series, motion_series_name)
    if acceleration_response == "velocity_derivative":
        measured_acceleration: Sequence[ScalarSample] = ()
        acceleration_role = motion_role
        acceleration_series_name = motion_series_name
        acceleration_method = (
            f"centered derivative of {motion_source} longitudinal speed"
        )
    else:
        measured_acceleration = data.series.localization_acceleration
        acceleration_role = "acceleration"
        acceleration_series_name = "localization_acceleration"
        acceleration_method = "localization acceleration (end-to-end filtered response)"

    for experiment in experiments:
        diagnostics.append(
            {
                "id": experiment.identifier,
                "kind": experiment.kind,
                "window_seconds": [experiment.start_seconds, experiment.end_seconds],
                "status": "labelled",
            }
        )
        if experiment.kind == "steering_step":
            problems = [
                _topic_problem(data, "control"),
                _topic_problem(data, "steering"),
                _series_problem(data, "steering_command"),
                _series_problem(data, "steering_response"),
            ]
            problem = next((item for item in problems if item), None)
            outputs = ("steering_gain", "steering_time_constant", "steering_delay")
            if problem:
                outcomes: dict[str, Candidate | str] = {
                    parameter: problem for parameter in outputs
                }
            else:
                outcomes = _step_candidates(
                    experiment,
                    data.series.steering_command,
                    data.series.steering_response,
                    origin_ns=data.analysis_origin_timestamp_ns,
                    gain_parameter="steering_gain",
                    delay_parameter="steering_delay",
                    time_constant_parameter="steering_time_constant",
                )
            _record_outcomes(experiment, outcomes, candidates, failures)
        elif experiment.kind == "acceleration_step":
            if acceleration_response == "velocity_derivative":
                measured_acceleration = _derive_acceleration(
                    motion,
                    experiment,
                    origin_ns=data.analysis_origin_timestamp_ns,
                )
            problems = [
                _topic_problem(data, "control"),
                _topic_problem(data, acceleration_role),
                _series_problem(data, "acceleration_command"),
                _series_problem(data, acceleration_series_name),
            ]
            problem = next((item for item in problems if item), None)
            outputs = ("acceleration_gain", "acceleration_delay")
            if problem:
                outcomes = {parameter: problem for parameter in outputs}
            else:
                outcomes = _step_candidates(
                    experiment,
                    data.series.acceleration_command,
                    measured_acceleration,
                    origin_ns=data.analysis_origin_timestamp_ns,
                    gain_parameter="acceleration_gain",
                    delay_parameter="acceleration_delay",
                    time_constant_parameter=None,
                )
                for outcome in outcomes.values():
                    if isinstance(outcome, Candidate):
                        outcome.evidence["acceleration_response_method"] = acceleration_method
            _record_outcomes(experiment, outcomes, candidates, failures)
        elif experiment.kind == "constant_speed_turn":
            problems = [
                _topic_problem(data, "steering"),
                _topic_problem(data, motion_role),
                _series_problem(data, "steering_response"),
                _series_problem(data, motion_series_name),
            ]
            problem = next((item for item in problems if item), None)
            outcome: Candidate | str
            if problem:
                outcome = problem
            else:
                outcome = _wheelbase_candidate(
                    experiment,
                    data.series.steering_response,
                    motion,
                    origin_ns=data.analysis_origin_timestamp_ns,
                )
            _record_outcomes(
                experiment,
                {"effective_wheelbase": outcome},
                candidates,
                failures,
            )
        elif experiment.kind == "coast":
            problems = [
                _topic_problem(data, "control"),
                _topic_problem(data, motion_role),
                _series_problem(data, "acceleration_command"),
                _series_problem(data, motion_series_name),
            ]
            problem = next((item for item in problems if item), None)
            if problem:
                outcome = problem
            else:
                outcome = _drag_candidate(
                    experiment,
                    data.series.acceleration_command,
                    motion,
                    origin_ns=data.analysis_origin_timestamp_ns,
                )
            _record_outcomes(
                experiment,
                {"velocity_drag": outcome},
                candidates,
                failures,
            )
        elif experiment.kind == "steering_sine_sweep":
            command_window = _scalar_window(
                data.series.steering_command,
                experiment,
                data.analysis_origin_timestamp_ns,
            )
            response_window = _scalar_window(
                data.series.steering_response,
                experiment,
                data.analysis_origin_timestamp_ns,
            )
            command_count = len(command_window)
            response_count = len(response_window)
            required = int(experiment.settings["minimum_samples"])
            problems = [
                _topic_problem(data, "control"),
                _topic_problem(data, "steering"),
                _series_problem(data, "steering_command"),
                _series_problem(data, "steering_response"),
            ]
            problem = next((item for item in problems if item), None)
            command_peak_to_peak = (
                max(value for _, value in command_window)
                - min(value for _, value in command_window)
                if command_window
                else None
            )
            response_peak_to_peak = (
                max(value for _, value in response_window)
                - min(value for _, value in response_window)
                if response_window
                else None
            )
            enough_samples = command_count >= required and response_count >= required
            enough_excitation = (
                command_peak_to_peak is not None
                and response_peak_to_peak is not None
                and command_peak_to_peak
                >= float(experiment.settings["minimum_command_peak_to_peak"])
                and response_peak_to_peak
                >= float(experiment.settings["minimum_response_peak_to_peak"])
            )
            validation_reason = problem
            if validation_reason is None and not enough_samples:
                validation_reason = (
                    "command or response sample count is below minimum_samples"
                )
            if validation_reason is None and not enough_excitation:
                validation_reason = (
                    "command or response peak-to-peak excitation is below the "
                    "explicit manifest threshold"
                )
            diagnostics[-1].update(
                status=(
                    "available_for_validation"
                    if problem is None and enough_samples and enough_excitation
                    else "insufficient_for_validation"
                ),
                steering_command_samples=command_count,
                steering_response_samples=response_count,
                minimum_samples=required,
                command_peak_to_peak=command_peak_to_peak,
                response_peak_to_peak=response_peak_to_peak,
                reason=validation_reason,
                note=(
                    "The sweep is retained as validation evidence; the seven model "
                    "parameters are not inferred from an unlabelled frequency sweep."
                ),
            )

    estimates = {
        parameter: _aggregate_parameter(
            parameter,
            candidates[parameter],
            failures[parameter],
            experiments,
        )
        for parameter in PARAMETERS
    }
    return estimates, diagnostics


def _record_outcomes(
    experiment: Experiment,
    outcomes: dict[str, Candidate | str],
    candidates: dict[str, list[Candidate]],
    failures: dict[str, list[dict[str, str]]],
) -> None:
    for parameter, outcome in outcomes.items():
        if isinstance(outcome, Candidate):
            candidates[parameter].append(outcome)
        else:
            failures[parameter].append(
                {"experiment_id": experiment.identifier, "reason": outcome}
            )


def build_report(
    *,
    bag_path: Path,
    manifest_path: Path,
    data: BagData,
    experiments: Sequence[Experiment],
    estimates: dict[str, Any],
    diagnostics: Sequence[dict[str, Any]],
    motion_source: str,
    acceleration_response: str,
) -> dict[str, Any]:
    missing_kinds = [
        kind for kind in EXPERIMENT_KINDS if not any(item.kind == kind for item in experiments)
    ]
    unavailable = [
        parameter for parameter, estimate in estimates.items() if estimate["value"] is None
    ]
    invalid_sine_sweeps = [
        item["id"]
        for item in diagnostics
        if item["kind"] == "steering_sine_sweep"
        and item["status"] != "available_for_validation"
    ]
    incomplete_reasons = []
    if unavailable:
        incomplete_reasons.append("unavailable vehicle-model parameters")
    if missing_kinds:
        incomplete_reasons.append("missing blueprint experiment kinds")
    if invalid_sine_sweeps:
        incomplete_reasons.append("invalid steering sine-sweep validation intervals")
    warnings = []
    if missing_kinds:
        warnings.append(
            "Blueprint experiment coverage is incomplete: " + ", ".join(missing_kinds)
        )
    if acceleration_response == "localization_acceleration":
        warnings.append(
            "/localization/acceleration includes localization filtering; acceleration "
            "gain/delay describe the end-to-end observed pipeline, not raw AWSIM alone."
        )
    warnings.append(
        "Review per-interval candidate spread before copying any measured value into training configuration."
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not incomplete_reasons else "incomplete_fail_closed",
        "source": {
            "bag_path": str(bag_path),
            "storage_id": data.storage_id,
            "experiment_manifest": str(manifest_path),
            "timestamp_source": data.timestamp_source,
            "analysis_origin_timestamp_ns": data.analysis_origin_timestamp_ns,
            "bag_start_timestamp_ns": data.bag_start_timestamp_ns,
            "bag_end_timestamp_ns": data.bag_end_timestamp_ns,
            "motion_source": motion_source,
            "acceleration_response": acceleration_response,
        },
        "topic_contract": {
            role: topic.as_report() for role, topic in sorted(data.topics.items())
        },
        "series_diagnostics": {
            "steering_command_samples": len(data.series.steering_command),
            "acceleration_command_samples": len(data.series.acceleration_command),
            "steering_response_samples": len(data.series.steering_response),
            "velocity_motion_samples": len(data.series.velocity_motion),
            "odometry_motion_samples": len(data.series.odometry_motion),
            "localization_acceleration_samples": len(
                data.series.localization_acceleration
            ),
            "nonmonotonic_timestamp_counts": dict(
                sorted(data.series.nonmonotonic_counts.items())
            ),
        },
        "experiment_coverage": {
            "required_kinds": list(EXPERIMENT_KINDS),
            "missing_kinds": missing_kinds,
            "experiments": [item.as_report() for item in experiments],
            "diagnostics": list(diagnostics),
        },
        "awsim_identification": {
            parameter: estimates[parameter]["value"] for parameter in PARAMETERS
        },
        "estimation": estimates,
        "unavailable_parameters": unavailable,
        "invalid_sine_sweep_experiments": invalid_sine_sweeps,
        "incomplete_reasons": incomplete_reasons,
        "warnings": warnings,
    }
    return report


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "null"
        rendered = repr(value)
        exponent_marker = "e" if "e" in rendered else "E" if "E" in rendered else None
        if exponent_marker is not None:
            mantissa, exponent = rendered.split(exponent_marker, maxsplit=1)
            if "." not in mantissa:
                mantissa += ".0"
            rendered = f"{mantissa}{exponent_marker}{exponent}"
        return rendered
    return json.dumps(str(value), ensure_ascii=False)


def _yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [prefix + "{}"]
        lines = []
        for key, item in value.items():
            rendered_key = str(key) if re.fullmatch(r"[A-Za-z0-9_]+", str(key)) else _yaml_scalar(key)
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{prefix}{rendered_key}:")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                rendered = "[]" if item == [] else "{}" if item == {} else _yaml_scalar(item)
                lines.append(f"{prefix}{rendered_key}: {rendered}")
        return lines
    if isinstance(value, list):
        if not value:
            return [prefix + "[]"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(prefix + "-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(prefix + "- " + _yaml_scalar(item))
        return lines
    return [prefix + _yaml_scalar(value)]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _topic_specs(args: argparse.Namespace) -> dict[str, tuple[str, str]]:
    specs = {
        "control": (args.control_topic, CONTROL_MESSAGE_TYPE),
        "steering": (args.steering_topic, STEERING_MESSAGE_TYPE),
        "velocity": (args.velocity_topic, VELOCITY_MESSAGE_TYPE),
        "odometry": (args.odometry_topic, ODOMETRY_MESSAGE_TYPE),
        "acceleration": (args.acceleration_topic, ACCELERATION_MESSAGE_TYPE),
    }
    topics = [topic for topic, _ in specs.values()]
    if len(topics) != len(set(topics)):
        raise AnalysisError("logical topic roles must resolve to distinct topic names")
    return specs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate AWSIM vehicle response only from explicitly labelled bag intervals; "
            "missing evidence is written as null with a reason."
        )
    )
    parser.add_argument("bag", type=Path, help="rosbag2 directory, .mcap, or .db3")
    parser.add_argument(
        "--experiments",
        type=Path,
        required=True,
        help="JSON interval manifest; see assets/calibration template",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--storage-id", choices=("auto", "mcap", "sqlite3"), default="auto"
    )
    parser.add_argument(
        "--timestamp-source",
        choices=("message", "bag"),
        default="message",
        help=(
            "message requires positive ROS stamps; bag explicitly identifies reception-level timing"
        ),
    )
    parser.add_argument(
        "--motion-source",
        choices=("velocity_status", "odometry"),
        default="velocity_status",
    )
    parser.add_argument(
        "--acceleration-response",
        choices=("velocity_derivative", "localization_acceleration"),
        default="velocity_derivative",
    )
    parser.add_argument("--control-topic", default=DEFAULT_CONTROL_TOPIC)
    parser.add_argument("--steering-topic", default=DEFAULT_STEERING_TOPIC)
    parser.add_argument("--velocity-topic", default=DEFAULT_VELOCITY_TOPIC)
    parser.add_argument("--odometry-topic", default=DEFAULT_ODOMETRY_TOPIC)
    parser.add_argument("--acceleration-topic", default=DEFAULT_ACCELERATION_TOPIC)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not args.bag.exists():
            raise AnalysisError(f"Bag path does not exist: {args.bag}")
        experiments = load_experiments(
            args.experiments,
            timestamp_source=args.timestamp_source,
            acceleration_response=args.acceleration_response,
        )
        storage_id = detect_storage_id(args.bag, args.storage_id)
        data = stream_bag(
            args.bag,
            storage_id=storage_id,
            timestamp_source=args.timestamp_source,
            topic_specs=_topic_specs(args),
        )
        estimates, diagnostics = identify(
            data,
            experiments,
            motion_source=args.motion_source,
            acceleration_response=args.acceleration_response,
        )
        report = build_report(
            bag_path=args.bag,
            manifest_path=args.experiments,
            data=data,
            experiments=experiments,
            estimates=estimates,
            diagnostics=diagnostics,
            motion_source=args.motion_source,
            acceleration_response=args.acceleration_response,
        )
        try:
            _atomic_write(args.output, "\n".join(_yaml_lines(report)) + "\n")
        except OSError as exc:
            raise AnalysisError(f"Could not write {args.output}: {exc}") from exc
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output}")
    if report["status"] != "complete":
        print(
            "Identification incomplete: " + ", ".join(report["incomplete_reasons"]),
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

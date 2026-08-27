#!/usr/bin/env python3
"""Convert one Bag Manager recording into an ego-LiDAR training sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


def nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    insertion = np.searchsorted(target_times, source_times)
    current = np.clip(insertion, 0, len(target_times) - 1)
    previous = np.clip(insertion - 1, 0, len(target_times) - 1)
    current_delta = np.abs(target_times[current] - source_times)
    previous_delta = np.abs(target_times[previous] - source_times)
    use_previous = previous_delta < current_delta
    indices = np.where(use_previous, previous, current)
    deltas = np.where(use_previous, previous_delta, current_delta)
    return indices, deltas


def preprocess(
    bag_dir: Path,
    output_dir: Path,
    scan_topic: str,
    control_topic: str,
    max_range: float,
    max_sync_delta: float,
) -> None:
    scan_times: list[int] = []
    scans: list[np.ndarray] = []
    control_times: list[int] = []
    accelerations: list[float] = []
    steers: list[float] = []
    angle_min: float | None = None
    angle_max: float | None = None
    angle_increment: float | None = None

    with AnyReader([bag_dir]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in {scan_topic, control_topic}
        ]
        available_topics = {connection.topic for connection in connections}
        missing_topics = {scan_topic, control_topic} - available_topics
        if missing_topics:
            raise RuntimeError(f"Required topics are missing from the bag: {sorted(missing_topics)}")

        for connection, timestamp, raw_message in reader.messages(connections=connections):
            message = reader.deserialize(raw_message, connection.msgtype)
            if connection.topic == scan_topic:
                if angle_min is None:
                    angle_min = float(message.angle_min)
                    angle_max = float(message.angle_max)
                    angle_increment = float(message.angle_increment)
                scan = np.asarray(message.ranges, dtype=np.float32)
                scan = np.nan_to_num(scan, nan=0.0, posinf=max_range, neginf=0.0)
                scans.append(np.clip(scan, 0.0, max_range))
                scan_times.append(timestamp)
            elif connection.topic == control_topic:
                accelerations.append(float(message.longitudinal.acceleration))
                steers.append(float(message.lateral.steering_tire_angle))
                control_times.append(timestamp)

    if not scans or not control_times:
        raise RuntimeError(
            f"Insufficient data: scans={len(scans)}, control_commands={len(control_times)}"
        )
    scan_lengths = {len(scan) for scan in scans}
    if len(scan_lengths) != 1:
        raise RuntimeError(f"LaserScan length changed inside the sequence: {sorted(scan_lengths)}")

    scan_times_array = np.asarray(scan_times, dtype=np.int64)
    control_times_array = np.asarray(control_times, dtype=np.int64)
    order = np.argsort(control_times_array)
    control_times_array = control_times_array[order]
    acceleration_array = np.asarray(accelerations, dtype=np.float32)[order]
    steer_array = np.asarray(steers, dtype=np.float32)[order]
    indices, deltas_ns = nearest_indices(scan_times_array, control_times_array)
    valid = deltas_ns <= int(max_sync_delta * 1.0e9)
    if not np.any(valid):
        raise RuntimeError(
            f"No scan/control pairs are within max_sync_delta={max_sync_delta:.3f}s"
        )

    scan_array = np.stack(scans).astype(np.float32, copy=False)[valid]
    acceleration_array = acceleration_array[indices][valid]
    steer_array = steer_array[indices][valid]
    delta_seconds = deltas_ns[valid].astype(np.float64) / 1.0e9

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "scans.npy", scan_array)
    np.save(output_dir / "accelerations.npy", acceleration_array)
    np.save(output_dir / "steers.npy", steer_array)
    np.save(output_dir / "sync_deltas.npy", delta_seconds.astype(np.float32))
    summary = {
        "source_bag": str(bag_dir),
        "scan_topic": scan_topic,
        "control_topic": control_topic,
        "samples": int(len(scan_array)),
        "scan_points": int(scan_array.shape[1]),
        "angle_min": angle_min,
        "angle_max": angle_max,
        "angle_increment": angle_increment,
        "max_range": max_range,
        "max_sync_delta_seconds": max_sync_delta,
        "sync_delta_mean_seconds": float(delta_seconds.mean()),
        "sync_delta_max_seconds": float(delta_seconds.max()),
        "discarded_scans": int(np.count_nonzero(~valid)),
    }
    (output_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"[OK] output={output_dir}")
    print(
        f"[OK] samples={len(scan_array)} scan_points={scan_array.shape[1]} "
        f"sync_mean={delta_seconds.mean():.4f}s sync_max={delta_seconds.max():.4f}s "
        f"discarded={np.count_nonzero(~valid)}"
    )
    print(
        f"[OK] acceleration=[{acceleration_array.min():.3f}, {acceleration_array.max():.3f}] "
        f"steering=[{steer_array.min():.3f}, {steer_array.max():.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-topic", default="/sensing/lidar/scan")
    parser.add_argument("--control-topic", default="/control/command/control_cmd")
    parser.add_argument("--max-range", type=float, default=30.0)
    parser.add_argument("--max-sync-delta", type=float, default=0.1)
    args = parser.parse_args()
    bag_dir = args.bag.expanduser().resolve()
    if not (bag_dir / "metadata.yaml").is_file():
        raise FileNotFoundError(f"ROS bag metadata.yaml not found: {bag_dir}")
    if args.max_range <= 0.0 or args.max_sync_delta < 0.0:
        raise ValueError("max-range must be positive and max-sync-delta must be non-negative")
    preprocess(
        bag_dir,
        args.output.expanduser().resolve(),
        args.scan_topic,
        args.control_topic,
        args.max_range,
        args.max_sync_delta,
    )


if __name__ == "__main__":
    main()

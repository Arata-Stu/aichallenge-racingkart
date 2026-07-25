#!/usr/bin/env python3
from __future__ import annotations

import argparse
from bisect import bisect_left
from pathlib import Path
from typing import Any

import numpy as np


def import_rosbag_tools() -> tuple[Any, Any, Any]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise SystemExit(
            "This script must run inside a ROS 2 environment with rosbag2_py available."
        ) from exc
    return rosbag2_py, deserialize_message, get_message


def resize_scan(ranges: list[float], scan_dim: int) -> np.ndarray:
    values = np.asarray(ranges, dtype=np.float32)
    values = np.nan_to_num(values, nan=np.inf, posinf=np.inf, neginf=0.0)
    if scan_dim <= 0 or len(values) == scan_dim:
        return values
    if len(values) == 0:
        return np.full((scan_dim,), np.inf, dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=scan_dim, dtype=np.float32)
    return np.interp(dst, src, values).astype(np.float32)


def timestamp_from_msg(msg: Any, fallback_ns: int) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return fallback_ns
    ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return ns if ns > 0 else fallback_ns


def control_target(msg: Any, mode: str) -> np.ndarray:
    lateral = getattr(msg, "lateral", None)
    longitudinal = getattr(msg, "longitudinal", None)
    steer = float(getattr(lateral, "steering_tire_angle", 0.0))
    if mode == "speed_steer":
        first = float(getattr(longitudinal, "speed", 0.0))
    else:
        first = float(getattr(longitudinal, "acceleration", 0.0))
    return np.asarray([first, steer], dtype=np.float32)


def nearest_index(times: list[int], stamp: int) -> int | None:
    if not times:
        return None
    index = bisect_left(times, stamp)
    candidates = []
    if index < len(times):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    return min(candidates, key=lambda i: abs(times[i] - stamp))


def parse_rsu_meta(text: str, rsu_count: int) -> np.ndarray:
    if not text:
        return np.zeros((rsu_count, 4), dtype=np.float32)
    rows = []
    for row in text.split(";"):
        values = [float(cell.strip()) for cell in row.split(",") if cell.strip()]
        if len(values) != 4:
            raise ValueError("--rsu-meta rows must be distance,rel_x,rel_y,rel_yaw")
        rows.append(values)
    if len(rows) != rsu_count:
        raise ValueError(f"--rsu-meta has {len(rows)} rows but {rsu_count} RSU topics were given")
    return np.asarray(rows, dtype=np.float32)


def read_bag(args: argparse.Namespace) -> dict[str, Any]:
    rosbag2_py, deserialize_message, get_message = import_rosbag_tools()
    storage_options = rosbag2_py.StorageOptions(uri=str(args.bag), storage_id=args.storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    wanted_topics = {args.ego_scan_topic, args.control_topic, *args.rsu_scan_topics}
    missing = sorted(topic for topic in wanted_topics if topic not in topic_types)
    if missing:
        raise SystemExit(f"Missing topics in bag: {missing}")

    msg_types = {topic: get_message(topic_types[topic]) for topic in wanted_topics}
    ego: list[tuple[int, np.ndarray]] = []
    controls: list[tuple[int, np.ndarray]] = []
    rsus: list[list[tuple[int, np.ndarray]]] = [[] for _ in args.rsu_scan_topics]
    rsu_topic_to_index = {topic: i for i, topic in enumerate(args.rsu_scan_topics)}

    while reader.has_next():
        topic, data, bag_stamp = reader.read_next()
        if topic not in wanted_topics:
            continue
        msg = deserialize_message(data, msg_types[topic])
        stamp = timestamp_from_msg(msg, int(bag_stamp))
        if topic == args.ego_scan_topic:
            ego.append((stamp, resize_scan(list(msg.ranges), args.scan_dim)))
        elif topic == args.control_topic:
            controls.append((stamp, control_target(msg, args.target_mode)))
        else:
            rsus[rsu_topic_to_index[topic]].append((stamp, resize_scan(list(msg.ranges), args.scan_dim)))

    ego.sort(key=lambda item: item[0])
    controls.sort(key=lambda item: item[0])
    for seq in rsus:
        seq.sort(key=lambda item: item[0])
    return {"ego": ego, "controls": controls, "rsus": rsus}


def synchronize(args: argparse.Namespace, streams: dict[str, Any]) -> dict[str, np.ndarray]:
    control_times = [stamp for stamp, _ in streams["controls"]]
    rsu_times = [[stamp for stamp, _ in seq] for seq in streams["rsus"]]
    static_meta = parse_rsu_meta(args.rsu_meta, len(args.rsu_scan_topics))
    max_dt_ns = int(args.max_sync_dt * 1_000_000_000)

    ego_scans = []
    rsu_scans = []
    rsu_meta = []
    rsu_mask = []
    targets = []

    for ego_stamp, ego_scan in streams["ego"]:
        control_index = nearest_index(control_times, ego_stamp)
        if control_index is None or abs(control_times[control_index] - ego_stamp) > max_dt_ns:
            continue

        sample_rsu_scans = []
        sample_meta = []
        sample_mask = []
        for sensor_index, seq in enumerate(streams["rsus"]):
            index = nearest_index(rsu_times[sensor_index], ego_stamp)
            if index is None:
                sample_rsu_scans.append(np.full_like(ego_scan, np.inf))
                sample_meta.append(np.concatenate([static_meta[sensor_index], np.asarray([np.inf], dtype=np.float32)]))
                sample_mask.append(False)
                continue
            rsu_stamp, scan = seq[index]
            age_s = abs(ego_stamp - rsu_stamp) / 1_000_000_000.0
            valid = age_s <= args.max_sync_dt
            sample_rsu_scans.append(scan if valid else np.full_like(ego_scan, np.inf))
            sample_meta.append(np.concatenate([static_meta[sensor_index], np.asarray([age_s], dtype=np.float32)]))
            sample_mask.append(valid)

        ego_scans.append(ego_scan)
        rsu_scans.append(np.stack(sample_rsu_scans, axis=0))
        rsu_meta.append(np.stack(sample_meta, axis=0))
        rsu_mask.append(np.asarray(sample_mask, dtype=np.bool_))
        targets.append(streams["controls"][control_index][1])

    return {
        "ego_scans": np.stack(ego_scans, axis=0).astype(np.float32),
        "rsu_scans": np.stack(rsu_scans, axis=0).astype(np.float32),
        "rsu_meta": np.stack(rsu_meta, axis=0).astype(np.float32),
        "rsu_mask": np.stack(rsu_mask, axis=0),
        "targets": np.stack(targets, axis=0).astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AIC RSU fusion rosbag to NumPy dataset.")
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--storage-id", default="mcap")
    parser.add_argument("--ego-scan-topic", default="/sensing/lidar/scan")
    parser.add_argument("--rsu-scan-topics", required=True, help="Comma-separated RSU LaserScan topics")
    parser.add_argument("--control-topic", default="/control/command/control_cmd")
    parser.add_argument("--target-mode", choices=["accel_steer", "speed_steer"], default="accel_steer")
    parser.add_argument("--scan-dim", type=int, default=1080)
    parser.add_argument("--max-sync-dt", type=float, default=0.1)
    parser.add_argument(
        "--rsu-meta",
        default="",
        help="Semicolon-separated rows: distance,rel_x,rel_y,rel_yaw per RSU",
    )
    args = parser.parse_args()
    args.rsu_scan_topics = [topic.strip() for topic in args.rsu_scan_topics.split(",") if topic.strip()]
    if not args.rsu_scan_topics:
        raise SystemExit("--rsu-scan-topics must contain at least one topic")

    streams = read_bag(args)
    dataset = synchronize(args, streams)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, array in dataset.items():
        np.save(args.output / f"{name}.npy", array)
    print(f"Saved {len(dataset['ego_scans'])} synchronized samples to {args.output}")


if __name__ == "__main__":
    main()

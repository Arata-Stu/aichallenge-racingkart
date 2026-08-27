#!/usr/bin/env python3
from __future__ import annotations

import argparse
from bisect import bisect_left
import math
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


def pack_bev_image(msg: Any) -> np.ndarray:
    """Pack an 8UC8 semantic BEV into one bit-mask byte per grid cell."""
    if str(getattr(msg, "encoding", "")).lower() != "8uc8":
        raise ValueError(f"BEV image encoding must be 8UC8, got {msg.encoding!r}")
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    row_bytes = width * 8
    if height <= 0 or width <= 0 or step < row_bytes:
        raise ValueError(f"Invalid BEV image geometry: {width}x{height}, step={step}")
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if raw.size < height * step:
        raise ValueError(f"BEV image data is truncated: {raw.size} < {height * step}")
    channels = raw[: height * step].reshape(height, step)[:, :row_bytes]
    channels = channels.reshape(height, width, 8)
    return np.packbits(channels != 0, axis=-1, bitorder="little")[..., 0]


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


def parse_rsu_poses(text: str, rsu_count: int) -> np.ndarray | None:
    if not text:
        return None
    rows = []
    for row in text.split(";"):
        values = [float(cell.strip()) for cell in row.split(",") if cell.strip()]
        if len(values) != 3:
            raise ValueError("--rsu-poses rows must be x,y,yaw_deg")
        rows.append([values[0], values[1], math.radians(values[2])])
    if len(rows) != rsu_count:
        raise ValueError(f"--rsu-poses has {len(rows)} rows but {rsu_count} RSU topics were given")
    return np.asarray(rows, dtype=np.float64)


def pose_xy_yaw(msg: Any) -> np.ndarray:
    pose = msg.pose.pose
    q = pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return np.asarray([pose.position.x, pose.position.y, yaw], dtype=np.float64)


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
    required_topics = {args.ego_scan_topic, args.control_topic, *args.rsu_scan_topics}
    if args.require_bev:
        required_topics.add(args.bev_topic)
    missing = sorted(topic for topic in required_topics if topic not in topic_types)
    if missing:
        raise SystemExit(f"Missing topics in bag: {missing}")

    wanted_topics = set(required_topics)
    if args.bev_topic in topic_types:
        wanted_topics.add(args.bev_topic)
    elif not args.require_bev:
        print(f"[WARN] BEV topic is absent; only scan-fusion training will be available: {args.bev_topic}")
    if args.pose_topic in topic_types:
        wanted_topics.add(args.pose_topic)
    else:
        print(f"[WARN] Pose topic is absent; RSU distance metadata falls back to --rsu-meta: {args.pose_topic}")
    velocity_topic = getattr(args, "velocity_topic", "/vehicle/status/velocity_status")
    if velocity_topic in topic_types:
        wanted_topics.add(velocity_topic)
    else:
        print(f"[WARN] Velocity topic is absent; speed will be derived from pose: {velocity_topic}")
    msg_types = {topic: get_message(topic_types[topic]) for topic in wanted_topics}
    ego: list[tuple[int, np.ndarray]] = []
    controls: list[tuple[int, np.ndarray]] = []
    rsus: list[list[tuple[int, np.ndarray]]] = [[] for _ in args.rsu_scan_topics]
    poses: list[tuple[int, np.ndarray]] = []
    velocities: list[tuple[int, float]] = []
    bev: list[tuple[int, np.ndarray]] = []
    rsu_topic_to_index = {topic: i for i, topic in enumerate(args.rsu_scan_topics)}

    while reader.has_next():
        topic, data, bag_stamp = reader.read_next()
        if topic not in wanted_topics:
            continue
        msg = deserialize_message(data, msg_types[topic])
        # Ego/RSU generators use simulation time while some control publishers
        # stamp messages with wall time. Rosbag receive timestamps share one
        # clock and are therefore the safe default for cross-topic alignment.
        stamp = (
            int(bag_stamp)
            if args.timestamp_source == "bag"
            else timestamp_from_msg(msg, int(bag_stamp))
        )
        if topic == args.bev_topic:
            bev.append((stamp, pack_bev_image(msg)))
        elif topic == args.pose_topic:
            poses.append((stamp, pose_xy_yaw(msg)))
        elif topic == velocity_topic:
            velocities.append((stamp, abs(float(getattr(msg, "longitudinal_velocity", 0.0)))))
        elif topic == args.ego_scan_topic:
            ego.append((stamp, resize_scan(list(msg.ranges), args.scan_dim)))
        elif topic == args.control_topic:
            controls.append((stamp, control_target(msg, args.target_mode)))
        else:
            rsus[rsu_topic_to_index[topic]].append((stamp, resize_scan(list(msg.ranges), args.scan_dim)))

    ego.sort(key=lambda item: item[0])
    controls.sort(key=lambda item: item[0])
    for seq in rsus:
        seq.sort(key=lambda item: item[0])
    poses.sort(key=lambda item: item[0])
    velocities.sort(key=lambda item: item[0])
    bev.sort(key=lambda item: item[0])
    return {
        "ego": ego, "controls": controls, "rsus": rsus, "poses": poses,
        "velocities": velocities, "bev": bev,
    }


def synchronize(args: argparse.Namespace, streams: dict[str, Any]) -> dict[str, np.ndarray]:
    if not streams["ego"]:
        raise SystemExit(f"No messages were read from ego scan topic: {args.ego_scan_topic}")
    if not streams["controls"]:
        raise SystemExit(f"No messages were read from control topic: {args.control_topic}")
    empty_rsus = [args.rsu_scan_topics[index] for index, stream in enumerate(streams["rsus"]) if not stream]
    if empty_rsus:
        raise SystemExit(f"No messages were read from RSU topics: {empty_rsus}")

    control_times = [stamp for stamp, _ in streams["controls"]]
    rsu_times = [[stamp for stamp, _ in seq] for seq in streams["rsus"]]
    static_meta = parse_rsu_meta(args.rsu_meta, len(args.rsu_scan_topics))
    rsu_poses = parse_rsu_poses(args.rsu_poses, len(args.rsu_scan_topics))
    pose_times = [stamp for stamp, _ in streams["poses"]]
    velocities = streams.get("velocities", [])
    velocity_times = [stamp for stamp, _ in velocities]
    bev_stream = streams.get("bev", [])
    bev_times = [stamp for stamp, _ in bev_stream]
    use_bev = bool(bev_stream)
    if args.require_bev and not use_bev:
        raise SystemExit(f"No messages were read from BEV topic: {args.bev_topic}")
    max_dt_ns = int(args.max_sync_dt * 1_000_000_000)

    ego_scans = []
    rsu_scans = []
    rsu_meta = []
    rsu_mask = []
    targets = []
    ego_poses = []
    timestamps_ns = []
    ego_speeds = []
    bev_frames = []

    for ego_stamp, ego_scan in streams["ego"]:
        control_index = nearest_index(control_times, ego_stamp)
        if control_index is None or abs(control_times[control_index] - ego_stamp) > max_dt_ns:
            continue

        bev_frame = None
        if use_bev:
            bev_index = nearest_index(bev_times, ego_stamp)
            if bev_index is None or abs(bev_times[bev_index] - ego_stamp) > max_dt_ns:
                # Keep every saved array exactly frame-aligned when BEV is present.
                continue
            bev_frame = bev_stream[bev_index][1]

        pose_index = nearest_index(pose_times, ego_stamp)
        pose_valid = (
            pose_index is not None
            and abs(pose_times[pose_index] - ego_stamp) <= max_dt_ns
        )
        ego_pose = streams["poses"][pose_index][1] if pose_valid else None
        sample_rsu_scans = []
        sample_meta = []
        sample_mask = []
        for sensor_index, seq in enumerate(streams["rsus"]):
            index = nearest_index(rsu_times[sensor_index], ego_stamp)
            if index is None:
                sample_rsu_scans.append(np.full_like(ego_scan, np.inf))
                base_meta = dynamic_rsu_meta(ego_pose, rsu_poses, sensor_index, static_meta)
                sample_meta.append(np.concatenate([base_meta, np.asarray([np.inf], dtype=np.float32)]))
                sample_mask.append(False)
                continue
            rsu_stamp, scan = seq[index]
            age_s = abs(ego_stamp - rsu_stamp) / 1_000_000_000.0
            valid = age_s <= args.max_sync_dt
            sample_rsu_scans.append(scan if valid else np.full_like(ego_scan, np.inf))
            base_meta = dynamic_rsu_meta(ego_pose, rsu_poses, sensor_index, static_meta)
            sample_meta.append(np.concatenate([base_meta, np.asarray([age_s], dtype=np.float32)]))
            sample_mask.append(valid)

        ego_scans.append(ego_scan)
        rsu_scans.append(np.stack(sample_rsu_scans, axis=0))
        rsu_meta.append(np.stack(sample_meta, axis=0))
        rsu_mask.append(np.asarray(sample_mask, dtype=np.bool_))
        targets.append(streams["controls"][control_index][1])
        ego_poses.append(
            ego_pose if ego_pose is not None else np.full((3,), np.nan, dtype=np.float64)
        )
        timestamps_ns.append(ego_stamp)
        velocity_index = nearest_index(velocity_times, ego_stamp)
        velocity_valid = (
            velocity_index is not None
            and abs(velocity_times[velocity_index] - ego_stamp) <= max_dt_ns
        )
        ego_speeds.append(
            velocities[velocity_index][1] if velocity_valid else np.nan
        )
        if bev_frame is not None:
            bev_frames.append(bev_frame)

    if not ego_scans:
        ego_times = [stamp for stamp, _ in streams["ego"]]
        nearest_deltas = []
        for stamp in ego_times[: min(len(ego_times), 1000)]:
            control_index = nearest_index(control_times, stamp)
            if control_index is not None:
                nearest_deltas.append(abs(control_times[control_index] - stamp) / 1_000_000_000.0)
        minimum = min(nearest_deltas) if nearest_deltas else float("inf")
        raise SystemExit(
            "No synchronized samples were produced: "
            f"ego={len(streams['ego'])}, controls={len(streams['controls'])}, "
            f"nearest_control_delta={minimum:.6f}s, allowed={args.max_sync_dt:.6f}s, "
            f"timestamp_source={args.timestamp_source}"
        )

    print(
        f"Synchronized {len(ego_scans)} / {len(streams['ego'])} ego scans "
        f"using timestamp_source={args.timestamp_source}"
    )
    result = {
        "ego_scans": np.stack(ego_scans, axis=0).astype(np.float32),
        "rsu_scans": np.stack(rsu_scans, axis=0).astype(np.float32),
        "rsu_meta": np.stack(rsu_meta, axis=0).astype(np.float32),
        "rsu_mask": np.stack(rsu_mask, axis=0),
        "targets": np.stack(targets, axis=0).astype(np.float32),
    }
    if use_bev:
        shapes = {frame.shape for frame in bev_frames}
        if len(shapes) != 1:
            raise SystemExit(f"BEV image shape changed within the bag: {sorted(shapes)}")
        result["bev_frames"] = np.stack(bev_frames, axis=0).astype(np.uint8)
    pose_array = np.stack(ego_poses, axis=0).astype(np.float64)
    if np.isfinite(pose_array).all():
        time_array = np.asarray(timestamps_ns, dtype=np.int64)
        derived_speed = pose_speed(pose_array, time_array)
        measured_speed = np.asarray(ego_speeds, dtype=np.float32)
        speed = np.where(np.isfinite(measured_speed), measured_speed, derived_speed)
        result["ego_poses"] = pose_array
        result["timestamps_ns"] = time_array
        # Raw m/s. Dataset normalization is stored in the training checkpoint.
        result["vehicle_state"] = speed[:, None].astype(np.float32)
    else:
        missing = int(np.count_nonzero(~np.isfinite(pose_array[:, 0])))
        print(
            f"[WARN] {missing} synchronized samples have no pose; "
            "trajectory training files were not written"
        )
    return result


def pose_speed(poses: np.ndarray, timestamps_ns: np.ndarray) -> np.ndarray:
    """Estimate ego speed from synchronized poses without differentiating yaw."""
    if len(poses) == 0:
        return np.empty((0,), dtype=np.float32)
    if len(poses) == 1:
        return np.zeros((1,), dtype=np.float32)
    dt = np.diff(timestamps_ns.astype(np.float64)) / 1_000_000_000.0
    distance = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    segment_speed = np.divide(
        distance,
        dt,
        out=np.zeros_like(distance, dtype=np.float64),
        where=dt > 1e-4,
    )
    segment_speed = np.nan_to_num(segment_speed, nan=0.0, posinf=0.0, neginf=0.0)
    speed = np.empty((len(poses),), dtype=np.float64)
    speed[0] = segment_speed[0]
    speed[-1] = segment_speed[-1]
    if len(poses) > 2:
        speed[1:-1] = 0.5 * (segment_speed[:-1] + segment_speed[1:])
    return speed.astype(np.float32)


def dynamic_rsu_meta(
    ego_pose: np.ndarray | None,
    rsu_poses: np.ndarray | None,
    sensor_index: int,
    static_meta: np.ndarray,
) -> np.ndarray:
    if ego_pose is None or rsu_poses is None:
        return static_meta[sensor_index]
    dx = float(rsu_poses[sensor_index, 0] - ego_pose[0])
    dy = float(rsu_poses[sensor_index, 1] - ego_pose[1])
    yaw = float(ego_pose[2])
    relative_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    relative_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    relative_yaw = math.atan2(
        math.sin(float(rsu_poses[sensor_index, 2]) - yaw),
        math.cos(float(rsu_poses[sensor_index, 2]) - yaw),
    )
    return np.asarray([math.hypot(dx, dy), relative_x, relative_y, relative_yaw], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AIC RSU fusion rosbag to NumPy dataset.")
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--storage-id", default="mcap")
    parser.add_argument("--ego-scan-topic", default="/sensing/lidar/scan")
    parser.add_argument("--rsu-scan-topics", required=True, help="Comma-separated RSU LaserScan topics")
    parser.add_argument("--control-topic", default="/control/command/control_cmd")
    parser.add_argument("--pose-topic", default="/localization/pose_with_covariance")
    parser.add_argument("--velocity-topic", default="/vehicle/status/velocity_status")
    parser.add_argument("--bev-topic", default="/perception/virtual_scan_bev/image")
    parser.add_argument(
        "--require-bev", action="store_true",
        help="Fail unless synchronized packed BEV frames are available",
    )
    parser.add_argument("--target-mode", choices=["accel_steer", "speed_steer"], default="accel_steer")
    parser.add_argument("--scan-dim", type=int, default=1080)
    parser.add_argument("--max-sync-dt", type=float, default=0.1)
    parser.add_argument(
        "--timestamp-source",
        choices=["bag", "header"],
        default="bag",
        help="Use rosbag receive timestamps by default; header clocks may differ between topics",
    )
    parser.add_argument(
        "--rsu-meta",
        default="",
        help="Semicolon-separated rows: distance,rel_x,rel_y,rel_yaw per RSU",
    )
    parser.add_argument(
        "--rsu-poses",
        default="",
        help="Semicolon-separated fixed map poses: x,y,yaw_deg per RSU",
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

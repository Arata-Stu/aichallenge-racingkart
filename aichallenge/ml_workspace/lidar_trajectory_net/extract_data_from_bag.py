from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rosbags.highlevel import AnyReader


@dataclass
class ExtractionConfig:
    free_scan_topic: str
    obstacle_scan_topic: str
    pose_topic: str
    scan_msg_type: str = "sensor_msgs/msg/LaserScan"
    max_scan_range: float = 30.0
    max_sync_delta_sec: float = 0.08
    target_num_rays: int = 0


def worker_init(debug_mode: bool) -> None:
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] [PID:%(process)d] %(message)s",
        force=True,
    )


def setup_logger(debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] [PID:%(process)d] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def clean_scan_array(scan_array: np.ndarray, max_range: float) -> np.ndarray:
    if not isinstance(scan_array, np.ndarray):
        scan_array = np.asarray(scan_array, dtype=np.float32)
    cleaned = np.nan_to_num(scan_array, nan=0.0, posinf=max_range, neginf=0.0)
    cleaned = np.clip(cleaned, 0.0, max_range)
    return cleaned.astype(np.float32)


def resize_scan(scan_array: np.ndarray, target_num_rays: int) -> np.ndarray:
    if target_num_rays <= 0 or len(scan_array) == target_num_rays:
        return scan_array

    src_x = np.linspace(0.0, 1.0, len(scan_array), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_num_rays, dtype=np.float32)
    return np.interp(dst_x, src_x, scan_array).astype(np.float32)


def synchronize_indices(src_times: np.ndarray, target_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(target_times) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    idx = np.searchsorted(target_times, src_times)
    idx = np.clip(idx, 0, len(target_times) - 1)
    prev_idx = np.clip(idx - 1, 0, len(target_times) - 1)

    curr_delta = np.abs(target_times[idx] - src_times)
    prev_delta = np.abs(target_times[prev_idx] - src_times)
    use_prev = prev_delta < curr_delta
    final_idx = np.where(use_prev, prev_idx, idx)
    final_delta = np.where(use_prev, prev_delta, curr_delta)
    return final_idx.astype(np.int64), final_delta.astype(np.int64)


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def extract_pose_xy_yaw(msg, msgtype: str) -> Tuple[float, float, float]:
    if msgtype == "nav_msgs/msg/Odometry":
        pose = msg.pose.pose
    elif msgtype == "geometry_msgs/msg/PoseWithCovarianceStamped":
        pose = msg.pose.pose
    elif msgtype == "geometry_msgs/msg/PoseStamped":
        pose = msg.pose
    else:
        if hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
            pose = msg.pose.pose
        elif hasattr(msg, "pose"):
            pose = msg.pose
        else:
            raise ValueError(f"Unsupported pose message type: {msgtype}")

    return (
        float(pose.position.x),
        float(pose.position.y),
        quaternion_to_yaw(pose.orientation),
    )


def sort_by_time(times: List[int], values: List) -> Tuple[np.ndarray, List]:
    order = np.argsort(np.asarray(times, dtype=np.int64))
    sorted_times = np.asarray(times, dtype=np.int64)[order]
    sorted_values = [values[i] for i in order]
    return sorted_times, sorted_values


def process_bag(
    bag_path: Path,
    output_root: Path,
    config: ExtractionConfig,
    debug: bool = False,
) -> None:
    logger = logging.getLogger(__name__)
    bag_name = bag_path.name
    out_dir = output_root / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    free_scans: List[np.ndarray] = []
    free_times: List[int] = []
    obstacle_scans: List[np.ndarray] = []
    obstacle_times: List[int] = []
    poses: List[Tuple[float, float, float]] = []
    pose_times: List[int] = []
    scan_angle_min = None
    scan_angle_max = None

    try:
        with AnyReader([bag_path]) as reader:
            target_topics = [config.free_scan_topic, config.obstacle_scan_topic, config.pose_topic]
            connections = [c for c in reader.connections if c.topic in target_topics]

            if not connections:
                logger.warning("%s: no relevant topics found.", bag_name)
                return

            for conn, timestamp, raw in reader.messages(connections=connections):
                try:
                    msg = reader.deserialize(raw, conn.msgtype)

                    if conn.topic in (config.free_scan_topic, config.obstacle_scan_topic):
                        if conn.msgtype != config.scan_msg_type:
                            continue
                        scan = clean_scan_array(np.asarray(msg.ranges, dtype=np.float32), config.max_scan_range)
                        scan = resize_scan(scan, config.target_num_rays)
                        if conn.topic == config.free_scan_topic:
                            if scan_angle_min is None:
                                scan_angle_min = float(msg.angle_min)
                                scan_angle_max = float(msg.angle_max)
                            free_scans.append(scan)
                            free_times.append(timestamp)
                        else:
                            obstacle_scans.append(scan)
                            obstacle_times.append(timestamp)

                    elif conn.topic == config.pose_topic:
                        poses.append(extract_pose_xy_yaw(msg, conn.msgtype))
                        pose_times.append(timestamp)
                except Exception as e:
                    if debug:
                        logger.debug("%s: skipped message on %s: %s", bag_name, conn.topic, e)
                    continue
    except Exception as e:
        logger.error("Failed to read %s: %s", bag_name, e)
        return

    if not free_scans or not obstacle_scans or not poses:
        logger.warning(
            "Skipping %s: insufficient data (free=%d, obstacle=%d, poses=%d).",
            bag_name, len(free_scans), len(obstacle_scans), len(poses),
        )
        return

    free_times_np, free_scans = sort_by_time(free_times, free_scans)
    obstacle_times_np, obstacle_scans = sort_by_time(obstacle_times, obstacle_scans)
    pose_times_np, poses = sort_by_time(pose_times, poses)

    obstacle_indices, obstacle_deltas = synchronize_indices(free_times_np, obstacle_times_np)
    pose_indices, pose_deltas = synchronize_indices(free_times_np, pose_times_np)
    max_delta_ns = int(config.max_sync_delta_sec * 1e9)

    scan_inputs = []
    synced_poses = []
    synced_times = []
    sync_deltas = []
    expected_num_rays = 0

    for i, timestamp in enumerate(free_times_np):
        if obstacle_deltas[i] > max_delta_ns or pose_deltas[i] > max_delta_ns:
            continue

        free_scan = free_scans[i]
        obstacle_scan = obstacle_scans[int(obstacle_indices[i])]
        if len(free_scan) != len(obstacle_scan):
            logger.warning(
                "%s: skipping scan pair with mismatched ray counts (%d vs %d).",
                bag_name, len(free_scan), len(obstacle_scan),
            )
            continue

        if expected_num_rays == 0:
            expected_num_rays = len(free_scan)
        if len(free_scan) != expected_num_rays:
            logger.warning(
                "%s: skipping scan with unexpected ray count (%d, expected %d).",
                bag_name, len(free_scan), expected_num_rays,
            )
            continue

        diff = np.clip(free_scan - obstacle_scan, 0.0, config.max_scan_range)
        scan_inputs.append(np.stack([free_scan, obstacle_scan, diff], axis=0))
        synced_poses.append(poses[int(pose_indices[i])])
        synced_times.append(timestamp)
        sync_deltas.append([obstacle_deltas[i] / 1e9, pose_deltas[i] / 1e9])

    if not scan_inputs:
        logger.warning("%s: no synchronized samples after filtering.", bag_name)
        return

    if scan_angle_min is None or scan_angle_max is None:
        scan_angle_min = -np.deg2rad(135.0)
        scan_angle_max = np.deg2rad(135.0)

    np_scan_inputs = np.asarray(scan_inputs, dtype=np.float32)
    np_poses = np.asarray(synced_poses, dtype=np.float32)
    np_timestamps = np.asarray(synced_times, dtype=np.int64)
    np_sync_deltas = np.asarray(sync_deltas, dtype=np.float32)

    np.save(out_dir / "scan_inputs.npy", np_scan_inputs)
    np.save(out_dir / "poses.npy", np_poses)
    np.save(out_dir / "timestamps.npy", np_timestamps)
    scan_angles = np.linspace(
        float(scan_angle_min),
        float(scan_angle_max),
        expected_num_rays,
        dtype=np.float32,
    )
    np.save(out_dir / "scan_angles.npy", scan_angles)
    if debug:
        np.save(out_dir / "sync_deltas.npy", np_sync_deltas)

    metadata = {
        "bag_name": bag_name,
        "num_samples": int(len(np_scan_inputs)),
        "scan_shape": list(np_scan_inputs.shape),
        "pose_shape": list(np_poses.shape),
        "channel_names": ["virtual_scan", "virtual_scan_with_obstacles", "diff"],
        "scan_geometry": {
            "angle_min": float(scan_angles[0]),
            "angle_max": float(scan_angles[-1]),
            "num_rays": int(len(scan_angles)),
        },
        "config": asdict(config),
        "sync_delta_mean_sec": np_sync_deltas.mean(axis=0).tolist(),
        "sync_delta_max_sec": np_sync_deltas.max(axis=0).tolist(),
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Saved %s: %d samples, shape=%s (%.2fs)",
        bag_name, len(np_scan_inputs), tuple(np_scan_inputs.shape), time.perf_counter() - start_time,
    )


def discover_bags(args) -> List[Path]:
    bag_dirs = []
    if args.bags_dir:
        root = args.bags_dir.expanduser().resolve()
        bag_dirs = [p.parent for p in root.rglob("metadata.yaml")]
        if not bag_dirs and (root / "metadata.yaml").exists():
            bag_dirs = [root]
    elif args.seq_dirs:
        for seq_dir in args.seq_dirs:
            seq_dir = seq_dir.expanduser().resolve()
            if (seq_dir / "metadata.yaml").exists():
                bag_dirs.append(seq_dir)
    return sorted(set(bag_dirs))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 3-channel virtual LiDAR trajectory data from ROS 2 bags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bags-dir", type=Path, help="Directory containing rosbag folders.")
    group.add_argument("--seq-dirs", type=Path, nargs="+", help="Specific rosbag directories.")
    parser.add_argument("--outdir", type=Path, required=True, help="Root directory for output sequences.")

    parser.add_argument("--free-scan-topic", type=str, default="/sensing/virtual_lidar/scan")
    parser.add_argument(
        "--obstacle-scan-topic",
        type=str,
        default="/sensing/virtual_lidar/scan_with_obstacles",
    )
    parser.add_argument("--pose-topic", type=str, default="/localization/kinematic_state")
    parser.add_argument("--max-scan-range", type=float, default=30.0)
    parser.add_argument("--max-sync-delta-sec", type=float, default=0.08)
    parser.add_argument(
        "--target-num-rays",
        type=int,
        default=0,
        help="Resample scans to this ray count. 0 keeps the original ray count.",
    )

    default_workers = min(os.cpu_count() or 1, 8)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    setup_logger(args.debug)
    logger = logging.getLogger(__name__)

    bag_dirs = discover_bags(args)
    if not bag_dirs:
        logger.error("No valid ROS 2 bag directories found.")
        return

    config = ExtractionConfig(
        free_scan_topic=args.free_scan_topic,
        obstacle_scan_topic=args.obstacle_scan_topic,
        pose_topic=args.pose_topic,
        max_scan_range=args.max_scan_range,
        max_sync_delta_sec=args.max_sync_delta_sec,
        target_num_rays=args.target_num_rays,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    num_workers = min(max(1, args.workers), len(bag_dirs))
    tasks = [(bag_dir, args.outdir, config, args.debug) for bag_dir in bag_dirs]

    logger.info("Found %d bags. Starting processing with %d workers.", len(bag_dirs), num_workers)
    start_time = time.time()
    with multiprocessing.Pool(processes=num_workers, initializer=worker_init, initargs=(args.debug,)) as pool:
        pool.starmap(process_bag, tasks)
    logger.info("All processing finished in %.2f seconds.", time.time() - start_time)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

#!/usr/bin/env python3
"""Local web dashboard for RSU fusion preprocessing, inspection and training."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import importlib.util
import json
import math
import mimetypes
import os
import re
import signal
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
STATIC_DIR = SCRIPT_DIR / "dashboard"
TINY_WORKSPACE = SCRIPT_DIR.parent / "tiny_lidar_net_pytorch"
_COMMON_SPEC = importlib.util.spec_from_file_location(
    "tiny_lidar_dashboard_common", TINY_WORKSPACE / "dashboard_backend.py"
)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise ImportError("Could not load TinyLiDAR dashboard process utilities")
common = importlib.util.module_from_spec(_COMMON_SPEC)
sys.modules[_COMMON_SPEC.name] = common
_COMMON_SPEC.loader.exec_module(common)

REQUIRED_FILES = ("ego_scans.npy", "rsu_scans.npy", "rsu_meta.npy", "targets.npy")
DEFAULT_VERSION = "default"
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
RSU_TOPICS = [f"/rsu/curve_{index:02d}/scan" for index in range(1, 7)]
EGO_FOV_DEG = 270.0
RSU_FOV_DEG = [135.0, 135.0, 150.0, 180.0, 150.0, 150.0]
SENSOR_MAX_RANGES = [45.0, 15.0, 20.0, 20.0, 15.0, 15.0, 30.0]
RSU_POSES = (
    "89620.233296,43157.347478,111.0;89639.094992,43147.310706,-60.0;"
    "89629.923709,43179.151500,110.0;89655.187999,43180.362834,30.0;"
    "89655.014956,43167.730346,-130.0;89665.916671,43154.751761,5.0"
)
LANE_CSV = SCRIPT_DIR.parents[1] / "workspace/src/aichallenge_submit/laserscan_generator/map/lane.csv"


@dataclass(frozen=True)
class Config:
    record_root: Path
    dataset_root: Path
    checkpoint_root: Path
    evaluation_root: Path | None = None

    def __post_init__(self) -> None:
        if self.evaluation_root is None:
            object.__setattr__(self, "evaluation_root", self.checkpoint_root.parent / "evaluations")


class PidFile(common.PidFile):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.identity["program"] = "rsu-fusion-dashboard"


def normalize_version(value: Any) -> str:
    version = str(value or DEFAULT_VERSION).strip()
    if version != DEFAULT_VERSION and not VERSION_PATTERN.fullmatch(version):
        raise ValueError("dataset_version must use 1-64 letters, numbers, '.', '_' or '-'")
    return version


def version_root(config: Config, version: str) -> Path:
    version = normalize_version(version)
    return config.dataset_root if version == DEFAULT_VERSION else config.dataset_root / "versions" / version


def version_ids(config: Config) -> list[str]:
    result = [DEFAULT_VERSION]
    root = config.dataset_root / "versions"
    if root.is_dir():
        result.extend(path.name for path in sorted(root.iterdir()) if path.is_dir() and VERSION_PATTERN.fullmatch(path.name))
    return result


def resolved_child(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("A non-empty relative path is required")
    candidate = (root.resolve() / unquote(relative)).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Path escapes the configured root")
    return candidate


def directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def recordings(config: Config) -> list[dict[str, Any]]:
    if not config.record_root.is_dir():
        return []
    result = [
        common.recording_info(config.record_root, metadata)
        for metadata in config.record_root.rglob("metadata.yaml")
    ]
    return sorted(result, key=lambda item: item["modified_at"], reverse=True)


def validate_sequence(path: Path) -> tuple[bool, int, int, int]:
    try:
        ego = np.load(path / "ego_scans.npy", mmap_mode="r")
        rsu = np.load(path / "rsu_scans.npy", mmap_mode="r")
        meta = np.load(path / "rsu_meta.npy", mmap_mode="r")
        targets = np.load(path / "targets.npy", mmap_mode="r")
        valid = (
            ego.ndim == 2 and rsu.ndim == 3 and meta.ndim == 3 and targets.ndim == 2
            and len(ego) == len(rsu) == len(meta) == len(targets)
            and rsu.shape[:2] == meta.shape[:2] and ego.shape[1] == rsu.shape[2]
        )
        return valid, int(len(ego)), int(ego.shape[1]), int(rsu.shape[1])
    except (OSError, ValueError, IndexError):
        return False, 0, 0, 0


def sequences(config: Config, selected_version: str | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    versions = [normalize_version(selected_version)] if selected_version else version_ids(config)
    for version in versions:
        root = version_root(config, version)
        for split in ("train", "val"):
            split_root = root / split
            if not split_root.is_dir():
                continue
            for ego_file in sorted(split_root.rglob("ego_scans.npy")):
                path = ego_file.parent
                if not all((path / name).is_file() for name in REQUIRED_FILES):
                    continue
                valid, samples, rays, rsu_count = validate_sequence(path)
                result.append({
                    "version": version, "split": split,
                    "id": path.relative_to(split_root).as_posix(), "name": path.name,
                    "samples": samples, "scan_points": rays, "rsu_count": rsu_count, "valid": valid,
                    "trajectory_ready": all((path / name).is_file() for name in (
                        "ego_poses.npy", "timestamps_ns.npy", "vehicle_state.npy"
                    )),
                    "bev_ready": (path / "bev_frames.npy").is_file(),
                })
    return result


def versions(config: Config, all_sequences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for version in version_ids(config):
        selected = [item for item in all_sequences if item["version"] == version]
        result.append({
            "id": version,
            "train_sequences": sum(item["split"] == "train" for item in selected),
            "val_sequences": sum(item["split"] == "val" for item in selected),
            "train_samples": sum(item["samples"] for item in selected if item["split"] == "train"),
            "val_samples": sum(item["samples"] for item in selected if item["split"] == "val"),
        })
    return result


def checkpoints(config: Config) -> list[dict[str, Any]]:
    if not config.checkpoint_root.is_dir():
        return []
    return [
        {"id": path.relative_to(config.checkpoint_root).as_posix(), "name": path.name,
         "size_bytes": path.stat().st_size, "modified_at": path.stat().st_mtime,
         "best": path.name == "best_model.pth"}
        for path in sorted(config.checkpoint_root.rglob("*.pth"), reverse=True) if path.is_file()
    ]


def evaluations(config: Config) -> list[dict[str, Any]]:
    if not config.evaluation_root.is_dir():
        return []
    result = []
    for report_path in sorted(config.evaluation_root.rglob("metrics.json"), reverse=True):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result.append({
                "id": report_path.parent.relative_to(config.evaluation_root).as_posix(),
                "version": report.get("version", DEFAULT_VERSION),
                "split": report.get("split", "val"),
                "checkpoint": report.get("checkpoint", ""),
                "metrics": report.get("metrics", {}),
                "trajectory": report.get("trajectory", {}),
                "sequences": len(report.get("sequences", [])),
                "modified_at": report_path.stat().st_mtime,
            })
        except (OSError, ValueError, TypeError):
            continue
    return result


def resolve_sequence(config: Config, version: str, split: str, sequence_id: str) -> Path:
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    path = resolved_child(version_root(config, version) / split, sequence_id)
    if not all((path / name).is_file() for name in REQUIRED_FILES):
        raise FileNotFoundError("Dataset sequence is incomplete")
    return path


def clean_scan(array: np.ndarray, max_range: float = 45.0) -> list[float]:
    return np.clip(np.nan_to_num(np.asarray(array, dtype=np.float32), nan=max_range, posinf=max_range, neginf=0.0), 0.0, max_range).astype(float).tolist()


def sequence_detail(config: Config, version: str, split: str, sequence_id: str) -> dict[str, Any]:
    path = resolve_sequence(config, normalize_version(version), split, sequence_id)
    ego = np.load(path / "ego_scans.npy", mmap_mode="r")
    rsu = np.load(path / "rsu_scans.npy", mmap_mode="r")
    meta = np.load(path / "rsu_meta.npy", mmap_mode="r")
    targets = np.asarray(np.load(path / "targets.npy", mmap_mode="r"), dtype=np.float64)
    mask_path = path / "rsu_mask.npy"
    mask = np.load(mask_path, mmap_mode="r") if mask_path.is_file() else np.isfinite(rsu).any(axis=2)
    availability = np.mean(mask.astype(bool), axis=0) if len(mask) else np.zeros(rsu.shape[1])
    finite_accelerations = targets[:, 0][np.isfinite(targets[:, 0])]
    finite_steers = targets[:, 1][np.isfinite(targets[:, 1])]
    if not len(finite_accelerations):
        finite_accelerations = np.zeros(1, dtype=np.float64)
    if not len(finite_steers):
        finite_steers = np.zeros(1, dtype=np.float64)
    accel_histogram, accel_edges = np.histogram(finite_accelerations, bins=25)
    steer_histogram, steer_edges = np.histogram(finite_steers, bins=31)
    distance_means = []
    age_means = []
    for sensor_index in range(rsu.shape[1]):
        valid = np.asarray(mask[:, sensor_index], dtype=bool)
        sensor_distance = np.asarray(meta[:, sensor_index, 0], dtype=np.float64)
        sensor_age = np.asarray(meta[:, sensor_index, -1], dtype=np.float64)
        distance_values = sensor_distance[valid & np.isfinite(sensor_distance)]
        age_values = sensor_age[valid & np.isfinite(sensor_age)]
        distance_means.append(float(np.mean(distance_values)) if len(distance_values) else None)
        age_means.append(float(np.mean(age_values)) if len(age_values) else None)
    return {
        "version": version, "split": split, "id": sequence_id, "samples": int(len(ego)),
        "scan_points": int(ego.shape[1]), "rsu_count": int(rsu.shape[1]), "meta_dim": int(meta.shape[2]),
        "availability": availability.astype(float).tolist(),
        "distance_means": distance_means, "age_means": age_means,
        "target": {"accel_min": float(np.min(finite_accelerations)), "accel_max": float(np.max(finite_accelerations)),
                   "steer_min": float(np.min(finite_steers)), "steer_max": float(np.max(finite_steers)),
                   "accel_mean": float(np.mean(finite_accelerations)),
                   "steer_mean": float(np.mean(finite_steers)),
                   "accel_histogram": accel_histogram.astype(int).tolist(),
                   "accel_edges": accel_edges.astype(float).tolist(),
                   "steer_histogram": steer_histogram.astype(int).tolist(),
                   "steer_edges": steer_edges.astype(float).tolist()},
    }


def sequence_frame(config: Config, version: str, split: str, sequence_id: str, index: int) -> dict[str, Any]:
    path = resolve_sequence(config, normalize_version(version), split, sequence_id)
    ego = np.load(path / "ego_scans.npy", mmap_mode="r")
    if index < 0 or index >= len(ego):
        raise IndexError(f"Frame index must be between 0 and {len(ego) - 1}")
    rsu = np.load(path / "rsu_scans.npy", mmap_mode="r")
    meta = np.load(path / "rsu_meta.npy", mmap_mode="r")
    targets = np.load(path / "targets.npy", mmap_mode="r")
    mask_path = path / "rsu_mask.npy"
    mask = np.load(mask_path, mmap_mode="r")[index].astype(bool) if mask_path.is_file() else np.isfinite(rsu[index]).any(axis=1)
    return {
        "index": index, "samples": int(len(ego)), "max_range": 45.0,
        "ego": clean_scan(ego[index]), "rsus": [clean_scan(scan) for scan in rsu[index]],
        "sensor_fov_deg": [EGO_FOV_DEG, *RSU_FOV_DEG],
        "sensor_max_ranges": SENSOR_MAX_RANGES,
        "mask": mask.tolist(), "meta": np.nan_to_num(meta[index], nan=0.0, posinf=999.0, neginf=-999.0).astype(float).tolist(),
        "acceleration": float(np.nan_to_num(targets[index, 0])), "steering": float(np.nan_to_num(targets[index, 1])),
    }


def evaluation_frame(config: Config, evaluation_id: str, sequence_id: str, index: int) -> dict[str, Any]:
    evaluation_dir = resolved_child(config.evaluation_root, evaluation_id)
    report = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    sequence = next((item for item in report.get("sequences", []) if item.get("id") == sequence_id), None)
    if sequence is None:
        return {"available": False, "reason": "Sequence was not included in this evaluation"}
    prediction_path = resolved_child(evaluation_dir / "sequences", str(sequence["file"]))
    with np.load(prediction_path) as prediction:
        sample_indices = prediction["sample_indices"]
        position = int(np.searchsorted(sample_indices, index))
        if position >= len(sample_indices) or int(sample_indices[position]) != index:
            return {"available": False, "reason": "Prediction is unavailable near the history/horizon boundary"}
        trajectories = np.asarray(prediction["trajectories"][position], dtype=np.float64)
        target = np.asarray(prediction["target_trajectories"][position], dtype=np.float64)
        probabilities = np.asarray(prediction["mode_probabilities"][position], dtype=np.float64)
        controls = np.asarray(prediction["controls"][position], dtype=np.float64)
        target_controls = np.asarray(prediction["target_controls"][position], dtype=np.float64)
        gates = np.asarray(prediction["gates"][position], dtype=np.float64)
    dataset_path = resolve_sequence(
        config, normalize_version(report.get("version")), str(report.get("split", "val")), sequence_id
    )
    pose_path = dataset_path / "ego_poses.npy"
    if not pose_path.is_file():
        raise FileNotFoundError("Sequence has no ego_poses.npy; preprocess it again")
    ego_pose = np.asarray(np.load(pose_path, mmap_mode="r")[index], dtype=np.float64)
    map_trajectories = local_trajectories_to_map(trajectories, ego_pose)
    map_target = local_trajectories_to_map(target[None, ...], ego_pose)[0]
    selected_mode = int(np.argmax(probabilities))
    selected = trajectories[selected_mode]
    displacement = np.linalg.norm(selected[:, :2] - target[:, :2], axis=1)
    return {
        "available": True, "sample_index": index,
        "trajectories": trajectories.tolist(), "target_trajectory": target.tolist(),
        "map_trajectories": map_trajectories.tolist(), "map_target_trajectory": map_target.tolist(),
        "ego_pose": ego_pose.tolist(),
        "mode_probabilities": probabilities.tolist(), "selected_mode": selected_mode,
        "control": controls.tolist(), "target_control": target_controls.tolist(),
        "gates": gates.tolist(), "metrics": sequence.get("metrics", {}),
        "frame_metrics": {"ade_m": float(displacement.mean()), "fde_m": float(displacement[-1])},
    }


def local_trajectories_to_map(trajectories: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    result = np.asarray(trajectories, dtype=np.float64).copy()
    forward = result[..., 0].copy()
    left = result[..., 1].copy()
    cosine, sine = math.cos(float(ego_pose[2])), math.sin(float(ego_pose[2]))
    result[..., 0] = ego_pose[0] + cosine * forward - sine * left
    result[..., 1] = ego_pose[1] + sine * forward + cosine * left
    return result


@lru_cache(maxsize=1)
def lane_polylines() -> list[list[list[float]]]:
    if not LANE_CSV.is_file():
        return []
    groups: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
    with LANE_CSV.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                key = (str(row["way_id"]), str(row["boundary_type"]))
                groups.setdefault(key, []).append(
                    (int(row["sequence_order"]), float(row["local_x"]), float(row["local_y"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
    return [
        [[x, y] for _, x, y in sorted(points)]
        for points in groups.values() if len(points) >= 2
    ]


def evaluation_course(config: Config, evaluation_id: str, requested_sequence: str = "") -> dict[str, Any]:
    evaluation_dir = resolved_child(config.evaluation_root, evaluation_id)
    report = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    sequence_options = [
        {"id": str(item.get("id", "")), "metrics": item.get("metrics", {})}
        for item in report.get("sequences", [])
    ]
    if not sequence_options:
        raise ValueError("Evaluation has no sequence results")
    sequence_id = requested_sequence or sequence_options[0]["id"]
    sequence = next((item for item in report["sequences"] if item.get("id") == sequence_id), None)
    if sequence is None:
        raise ValueError("Sequence was not included in this evaluation")
    prediction_path = resolved_child(evaluation_dir / "sequences", str(sequence["file"]))
    with np.load(prediction_path) as prediction:
        sample_indices = np.asarray(prediction["sample_indices"], dtype=np.int64)
        trajectories = np.asarray(prediction["trajectories"], dtype=np.float64)
        targets = np.asarray(prediction["target_trajectories"], dtype=np.float64)
        probabilities = np.asarray(prediction["mode_probabilities"], dtype=np.float64)
    dataset_path = resolve_sequence(
        config, normalize_version(report.get("version")), str(report.get("split", "val")), sequence_id
    )
    poses = np.asarray(np.load(dataset_path / "ego_poses.npy", mmap_mode="r"), dtype=np.float64)
    selected_modes = probabilities.argmax(axis=1)
    selected = trajectories[np.arange(len(trajectories)), selected_modes]
    displacement = np.linalg.norm(selected[..., :2] - targets[..., :2], axis=-1)
    ade = displacement.mean(axis=1)
    fde = displacement[:, -1]
    point_stride = max(1, math.ceil(len(sample_indices) / 1800))
    route_stride = max(1, math.ceil(len(poses) / 4000))
    positions = np.arange(0, len(sample_indices), point_stride, dtype=np.int64)
    heat_points = np.column_stack((
        poses[sample_indices[positions], 0], poses[sample_indices[positions], 1],
        ade[positions], fde[positions], selected_modes[positions], positions,
    ))
    return {
        "evaluation": evaluation_id, "sequence": sequence_id,
        "sequence_options": sequence_options, "sample_indices": sample_indices.astype(int).tolist(),
        "lanes": lane_polylines(), "route": poses[::route_stride, :2].astype(float).tolist(),
        "heat_points": heat_points.astype(float).tolist(),
        "error_scale_m": float(max(1.0, np.percentile(ade, 90))),
        "metrics": sequence.get("metrics", {}), "trajectory": report.get("trajectory", {}),
    }


def preprocess_worker(config: Config, payload: dict[str, Any], manager: common.JobManager) -> Callable[[common.Job], int]:
    train_ids, val_ids = payload.get("train", []), payload.get("val", [])
    if not isinstance(train_ids, list) or not isinstance(val_ids, list) or not (train_ids or val_ids):
        raise ValueError("Select at least one recording")
    version = normalize_version(payload.get("dataset_version"))
    policy = str(payload.get("existing_policy", "skip"))
    if policy not in {"skip", "overwrite"}:
        raise ValueError("existing_policy must be skip or overwrite")
    ego_topic = str(payload.get("ego_scan_topic", "/sensing/lidar/scan"))
    if ego_topic not in {"/sensing/lidar/scan", "/sensing/lidar/scan_without_obstacles"}:
        raise ValueError("Unsupported ego scan topic")
    sync = common.finite_float(payload.get("max_sync_dt", 0.1), "max_sync_dt", 0.0, 10.0)
    scan_dim = common.bounded_int(payload.get("scan_dim", 1080), "scan_dim", 64, 10000)
    input_mode = str(payload.get("input_mode", "scan_fusion"))
    if input_mode not in {"bev", "scan_fusion"}:
        raise ValueError("input_mode must be bev or scan_fusion")
    selected: list[tuple[str, str, Path, Path]] = []
    for split, ids in (("train", train_ids), ("val", val_ids)):
        for raw_id in ids:
            record_id = str(raw_id)
            bag = resolved_child(config.record_root, record_id)
            if not (bag / "metadata.yaml").is_file():
                raise FileNotFoundError(f"Recording is missing metadata.yaml: {record_id}")
            output = version_root(config, version) / split / record_id.strip("/").replace("/", "_")
            selected.append((split, record_id, bag, output))

    def worker(job: common.Job) -> int:
        job.append(
            f"[INFO] version={version} input={input_mode} ego_topic={ego_topic} "
            f"train={len(train_ids)} val={len(val_ids)}"
        )
        for number, (split, record_id, bag, output) in enumerate(selected, 1):
            if job.cancel_requested:
                return 130
            job.append(f"[INFO] [{number}/{len(selected)}] {split}: {record_id}")
            if output.exists() and policy == "skip":
                job.append(f"[SKIP] {output}")
                continue
            command = [sys.executable, str(SCRIPT_DIR / "preprocess_bag_to_npy.py"), "--bag", str(bag),
                       "--output", str(output), "--ego-scan-topic", ego_topic,
                       "--rsu-scan-topics", ",".join(RSU_TOPICS), "--control-topic", "/control/command/control_cmd",
                       "--pose-topic", "/localization/pose_with_covariance", "--rsu-poses", RSU_POSES,
                       "--velocity-topic", "/vehicle/status/velocity_status",
                       "--target-mode", "accel_steer", "--scan-dim", str(scan_dim), "--max-sync-dt", str(sync),
                       "--timestamp-source", "bag"]
            if input_mode == "bev":
                command.append("--require-bev")
            status = manager.run_command(job, command, SCRIPT_DIR)
            if status != 0:
                return status
        job.append("[OK] RSU fusion preprocessing completed")
        return 0
    return worker


def training_worker(config: Config, payload: dict[str, Any], manager: common.JobManager) -> Callable[[common.Job], int]:
    version = normalize_version(payload.get("dataset_version"))
    input_mode = str(payload.get("input_mode", "scan_fusion"))
    if input_mode not in {"bev", "scan_fusion"}:
        raise ValueError("input_mode must be bev or scan_fusion")
    available = sequences(config, version)
    for split in ("train", "val"):
        if not any(
            item["split"] == split and item["valid"] and item["trajectory_ready"]
            and (input_mode != "bev" or item["bev_ready"])
            for item in available
        ):
            requirement = "BEV trajectory-ready" if input_mode == "bev" else "trajectory-ready"
            raise RuntimeError(
                f"No {requirement} {split} sequence is available in {version}; preprocess the bags again"
            )
    history = common.bounded_int(payload.get("history_len", 5), "history_len", 1, 100)
    batch = common.bounded_int(payload.get("batch_size", 64), "batch_size", 1, 65536)
    workers = common.bounded_int(payload.get("workers", 4), "workers", 0, 64)
    epochs = common.bounded_int(payload.get("epochs", 80), "epochs", 1, 10000)
    patience = common.bounded_int(payload.get("patience", 12), "patience", 0, 10000)
    top_k = common.bounded_int(payload.get("top_k_rsus", 2), "top_k_rsus", 0, 6)
    lr = common.finite_float(payload.get("learning_rate", 3e-4), "learning_rate", 1e-8, 10.0)
    decay = common.finite_float(payload.get("distance_decay_m", 35.0), "distance_decay_m", 0.01, 10000.0)
    trajectory_steps = common.bounded_int(payload.get("trajectory_steps", 12), "trajectory_steps", 2, 100)
    trajectory_modes = common.bounded_int(payload.get("trajectory_modes", 4), "trajectory_modes", 1, 12)
    trajectory_anchors = common.bounded_int(payload.get("trajectory_anchors", 4), "trajectory_anchors", 1, 20)
    if trajectory_anchors > trajectory_steps:
        raise ValueError("trajectory_anchors must not exceed trajectory_steps")
    trajectory_dt = common.finite_float(payload.get("trajectory_dt", 0.25), "trajectory_dt", 0.05, 2.0)
    ade_weight = common.finite_float(payload.get("ade_weight", 1.0), "ade_weight", 0.0, 100.0)
    fde_weight = common.finite_float(payload.get("fde_weight", 1.5), "fde_weight", 0.0, 100.0)
    if ade_weight + fde_weight <= 0.0:
        raise ValueError("At least one of ade_weight and fde_weight must be positive")
    pretrained_id = str(payload.get("pretrained", "")).strip()
    pretrained = resolved_child(config.checkpoint_root, pretrained_id) if pretrained_id else None
    if pretrained is not None and (not pretrained.is_file() or pretrained.suffix != ".pth"):
        raise FileNotFoundError("Selected pretrained checkpoint does not exist")
    root = version_root(config, version)
    save_root = config.checkpoint_root if version == DEFAULT_VERSION else config.checkpoint_root / "versions" / version
    run_name = f"h{history}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    anchor_dt = trajectory_steps * trajectory_dt / trajectory_anchors
    max_anchor_step = min(1.0, 1.1 * 15.0 * anchor_dt / 50.0)
    command = [sys.executable, str(SCRIPT_DIR / "train.py"),
               f"data.train_dir={root / 'train'}", f"data.val_dir={root / 'val'}", f"data.history_len={history}",
               f"train.batch_size={batch}", f"train.num_workers={workers}", f"train.epochs={epochs}",
               f"train.lr={lr}", f"train.early_stop_patience={patience}", f"train.save_dir={save_root}",
               f"train.log_dir={SCRIPT_DIR / 'logs'}", f"train.run_name={run_name}",
               f"model.top_k_rsus={top_k}", f"model.distance_decay_m={decay}",
               f"model.trajectory_modes={trajectory_modes}", f"data.trajectory_steps={trajectory_steps}",
               f"model.trajectory_anchor_count={trajectory_anchors}",
               f"model.max_anchor_step_normalized={max_anchor_step:.6f}",
               f"data.trajectory_dt={trajectory_dt}",
               f"loss.average_displacement_weight={ade_weight}", f"loss.endpoint_weight={fde_weight}",
               "loss.acceleration_weight=1.0", "loss.steering_weight=1.0"]
    command.append(
        "model.architecture=bev_trajectory_bezier_v1"
        if input_mode == "bev" else "model.architecture=trajectory_bezier_v2"
    )
    if pretrained is not None:
        command.append(f"train.pretrained={pretrained}")

    def worker(job: common.Job) -> int:
        horizon = trajectory_steps * trajectory_dt
        job.append(
            f"[INFO] version={version} input={input_mode} history={history} top_k={top_k} "
            f"Bezier={trajectory_modes} modes, {trajectory_steps} points/{horizon:.2f}s, "
            f"{trajectory_anchors} anchors/{anchor_dt:.2f}s, ADE/FDE={ade_weight:g}/{fde_weight:g}"
        )
        return manager.run_command(job, command, SCRIPT_DIR)
    return worker


def evaluation_worker(config: Config, payload: dict[str, Any], manager: common.JobManager) -> Callable[[common.Job], int]:
    version = normalize_version(payload.get("dataset_version"))
    split = str(payload.get("split", "val"))
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    checkpoint_id = str(payload.get("checkpoint", "")).strip()
    if not checkpoint_id:
        raise ValueError("Select a checkpoint")
    checkpoint = resolved_child(config.checkpoint_root, checkpoint_id)
    if not checkpoint.is_file() or checkpoint.suffix != ".pth":
        raise FileNotFoundError("Selected checkpoint does not exist")
    batch = common.bounded_int(payload.get("batch_size", 64), "batch_size", 1, 4096)
    device = str(payload.get("device", "auto"))
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    dataset_dir = version_root(config, version) / split
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset split does not exist: {version}/{split}")
    run_name = f"{version}_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = config.evaluation_root / run_name
    command = [
        sys.executable, str(SCRIPT_DIR / "evaluate.py"), "--checkpoint", str(checkpoint),
        "--dataset-dir", str(dataset_dir), "--output", str(output), "--batch-size", str(batch),
        "--device", device, "--version", version, "--split", split,
    ]

    def worker(job: common.Job) -> int:
        job.append(f"[INFO] offline evaluation version={version} split={split} checkpoint={checkpoint_id}")
        return manager.run_command(job, command, SCRIPT_DIR)
    return worker


class Handler(BaseHTTPRequestHandler):
    config: Config
    jobs: common.JobManager

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(f"[HTTP] {self.address_string()} {fmt % args}\n")

    def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def error(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("Invalid JSON request size")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def serve_static(self, path: str) -> None:
        name = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/rsu.css": "rsu.css"}.get(path)
        source = STATIC_DIR / name if name else (TINY_WORKSPACE / "dashboard/style.css" if path == "/style.css" else None)
        if source is None or not source.is_file():
            self.error(HTTPStatus.NOT_FOUND, "Not found"); return
        body = source.read_bytes(); content_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        self.send_response(200); self.send_header("Content-Type", f"{content_type}; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if not parsed.path.startswith("/api/"):
                self.serve_static(parsed.path); return
            if parsed.path == "/api/overview":
                all_sequences = sequences(self.config)
                self.send_json({"recordings": recordings(self.config), "sequences": all_sequences,
                                "versions": versions(self.config, all_sequences), "checkpoints": checkpoints(self.config),
                                "evaluations": evaluations(self.config),
                                "jobs": self.jobs.list(), "server": {"pid": os.getpid()}}); return
            query = parse_qs(parsed.query)
            if parsed.path == "/api/recordings/latest":
                self.send_json({"recording": common.latest_recording(self.config.record_root)}); return
            args = (query.get("version", [DEFAULT_VERSION])[0], query.get("split", [""])[0], query.get("id", [""])[0])
            if parsed.path == "/api/sequence":
                self.send_json(sequence_detail(self.config, *args)); return
            if parsed.path == "/api/frame":
                self.send_json(sequence_frame(self.config, *args, int(query.get("index", ["0"])[0]))); return
            if parsed.path == "/api/prediction":
                self.send_json(evaluation_frame(
                    self.config, query.get("evaluation", [""])[0], query.get("id", [""])[0],
                    int(query.get("index", ["0"])[0]),
                )); return
            if parsed.path == "/api/evaluation-course":
                self.send_json(evaluation_course(
                    self.config, query.get("evaluation", [""])[0],
                    query.get("sequence", [""])[0],
                )); return
            if parsed.path.startswith("/api/jobs/"):
                self.send_json(self.jobs.get(parsed.path.rsplit("/", 1)[-1])); return
            self.error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
        except KeyError:
            self.error(HTTPStatus.NOT_FOUND, "Job not found")
        except (ValueError, FileNotFoundError, IndexError) as exc:
            self.error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            traceback.print_exc(); self.error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/recordings/annotate":
                payload = self.read_json()
                annotation = common.save_collection_annotation(
                    self.config.record_root, payload.get("recording_id"), payload
                )
                self.send_json({"annotation": annotation}); return
            if self.path == "/api/jobs/preprocess":
                job = self.jobs.start("preprocess", preprocess_worker(self.config, self.read_json(), self.jobs)); self.send_json(job.snapshot(), HTTPStatus.ACCEPTED); return
            if self.path == "/api/jobs/train":
                job = self.jobs.start("train", training_worker(self.config, self.read_json(), self.jobs)); self.send_json(job.snapshot(), HTTPStatus.ACCEPTED); return
            if self.path == "/api/jobs/evaluate":
                job = self.jobs.start("evaluate", evaluation_worker(self.config, self.read_json(), self.jobs)); self.send_json(job.snapshot(), HTTPStatus.ACCEPTED); return
            if self.path.startswith("/api/jobs/") and self.path.endswith("/cancel"):
                self.send_json(self.jobs.cancel(self.path.split("/")[-2])); return
            self.error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
        except RuntimeError as exc:
            self.error(HTTPStatus.CONFLICT, str(exc))
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            self.error(HTTPStatus.BAD_REQUEST, str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--record-root", type=Path, default=SCRIPT_DIR / "../../record")
    parser.add_argument("--dataset-root", type=Path, default=SCRIPT_DIR / "datasets")
    parser.add_argument("--checkpoint-root", type=Path, default=SCRIPT_DIR / "checkpoints")
    parser.add_argument("--evaluation-root", type=Path, default=SCRIPT_DIR / "evaluations")
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args()
    config = Config(*(path.expanduser().resolve() for path in (args.record_root, args.dataset_root, args.checkpoint_root, args.evaluation_root)))
    config.dataset_root.mkdir(parents=True, exist_ok=True); config.checkpoint_root.mkdir(parents=True, exist_ok=True); config.evaluation_root.mkdir(parents=True, exist_ok=True)
    pid_file = PidFile(args.pid_file) if args.pid_file else None
    if pid_file:
        try: pid_file.acquire()
        except RuntimeError as exc: raise SystemExit(f"[ERROR] {exc}") from None
    jobs = common.JobManager(); Handler.config = config; Handler.jobs = jobs
    server: ThreadingHTTPServer | None = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        print(f"[READY] RSU Fusion Studio: http://localhost:{args.port}")
        print(f"[INFO] pid={os.getpid()} recordings={config.record_root}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping RSU Fusion Studio and child jobs...")
    finally:
        if server: server.server_close()
        jobs.shutdown()
        if pid_file: pid_file.release()
        print("[OK] RSU Fusion Studio stopped")


if __name__ == "__main__":
    main()

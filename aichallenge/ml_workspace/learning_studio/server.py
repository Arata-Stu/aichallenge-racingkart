#!/usr/bin/env python3
"""Small local web server for the end-to-end driving learning workflow."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
ML_ROOT = APP_DIR.parent
PILOT_ROOT = ML_ROOT / "pilot_net"
TINY_LIDAR_ROOT = ML_ROOT / "tiny_lidar_net"


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


RECORD_ROOT = env_path("E2E_RECORD_ROOT", Path("/aichallenge/record"))
MODEL_SPECS = {
    "pilot_net": {
        "id": "pilot_net",
        "label": "PilotNet",
        "modality": "camera",
        "workspace": PILOT_ROOT,
        "datasets_root": env_path(
            "E2E_PILOT_DATASETS_ROOT",
            env_path("E2E_DATASETS_ROOT", PILOT_ROOT / "datasets"),
        ),
        "output_root": env_path(
            "E2E_PILOT_OUTPUT_ROOT",
            env_path(
                "E2E_STUDIO_OUTPUT_ROOT",
                PILOT_ROOT / "outputs" / "learning_studio",
            ),
        ),
        "sample_file": "images.npy",
        "required_files": ("images.npy", "steers.npy", "accelerations.npy"),
        "defaults": {
            "image_height": 66,
            "image_width": 200,
            "output_dim": 2,
            "color_space": "yuv",
        },
    },
    "tiny_lidar_net": {
        "id": "tiny_lidar_net",
        "label": "TinyLiDARNet",
        "modality": "lidar",
        "workspace": TINY_LIDAR_ROOT,
        "datasets_root": env_path(
            "E2E_TINY_LIDAR_DATASETS_ROOT",
            TINY_LIDAR_ROOT / "datasets",
        ),
        "output_root": env_path(
            "E2E_TINY_LIDAR_OUTPUT_ROOT",
            TINY_LIDAR_ROOT / "outputs" / "learning_studio",
        ),
        "sample_file": "scans.npy",
        "required_files": ("scans.npy", "steers.npy", "accelerations.npy"),
        "defaults": {
            "architecture": "TinyLidarNet",
            "input_dim": 750,
            "output_dim": 2,
            "max_range": 30.0,
        },
    },
}

SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
MIME_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: Any, field_name: str) -> str:
    value = str(value or "").strip()
    if not SLUG_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} は英数字、'.'、'_'、'-' のみで 1〜80 文字にしてください。"
        )
    return value


def get_model_spec(value: Any) -> dict[str, Any]:
    model_type = str(value or "pilot_net")
    try:
        return MODEL_SPECS[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported model_type: {model_type}") from exc


def runs_root(spec: dict[str, Any]) -> Path:
    return spec["output_root"] / "runs"


def evaluations_root(spec: dict[str, Any]) -> Path:
    return spec["output_root"] / "evaluations"


def as_int(payload: dict[str, Any], key: str, minimum: int, maximum: int) -> int:
    value = int(payload[key])
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def as_float(
    payload: dict[str, Any],
    key: str,
    minimum: float,
    maximum: float,
) -> float:
    value = float(payload[key])
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def require_child(path_value: Any, root: Path, kind: str = "path") -> Path:
    path = Path(str(path_value)).expanduser().resolve()
    if not is_relative_to(path, root.resolve()):
        raise ValueError(f"{kind} is outside the allowed root")
    return path


def format_duration(nanoseconds: int) -> str:
    seconds = max(0.0, nanoseconds / 1_000_000_000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}h {int(minutes):02d}m {seconds:04.1f}s"
    if minutes:
        return f"{int(minutes)}m {seconds:04.1f}s"
    return f"{seconds:.1f}s"


def read_bag_metadata(metadata_path: Path) -> dict[str, Any]:
    import yaml

    with metadata_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    info = raw.get("rosbag2_bagfile_information", raw)
    duration = info.get("duration", {}) or {}
    duration_ns = int(duration.get("nanoseconds", 0) or 0)
    topics = []
    for item in info.get("topics_with_message_count", []) or []:
        metadata = item.get("topic_metadata", {}) or {}
        topics.append(
            {
                "name": metadata.get("name", ""),
                "type": metadata.get("type", ""),
                "count": int(item.get("message_count", 0) or 0),
            }
        )
    return {
        "storage": info.get("storage_identifier", ""),
        "duration_ns": duration_ns,
        "duration": format_duration(duration_ns),
        "messages": int(info.get("message_count", 0) or 0),
        "topics": topics,
    }


def discover_sequences() -> list[dict[str, Any]]:
    if not RECORD_ROOT.exists():
        return []

    sequences = []
    for metadata_path in sorted(RECORD_ROOT.rglob("metadata.yaml")):
        sequence_dir = metadata_path.parent.resolve()
        try:
            metadata = read_bag_metadata(metadata_path)
        except Exception as exc:
            metadata = {
                "storage": "",
                "duration_ns": 0,
                "duration": "unknown",
                "messages": 0,
                "topics": [],
                "error": str(exc),
            }
        relative = str(sequence_dir.relative_to(RECORD_ROOT))
        sequence_id = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:12]
        sequences.append(
            {
                "id": sequence_id,
                "name": sequence_dir.name,
                "relative_path": relative,
                "path": str(sequence_dir),
                **metadata,
            }
        )
    return sequences


def count_samples(split_dir: Path, sample_file: str) -> int:
    import numpy as np

    total = 0
    if not split_dir.exists():
        return 0
    for sample_path in split_dir.glob(f"*/{sample_file}"):
        try:
            total += int(np.load(sample_path, mmap_mode="r").shape[0])
        except Exception:
            continue
    return total


def list_datasets() -> list[dict[str, Any]]:
    datasets = []
    for model_type, spec in MODEL_SPECS.items():
        dataset_root = spec["datasets_root"]
        if not dataset_root.exists():
            continue
        for dataset_dir in (
            path for path in dataset_root.iterdir() if path.is_dir()
        ):
            train_count = count_samples(
                dataset_dir / "train",
                spec["sample_file"],
            )
            val_count = count_samples(
                dataset_dir / "val",
                spec["sample_file"],
            )
            if (
                train_count == 0
                and val_count == 0
                and not (dataset_dir / "dataset.json").exists()
            ):
                continue
            input_shape = None
            for split in ("train", "val"):
                sample_paths = list(
                    (dataset_dir / split).glob(f"*/{spec['sample_file']}")
                )
                if not sample_paths:
                    continue
                try:
                    import numpy as np

                    input_shape = list(
                        np.load(sample_paths[0], mmap_mode="r").shape[1:]
                    )
                except Exception:
                    pass
                break
            datasets.append(
                {
                    "name": dataset_dir.name,
                    "path": str(dataset_dir.resolve()),
                    "model_type": model_type,
                    "model_label": spec["label"],
                    "train_samples": train_count,
                    "val_samples": val_count,
                    "input_shape": input_shape,
                    "updated_at": dataset_dir.stat().st_mtime,
                }
            )
    datasets.sort(key=lambda item: item["updated_at"], reverse=True)
    return datasets


def list_checkpoints() -> list[dict[str, Any]]:
    checkpoints = []
    for model_type, spec in MODEL_SPECS.items():
        model_runs_root = runs_root(spec)
        candidates: set[Path] = set()
        for root in (spec["workspace"] / "checkpoints", model_runs_root):
            if root.exists():
                candidates.update(path.resolve() for path in root.rglob("*.pth"))
        for path in candidates:
            label = path.name
            model_config = dict(spec["defaults"])
            if is_relative_to(path, model_runs_root):
                relative = path.relative_to(model_runs_root)
                label = f"{relative.parts[0]} / {path.name}"
                run_config_path = model_runs_root / relative.parts[0] / "run.json"
                if run_config_path.exists():
                    try:
                        run_config = json.loads(
                            run_config_path.read_text(encoding="utf-8")
                        )
                        model_config = {
                            key: run_config.get(key, default)
                            for key, default in model_config.items()
                        }
                    except Exception:
                        pass
            checkpoints.append(
                {
                    "name": label,
                    "path": str(path),
                    "model_type": model_type,
                    "model_label": spec["label"],
                    "size": path.stat().st_size,
                    "updated_at": path.stat().st_mtime,
                    "model": model_config,
                }
            )
    checkpoints.sort(key=lambda item: item["updated_at"], reverse=True)
    return checkpoints


def list_evaluations() -> list[dict[str, Any]]:
    evaluations = []
    for model_type, spec in MODEL_SPECS.items():
        for manifest_path in evaluations_root(spec).glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                name = manifest_path.parent.name
                evaluations.append(
                    {
                        "id": f"{model_type}:{name}",
                        "name": name,
                        "model_type": model_type,
                        "model_label": spec["label"],
                        "path": str(manifest_path.parent.resolve()),
                        **manifest.get("summary", {}),
                        "created_at": manifest.get("created_at", ""),
                        "dataset_name": manifest.get("dataset_name", ""),
                        "split": manifest.get("split", ""),
                        "checkpoint_name": Path(
                            manifest.get("checkpoint", "")
                        ).name,
                    }
                )
            except Exception:
                continue
    evaluations.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return evaluations


@dataclass
class Job:
    id: str
    kind: str
    name: str
    status: str = "queued"
    progress: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    error: str | None = None
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=1500))
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    cancel_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, message: str) -> None:
        text = message.rstrip()
        if text:
            with self.lock:
                self.lines.append(text)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "name": self.name,
                "status": self.status,
                "progress": self.progress,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "return_code": self.return_code,
                "error": self.error,
                "log": list(self.lines),
            }

    def run_command(self, command: list[str], cwd: Path) -> None:
        if self.cancel_requested:
            raise RuntimeError("Job cancelled")
        self.append("$ " + " ".join(command))
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.process = process
        assert process.stdout is not None
        for line in process.stdout:
            self.append(line)
            if self.cancel_requested and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        self.return_code = process.wait()
        self.process = None
        if self.cancel_requested:
            raise RuntimeError("Job cancelled")
        if self.return_code != 0:
            raise RuntimeError(f"Command failed with exit code {self.return_code}")


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Job | None = None
        self._latest: Job | None = None

    def start(self, kind: str, name: str, task: Callable[[Job], None]) -> Job:
        with self._lock:
            if self._active and self._active.status in {"queued", "running"}:
                raise ValueError("別の job が実行中です。完了または停止してから開始してください。")
            job = Job(id=uuid.uuid4().hex[:10], kind=kind, name=name)
            self._active = job
            self._latest = job

        def runner() -> None:
            job.status = "running"
            job.started_at = utc_now()
            try:
                task(job)
                job.progress = 1.0
                job.status = "cancelled" if job.cancel_requested else "succeeded"
            except Exception as exc:
                job.error = str(exc)
                job.append(f"[ERROR] {exc}")
                if not isinstance(exc, RuntimeError) or str(exc) != "Job cancelled":
                    job.append(traceback.format_exc())
                job.status = "cancelled" if job.cancel_requested else "failed"
            finally:
                job.finished_at = utc_now()
                with self._lock:
                    if self._active is job:
                        self._active = None

        threading.Thread(target=runner, daemon=True, name=f"studio-{kind}").start()
        return job

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            job = self._latest
        return job.snapshot() if job else None

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            job = self._active
        if not job:
            raise ValueError("実行中の job はありません。")
        job.cancel_requested = True
        job.append("[INFO] Stop requested...")
        process = job.process
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        return job.snapshot()


JOBS = JobManager()
SEQUENCE_CACHE: dict[str, dict[str, Any]] = {}
GRAD_CAM_LOCK = threading.Lock()


def refresh_sequence_cache() -> list[dict[str, Any]]:
    sequences = discover_sequences()
    SEQUENCE_CACHE.clear()
    SEQUENCE_CACHE.update({item["id"]: item for item in sequences})
    return sequences


def start_extraction(payload: dict[str, Any]) -> Job:
    spec = get_model_spec(payload.get("model_type"))
    model_type = spec["id"]
    dataset_name = safe_slug(payload.get("dataset_name"), "dataset_name")
    dataset_dir = (spec["datasets_root"] / dataset_name).resolve()
    if dataset_dir.exists():
        raise ValueError(f"dataset '{dataset_name}' は既に存在します。別名を指定してください。")

    assignments = payload.get("assignments", [])
    selected: list[tuple[dict[str, Any], str]] = []
    for assignment in assignments:
        sequence = SEQUENCE_CACHE.get(str(assignment.get("id")))
        split = str(assignment.get("split", "unused"))
        if not sequence or split not in {"train", "val", "both", "unused"}:
            continue
        if split != "unused":
            selected.append((sequence, split))
    if not selected:
        raise ValueError("train または val に割り当てた sequence がありません。")

    control_topic = str(payload.get("control_topic", "/control/command/control_cmd")).strip()
    if not control_topic.startswith("/"):
        raise ValueError("topic 名は '/' から始めてください。")
    workers = as_int(payload, "workers", 1, 32)

    manifest = {
        "name": dataset_name,
        "model_type": model_type,
        "model_label": spec["label"],
        "created_at": utc_now(),
        "record_root": str(RECORD_ROOT),
        "control_topic": control_topic,
        "assignments": [
            {
                "id": sequence["id"],
                "path": sequence["path"],
                "relative_path": sequence["relative_path"],
                "split": split,
            }
            for sequence, split in selected
        ],
    }
    if model_type == "pilot_net":
        image_topic = str(
            payload.get("image_topic", "/sensing/camera/image_raw")
        ).strip()
        if not image_topic.startswith("/"):
            raise ValueError("topic 名は '/' から始めてください。")
        image_height = as_int(payload, "image_height", 16, 2160)
        image_width = as_int(payload, "image_width", 16, 3840)
        crop_top_ratio = as_float(payload, "crop_top_ratio", 0.0, 0.95)
        manifest.update(
            {
                "image_topic": image_topic,
                "image_height": image_height,
                "image_width": image_width,
                "crop_top_ratio": crop_top_ratio,
            }
        )
    else:
        scan_topic = str(
            payload.get("scan_topic", "/sensing/lidar/scan")
        ).strip()
        if not scan_topic.startswith("/"):
            raise ValueError("topic 名は '/' から始めてください。")
        max_range = as_float(payload, "max_range", 0.1, 10000.0)
        manifest.update(
            {
                "scan_topic": scan_topic,
                "max_range": max_range,
            }
        )

    def task(job: Job) -> None:
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "train").mkdir()
        (dataset_dir / "val").mkdir()
        (dataset_dir / "dataset.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        operations = [
            (sequence, target_split)
            for sequence, split in selected
            for target_split in (("train", "val") if split == "both" else (split,))
        ]
        for index, (sequence, target_split) in enumerate(operations, start=1):
            target_root = dataset_dir / target_split
            temp_root = dataset_dir / ".extracting" / f"{sequence['id']}-{target_split}"
            target_root.mkdir(parents=True, exist_ok=True)
            temp_root.mkdir(parents=True, exist_ok=True)
            job.append(
                f"[{index}/{len(operations)}] {sequence['relative_path']} -> {target_split}"
            )
            command = [
                sys.executable,
                "-u",
                str(spec["workspace"] / "extract_data_from_bag.py"),
                "--seq-dirs",
                sequence["path"],
                "--outdir",
                str(temp_root),
                "--control-topic",
                control_topic,
                "--workers",
                str(workers),
            ]
            if model_type == "pilot_net":
                command.extend(
                    [
                        "--image-topic",
                        image_topic,
                        "--image-height",
                        str(image_height),
                        "--image-width",
                        str(image_width),
                        "--crop-top-ratio",
                        str(crop_top_ratio),
                    ]
                )
            else:
                command.extend(
                    [
                        "--scan-topic",
                        scan_topic,
                        "--max-scan-range",
                        str(max_range),
                    ]
                )
            job.run_command(command, spec["workspace"])
            extracted_dir = temp_root / Path(sequence["path"]).name
            if not extracted_dir.exists():
                raise RuntimeError(
                    f"{sequence['relative_path']} から学習データを生成できませんでした。"
                )
            target_name = f"{Path(sequence['path']).name}__{sequence['id']}"
            shutil.move(str(extracted_dir), str(target_root / target_name))
            shutil.rmtree(temp_root, ignore_errors=True)
            job.progress = index / len(operations)
        shutil.rmtree(dataset_dir / ".extracting", ignore_errors=True)
        job.append(f"[DONE] Dataset created: {dataset_dir}")

    return JOBS.start("extract", dataset_name, task)


def start_training(payload: dict[str, Any]) -> Job:
    spec = get_model_spec(payload.get("model_type"))
    model_type = spec["id"]
    dataset_name = safe_slug(payload.get("dataset_name"), "dataset_name")
    run_name = safe_slug(payload.get("run_name"), "run_name")
    dataset_dir = require_child(
        spec["datasets_root"] / dataset_name,
        spec["datasets_root"],
        "dataset",
    )
    if not dataset_dir.exists():
        raise ValueError(f"dataset '{dataset_name}' が見つかりません。")
    train_samples = count_samples(dataset_dir / "train", spec["sample_file"])
    val_samples = count_samples(dataset_dir / "val", spec["sample_file"])
    if train_samples == 0:
        raise ValueError("training split に sample がありません。")
    if model_type == "tiny_lidar_net" and val_samples == 0:
        raise ValueError(
            "TinyLiDARNet の学習には validation split が必要です。"
        )
    run_dir = (runs_root(spec) / run_name).resolve()
    if run_dir.exists():
        raise ValueError(f"run '{run_name}' は既に存在します。別名を指定してください。")

    epochs = as_int(payload, "epochs", 1, 10000)
    batch_size = as_int(payload, "batch_size", 1, 4096)
    num_workers = as_int(payload, "num_workers", 0, 64)
    early_stop = as_int(payload, "early_stop_patience", 1, 10000)
    lr = as_float(payload, "lr", 1e-8, 10.0)
    steer_weight = as_float(payload, "steer_weight", 0.0, 1000.0)
    accel_weight = as_float(payload, "accel_weight", 0.0, 1000.0)

    pretrained = str(payload.get("pretrained_path", "")).strip()
    pretrained_path = None
    if pretrained:
        pretrained_path = require_child(
            pretrained,
            spec["workspace"],
            "checkpoint",
        )
        if not pretrained_path.is_file():
            raise ValueError("pretrained checkpoint が見つかりません。")

    config = {
        "model_type": model_type,
        "run_name": run_name,
        "dataset_name": dataset_name,
        "created_at": utc_now(),
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "early_stop_patience": early_stop,
        "lr": lr,
        "steer_weight": steer_weight,
        "accel_weight": accel_weight,
        "pretrained_path": str(pretrained_path) if pretrained_path else None,
    }

    common_command = [
        sys.executable,
        "-u",
        str(spec["workspace"] / "train.py"),
        f"data.train_dir={dataset_dir / 'train'}",
        f"data.val_dir={dataset_dir / 'val'}",
        f"train.epochs={epochs}",
        f"train.batch_size={batch_size}",
        f"train.num_workers={num_workers}",
        f"train.lr={lr}",
        f"train.early_stop_patience={early_stop}",
        f"train.save_dir={run_dir / 'checkpoints'}",
        f"train.log_dir={run_dir / 'logs'}",
        f"train.loss.steer_weight={steer_weight}",
        f"train.loss.accel_weight={accel_weight}",
        f"hydra.run.dir={run_dir / 'hydra'}",
    ]
    if pretrained_path:
        common_command.append(f"train.pretrained_path={pretrained_path}")

    if model_type == "pilot_net":
        image_height = as_int(payload, "image_height", 16, 2160)
        image_width = as_int(payload, "image_width", 16, 3840)
        output_dim = as_int(payload, "output_dim", 1, 2)
        weight_decay = as_float(payload, "weight_decay", 0.0, 10.0)
        shift_range = as_float(payload, "shift_range", 0.0, 10000.0)
        steer_correction = as_float(
            payload,
            "steer_correction_per_pixel",
            0.0,
            100.0,
        )
        color_space = str(payload.get("color_space", "yuv")).lower()
        if color_space not in {"rgb", "yuv"}:
            raise ValueError("color_space must be rgb or yuv")
        loss_type = str(payload.get("loss_type", "smooth_l1")).lower()
        if loss_type not in {"smooth_l1", "mse"}:
            raise ValueError("loss_type must be smooth_l1 or mse")
        config.update(
            {
                "image_height": image_height,
                "image_width": image_width,
                "output_dim": output_dim,
                "color_space": color_space,
                "weight_decay": weight_decay,
                "shift_range": shift_range,
                "steer_correction_per_pixel": steer_correction,
                "loss_type": loss_type,
            }
        )
        command = common_command + [
            f"model.image_height={image_height}",
            f"model.image_width={image_width}",
            f"model.output_dim={output_dim}",
            f"model.color_space={color_space}",
            "model.crop_top_ratio=0.0",
            "model.crop_bottom_ratio=0.0",
            f"train.weight_decay={weight_decay}",
            f"train.shift_range={shift_range}",
            f"train.steer_correction_per_pixel={steer_correction}",
        ]
        if loss_type == "mse":
            command.append("+train.loss_type=mse")
    else:
        architecture = str(payload.get("architecture", "TinyLidarNet"))
        if architecture not in {"TinyLidarNet", "TinyLidarNetSmall"}:
            raise ValueError("Unsupported TinyLiDARNet architecture")
        input_dim = as_int(payload, "input_dim", 32, 100000)
        max_range = as_float(payload, "max_range", 0.1, 10000.0)
        device = str(payload.get("device", "auto")).lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        config.update(
            {
                "architecture": architecture,
                "input_dim": input_dim,
                "output_dim": 2,
                "max_range": max_range,
                "device": device,
            }
        )
        command = common_command + [
            f"model.name={architecture}",
            f"model.input_dim={input_dim}",
            "model.output_dim=2",
            f"data.max_range={max_range}",
            f"train.device={device}",
        ]

    def task(job: Job) -> None:
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        job.run_command(command, spec["workspace"])
        job.append(f"[DONE] Training completed: {run_dir}")

    return JOBS.start("train", run_name, task)


def start_evaluation(payload: dict[str, Any]) -> Job:
    spec = get_model_spec(payload.get("model_type"))
    model_type = spec["id"]
    evaluation_name = safe_slug(payload.get("evaluation_name"), "evaluation_name")
    dataset_name = safe_slug(payload.get("dataset_name"), "dataset_name")
    split = str(payload.get("split", "val"))
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    dataset_dir = require_child(
        spec["datasets_root"] / dataset_name,
        spec["datasets_root"],
        "dataset",
    )
    split_dir = dataset_dir / split
    if not split_dir.exists():
        raise ValueError(f"{dataset_name}/{split} が見つかりません。")
    checkpoint = require_child(
        payload.get("checkpoint"),
        spec["workspace"],
        "checkpoint",
    )
    if not checkpoint.is_file():
        raise ValueError("checkpoint が見つかりません。")
    output_dir = (evaluations_root(spec) / evaluation_name).resolve()
    if output_dir.exists():
        raise ValueError(
            f"evaluation '{evaluation_name}' は既に存在します。別名を指定してください。"
        )

    batch_size = as_int(payload, "batch_size", 1, 4096)
    command_options: list[str]
    if model_type == "pilot_net":
        image_height = as_int(payload, "image_height", 16, 2160)
        image_width = as_int(payload, "image_width", 16, 3840)
        output_dim = as_int(payload, "output_dim", 1, 2)
        color_space = str(payload.get("color_space", "yuv")).lower()
        if color_space not in {"rgb", "yuv"}:
            raise ValueError("color_space must be rgb or yuv")
        command_options = [
            "--image-height",
            str(image_height),
            "--image-width",
            str(image_width),
            "--output-dim",
            str(output_dim),
            "--color-space",
            color_space,
        ]
    else:
        architecture = str(payload.get("architecture", "TinyLidarNet"))
        if architecture not in {"TinyLidarNet", "TinyLidarNetSmall"}:
            raise ValueError("Unsupported TinyLiDARNet architecture")
        input_dim = as_int(payload, "input_dim", 32, 100000)
        max_range = as_float(payload, "max_range", 0.1, 10000.0)
        command_options = [
            "--architecture",
            architecture,
            "--input-dim",
            str(input_dim),
            "--output-dim",
            "2",
            "--max-range",
            str(max_range),
        ]

    def task(job: Job) -> None:
        output_dir.mkdir(parents=True)
        command = [
            sys.executable,
            "-u",
            str(APP_DIR / "evaluate.py"),
            "--model-type",
            model_type,
            "--dataset-dir",
            str(split_dir),
            "--dataset-name",
            dataset_name,
            "--split",
            split,
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            str(batch_size),
        ] + command_options
        job.run_command(command, APP_DIR)
        load_evaluation.cache_clear()
        job.append(f"[DONE] Evaluation completed: {output_dir}")

    return JOBS.start("evaluate", evaluation_name, task)


def parse_evaluation_ref(reference: str) -> tuple[dict[str, Any], str]:
    if ":" in reference:
        model_type, name = reference.split(":", 1)
    else:
        model_type, name = "pilot_net", reference
    return get_model_spec(model_type), safe_slug(name, "evaluation")


@lru_cache(maxsize=8)
def load_evaluation(reference: str) -> tuple[dict[str, Any], Any]:
    import numpy as np

    spec, name = parse_evaluation_ref(reference)
    root = evaluations_root(spec)
    evaluation_dir = require_child(root / name, root, "evaluation")
    manifest = json.loads((evaluation_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.setdefault("model_type", spec["id"])
    manifest.setdefault("model_label", spec["label"])
    results = np.load(evaluation_dir / "results.npz", allow_pickle=False)
    return manifest, results


def evaluation_detail(reference: str) -> dict[str, Any]:
    import numpy as np

    spec, name = parse_evaluation_ref(reference)
    manifest, results = load_evaluation(reference)
    frame_count = int(manifest["summary"]["frame_count"])
    sample_count = min(frame_count, 700)
    sample_indices = (
        np.linspace(0, frame_count - 1, sample_count, dtype=np.int64)
        if frame_count
        else np.array([], dtype=np.int64)
    )
    worst_count = min(frame_count, 60)
    worst_indices = (
        np.argsort(results["mae"])[-worst_count:][::-1]
        if worst_count
        else np.array([], dtype=np.int64)
    )
    return {
        "id": f"{spec['id']}:{name}",
        "name": name,
        **manifest,
        "series": {
            "indices": sample_indices.tolist(),
            "mae": results["mae"][sample_indices].astype(float).tolist(),
            "steer_error": results["steer_error"][sample_indices].astype(float).tolist(),
        },
        "worst": [frame_info(reference, int(index)) for index in worst_indices],
    }


def locate_frame(manifest: dict[str, Any], index: int) -> tuple[dict[str, Any], int]:
    frame_count = int(manifest["summary"]["frame_count"])
    if not 0 <= index < frame_count:
        raise ValueError(f"frame index must be between 0 and {max(0, frame_count - 1)}")
    for sequence in manifest["sequences"]:
        start = int(sequence["start"])
        count = int(sequence["count"])
        if start <= index < start + count:
            return sequence, index - start
    raise ValueError("frame mapping not found")


def frame_info(reference: str, index: int) -> dict[str, Any]:
    manifest, results = load_evaluation(reference)
    sequence, local_index = locate_frame(manifest, index)
    target = results["targets"][index].astype(float).tolist()
    prediction = results["predictions"][index].astype(float).tolist()
    output_dim = int(manifest["model"]["output_dim"])
    if output_dim == 1:
        target_values = {"steer": target[0]}
        prediction_values = {"steer": prediction[0]}
    else:
        target_values = {"acceleration": target[0], "steer": target[1]}
        prediction_values = {"acceleration": prediction[0], "steer": prediction[1]}
    return {
        "index": index,
        "local_index": local_index,
        "sequence": sequence["name"],
        "mae": float(results["mae"][index]),
        "steer_error": float(results["steer_error"][index]),
        "accel_error": float(results["accel_error"][index]),
        "target": target_values,
        "prediction": prediction_values,
    }


@lru_cache(maxsize=4)
def load_pilot_grad_cam_model(
    checkpoint_value: str,
    image_height: int,
    image_width: int,
    output_dim: int,
) -> Any:
    import torch

    model_path = PILOT_ROOT / "lib" / "model.py"
    module_spec = importlib.util.spec_from_file_location(
        "learning_studio_grad_cam_pilot_model",
        model_path,
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot import PilotNet model: {model_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.PilotNet(
        image_height=image_height,
        image_width=image_width,
        output_dim=output_dim,
    )
    state_dict = torch.load(
        checkpoint_value,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    return model.eval()


def render_pilot_grad_cam(
    rgb: Any,
    manifest: dict[str, Any],
) -> Any:
    import cv2
    import numpy as np
    import torch

    model_config = manifest["model"]
    image_height = int(model_config["image_height"])
    image_width = int(model_config["image_width"])
    output_dim = int(model_config["output_dim"])
    color_space = str(model_config.get("color_space", "yuv")).lower()
    checkpoint = require_child(
        manifest.get("checkpoint"),
        PILOT_ROOT,
        "checkpoint",
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model_input = np.asarray(rgb)
    if model_input.shape[:2] != (image_height, image_width):
        model_input = cv2.resize(
            model_input,
            (image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )
    if color_space == "yuv":
        model_input = cv2.cvtColor(model_input, cv2.COLOR_RGB2YUV)
    elif color_space != "rgb":
        raise ValueError(f"Unsupported color space: {color_space}")
    batch = np.ascontiguousarray(
        (model_input.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    )
    tensor = torch.from_numpy(batch).requires_grad_(True)

    with GRAD_CAM_LOCK, torch.enable_grad():
        model = load_pilot_grad_cam_model(
            str(checkpoint),
            image_height,
            image_width,
            output_dim,
        )
        captured: dict[str, Any] = {}

        def capture_activation(
            _module: Any,
            _inputs: Any,
            output: Any,
        ) -> None:
            captured["activation"] = output
            output.register_hook(
                lambda gradient: captured.__setitem__("gradient", gradient)
            )

        hook = model.conv5.register_forward_hook(capture_activation)
        try:
            model.zero_grad(set_to_none=True)
            output = model(tensor)
            steer_index = 0 if output_dim == 1 else 1
            output[0, steer_index].backward()
        finally:
            hook.remove()

        activation = captured.get("activation")
        gradient = captured.get("gradient")
        if activation is None or gradient is None:
            raise RuntimeError("Grad-CAM feature map could not be captured")
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1))[0]
        cam = cam.detach().cpu().numpy()

    cam -= float(cam.min())
    maximum = float(cam.max())
    if maximum > 1e-8:
        cam /= maximum
    cam = cv2.resize(
        cam,
        (rgb.shape[1], rgb.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    heatmap = cv2.applyColorMap(
        np.uint8(np.clip(cam, 0.0, 1.0) * 255),
        cv2.COLORMAP_JET,
    )
    original = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(original, 0.55, heatmap, 0.45, 0.0)


def render_frame(
    reference: str,
    index: int,
    overlay: bool,
    grad_cam: bool,
) -> bytes:
    import cv2
    import numpy as np

    manifest, results = load_evaluation(reference)
    sequence, local_index = locate_frame(manifest, index)
    model_type = manifest.get("model_type", "pilot_net")
    if grad_cam and model_type != "pilot_net":
        raise ValueError("Grad-CAM is currently available for PilotNet only")
    if model_type == "tiny_lidar_net":
        scans = np.load(Path(sequence["path"]) / "scans.npy", mmap_mode="r")
        scan = np.asarray(scans[local_index], dtype=np.float32)
        max_range = float(manifest["model"].get("max_range", 30.0))
        bgr = np.full((620, 1000, 3), (10, 15, 20), dtype=np.uint8)
        origin = (500, 560)
        radius = 460
        for fraction in (0.25, 0.5, 0.75, 1.0):
            cv2.ellipse(
                bgr,
                origin,
                (int(radius * fraction), int(radius * fraction)),
                0,
                200,
                140,
                (39, 54, 66),
                1,
                cv2.LINE_AA,
            )
        angles = np.linspace(-3 * np.pi / 4, 3 * np.pi / 4, len(scan))
        distances = np.clip(np.nan_to_num(scan), 0.0, max_range)
        pixels = distances / max_range * radius
        points = np.column_stack(
            (
                origin[0] + np.sin(angles) * pixels,
                origin[1] - np.cos(angles) * pixels,
            )
        ).astype(np.int32)
        valid = distances > 0.02
        for point in points[valid][:: max(1, len(points) // 1400)]:
            cv2.circle(
                bgr,
                (int(point[0]), int(point[1])),
                2,
                (173, 211, 109),
                -1,
                cv2.LINE_AA,
            )
        cv2.circle(bgr, origin, 7, (112, 213, 159), -1, cv2.LINE_AA)
        cv2.putText(
            bgr,
            f"LiDAR scan  {len(scan)} points  max {max_range:g} m",
            (22, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (145, 160, 174),
            2,
            cv2.LINE_AA,
        )
    else:
        images = np.load(Path(sequence["path"]) / "images.npy", mmap_mode="r")
        rgb = np.asarray(images[local_index]).copy()
        bgr = (
            render_pilot_grad_cam(rgb, manifest)
            if grad_cam
            else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
        scale = max(2, min(6, 1000 // max(1, bgr.shape[1])))
        bgr = cv2.resize(
            bgr,
            (bgr.shape[1] * scale, bgr.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    if overlay:
        info = frame_info(reference, index)
        bgr = cv2.copyMakeBorder(
            bgr,
            64,
            100,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(10, 15, 20),
        )
        width = bgr.shape[1]
        cv2.putText(
            bgr,
            f"{info['sequence']}  frame {info['local_index']}",
            (18, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (240, 244, 248),
            2,
            cv2.LINE_AA,
        )
        if grad_cam:
            cv2.putText(
                bgr,
                "Grad-CAM  conv5 / steering",
                (max(18, width - 315), 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (112, 213, 159),
                2,
                cv2.LINE_AA,
            )
        target_steer = float(info["target"]["steer"])
        predicted_steer = float(info["prediction"]["steer"])
        origin = (width // 2, bgr.shape[0] - 14)
        line_length = min(120, width // 4)
        for steer, color in (
            (target_steer, (218, 177, 71)),
            (predicted_steer, (112, 213, 159)),
        ):
            endpoint = (
                int(origin[0] + np.sin(steer) * line_length),
                int(origin[1] - np.cos(steer) * line_length),
            )
            cv2.line(bgr, origin, endpoint, color, 4, cv2.LINE_AA)
        label = (
            f"target steer {target_steer:+.4f}   "
            f"prediction {predicted_steer:+.4f}   MAE {info['mae']:.4f}"
        )
        cv2.putText(
            bgr,
            label,
            (18, bgr.shape[0] - 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 234, 239),
            2,
            cv2.LINE_AA,
        )
    success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Failed to encode frame")
    return encoded.tobytes()


def app_state() -> dict[str, Any]:
    sequences = refresh_sequence_cache()
    return {
        "config": {
            "record_root": str(RECORD_ROOT),
            "record_root_exists": RECORD_ROOT.exists(),
            "python": sys.executable,
            "models": [
                {
                    "id": spec["id"],
                    "label": spec["label"],
                    "modality": spec["modality"],
                    "datasets_root": str(spec["datasets_root"]),
                    "output_root": str(spec["output_root"]),
                    "defaults": spec["defaults"],
                }
                for spec in MODEL_SPECS.values()
            ],
        },
        "sequences": sequences,
        "datasets": list_datasets(),
        "checkpoints": list_checkpoints(),
        "evaluations": list_evaluations(),
        "job": JOBS.latest(),
    }


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "E2ELearningStudio/1.0"

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stderr.write(
            f"[{self.log_date_time_string()}] {format_string % args}\n"
        )

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, exc: Exception, status: int = 400) -> None:
        self.send_json({"error": str(exc)}, status)

    def read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > 1_000_000:
            raise ValueError("Request body is too large")
        raw = self.rfile.read(content_length) if content_length else b"{}"
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON object is required")
        return payload

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        try:
            if path == "/api/state":
                self.send_json(app_state())
                return
            if path == "/api/job":
                self.send_json({"job": JOBS.latest()})
                return
            if path.startswith("/api/evaluations/"):
                parts = [part for part in path.split("/") if part]
                if len(parts) == 3:
                    self.send_json(evaluation_detail(parts[2]))
                    return
                if len(parts) == 4 and parts[3] == "frame-info":
                    index = int(query.get("index", ["0"])[0])
                    self.send_json(frame_info(parts[2], index))
                    return
                if len(parts) == 4 and parts[3] == "frame.jpg":
                    index = int(query.get("index", ["0"])[0])
                    overlay = query.get("overlay", ["1"])[0] != "0"
                    grad_cam = query.get("gradcam", ["0"])[0] != "0"
                    body = render_frame(parts[2], index, overlay, grad_cam)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self.serve_static(path)
        except FileNotFoundError:
            self.send_error_json(FileNotFoundError("Not found"), 404)
        except Exception as exc:
            self.send_error_json(exc, 400)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_json()
            if parsed.path == "/api/extract":
                self.send_json({"job": start_extraction(payload).snapshot()}, 202)
                return
            if parsed.path == "/api/train":
                self.send_json({"job": start_training(payload).snapshot()}, 202)
                return
            if parsed.path == "/api/evaluate":
                self.send_json({"job": start_evaluation(payload).snapshot()}, 202)
                return
            if parsed.path == "/api/job/cancel":
                self.send_json({"job": JOBS.cancel()})
                return
            self.send_error_json(FileNotFoundError("Not found"), 404)
        except json.JSONDecodeError:
            self.send_error_json(ValueError("Invalid JSON"), 400)
        except Exception as exc:
            self.send_error_json(exc, 400)

    def serve_static(self, requested_path: str) -> None:
        relative = "index.html" if requested_path in {"", "/"} else requested_path.lstrip("/")
        static_path = (STATIC_DIR / relative).resolve()
        if not is_relative_to(static_path, STATIC_DIR.resolve()) or not static_path.is_file():
            raise FileNotFoundError(relative)
        body = static_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            MIME_TYPES.get(static_path.suffix.lower(), "application/octet-stream"),
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Cache-Control",
            "no-store" if static_path.suffix == ".html" else "public, max-age=60",
        )
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E2E Learning Studio.")
    parser.add_argument("--host", default=os.environ.get("E2E_STUDIO_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("E2E_STUDIO_PORT", "8765")),
    )
    args = parser.parse_args()

    for spec in MODEL_SPECS.values():
        spec["datasets_root"].mkdir(parents=True, exist_ok=True)
        runs_root(spec).mkdir(parents=True, exist_ok=True)
        evaluations_root(spec).mkdir(parents=True, exist_ok=True)
    refresh_sequence_cache()

    server = ThreadingHTTPServer((args.host, args.port), StudioHandler)
    print(f"E2E Learning Studio: http://{args.host}:{args.port}")
    print(f"Record root: {RECORD_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

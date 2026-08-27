#!/usr/bin/env python3
"""Local web dashboard for TinyLiDARNet dataset inspection and training."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import mimetypes
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
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
REQUIRED_SEQUENCE_FILES = ("scans.npy", "accelerations.npy", "steers.npy")
DEFAULT_DATASET_VERSION = "default"
DATASET_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
COLLECTION_ANNOTATION_FILE = "collection_annotation.json"
COLLECTION_CATEGORIES = {
    "free_optimal",
    "free_diverse",
    "follow",
    "pass_left",
    "pass_right",
    "abort",
    "recovery",
    "other",
}
COLLECTION_OUTCOMES = {"success", "failure", "review"}
COLLECTION_QUALITIES = {"accepted", "review", "rejected"}


@dataclass(frozen=True)
class DashboardConfig:
    record_root: Path
    dataset_root: Path
    checkpoint_root: Path


def process_start_ticks(pid: int) -> int | None:
    """Return Linux /proc start ticks so a reused PID is never mistaken for our process."""
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        # Fields after the final ')' start at field 3 (state); starttime is field 22.
        return int(value[value.rfind(")") + 2 :].split()[19])
    except (OSError, ValueError, IndexError):
        return None


class PidFile:
    """Atomic PID ownership with PID-reuse protection."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.lock_descriptor: int | None = None
        self.identity = {
            "pid": os.getpid(),
            "start_ticks": process_start_ticks(os.getpid()),
            "program": "tiny-lidar-dashboard",
        }
        self.acquired = False

    @staticmethod
    def _is_live(identity: dict[str, Any]) -> bool:
        try:
            pid = int(identity["pid"])
            expected_ticks = int(identity["start_ticks"])
        except (KeyError, TypeError, ValueError):
            return False
        return process_start_ticks(pid) == expected_ticks

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(self.identity, sort_keys=True) + "\n").encode("utf-8")
        lock_descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(lock_descriptor)
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
                existing_pid = existing.get("pid", "unknown")
            except (OSError, json.JSONDecodeError, AttributeError):
                existing_pid = "unknown"
            raise RuntimeError(
                f"TinyLiDAR dashboard is already running with PID {existing_pid}"
            ) from None

        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except Exception:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
            raise
        self.lock_descriptor = lock_descriptor
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if current == self.identity:
                self.path.unlink()
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        if self.lock_descriptor is not None:
            fcntl.flock(self.lock_descriptor, fcntl.LOCK_UN)
            os.close(self.lock_descriptor)
            self.lock_descriptor = None
        self.acquired = False


def resolved_child(root: Path, relative: str) -> Path:
    """Resolve a user-provided relative path without allowing root escape."""
    if not relative or Path(relative).is_absolute():
        raise ValueError("A non-empty relative path is required")
    root = root.resolve()
    candidate = (root / unquote(relative)).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Path escapes the configured root")
    return candidate


def directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def load_collection_annotation(bag: Path) -> dict[str, Any] | None:
    path = bag / COLLECTION_ANNOTATION_FILE
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def recording_info(record_root: Path, metadata: Path, include_size: bool = True) -> dict[str, Any]:
    bag = metadata.parent
    return {
        "id": bag.relative_to(record_root).as_posix(),
        "name": bag.name,
        "size_bytes": directory_size(bag) if include_size else 0,
        "modified_at": metadata.stat().st_mtime,
        "annotation": load_collection_annotation(bag),
    }


def latest_recording(record_root: Path) -> dict[str, Any] | None:
    if not record_root.is_dir():
        return None
    latest: Path | None = None
    latest_mtime = -1.0
    for metadata in record_root.rglob("metadata.yaml"):
        try:
            modified_at = metadata.stat().st_mtime
        except OSError:
            continue
        if modified_at > latest_mtime:
            latest = metadata
            latest_mtime = modified_at
    return recording_info(record_root, latest, include_size=False) if latest else None


def save_collection_annotation(
    record_root: Path, recording_id: Any, payload: dict[str, Any]
) -> dict[str, Any]:
    recording_id = str(recording_id or "").strip()
    bag = resolved_child(record_root, recording_id)
    if not (bag / "metadata.yaml").is_file():
        raise FileNotFoundError(f"Recording is missing metadata.yaml: {recording_id}")

    category = str(payload.get("category", "other")).strip()
    outcome = str(payload.get("outcome", "review")).strip()
    quality = str(payload.get("quality", "review")).strip()
    if category not in COLLECTION_CATEGORIES:
        raise ValueError("Unsupported collection category")
    if outcome not in COLLECTION_OUTCOMES:
        raise ValueError("Unsupported collection outcome")
    if quality not in COLLECTION_QUALITIES:
        raise ValueError("Unsupported collection quality")

    raw_versions = payload.get("dataset_versions", [])
    if isinstance(raw_versions, str):
        raw_versions = raw_versions.split(",")
    if not isinstance(raw_versions, list):
        raise ValueError("dataset_versions must be a list or comma-separated string")
    dataset_versions: list[str] = []
    for raw_version in raw_versions:
        raw_version = str(raw_version).strip()
        if not raw_version:
            continue
        version = normalize_dataset_version(raw_version)
        if version not in dataset_versions:
            dataset_versions.append(version)
    if len(dataset_versions) > 20:
        raise ValueError("At most 20 dataset versions may be attached")

    notes = str(payload.get("notes", "")).strip()
    if len(notes) > 4000:
        raise ValueError("notes must be at most 4000 characters")
    previous = load_collection_annotation(bag) or {}
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    annotation = {
        "schema_version": 1,
        "recording_id": recording_id,
        "category": category,
        "outcome": outcome,
        "quality": quality,
        "dataset_versions": dataset_versions,
        "notes": notes,
        "created_at": previous.get("created_at", now),
        "updated_at": now,
    }
    destination = bag / COLLECTION_ANNOTATION_FILE
    temporary = bag / (
        f".{COLLECTION_ANNOTATION_FILE}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return annotation


def sequence_name_for_recording(recording_id: str) -> str:
    return recording_id.strip("/").replace("/", "_")


def normalize_dataset_version(value: Any) -> str:
    version = str(value or DEFAULT_DATASET_VERSION).strip()
    if version == DEFAULT_DATASET_VERSION:
        return version
    if not DATASET_VERSION_PATTERN.fullmatch(version):
        raise ValueError(
            "dataset_version must be 1-64 ASCII letters, numbers, '.', '_' or '-' "
            "and must start with a letter or number"
        )
    return version


def dataset_version_root(config: DashboardConfig, version: str) -> Path:
    normalized = normalize_dataset_version(version)
    if normalized == DEFAULT_DATASET_VERSION:
        return config.dataset_root
    return config.dataset_root / "versions" / normalized


def discover_version_ids(config: DashboardConfig) -> list[str]:
    versions = [DEFAULT_DATASET_VERSION]
    versions_root = config.dataset_root / "versions"
    if versions_root.is_dir():
        for path in sorted(versions_root.iterdir()):
            if path.is_dir() and DATASET_VERSION_PATTERN.fullmatch(path.name):
                versions.append(path.name)
    return versions


def discover_recordings(config: DashboardConfig) -> list[dict[str, Any]]:
    if not config.record_root.is_dir():
        return []
    recordings = [
        recording_info(config.record_root, metadata)
        for metadata in config.record_root.rglob("metadata.yaml")
    ]
    return sorted(recordings, key=lambda item: item["modified_at"], reverse=True)


def load_summary(sequence_dir: Path) -> dict[str, Any]:
    path = sequence_dir / "preprocess_summary.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def discover_sequences(
    config: DashboardConfig, version: str | None = None
) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    version_ids = (
        [normalize_dataset_version(version)]
        if version is not None
        else discover_version_ids(config)
    )
    for version_id in version_ids:
        version_root = dataset_version_root(config, version_id)
        for split in ("train", "val"):
            split_root = version_root / split
            if not split_root.is_dir():
                continue
            for scans_path in sorted(split_root.rglob("scans.npy")):
                sequence_dir = scans_path.parent
                if not all((sequence_dir / name).is_file() for name in REQUIRED_SEQUENCE_FILES):
                    continue
                try:
                    scans = np.load(scans_path, mmap_mode="r")
                    steers = np.load(sequence_dir / "steers.npy", mmap_mode="r")
                    accelerations = np.load(sequence_dir / "accelerations.npy", mmap_mode="r")
                    valid = scans.ndim == 2 and len(scans) == len(steers) == len(accelerations)
                    sample_count = int(len(scans))
                    scan_points = int(scans.shape[1]) if scans.ndim == 2 else 0
                except (OSError, ValueError):
                    valid = False
                    sample_count = 0
                    scan_points = 0
                sequence_id = sequence_dir.relative_to(split_root).as_posix()
                sequences.append(
                    {
                        "version": version_id,
                        "split": split,
                        "id": sequence_id,
                        "name": sequence_dir.name,
                        "samples": sample_count,
                        "scan_points": scan_points,
                        "valid": valid,
                        "summary": load_summary(sequence_dir),
                    }
                )
    return sequences


def discover_versions(
    config: DashboardConfig, sequences: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    all_sequences = discover_sequences(config) if sequences is None else sequences
    result = []
    for version in discover_version_ids(config):
        selected = [item for item in all_sequences if item["version"] == version]
        result.append(
            {
                "id": version,
                "train_sequences": sum(item["split"] == "train" for item in selected),
                "val_sequences": sum(item["split"] == "val" for item in selected),
                "train_samples": sum(
                    item["samples"] for item in selected if item["split"] == "train"
                ),
                "val_samples": sum(
                    item["samples"] for item in selected if item["split"] == "val"
                ),
            }
        )
    return result


def discover_checkpoints(config: DashboardConfig) -> list[dict[str, Any]]:
    if not config.checkpoint_root.is_dir():
        return []
    checkpoints = []
    for path in sorted(config.checkpoint_root.rglob("*.pth"), reverse=True):
        if not path.is_file():
            continue
        stat = path.stat()
        checkpoints.append(
            {
                "id": path.relative_to(config.checkpoint_root).as_posix(),
                "name": path.name,
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
                "best": path.name == "best_model.pth",
            }
        )
    return checkpoints


def resolve_sequence(
    config: DashboardConfig,
    split: str,
    sequence_id: str,
    version: str = DEFAULT_DATASET_VERSION,
) -> Path:
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    path = resolved_child(dataset_version_root(config, version) / split, sequence_id)
    if not all((path / name).is_file() for name in REQUIRED_SEQUENCE_FILES):
        raise FileNotFoundError("Dataset sequence is incomplete or missing")
    return path


def finite_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


class Job:
    def __init__(self, kind: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "queued"
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.return_code: int | None = None
        self.log: list[str] = []
        self.process: subprocess.Popen[str] | None = None
        self.process_pid: int | None = None
        self.process_start_ticks: int | None = None
        self.thread: threading.Thread | None = None
        self.cancel_requested = False

    def append(self, line: str) -> None:
        self.log.append(line.rstrip("\n"))
        if len(self.log) > 4000:
            del self.log[:1000]

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "pid": self.process_pid if self.status == "running" else None,
            "log": "\n".join(self.log),
        }


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.shutting_down = False

    def start(self, kind: str, worker: Callable[[Job], int]) -> Job:
        with self.lock:
            if self.shutting_down:
                raise RuntimeError("The dashboard is shutting down")
            if any(job.status in {"queued", "running"} for job in self.jobs.values()):
                raise RuntimeError("Another preprocessing or training job is already running")
            job = Job(kind)
            self.jobs[job.id] = job

        def run() -> None:
            with self.lock:
                job.status = "running"
                job.started_at = time.time()
            try:
                return_code = worker(job)
                with self.lock:
                    job.return_code = return_code
                    if job.cancel_requested:
                        job.status = "cancelled"
                    else:
                        job.status = "completed" if return_code == 0 else "failed"
            except Exception as exc:  # Keep the server alive and expose errors in the job log.
                with self.lock:
                    job.append(f"[ERROR] {exc}")
                    job.append(traceback.format_exc())
                    job.return_code = 1
                    job.status = "failed"
            finally:
                with self.lock:
                    job.process = None
                    job.finished_at = time.time()

        job.thread = threading.Thread(target=run, name=f"tiny-lidar-{job.id}", daemon=True)
        job.thread.start()
        return job

    def run_command(self, job: Job, command: list[str], cwd: Path) -> int:
        job.append("[COMMAND] " + " ".join(command))
        guarded_command = [
            sys.executable,
            str(SCRIPT_DIR / "process_guard.py"),
            str(os.getpid()),
            *command,
        ]
        process = subprocess.Popen(
            guarded_command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with self.lock:
            job.process = process
            job.process_pid = process.pid
            job.process_start_ticks = process_start_ticks(process.pid)
            job.append(f"[PID] {process.pid}")
        assert process.stdout is not None
        try:
            for line in process.stdout:
                with self.lock:
                    job.append(line)
        finally:
            process.stdout.close()
        return_code = process.wait()
        with self.lock:
            job.process = None
            job.append(f"[EXIT] status={return_code}")
        return return_code

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return [job.snapshot() for job in reversed(list(self.jobs.values()))]

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return self.jobs[job_id].snapshot()

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = self.jobs[job_id]
            job.cancel_requested = True
            process = job.process
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                job.append("[INFO] Cancellation requested")
            elif job.status == "queued":
                job.status = "cancelled"
            return job.snapshot()

    def shutdown(self, grace_seconds: float = 5.0) -> None:
        """Terminate, reap and join every child before the backend exits."""
        with self.lock:
            self.shutting_down = True
            active = [
                (job, job.process)
                for job in self.jobs.values()
                if job.process is not None and job.process.poll() is None
            ]
            threads = [job.thread for job in self.jobs.values() if job.thread is not None]
            for job, _ in active:
                job.cancel_requested = True
                job.append("[INFO] Dashboard shutdown: sending SIGTERM to the process group")

        for _, process in active:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        deadline = time.monotonic() + max(0.0, grace_seconds)
        for _, process in active:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        for _, process in active:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        for thread in threads:
            thread.join(timeout=2.0)


def preprocess_worker(config: DashboardConfig, payload: dict[str, Any], manager: JobManager) -> Callable[[Job], int]:
    train_ids = payload.get("train", [])
    val_ids = payload.get("val", [])
    if not isinstance(train_ids, list) or not isinstance(val_ids, list):
        raise ValueError("train and val must be arrays")
    if not train_ids and not val_ids:
        raise ValueError("Select at least one recording")
    existing_policy = str(payload.get("existing_policy", "skip"))
    if existing_policy not in {"skip", "overwrite"}:
        raise ValueError("existing_policy must be skip or overwrite")
    max_range = finite_float(payload.get("max_range", 30.0), "max_range", 0.1, 1000.0)
    max_sync_delta = finite_float(
        payload.get("max_sync_delta", 0.1), "max_sync_delta", 0.0, 10.0
    )
    dataset_version = normalize_dataset_version(payload.get("dataset_version"))
    version_root = dataset_version_root(config, dataset_version)
    selections: list[tuple[str, str, Path, Path]] = []
    for split, recording_ids in (("train", train_ids), ("val", val_ids)):
        for raw_id in recording_ids:
            recording_id = str(raw_id)
            bag = resolved_child(config.record_root, recording_id)
            if not (bag / "metadata.yaml").is_file():
                raise FileNotFoundError(f"Recording is missing metadata.yaml: {recording_id}")
            output = version_root / split / sequence_name_for_recording(recording_id)
            selections.append((split, recording_id, bag, output))

    def worker(job: Job) -> int:
        job.append(
            f"[INFO] dataset_version={dataset_version} train={len(train_ids)} "
            f"val={len(val_ids)} policy={existing_policy}"
        )
        for index, (split, recording_id, bag, output) in enumerate(selections, start=1):
            if job.cancel_requested:
                return 130
            job.append(f"[INFO] [{index}/{len(selections)}] {split}: {recording_id}")
            if output.exists() and existing_policy == "skip":
                job.append(f"[SKIP] {output}")
                continue
            command = [
                sys.executable,
                str(SCRIPT_DIR / "preprocess_bag.py"),
                "--bag",
                str(bag),
                "--output",
                str(output),
                "--max-range",
                str(max_range),
                "--max-sync-delta",
                str(max_sync_delta),
            ]
            status = manager.run_command(job, command, SCRIPT_DIR)
            if status != 0:
                return status
        job.append("[OK] All selected recordings were preprocessed")
        return 0

    return worker


def unique_training_output(checkpoint_root: Path, dataset_version: str) -> Path:
    run_root = (
        checkpoint_root
        if dataset_version == DEFAULT_DATASET_VERSION
        else checkpoint_root / "versions" / dataset_version
    )
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = run_root / base
    suffix = 1
    while candidate.exists():
        candidate = run_root / f"{base}_{suffix}"
        suffix += 1
    return candidate


def training_worker(config: DashboardConfig, payload: dict[str, Any], manager: JobManager) -> Callable[[Job], int]:
    dataset_version = normalize_dataset_version(payload.get("dataset_version"))
    version_root = dataset_version_root(config, dataset_version)
    architecture = str(payload.get("architecture", "normal"))
    if architecture not in {"normal", "small"}:
        raise ValueError("architecture must be normal or small")
    epochs = bounded_int(payload.get("epochs", 100), "epochs", 1, 10000)
    batch_size = bounded_int(payload.get("batch_size", 64), "batch_size", 1, 65536)
    workers = bounded_int(payload.get("workers", 4), "workers", 0, 64)
    patience = bounded_int(payload.get("patience", 15), "patience", 0, 10000)
    learning_rate = finite_float(
        payload.get("learning_rate", 1.0e-3), "learning_rate", 1.0e-8, 10.0
    )
    device = str(payload.get("device", "auto")).strip()
    if device not in {"auto", "cpu", "cuda"} and not (
        device.startswith("cuda:") and device[5:].isdigit()
    ):
        raise ValueError("device must be auto, cpu, cuda, or cuda:N")
    steering_only = bool(payload.get("steering_only", True))
    pretrained_id = str(payload.get("pretrained", "")).strip()
    pretrained: Path | None = None
    if pretrained_id:
        pretrained = resolved_child(config.checkpoint_root, pretrained_id)
        if not pretrained.is_file() or pretrained.suffix != ".pth":
            raise FileNotFoundError("Selected pretrained checkpoint does not exist")
    version_sequences = discover_sequences(config, dataset_version)
    if not version_sequences:
        raise RuntimeError("No preprocessed dataset is available")
    for split in ("train", "val"):
        if not any(item["split"] == split and item["valid"] for item in version_sequences):
            raise RuntimeError(f"No valid {split} sequence is available in {dataset_version}")

    output = unique_training_output(config.checkpoint_root, dataset_version)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "train.py"),
        "--train-dir",
        str(version_root / "train"),
        "--val-dir",
        str(version_root / "val"),
        "--output-dir",
        str(output),
        "--architecture",
        architecture,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--workers",
        str(workers),
        "--patience",
        str(patience),
        "--learning-rate",
        str(learning_rate),
        "--device",
        device,
        "--acceleration-weight",
        "0.0" if steering_only else "1.0",
        "--steering-weight",
        "1.0",
        "--dataset-version",
        dataset_version,
    ]
    if pretrained is not None:
        command.extend(("--pretrained", str(pretrained)))

    def worker(job: Job) -> int:
        output.mkdir(parents=True, exist_ok=False)
        job.append(f"[INFO] dataset_version={dataset_version}")
        job.append(f"[INFO] output={output}")
        job.append(f"[INFO] steering_only={steering_only} pretrained={pretrained or 'none'}")
        status = manager.run_command(job, command, SCRIPT_DIR)
        if status == 0 and (output / "best_model.pth").is_file():
            latest = config.checkpoint_root / "latest"
            if latest.is_symlink():
                latest.unlink()
            if not latest.exists():
                latest_target = output.relative_to(config.checkpoint_root)
                latest.symlink_to(latest_target, target_is_directory=True)
                job.append(f"[OK] latest -> {latest_target}")
        return status

    return worker


def sequence_detail(
    config: DashboardConfig,
    split: str,
    sequence_id: str,
    version: str = DEFAULT_DATASET_VERSION,
) -> dict[str, Any]:
    version = normalize_dataset_version(version)
    path = resolve_sequence(config, split, sequence_id, version)
    scans = np.load(path / "scans.npy", mmap_mode="r")
    steers = np.asarray(np.load(path / "steers.npy", mmap_mode="r"), dtype=np.float64)
    accelerations = np.asarray(
        np.load(path / "accelerations.npy", mmap_mode="r"), dtype=np.float64
    )
    if scans.ndim != 2 or len(scans) != len(steers) or len(scans) != len(accelerations):
        raise ValueError("Sequence arrays have incompatible shapes")
    finite_steers = steers[np.isfinite(steers)]
    if not len(finite_steers):
        finite_steers = np.zeros(1, dtype=np.float64)
    finite_accelerations = accelerations[np.isfinite(accelerations)]
    if not len(finite_accelerations):
        finite_accelerations = np.zeros(1, dtype=np.float64)
    histogram, edges = np.histogram(finite_steers, bins=31)
    summary = load_summary(path)
    return {
        "version": version,
        "split": split,
        "id": sequence_id,
        "samples": int(len(scans)),
        "scan_points": int(scans.shape[1]),
        "steering": {
            "min": float(np.min(finite_steers)),
            "max": float(np.max(finite_steers)),
            "mean": float(np.mean(finite_steers)),
            "median": float(np.median(finite_steers)),
            "straight_ratio": float(np.mean(np.abs(finite_steers) <= 0.02)),
            "left_ratio": float(np.mean(finite_steers < -0.02)),
            "right_ratio": float(np.mean(finite_steers > 0.02)),
            "histogram": histogram.astype(int).tolist(),
            "edges": edges.astype(float).tolist(),
        },
        "acceleration": {
            "min": float(np.min(finite_accelerations)),
            "max": float(np.max(finite_accelerations)),
            "mean": float(np.mean(finite_accelerations)),
        },
        "summary": summary,
    }


def sequence_frame(
    config: DashboardConfig,
    split: str,
    sequence_id: str,
    index: int,
    version: str = DEFAULT_DATASET_VERSION,
) -> dict[str, Any]:
    version = normalize_dataset_version(version)
    path = resolve_sequence(config, split, sequence_id, version)
    scans = np.load(path / "scans.npy", mmap_mode="r")
    steers = np.load(path / "steers.npy", mmap_mode="r")
    accelerations = np.load(path / "accelerations.npy", mmap_mode="r")
    if index < 0 or index >= len(scans):
        raise IndexError(f"Frame index must be between 0 and {len(scans) - 1}")
    summary = load_summary(path)
    max_range = float(summary.get("max_range", 30.0))
    values = np.nan_to_num(
        np.asarray(scans[index], dtype=np.float32),
        nan=0.0,
        posinf=max_range,
        neginf=0.0,
    )
    return {
        "version": version,
        "index": index,
        "samples": int(len(scans)),
        "ranges": values.astype(float).tolist(),
        "steering": float(np.nan_to_num(steers[index])),
        "acceleration": float(np.nan_to_num(accelerations[index])),
        "angle_min": float(summary.get("angle_min", math.radians(-135.0))),
        "angle_max": float(summary.get("angle_max", math.radians(135.0))),
        "max_range": max_range,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    config: DashboardConfig
    jobs: JobManager

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(f"[HTTP] {self.address_string()} {fmt % args}\n")

    def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("Request body must contain JSON and be smaller than 1 MB")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON request body must be an object")
        return value

    def serve_static(self, request_path: str) -> None:
        file_name = {
            "/": "index.html",
            "/index.html": "index.html",
            "/app.js": "app.js",
            "/style.css": "style.css",
        }.get(request_path)
        if file_name is None:
            self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")
            return
        path = STATIC_DIR / file_name
        if not path.is_file():
            self.send_error_json(HTTPStatus.NOT_FOUND, "Static asset is missing")
            return
        body = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        try:
            if not parsed.path.startswith("/api/"):
                self.serve_static(parsed.path)
                return
            if parsed.path == "/api/overview":
                sequences = discover_sequences(self.config)
                self.send_json(
                    {
                        "recordings": discover_recordings(self.config),
                        "sequences": sequences,
                        "versions": discover_versions(self.config, sequences),
                        "checkpoints": discover_checkpoints(self.config),
                        "jobs": self.jobs.list(),
                        "server": {"pid": os.getpid()},
                        "totals": {
                            "train_samples": sum(
                                item["samples"]
                                for item in sequences
                                if item["version"] == DEFAULT_DATASET_VERSION
                                and item["split"] == "train"
                            ),
                            "val_samples": sum(
                                item["samples"]
                                for item in sequences
                                if item["version"] == DEFAULT_DATASET_VERSION
                                and item["split"] == "val"
                            ),
                        },
                    }
                )
                return
            query = parse_qs(parsed.query)
            if parsed.path == "/api/recordings/latest":
                self.send_json({"recording": latest_recording(self.config.record_root)})
                return
            if parsed.path == "/api/sequence":
                self.send_json(
                    sequence_detail(
                        self.config,
                        query.get("split", [""])[0],
                        query.get("id", [""])[0],
                        query.get("version", [DEFAULT_DATASET_VERSION])[0],
                    )
                )
                return
            if parsed.path == "/api/frame":
                self.send_json(
                    sequence_frame(
                        self.config,
                        query.get("split", [""])[0],
                        query.get("id", [""])[0],
                        int(query.get("index", ["0"])[0]),
                        query.get("version", [DEFAULT_DATASET_VERSION])[0],
                    )
                )
                return
            if parsed.path.startswith("/api/jobs/"):
                self.send_json(self.jobs.get(parsed.path.rsplit("/", 1)[-1]))
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
        except KeyError:
            self.send_error_json(HTTPStatus.NOT_FOUND, "Job not found")
        except (ValueError, FileNotFoundError, IndexError) as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/recordings/annotate":
                payload = self.read_json()
                annotation = save_collection_annotation(
                    self.config.record_root, payload.get("recording_id"), payload
                )
                self.send_json({"annotation": annotation})
                return
            if parsed.path == "/api/jobs/preprocess":
                payload = self.read_json()
                worker = preprocess_worker(self.config, payload, self.jobs)
                job = self.jobs.start("preprocess", worker)
                self.send_json(job.snapshot(), HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/jobs/train":
                payload = self.read_json()
                worker = training_worker(self.config, payload, self.jobs)
                job = self.jobs.start("train", worker)
                self.send_json(job.snapshot(), HTTPStatus.ACCEPTED)
                return
            if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
                job_id = parsed.path.split("/")[-2]
                self.send_json(self.jobs.cancel(job_id))
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
        except KeyError:
            self.send_error_json(HTTPStatus.NOT_FOUND, "Job not found")
        except RuntimeError as exc:
            self.send_error_json(HTTPStatus.CONFLICT, str(exc))
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--record-root", type=Path, default=SCRIPT_DIR / "../../record")
    parser.add_argument("--dataset-root", type=Path, default=SCRIPT_DIR / "datasets")
    parser.add_argument("--checkpoint-root", type=Path, default=SCRIPT_DIR / "checkpoints")
    parser.add_argument("--pid-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = DashboardConfig(
        record_root=args.record_root.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        checkpoint_root=args.checkpoint_root.expanduser().resolve(),
    )
    config.dataset_root.mkdir(parents=True, exist_ok=True)
    config.checkpoint_root.mkdir(parents=True, exist_ok=True)
    pid_file = PidFile(args.pid_file) if args.pid_file is not None else None
    if pid_file is not None:
        try:
            pid_file.acquire()
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            raise SystemExit(1) from None
    jobs = JobManager()
    DashboardHandler.config = config
    DashboardHandler.jobs = jobs
    server: ThreadingHTTPServer | None = None

    def terminate(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
        print(f"[READY] TinyLiDARNet dashboard: http://localhost:{args.port}")
        print(f"[INFO] pid={os.getpid()} pid_file={args.pid_file or 'disabled'}")
        print(f"[INFO] recordings={config.record_root}")
        print(f"[INFO] datasets={config.dataset_root}")
        print(f"[INFO] checkpoints={config.checkpoint_root}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down dashboard and child jobs...")
    finally:
        if server is not None:
            server.server_close()
        jobs.shutdown()
        if pid_file is not None:
            pid_file.release()
        print("[OK] Dashboard stopped; all child processes were reaped")


if __name__ == "__main__":
    main()

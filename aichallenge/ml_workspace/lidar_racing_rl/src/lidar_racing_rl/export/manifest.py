"""Deployment manifest shared by export and the fail-closed ROS runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARCHITECTURE_VERSION = "lidar_actor_conv1d_v1"


def _validate_lowercase_hex(
    value: str,
    *,
    lengths: tuple[int, ...],
    label: str,
) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        expected = " or ".join(str(length) for length in lengths)
        raise ValueError(f"{label} must be {expected} lowercase hexadecimal characters")


def sha256_file(path: Path) -> str:
    """Stream a file checksum without loading the artifact twice into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_policy_manifest(
    *,
    model_checksum_sha256: str,
    training_config_hash: str,
    root_repository_commit: str,
    f1tenth_gym_jax_commit: str,
    beam_count: int,
    frame_stack: int,
    scan_channels: int,
    field_of_view: float,
    range_max: float,
    steering_max_abs: float,
    acceleration_min: float,
    acceleration_max: float,
    conv_channels: tuple[int, int, int] = (32, 64, 64),
    kernel_sizes: tuple[int, int, int] = (8, 4, 3),
    strides: tuple[int, int, int] = (4, 2, 1),
    hidden_dim: int = 256,
    action_dim: int = 2,
    log_std_min: float = -5.0,
    log_std_max: float = 2.0,
    export_timestamp: str | None = None,
) -> dict[str, Any]:
    """Build and validate the exact JSON contract consumed by ROS 2."""

    _validate_lowercase_hex(
        model_checksum_sha256,
        lengths=(64,),
        label="model checksum",
    )
    _validate_lowercase_hex(
        training_config_hash,
        lengths=(64,),
        label="training config hash",
    )
    _validate_lowercase_hex(
        root_repository_commit,
        lengths=(40, 64),
        label="root repository commit",
    )
    _validate_lowercase_hex(
        f1tenth_gym_jax_commit,
        lengths=(40, 64),
        label="F1TENTH Gym JAX commit",
    )
    if (beam_count, frame_stack, scan_channels, action_dim) != (360, 4, 2, 2):
        raise ValueError("lidar_actor_conv1d_v1 requires 360 beams, four frames, and 2x2 I/O")
    finite_values = (
        field_of_view,
        range_max,
        steering_max_abs,
        acceleration_min,
        acceleration_max,
        log_std_min,
        log_std_max,
    )
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("manifest numeric values must be finite")
    if field_of_view <= 0.0 or range_max <= 0.0 or steering_max_abs <= 0.0:
        raise ValueError("FOV, range_max, and steering scale must be positive")
    if acceleration_min >= acceleration_max or log_std_min >= log_std_max:
        raise ValueError("manifest numeric bounds must be ordered")
    if not (
        len(conv_channels) == len(kernel_sizes) == len(strides) == 3
        and all(value > 0 for value in (*conv_channels, *kernel_sizes, *strides))
    ):
        raise ValueError("architecture requires three positive Conv1D specifications")
    timestamp = export_timestamp or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture": {
            "conv_channels": list(conv_channels),
            "kernel_sizes": list(kernel_sizes),
            "strides": list(strides),
            "hidden_dim": hidden_dim,
            "action_dim": action_dim,
            "log_std_min": log_std_min,
            "log_std_max": log_std_max,
        },
        "beam_count": beam_count,
        "frame_stack": frame_stack,
        "scan_channels": scan_channels,
        "field_of_view": field_of_view,
        "range_normalization": {
            "type": "divide_by_range_max",
            "range_max": range_max,
            "output_min": 0.0,
            "output_max": 1.0,
        },
        "validity": {"valid": 1.0, "invalid": 0.0},
        "action_scaling": {
            "steering_max_abs": steering_max_abs,
            "acceleration_min": acceleration_min,
            "acceleration_max": acceleration_max,
        },
        "training_config_hash": training_config_hash,
        "root_repository_commit": root_repository_commit,
        "f1tenth_gym_jax_commit": f1tenth_gym_jax_commit,
        "model_checksum": {
            "algorithm": "sha256",
            "value": model_checksum_sha256,
        },
        "export_timestamp": timestamp,
    }


def write_policy_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomically write a strict, finite JSON manifest."""

    serialized = (
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "ARCHITECTURE_VERSION",
    "build_policy_manifest",
    "sha256_file",
    "write_policy_manifest",
]

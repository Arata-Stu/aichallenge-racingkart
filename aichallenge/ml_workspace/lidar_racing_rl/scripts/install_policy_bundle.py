#!/usr/bin/env python3
"""Verify and install an exported policy into the ROS 2 submission package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BUNDLE = PROJECT_ROOT / "exported"
DEFAULT_DESTINATION = (
    REPOSITORY_ROOT
    / "aichallenge"
    / "workspace"
    / "src"
    / "aichallenge_submit"
    / "lidar_racing_controller"
    / "models"
)
MODEL_FILENAME = "policy_torch.pt"
MANIFEST_FILENAME = "policy_manifest.json"
ARCHITECTURE_VERSION = "lidar_actor_conv1d_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_checksum(manifest: Mapping[str, Any]) -> str:
    raw = manifest.get("model_checksum")
    if isinstance(raw, Mapping):
        if raw.get("algorithm") != "sha256":
            raise ValueError("manifest model_checksum.algorithm must be sha256")
        raw = raw.get("value")
    if not isinstance(raw, str):
        raise ValueError("manifest model_checksum must contain a SHA-256 string")
    checksum = raw.lower().removeprefix("sha256:")
    if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
        raise ValueError("manifest model checksum is not a lowercase SHA-256")
    return checksum


def _require_mapping(manifest: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = manifest.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest {key} must be an object")
    return value


def _require_lowercase_hex(
    manifest: Mapping[str, Any],
    key: str,
    *,
    lengths: tuple[int, ...],
) -> None:
    value = manifest.get(key)
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"manifest {key} is not a traceable hexadecimal identity")


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    """Reject a bundle that the fixed ROS runtime could not activate."""

    if manifest.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(f"manifest architecture_version must be {ARCHITECTURE_VERSION}")
    expected_architecture = {
        "conv_channels": [32, 64, 64],
        "kernel_sizes": [8, 4, 3],
        "strides": [4, 2, 1],
        "hidden_dim": 256,
        "action_dim": 2,
        "log_std_min": -5.0,
        "log_std_max": 2.0,
    }
    architecture = _require_mapping(manifest, "architecture")
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(f"manifest architecture.{key} must be {expected!r}")
    for key, expected in (
        ("beam_count", 360),
        ("frame_stack", 4),
        ("scan_channels", 2),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key} must be {expected}")

    field_of_view = manifest.get("field_of_view")
    if (
        isinstance(field_of_view, bool)
        or not isinstance(field_of_view, (int, float))
        or not math.isclose(
            float(field_of_view),
            1.5 * math.pi,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError("manifest field_of_view must be 270 degrees")

    normalization = _require_mapping(manifest, "range_normalization")
    if normalization != {
        "type": "divide_by_range_max",
        "range_max": 30.0,
        "output_min": 0.0,
        "output_max": 1.0,
    }:
        raise ValueError("manifest range_normalization does not match deployment")
    if _require_mapping(manifest, "validity") != {"valid": 1.0, "invalid": 0.0}:
        raise ValueError("manifest validity encoding does not match deployment")
    action_scaling = _require_mapping(manifest, "action_scaling")
    for key, expected in (
        ("steering_max_abs", 0.64),
        ("acceleration_min", -3.2),
        ("acceleration_max", 3.2),
    ):
        actual = action_scaling.get(key)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
        ):
            raise ValueError(f"manifest action_scaling.{key} does not match deployment")

    _require_lowercase_hex(manifest, "training_config_hash", lengths=(64,))
    _require_lowercase_hex(manifest, "root_repository_commit", lengths=(40, 64))
    _require_lowercase_hex(manifest, "f1tenth_gym_jax_commit", lengths=(40, 64))
    timestamp = manifest.get("export_timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("manifest export_timestamp must be a non-empty string")


def verify_policy_bundle(bundle_directory: Path) -> tuple[Path, Path, str]:
    """Return verified model/manifest paths and the model checksum."""

    bundle = bundle_directory.resolve()
    model = bundle / MODEL_FILENAME
    manifest_path = bundle / MANIFEST_FILENAME
    if not model.is_file():
        raise FileNotFoundError(f"exported PyTorch policy is missing: {model}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"exported policy manifest is missing: {manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("policy manifest root must be an object")
    _validate_manifest_contract(raw)
    expected = _manifest_checksum(raw)
    actual = _sha256(model)
    if actual != expected:
        raise ValueError(
            f"exported policy checksum mismatch: expected {expected}, actual {actual}"
        )
    return model, manifest_path, actual


def _copy_to_temporary(source: Path, destination_directory: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{source.name}.",
        dir=destination_directory,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def install_policy_bundle(
    bundle_directory: Path,
    destination_directory: Path,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Verify, copy, and publish model first and manifest last."""

    model, manifest, checksum = verify_policy_bundle(bundle_directory)
    destination = destination_directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model_destination = destination / MODEL_FILENAME
    manifest_destination = destination / MANIFEST_FILENAME
    existing = [path for path in (model_destination, manifest_destination) if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"destination already contains {names}; pass --force to replace a reviewed policy"
        )

    temporary_model = _copy_to_temporary(model, destination)
    temporary_manifest = _copy_to_temporary(manifest, destination)
    try:
        if _sha256(temporary_model) != checksum:
            raise OSError("copied policy checksum changed before publication")
        # Publish the manifest last so the ROS loader never sees a new manifest
        # referring to an incompletely copied model.
        os.replace(temporary_model, model_destination)
        os.replace(temporary_manifest, manifest_destination)
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)

    return {
        "model": str(model_destination),
        "manifest": str(manifest_destination),
        "model_sha256": checksum,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and install an exported policy into lidar_racing_controller.",
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = install_policy_bundle(
            args.bundle,
            args.destination,
            force=args.force,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print(f"ERROR: policy installation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

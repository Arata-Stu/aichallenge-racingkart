"""Standard-library contract tests for policy bundle installation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts.install_policy_bundle import install_policy_bundle


def _bundle(path: Path, payload: bytes = b"weights") -> Path:
    path.mkdir()
    (path / "policy_torch.pt").write_bytes(payload)
    checksum = hashlib.sha256(payload).hexdigest()
    (path / "policy_manifest.json").write_text(
        json.dumps(
            {
                "architecture_version": "lidar_actor_conv1d_v1",
                "architecture": {
                    "conv_channels": [32, 64, 64],
                    "kernel_sizes": [8, 4, 3],
                    "strides": [4, 2, 1],
                    "hidden_dim": 256,
                    "action_dim": 2,
                    "log_std_min": -5.0,
                    "log_std_max": 2.0,
                },
                "beam_count": 360,
                "frame_stack": 4,
                "scan_channels": 2,
                "field_of_view": 1.5 * math.pi,
                "range_normalization": {
                    "type": "divide_by_range_max",
                    "range_max": 30.0,
                    "output_min": 0.0,
                    "output_max": 1.0,
                },
                "validity": {"valid": 1.0, "invalid": 0.0},
                "action_scaling": {
                    "steering_max_abs": 0.64,
                    "acceleration_min": -3.2,
                    "acceleration_max": 3.2,
                },
                "training_config_hash": "a" * 64,
                "root_repository_commit": "b" * 40,
                "f1tenth_gym_jax_commit": "c" * 40,
                "model_checksum": {
                    "algorithm": "sha256",
                    "value": checksum,
                },
                "export_timestamp": "2026-08-28T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_installs_verified_model_and_manifest(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    destination = tmp_path / "models"

    result = install_policy_bundle(bundle, destination)

    assert Path(result["model"]).read_bytes() == b"weights"
    assert Path(result["manifest"]).is_file()


def test_rejects_checksum_mismatch(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    (bundle / "policy_torch.pt").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        install_policy_bundle(bundle, tmp_path / "models")


def test_requires_force_to_replace_existing_policy(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    destination = tmp_path / "models"
    install_policy_bundle(bundle, destination)

    with pytest.raises(FileExistsError, match="--force"):
        install_policy_bundle(bundle, destination)

    install_policy_bundle(bundle, destination, force=True)


def test_rejects_manifest_that_cannot_activate_in_ros(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    manifest_path = bundle / "policy_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_stack"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="frame_stack"):
        install_policy_bundle(bundle, tmp_path / "models")

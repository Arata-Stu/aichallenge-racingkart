"""Deployment manifest schema tests independent of JAX and PyTorch."""

from __future__ import annotations

import json
import math

import pytest

from lidar_racing_rl.export.manifest import build_policy_manifest, write_policy_manifest


def _manifest() -> dict[str, object]:
    return build_policy_manifest(
        model_checksum_sha256="a" * 64,
        training_config_hash="b" * 64,
        root_repository_commit="c" * 40,
        f1tenth_gym_jax_commit="d" * 40,
        beam_count=360,
        frame_stack=4,
        scan_channels=2,
        field_of_view=1.5 * math.pi,
        range_max=30.0,
        steering_max_abs=0.64,
        acceleration_min=-3.2,
        acceleration_max=3.2,
        export_timestamp="2026-08-27T00:00:00Z",
    )


def test_manifest_matches_ros_runtime_contract(tmp_path) -> None:
    manifest = _manifest()
    output = tmp_path / "policy_manifest.json"
    write_policy_manifest(output, manifest)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded["architecture_version"] == "lidar_actor_conv1d_v1"
    assert loaded["beam_count"] == 360
    assert loaded["frame_stack"] == 4
    assert loaded["range_normalization"]["range_max"] == 30.0
    assert loaded["validity"] == {"invalid": 0.0, "valid": 1.0}


def test_manifest_rejects_wrong_observation_contract() -> None:
    with pytest.raises(ValueError, match="requires 360"):
        build_policy_manifest(
            model_checksum_sha256="a" * 64,
            training_config_hash="b" * 64,
            root_repository_commit="c" * 40,
            f1tenth_gym_jax_commit="d" * 40,
            beam_count=1080,
            frame_stack=4,
            scan_channels=2,
            field_of_view=1.5 * math.pi,
            range_max=30.0,
            steering_max_abs=1.0,
            acceleration_min=-3.2,
            acceleration_max=3.2,
        )


def test_manifest_rejects_untraceable_repository_identity() -> None:
    with pytest.raises(ValueError, match="root repository commit"):
        build_policy_manifest(
            model_checksum_sha256="a" * 64,
            training_config_hash="b" * 64,
            root_repository_commit="not-a-git-object-id",
            f1tenth_gym_jax_commit="d" * 40,
            beam_count=360,
            frame_stack=4,
            scan_channels=2,
            field_of_view=1.5 * math.pi,
            range_max=30.0,
            steering_max_abs=1.0,
            acceleration_min=-3.2,
            acceleration_max=3.2,
        )

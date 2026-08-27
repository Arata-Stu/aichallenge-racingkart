import hashlib
import json
import math

import numpy as np
import pytest
import torch

from lidar_racing_controller.policy import (
    ActorArchitecture,
    LidarActor,
    PolicyLoadError,
    PolicyManifest,
    PolicyRuntime,
)


def _manifest(checksum: str) -> dict[str, object]:
    return {
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
        "field_of_view": 3.0 * math.pi / 2.0,
        "range_normalization": {
            "type": "divide_by_range_max",
            "range_max": 30.0,
            "output_min": 0.0,
            "output_max": 1.0,
        },
        "validity": {"valid": 1.0, "invalid": 0.0},
        "action_scaling": {
            "steering_max_abs": 1.0,
            "acceleration_min": -3.2,
            "acceleration_max": 3.2,
        },
        "training_config_hash": "a" * 64,
        "root_repository_commit": "b" * 40,
        "f1tenth_gym_jax_commit": "c" * 40,
        "model_checksum": {"algorithm": "sha256", "value": checksum},
        "export_timestamp": "2026-08-27T00:00:00Z",
    }


def test_verified_state_dict_loads_and_produces_finite_action(tmp_path) -> None:
    model_path = tmp_path / "policy_torch.pt"
    manifest_path = tmp_path / "policy_manifest.json"
    torch.save(LidarActor(ActorArchitecture()).state_dict(), model_path)
    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(_manifest(checksum)), encoding="utf-8")

    runtime = PolicyRuntime.load(
        model_path=model_path,
        manifest_path=manifest_path,
        device="cpu",
        expected_beam_count=360,
        expected_frame_stack=4,
        expected_scan_channels=2,
        expected_range_max=30.0,
        expected_angle_min=-3.0 * math.pi / 4.0,
        expected_angle_max=3.0 * math.pi / 4.0,
        expected_steering_max_abs=1.0,
        expected_acceleration_min=-3.2,
        expected_acceleration_max=3.2,
    )
    action = runtime.predict(np.zeros((8, 360), dtype=np.float32))

    assert action.shape == (2,)
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0)


def test_checksum_mismatch_fails_before_model_activation(tmp_path) -> None:
    model_path = tmp_path / "policy_torch.pt"
    manifest_path = tmp_path / "policy_manifest.json"
    torch.save(LidarActor(ActorArchitecture()).state_dict(), model_path)
    manifest_path.write_text(json.dumps(_manifest("0" * 64)), encoding="utf-8")

    with pytest.raises(PolicyLoadError, match="checksum"):
        PolicyRuntime.load(
            model_path=model_path,
            manifest_path=manifest_path,
            device="cpu",
            expected_beam_count=360,
            expected_frame_stack=4,
            expected_scan_channels=2,
            expected_range_max=30.0,
            expected_angle_min=-3.0 * math.pi / 4.0,
            expected_angle_max=3.0 * math.pi / 4.0,
            expected_steering_max_abs=1.0,
            expected_acceleration_min=-3.2,
            expected_acceleration_max=3.2,
        )


def test_range_max_mismatch_prevents_model_activation(tmp_path) -> None:
    model_path = tmp_path / "policy_torch.pt"
    manifest_path = tmp_path / "policy_manifest.json"
    torch.save(LidarActor(ActorArchitecture()).state_dict(), model_path)
    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = _manifest(checksum)
    manifest["range_normalization"]["range_max"] = 20.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PolicyLoadError, match="range_max"):
        PolicyRuntime.load(
            model_path=model_path,
            manifest_path=manifest_path,
            device="cpu",
            expected_beam_count=360,
            expected_frame_stack=4,
            expected_scan_channels=2,
            expected_range_max=30.0,
            expected_angle_min=-3.0 * math.pi / 4.0,
            expected_angle_max=3.0 * math.pi / 4.0,
            expected_steering_max_abs=1.0,
            expected_acceleration_min=-3.2,
            expected_acceleration_max=3.2,
        )


def test_missing_range_max_is_rejected_while_loading_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "policy_manifest.json"
    manifest = _manifest("0" * 64)
    del manifest["range_normalization"]["range_max"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PolicyLoadError, match="range_max"):
        PolicyManifest.load(manifest_path)


def test_field_of_view_mismatch_prevents_model_activation(tmp_path) -> None:
    model_path = tmp_path / "policy_torch.pt"
    manifest_path = tmp_path / "policy_manifest.json"
    torch.save(LidarActor(ActorArchitecture()).state_dict(), model_path)
    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = _manifest(checksum)
    manifest["field_of_view"] = math.pi
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PolicyLoadError, match="field_of_view"):
        PolicyRuntime.load(
            model_path=model_path,
            manifest_path=manifest_path,
            device="cpu",
            expected_beam_count=360,
            expected_frame_stack=4,
            expected_scan_channels=2,
            expected_range_max=30.0,
            expected_angle_min=-3.0 * math.pi / 4.0,
            expected_angle_max=3.0 * math.pi / 4.0,
            expected_steering_max_abs=1.0,
            expected_acceleration_min=-3.2,
            expected_acceleration_max=3.2,
        )


def test_frame_stack_mismatch_prevents_model_activation(tmp_path) -> None:
    model_path = tmp_path / "policy_torch.pt"
    manifest_path = tmp_path / "policy_manifest.json"
    torch.save(LidarActor(ActorArchitecture()).state_dict(), model_path)
    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = _manifest(checksum)
    manifest["frame_stack"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PolicyLoadError, match="observation shape"):
        PolicyRuntime.load(
            model_path=model_path,
            manifest_path=manifest_path,
            device="cpu",
            expected_beam_count=360,
            expected_frame_stack=4,
            expected_scan_channels=2,
            expected_range_max=30.0,
            expected_angle_min=-3.0 * math.pi / 4.0,
            expected_angle_max=3.0 * math.pi / 4.0,
            expected_steering_max_abs=1.0,
            expected_acceleration_min=-3.2,
            expected_acceleration_max=3.2,
        )


def test_v1_manifest_rejects_altered_network_architecture(tmp_path) -> None:
    manifest_path = tmp_path / "policy_manifest.json"
    manifest = _manifest("0" * 64)
    manifest["architecture"]["hidden_dim"] = 128
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PolicyLoadError, match="lidar_actor_conv1d_v1"):
        PolicyManifest.load(manifest_path)

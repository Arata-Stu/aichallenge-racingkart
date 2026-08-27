from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from rsu_fusion_net_pytorch.model import (
    DistanceGatedRsuFusionNet, RsuBezierTrajectoryFusionNet, RsuTrajectoryFusionNet,
)
from rsu_fusion_net_pytorch.policy import RsuFusionTorchPolicy


def test_checkpoint_round_trip(tmp_path: Path):
    config = {
        "data": {"history_len": 3, "max_range": 45.0, "acceleration_scale": 2.0},
        "loss": {"acceleration_weight": 1.0, "steering_weight": 1.0},
        "model": {
            "scan_dim": 256, "rsu_count": 2, "rsu_meta_dim": 5,
            "vehicle_state_dim": 0, "output_dim": 2,
            "frame_feature_dim": 8, "temporal_hidden_dim": 8,
            "fusion_hidden_dim": 8, "distance_decay_m": 35.0, "top_k_rsus": 1,
        },
    }
    model = DistanceGatedRsuFusionNet(**config["model"])
    path = tmp_path / "best_model.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
    policy = RsuFusionTorchPolicy(path, device="cpu")
    assert policy.learns_acceleration
    scan = policy.preprocess_scan([float("inf")] * 32)
    ego = [scan] * 3
    rsu = [[scan, scan] for _ in range(3)]
    acceleration, steering, gates = policy.predict(
        ego, rsu, [[5.0, 0.0, 0.0, 0.0, 0.01]] * 2, [True, True]
    )
    assert -2.0 <= acceleration <= 2.0
    assert -1.0 <= steering <= 1.0
    assert len(gates) == 2


def test_trajectory_checkpoint_returns_candidates(tmp_path: Path):
    config = {
        "data": {
            "history_len": 2, "max_range": 45.0, "acceleration_scale": 2.0,
            "trajectory_steps": 3, "trajectory_dt": 0.25,
            "trajectory_distance_scale": 50.0, "max_speed": 15.0,
        },
        "loss": {"acceleration_weight": 1.0, "steering_weight": 1.0},
        "model": {
            "architecture": "trajectory_multimodal", "scan_dim": 256,
            "rsu_count": 2, "rsu_meta_dim": 5, "vehicle_state_dim": 1,
            "output_dim": 2, "frame_feature_dim": 8, "temporal_hidden_dim": 8,
            "fusion_hidden_dim": 8, "distance_decay_m": 35.0, "top_k_rsus": 1,
            "trajectory_modes": 4, "trajectory_dim": 3,
        },
    }
    model_args = dict(config["model"])
    model_args.pop("architecture")
    model_args["trajectory_steps"] = config["data"]["trajectory_steps"]
    model = RsuTrajectoryFusionNet(**model_args)
    path = tmp_path / "trajectory.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
    policy = RsuFusionTorchPolicy(path, device="cpu")
    scan = policy.preprocess_scan([float("inf")] * 32)
    prediction = policy.predict_full(
        [scan] * 2, [[scan, scan], [scan, scan]],
        [[5.0, 0.0, 0.0, 0.0, 0.01]] * 2, [True, True], ego_speed=3.0,
    )
    assert prediction.selected_mode in range(4)
    assert len(prediction.trajectories) == 4
    assert len(prediction.trajectories[0]) == 3
    assert len(prediction.mode_probabilities) == 4


def test_bezier_v2_checkpoint_returns_smooth_candidates(tmp_path: Path):
    config = {
        "data": {
            "history_len": 2, "max_range": 45.0, "acceleration_scale": 2.0,
            "trajectory_steps": 12, "trajectory_dt": 0.25,
            "trajectory_distance_scale": 50.0, "max_speed": 15.0,
        },
        "loss": {"acceleration_weight": 1.0, "steering_weight": 1.0},
        "model": {
            "architecture": "trajectory_bezier_v2", "scan_dim": 256,
            "rsu_count": 2, "rsu_meta_dim": 5, "vehicle_state_dim": 1,
            "output_dim": 2, "frame_feature_dim": 8, "temporal_hidden_dim": 8,
            "fusion_hidden_dim": 8, "distance_decay_m": 35.0, "top_k_rsus": 1,
            "trajectory_modes": 4, "trajectory_dim": 3, "trajectory_anchor_count": 4,
            "max_anchor_step_normalized": 0.24, "max_anchor_heading_delta": 1.2,
        },
    }
    args = dict(config["model"]); args.pop("architecture"); args["trajectory_steps"] = 12
    path = tmp_path / "bezier.pth"
    torch.save({"model_state_dict": RsuBezierTrajectoryFusionNet(**args).state_dict(), "config": config}, path)
    policy = RsuFusionTorchPolicy(path, device="cpu")
    scan = policy.preprocess_scan([float("inf")] * 32)
    result = policy.predict_full(
        [scan] * 2, [[scan, scan], [scan, scan]],
        [[5.0, 0.0, 0.0, 0.0, 0.01]] * 2, [True, True], ego_speed=3.0,
    )
    assert len(result.trajectories) == 4
    assert len(result.trajectories[0]) == 12
    assert result.selected_mode in range(4)

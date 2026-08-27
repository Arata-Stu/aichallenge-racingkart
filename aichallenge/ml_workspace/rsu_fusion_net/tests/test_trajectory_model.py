from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.data import RsuTrajectorySequenceDataset  # noqa: E402
from lib.model import BevBezierTrajectoryNet, RsuBezierTrajectoryFusionNet, RsuTrajectoryFusionNet  # noqa: E402
from train import MultiTaskTrajectoryLoss  # noqa: E402
from evaluate import dataset_from_config, evaluate_sequence, load_checkpoint  # noqa: E402


def test_trajectory_model_forward_and_backward() -> None:
    model = RsuTrajectoryFusionNet(
        scan_dim=256, rsu_count=2, rsu_meta_dim=5, vehicle_state_dim=1,
        frame_feature_dim=8, temporal_hidden_dim=8, fusion_hidden_dim=16,
        trajectory_modes=4, trajectory_steps=5,
    )
    output = model(
        torch.rand(3, 2, 256), torch.rand(3, 2, 2, 256),
        torch.zeros(3, 2, 5), vehicle_state=torch.rand(3, 1),
        rsu_mask=torch.ones(3, 2, dtype=torch.bool),
    )
    assert output.control.shape == (3, 2)
    assert output.trajectories.shape == (3, 4, 5, 3)
    assert output.mode_logits.shape == (3, 4)
    assert torch.all((0.0 <= output.trajectories[..., 2]) & (output.trajectories[..., 2] <= 1.0))
    criterion = MultiTaskTrajectoryLoss(SimpleNamespace(
        acceleration_weight=1.0, steering_weight=1.0, control_weight=1.0,
        trajectory_weight=2.0, speed_weight=1.0, mode_weight=0.2,
        average_displacement_weight=1.0, endpoint_weight=1.5,
        smoothness_weight=0.05, diversity_weight=0.05, diversity_margin=0.08,
    ))
    loss, metrics = criterion(output, torch.rand(3, 2), torch.rand(3, 5, 3))
    loss.backward()
    assert set(metrics) == {
        "total", "control", "trajectory", "ade", "fde", "mode", "smoothness", "diversity"
    }


def test_bezier_model_outputs_ordered_smooth_trajectory() -> None:
    model = RsuBezierTrajectoryFusionNet(
        scan_dim=256, rsu_count=2, rsu_meta_dim=5, vehicle_state_dim=1,
        frame_feature_dim=8, temporal_hidden_dim=8, fusion_hidden_dim=16,
        trajectory_modes=3, trajectory_steps=12, trajectory_anchor_count=4,
        max_anchor_step_normalized=0.24,
    )
    output = model(
        torch.rand(2, 2, 256), torch.rand(2, 2, 2, 256),
        torch.zeros(2, 2, 5), vehicle_state=torch.full((2, 1), 0.2),
        rsu_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    assert output.trajectories.shape == (2, 3, 12, 3)
    assert torch.isfinite(output.trajectories).all()
    assert torch.all((0.0 <= output.trajectories[..., 2]) & (output.trajectories[..., 2] <= 1.0))
    point_steps = torch.linalg.vector_norm(
        output.trajectories[..., 1:, :2] - output.trajectories[..., :-1, :2], dim=-1
    )
    assert float(point_steps.max()) < 0.24
    (output.trajectories.mean() + output.control.mean() + output.mode_logits.mean()).backward()
    assert model.anchor_head.weight.grad is not None


def test_bev_model_forward_and_backward() -> None:
    model = BevBezierTrajectoryNet(
        bev_channels=6, vehicle_state_dim=1, frame_feature_dim=16,
        temporal_hidden_dim=16, fusion_hidden_dim=24, trajectory_modes=3,
        trajectory_steps=8, trajectory_anchor_count=4,
    )
    output = model(torch.rand(2, 3, 6, 32, 40), vehicle_state=torch.rand(2, 1))
    assert output.control.shape == (2, 2)
    assert output.trajectories.shape == (2, 3, 8, 3)
    assert output.gates.shape == (2, 0)
    (output.control.mean() + output.trajectories.mean() + output.mode_logits.mean()).backward()
    assert model.frame_encoder[0].weight.grad is not None


def test_trajectory_loss_reports_average_and_endpoint_error() -> None:
    prediction = SimpleNamespace(
        trajectories=torch.zeros(1, 1, 3, 3),
        control=torch.zeros(1, 2), mode_logits=torch.zeros(1, 1),
    )
    target_trajectory = torch.tensor([[[0.02, 0.0, 0.0], [0.04, 0.0, 0.0], [0.09, 0.0, 0.0]]])
    criterion = MultiTaskTrajectoryLoss(SimpleNamespace(
        acceleration_weight=1.0, steering_weight=1.0, control_weight=0.0,
        trajectory_weight=1.0, average_displacement_weight=1.0, endpoint_weight=2.0,
        speed_weight=0.0, mode_weight=0.0, smoothness_weight=0.0,
        diversity_weight=0.0, diversity_margin=0.08,
    ))
    _, metrics = criterion(prediction, torch.zeros(1, 2), target_trajectory)
    assert metrics["ade"].item() == pytest.approx(0.05)
    assert metrics["fde"].item() == pytest.approx(0.09)
    assert metrics["trajectory"].item() == pytest.approx(0.23)


def test_trajectory_dataset_has_no_samples_without_full_horizon(tmp_path: Path) -> None:
    count = 6
    np.save(tmp_path / "ego_scans.npy", np.ones((count, 256), dtype=np.float32))
    np.save(tmp_path / "rsu_scans.npy", np.ones((count, 2, 256), dtype=np.float32))
    np.save(tmp_path / "rsu_meta.npy", np.zeros((count, 2, 5), dtype=np.float32))
    np.save(tmp_path / "targets.npy", np.zeros((count, 2), dtype=np.float32))
    np.save(tmp_path / "ego_poses.npy", np.zeros((count, 3), dtype=np.float64))
    np.save(tmp_path / "timestamps_ns.npy", np.arange(count, dtype=np.int64) * 100_000_000)
    np.save(tmp_path / "vehicle_state.npy", np.zeros((count, 1), dtype=np.float32))
    dataset = RsuTrajectorySequenceDataset(
        tmp_path, history_len=2, trajectory_steps=5, trajectory_dt=0.1,
    )
    assert len(dataset) == 0


def test_offline_evaluation_exports_metric_arrays(tmp_path: Path) -> None:
    count = 10
    sequence = tmp_path / "sequence"
    sequence.mkdir()
    np.save(sequence / "ego_scans.npy", np.ones((count, 256), dtype=np.float32))
    np.save(sequence / "rsu_scans.npy", np.ones((count, 2, 256), dtype=np.float32))
    np.save(sequence / "rsu_meta.npy", np.zeros((count, 2, 5), dtype=np.float32))
    np.save(sequence / "rsu_mask.npy", np.ones((count, 2), dtype=np.bool_))
    np.save(sequence / "targets.npy", np.zeros((count, 2), dtype=np.float32))
    np.save(sequence / "ego_poses.npy", np.column_stack((np.arange(count), np.zeros(count), np.zeros(count))))
    np.save(sequence / "timestamps_ns.npy", np.arange(count, dtype=np.int64) * 100_000_000)
    np.save(sequence / "vehicle_state.npy", np.ones((count, 1), dtype=np.float32))
    config = {
        "data": {"history_len": 2, "max_range": 45.0, "acceleration_scale": 2.0,
                 "trajectory_steps": 2, "trajectory_dt": 0.1,
                 "trajectory_distance_scale": 50.0, "max_speed": 15.0},
        "model": {"architecture": "trajectory_multimodal", "scan_dim": 256,
                  "rsu_count": 2, "rsu_meta_dim": 5, "vehicle_state_dim": 1,
                  "output_dim": 2, "frame_feature_dim": 8, "temporal_hidden_dim": 8,
                  "fusion_hidden_dim": 16, "distance_decay_m": 35.0, "top_k_rsus": 1,
                  "trajectory_modes": 3, "trajectory_dim": 3},
    }
    model_args = dict(config["model"])
    model_args.pop("architecture")
    model_args["trajectory_steps"] = 2
    checkpoint = tmp_path / "model.pth"
    torch.save({"model_state_dict": RsuTrajectoryFusionNet(**model_args).state_dict(), "config": config}, checkpoint)
    loaded_model, loaded_config = load_checkpoint(checkpoint, torch.device("cpu"))
    dataset = dataset_from_config(sequence, loaded_config)
    metrics, arrays = evaluate_sequence(loaded_model, dataset, torch.device("cpu"), 4)
    assert metrics["samples"] == len(dataset)
    assert arrays["trajectories"].shape[1:] == (3, 2, 3)
    assert np.isfinite(metrics["ade_m"])


def test_offline_evaluation_loads_bezier_v2_checkpoint(tmp_path: Path) -> None:
    config = {
        "data": {"trajectory_steps": 12, "trajectory_dt": 0.25},
        "model": {
            "architecture": "trajectory_bezier_v2", "scan_dim": 256,
            "rsu_count": 2, "rsu_meta_dim": 5, "vehicle_state_dim": 1,
            "output_dim": 2, "frame_feature_dim": 8, "temporal_hidden_dim": 8,
            "fusion_hidden_dim": 16, "distance_decay_m": 35.0, "top_k_rsus": 1,
            "trajectory_modes": 3, "trajectory_dim": 3, "trajectory_anchor_count": 4,
            "max_anchor_step_normalized": 0.24, "max_anchor_heading_delta": 1.2,
        },
    }
    args = dict(config["model"]); args.pop("architecture"); args["trajectory_steps"] = 12
    checkpoint = tmp_path / "bezier.pth"
    torch.save({"model_state_dict": RsuBezierTrajectoryFusionNet(**args).state_dict(), "config": config}, checkpoint)
    model, loaded = load_checkpoint(checkpoint, torch.device("cpu"))
    assert isinstance(model, RsuBezierTrajectoryFusionNet)
    assert loaded["model"]["trajectory_anchor_count"] == 4


def test_offline_evaluation_supports_bev_checkpoint(tmp_path: Path) -> None:
    count = 8
    sequence = tmp_path / "sequence"
    sequence.mkdir()
    np.save(sequence / "bev_frames.npy", np.zeros((count, 32, 40), dtype=np.uint8))
    np.save(sequence / "targets.npy", np.zeros((count, 2), dtype=np.float32))
    np.save(
        sequence / "ego_poses.npy",
        np.column_stack((np.arange(count), np.zeros(count), np.zeros(count))),
    )
    np.save(sequence / "timestamps_ns.npy", np.arange(count, dtype=np.int64) * 100_000_000)
    np.save(sequence / "vehicle_state.npy", np.ones((count, 1), dtype=np.float32))
    config = {
        "data": {
            "history_len": 2, "acceleration_scale": 2.0, "trajectory_steps": 2,
            "trajectory_dt": 0.1, "trajectory_distance_scale": 50.0,
            "max_speed": 15.0, "bev_channels": [0, 1, 2, 3, 4, 5],
            "bev_height": 32, "bev_width": 40,
        },
        "model": {
            "architecture": "bev_trajectory_bezier_v1", "vehicle_state_dim": 1,
            "output_dim": 2, "frame_feature_dim": 8, "temporal_hidden_dim": 8,
            "fusion_hidden_dim": 16, "trajectory_modes": 2, "trajectory_dim": 3,
            "trajectory_anchor_count": 2, "max_anchor_step_normalized": 0.1,
            "max_anchor_heading_delta": 1.2,
        },
    }
    model = BevBezierTrajectoryNet(
        bev_channels=6, vehicle_state_dim=1, frame_feature_dim=8, temporal_hidden_dim=8,
        fusion_hidden_dim=16, trajectory_modes=2, trajectory_steps=2,
        trajectory_anchor_count=2, max_anchor_step_normalized=0.1,
    )
    checkpoint = tmp_path / "bev.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint)
    loaded_model, loaded_config = load_checkpoint(checkpoint, torch.device("cpu"))
    dataset = dataset_from_config(sequence, loaded_config)
    metrics, arrays = evaluate_sequence(loaded_model, dataset, torch.device("cpu"), 2)
    assert isinstance(loaded_model, BevBezierTrajectoryNet)
    assert metrics["samples"] == len(dataset)
    assert arrays["gates"].shape == (len(dataset), 0)

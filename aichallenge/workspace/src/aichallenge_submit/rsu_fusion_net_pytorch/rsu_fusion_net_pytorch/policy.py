"""Checkpoint loading, scan preprocessing and framework-native inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from rsu_fusion_net_pytorch.model import (
    DistanceGatedRsuFusionNet, RsuBezierTrajectoryFusionNet, RsuFusionOutput, RsuTrajectoryFusionNet,
    RsuTrajectoryOutput,
)


@dataclass(frozen=True)
class RsuPolicyPrediction:
    acceleration: float
    steering: float
    gates: list[float]
    trajectories: list[list[list[float]]]
    mode_probabilities: list[float]
    selected_mode: int


def resolve_device(requested: str) -> torch.device:
    name = requested.strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({requested!r}), but CUDA is unavailable")
    return device


def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class RsuFusionTorchPolicy:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"RSU fusion checkpoint not found: {self.checkpoint_path}")
        self.device = resolve_device(device)
        checkpoint = torch_load(self.checkpoint_path, self.device)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint must contain model_state_dict and its training config")
        config = checkpoint.get("config", {})
        model_config = config.get("model", {}) if isinstance(config, dict) else {}
        data_config = config.get("data", {}) if isinstance(config, dict) else {}
        loss_config = config.get("loss", {}) if isinstance(config, dict) else {}
        self.learns_acceleration = float(loss_config.get("acceleration_weight", 0.0)) > 0.0
        self.architecture = str(model_config.get("architecture", "control"))
        self.scan_dim = int(model_config.get("scan_dim", 1080))
        self.rsu_count = int(model_config.get("rsu_count", 6))
        self.rsu_meta_dim = int(model_config.get("rsu_meta_dim", 5))
        self.output_dim = int(model_config.get("output_dim", 2))
        self.history_len = int(data_config.get("history_len", 5))
        self.max_range = float(data_config.get("max_range", 45.0))
        self.acceleration_scale = float(data_config.get("acceleration_scale", 1.0))
        self.trajectory_distance_scale = float(data_config.get("trajectory_distance_scale", 50.0))
        self.max_speed = float(data_config.get("max_speed", 15.0))
        self.trajectory_dt = float(data_config.get("trajectory_dt", 0.25))
        self.vehicle_state_dim = int(model_config.get("vehicle_state_dim", 0))
        if self.history_len < 1 or self.max_range <= 0.0 or self.acceleration_scale <= 0.0:
            raise ValueError("Invalid history_len, max_range, or acceleration_scale in checkpoint")
        common_model_args = dict(
            scan_dim=self.scan_dim,
            rsu_count=self.rsu_count,
            rsu_meta_dim=self.rsu_meta_dim,
            vehicle_state_dim=self.vehicle_state_dim,
            output_dim=self.output_dim,
            frame_feature_dim=int(model_config.get("frame_feature_dim", 128)),
            temporal_hidden_dim=int(model_config.get("temporal_hidden_dim", 128)),
            fusion_hidden_dim=int(model_config.get("fusion_hidden_dim", 128)),
            distance_decay_m=float(model_config.get("distance_decay_m", 35.0)),
            top_k_rsus=int(model_config.get("top_k_rsus", 2)),
        )
        if self.architecture in {"trajectory_multimodal", "trajectory_bezier_v2"}:
            model_class = (
                RsuBezierTrajectoryFusionNet
                if self.architecture == "trajectory_bezier_v2"
                else RsuTrajectoryFusionNet
            )
            trajectory_args = dict(
                **common_model_args,
                trajectory_modes=int(model_config.get("trajectory_modes", 4)),
                trajectory_steps=int(data_config.get("trajectory_steps", 20)),
                trajectory_dim=int(model_config.get("trajectory_dim", 3)),
            )
            if model_class is RsuBezierTrajectoryFusionNet:
                trajectory_args.update(
                    trajectory_anchor_count=int(model_config.get("trajectory_anchor_count", 4)),
                    max_anchor_step_normalized=float(model_config.get("max_anchor_step_normalized", 0.24)),
                    max_anchor_heading_delta=float(model_config.get("max_anchor_heading_delta", 1.2)),
                )
            self.model = model_class(**trajectory_args).to(self.device)
        else:
            self.model = DistanceGatedRsuFusionNet(**common_model_args).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

    def preprocess_scan(self, ranges: Sequence[float] | torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(ranges, dtype=torch.float32).detach().cpu().clone().flatten()
        if values.numel() == 0:
            raise ValueError("LaserScan ranges must not be empty")
        values = torch.nan_to_num(
            values, nan=self.max_range, posinf=self.max_range, neginf=self.max_range
        ).clamp_(0.0, self.max_range)
        if values.numel() != self.scan_dim:
            values = F.interpolate(
                values.reshape(1, 1, -1), size=self.scan_dim, mode="linear", align_corners=True
            ).reshape(-1)
        return values / self.max_range

    def predict(
        self,
        ego_history: Sequence[torch.Tensor],
        rsu_history: Sequence[Sequence[torch.Tensor]],
        rsu_meta: Sequence[Sequence[float]],
        rsu_mask: Sequence[bool],
        ego_speed: float = 0.0,
    ) -> tuple[float, float, list[float]]:
        result = self.predict_full(ego_history, rsu_history, rsu_meta, rsu_mask, ego_speed)
        return result.acceleration, result.steering, result.gates

    def predict_full(
        self,
        ego_history: Sequence[torch.Tensor],
        rsu_history: Sequence[Sequence[torch.Tensor]],
        rsu_meta: Sequence[Sequence[float]],
        rsu_mask: Sequence[bool],
        ego_speed: float = 0.0,
    ) -> RsuPolicyPrediction:
        if len(ego_history) != self.history_len or len(rsu_history) != self.history_len:
            raise ValueError(f"Expected exactly {self.history_len} history frames")
        ego = torch.stack(list(ego_history)).unsqueeze(0).to(self.device)
        rsu = torch.stack([torch.stack(list(frame)) for frame in rsu_history])
        rsu = rsu.permute(1, 0, 2).unsqueeze(0).to(self.device)
        meta = torch.as_tensor(rsu_meta, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(rsu_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        vehicle_state = None
        if self.vehicle_state_dim > 0:
            normalized_speed = max(0.0, min(float(ego_speed) / self.max_speed, 1.0))
            vehicle_state = torch.zeros((1, self.vehicle_state_dim), dtype=torch.float32, device=self.device)
            vehicle_state[0, 0] = normalized_speed
        with torch.inference_mode():
            result = self.model(
                ego, rsu, meta, vehicle_state=vehicle_state, rsu_mask=mask, return_gates=True
            )
        assert isinstance(result, (RsuFusionOutput, RsuTrajectoryOutput))
        output = result.control[0].detach().cpu()
        if output.numel() < 2:
            raise RuntimeError("RSU model must output acceleration and steering")
        trajectories: list[list[list[float]]] = []
        probabilities: list[float] = []
        selected_mode = -1
        if isinstance(result, RsuTrajectoryOutput):
            trajectory = result.trajectories[0].detach().cpu().clone()
            trajectory[..., :2] *= self.trajectory_distance_scale
            trajectory[..., 2] *= self.max_speed
            trajectories = trajectory.tolist()
            probability_tensor = torch.softmax(result.mode_logits[0], dim=-1).detach().cpu()
            probabilities = probability_tensor.tolist()
            selected_mode = int(torch.argmax(probability_tensor).item())
        return RsuPolicyPrediction(
            acceleration=float(output[0]) * self.acceleration_scale,
            steering=float(output[1]), gates=result.gates[0].detach().cpu().tolist(),
            trajectories=trajectories, mode_probabilities=probabilities,
            selected_mode=selected_mode,
        )

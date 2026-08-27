"""Model definition kept compatible with ml_workspace/rsu_fusion_net."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RsuFusionOutput:
    control: torch.Tensor
    gates: torch.Tensor


@dataclass(frozen=True)
class RsuTrajectoryOutput:
    control: torch.Tensor
    trajectories: torch.Tensor
    mode_logits: torch.Tensor
    gates: torch.Tensor


class ScanFrameEncoder(nn.Module):
    def __init__(self, scan_dim: int, feature_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=10, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(24, 36, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(36, 48, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(48, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, feature_dim)
        with torch.no_grad():
            self.net(torch.zeros(1, 1, scan_dim))

    def forward(self, scan: torch.Tensor) -> torch.Tensor:
        return F.relu(self.proj(self.net(scan.unsqueeze(1)).flatten(1)), inplace=True)


class TemporalScanEncoder(nn.Module):
    def __init__(self, scan_dim: int, frame_feature_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.frame_encoder = ScanFrameEncoder(scan_dim, frame_feature_dim)
        self.gru = nn.GRU(frame_feature_dim, hidden_dim, batch_first=True)

    def forward(self, scans: torch.Tensor) -> torch.Tensor:
        batch, history, rays = scans.shape
        encoded = self.frame_encoder(scans.reshape(batch * history, rays))
        _, hidden = self.gru(encoded.reshape(batch, history, -1))
        return hidden[-1]


class DistanceGatedRsuFusionNet(nn.Module):
    def __init__(
        self,
        scan_dim: int = 1080,
        rsu_count: int = 6,
        rsu_meta_dim: int = 5,
        vehicle_state_dim: int = 0,
        output_dim: int = 2,
        frame_feature_dim: int = 128,
        temporal_hidden_dim: int = 128,
        fusion_hidden_dim: int = 128,
        distance_decay_m: float = 35.0,
        top_k_rsus: int = 2,
    ) -> None:
        super().__init__()
        self.rsu_count = rsu_count
        self.rsu_meta_dim = rsu_meta_dim
        self.vehicle_state_dim = vehicle_state_dim
        self.distance_decay_m = distance_decay_m
        self.top_k_rsus = top_k_rsus
        self.scan_encoder = TemporalScanEncoder(scan_dim, frame_feature_dim, temporal_hidden_dim)
        gate_input_dim = temporal_hidden_dim * 2 + rsu_meta_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, fusion_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, 1),
        )
        head_input_dim = temporal_hidden_dim * 2 + vehicle_state_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, fusion_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim // 2, output_dim), nn.Tanh(),
        )

    def forward(
        self,
        ego_scans: torch.Tensor,
        rsu_scans: torch.Tensor,
        rsu_meta: torch.Tensor,
        vehicle_state: Optional[torch.Tensor] = None,
        rsu_mask: Optional[torch.Tensor] = None,
        return_gates: bool = False,
    ) -> torch.Tensor | RsuFusionOutput:
        batch, sensors, history, rays = rsu_scans.shape
        ego_feature = self.scan_encoder(ego_scans)
        rsu_features = self.scan_encoder(rsu_scans.reshape(batch * sensors, history, rays))
        rsu_features = rsu_features.reshape(batch, sensors, -1)
        ego_repeated = ego_feature.unsqueeze(1).expand(-1, sensors, -1)
        learned_gate = torch.sigmoid(
            self.gate(torch.cat([ego_repeated, rsu_features, rsu_meta], dim=-1)).squeeze(-1)
        )
        distance = torch.clamp(rsu_meta[..., 0], min=0.0)
        gates = learned_gate * torch.exp(-distance / max(self.distance_decay_m, 1e-6))
        if rsu_mask is not None:
            gates = gates.masked_fill(~rsu_mask.bool(), 0.0)
        if 0 < self.top_k_rsus < sensors:
            indices = torch.topk(gates, k=self.top_k_rsus, dim=1).indices
            topk_mask = torch.zeros_like(gates, dtype=torch.bool)
            topk_mask.scatter_(1, indices, True)
            gates = gates.masked_fill(~topk_mask, 0.0)
        weights = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        fused_rsu = torch.sum(rsu_features * weights.unsqueeze(-1), dim=1)
        features = [ego_feature, fused_rsu]
        if self.vehicle_state_dim > 0:
            if vehicle_state is None:
                vehicle_state = ego_feature.new_zeros(batch, self.vehicle_state_dim)
            features.append(vehicle_state)
        control = self.head(torch.cat(features, dim=-1))
        return RsuFusionOutput(control, gates) if return_gates else control


class RsuTrajectoryFusionNet(nn.Module):
    """Multi-modal future trajectory planner with a learned direct-control head."""

    def __init__(
        self, scan_dim: int = 1080, rsu_count: int = 6, rsu_meta_dim: int = 5,
        vehicle_state_dim: int = 1, output_dim: int = 2, frame_feature_dim: int = 128,
        temporal_hidden_dim: int = 128, fusion_hidden_dim: int = 128,
        distance_decay_m: float = 35.0, top_k_rsus: int = 2,
        trajectory_modes: int = 4, trajectory_steps: int = 20, trajectory_dim: int = 3,
    ) -> None:
        super().__init__()
        if trajectory_modes < 1 or trajectory_steps < 1 or trajectory_dim != 3:
            raise ValueError("trajectory_modes/steps must be positive and trajectory_dim must be 3")
        self.rsu_count = int(rsu_count)
        self.rsu_meta_dim = int(rsu_meta_dim)
        self.vehicle_state_dim = int(vehicle_state_dim)
        self.distance_decay_m = float(distance_decay_m)
        self.top_k_rsus = int(top_k_rsus)
        self.trajectory_modes = int(trajectory_modes)
        self.trajectory_steps = int(trajectory_steps)
        self.trajectory_dim = int(trajectory_dim)
        self.scan_encoder = TemporalScanEncoder(scan_dim, frame_feature_dim, temporal_hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(temporal_hidden_dim * 2 + rsu_meta_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(fusion_hidden_dim, 1),
        )
        feature_dim = temporal_hidden_dim * 2 + vehicle_state_dim
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, fusion_hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim), nn.ReLU(inplace=True),
        )
        self.control_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim // 2, output_dim), nn.Tanh(),
        )
        self.trajectory_head = nn.Linear(
            fusion_hidden_dim, trajectory_modes * trajectory_steps * trajectory_dim
        )
        self.mode_head = nn.Linear(fusion_hidden_dim, trajectory_modes)

    def _encode_fusion(
        self, ego_scans: torch.Tensor, rsu_scans: torch.Tensor, rsu_meta: torch.Tensor,
        vehicle_state: Optional[torch.Tensor] = None, rsu_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sensors, history, rays = rsu_scans.shape
        ego_feature = self.scan_encoder(ego_scans)
        rsu_features = self.scan_encoder(rsu_scans.reshape(batch * sensors, history, rays))
        rsu_features = rsu_features.reshape(batch, sensors, -1)
        ego_repeated = ego_feature.unsqueeze(1).expand(-1, sensors, -1)
        gates = torch.sigmoid(
            self.gate(torch.cat([ego_repeated, rsu_features, rsu_meta], dim=-1)).squeeze(-1)
        )
        distance = torch.clamp(rsu_meta[..., 0], min=0.0)
        gates = gates * torch.exp(-distance / max(self.distance_decay_m, 1e-6))
        if rsu_mask is not None:
            gates = gates.masked_fill(~rsu_mask.bool(), 0.0)
        if 0 < self.top_k_rsus < sensors:
            indices = torch.topk(gates, k=self.top_k_rsus, dim=1).indices
            topk_mask = torch.zeros_like(gates, dtype=torch.bool)
            topk_mask.scatter_(1, indices, True)
            gates = gates.masked_fill(~topk_mask, 0.0)
        weights = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        fused_rsu = torch.sum(rsu_features * weights.unsqueeze(-1), dim=1)
        features = [ego_feature, fused_rsu]
        if self.vehicle_state_dim > 0:
            if vehicle_state is None:
                vehicle_state = ego_feature.new_zeros(batch, self.vehicle_state_dim)
            features.append(vehicle_state)
        hidden = self.trunk(torch.cat(features, dim=-1))
        return hidden, gates

    def forward(
        self, ego_scans: torch.Tensor, rsu_scans: torch.Tensor, rsu_meta: torch.Tensor,
        vehicle_state: Optional[torch.Tensor] = None, rsu_mask: Optional[torch.Tensor] = None,
        return_gates: bool = True,
    ) -> RsuTrajectoryOutput:
        hidden, gates = self._encode_fusion(
            ego_scans, rsu_scans, rsu_meta, vehicle_state, rsu_mask
        )
        batch = hidden.shape[0]
        raw = self.trajectory_head(hidden).reshape(
            batch, self.trajectory_modes, self.trajectory_steps, self.trajectory_dim
        )
        trajectories = torch.cat([torch.tanh(raw[..., :2]), torch.sigmoid(raw[..., 2:3])], dim=-1)
        return RsuTrajectoryOutput(
            self.control_head(hidden), trajectories, self.mode_head(hidden), gates
        )


class RsuBezierTrajectoryFusionNet(RsuTrajectoryFusionNet):
    """Ordered anchors decoded as a C1-continuous piecewise cubic Bezier path."""

    def __init__(
        self, *args, trajectory_anchor_count: int = 4,
        max_anchor_step_normalized: float = 0.24,
        max_anchor_heading_delta: float = 1.2, **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if trajectory_anchor_count < 1:
            raise ValueError("trajectory_anchor_count must be positive")
        if max_anchor_step_normalized <= 0.0 or max_anchor_heading_delta <= 0.0:
            raise ValueError("anchor step and heading bounds must be positive")
        self.trajectory_anchor_count = int(trajectory_anchor_count)
        self.max_anchor_step_normalized = float(max_anchor_step_normalized)
        self.max_anchor_heading_delta = float(max_anchor_heading_delta)
        hidden_dim = self.mode_head.in_features
        del self.trajectory_head
        self.anchor_head = nn.Linear(
            hidden_dim, self.trajectory_modes * self.trajectory_anchor_count * 2
        )
        self.speed_anchor_head = nn.Linear(
            hidden_dim, self.trajectory_modes * self.trajectory_anchor_count
        )
        scaled_time = (
            torch.arange(1, self.trajectory_steps + 1, dtype=torch.float32)
            * self.trajectory_anchor_count / self.trajectory_steps
        )
        segment_index = torch.ceil(scaled_time).to(torch.long) - 1
        segment_index.clamp_(0, self.trajectory_anchor_count - 1)
        self.register_buffer("sample_segment_index", segment_index, persistent=False)
        self.register_buffer(
            "sample_local_time", scaled_time - segment_index.to(torch.float32), persistent=False
        )

    def _sample_bezier(self, anchors: torch.Tensor) -> torch.Tensor:
        differences = anchors[..., 1:, :] - anchors[..., :-1, :]
        if self.trajectory_anchor_count == 1:
            tangents = torch.cat([differences, differences], dim=-2)
        else:
            middle = 0.5 * (anchors[..., 2:, :] - anchors[..., :-2, :])
            tangents = torch.cat(
                [differences[..., :1, :], middle, differences[..., -1:, :]], dim=-2
            )
        index = self.sample_segment_index
        p0, p3 = anchors[..., index, :], anchors[..., index + 1, :]
        p1 = p0 + tangents[..., index, :] / 3.0
        p2 = p3 - tangents[..., index + 1, :] / 3.0
        u = self.sample_local_time.reshape(1, 1, -1, 1)
        v = 1.0 - u
        return v.pow(3) * p0 + 3 * v.pow(2) * u * p1 + 3 * v * u.pow(2) * p2 + u.pow(3) * p3

    def forward(
        self, ego_scans: torch.Tensor, rsu_scans: torch.Tensor, rsu_meta: torch.Tensor,
        vehicle_state: Optional[torch.Tensor] = None, rsu_mask: Optional[torch.Tensor] = None,
        return_gates: bool = True,
    ) -> RsuTrajectoryOutput:
        hidden, gates = self._encode_fusion(
            ego_scans, rsu_scans, rsu_meta, vehicle_state, rsu_mask
        )
        batch = hidden.shape[0]
        raw = self.anchor_head(hidden).reshape(
            batch, self.trajectory_modes, self.trajectory_anchor_count, 2
        )
        lengths = torch.sigmoid(raw[..., 0]) * self.max_anchor_step_normalized
        headings = torch.cumsum(
            torch.tanh(raw[..., 1]) * self.max_anchor_heading_delta, dim=-1
        )
        displacement = torch.stack(
            [lengths * torch.cos(headings), lengths * torch.sin(headings)], dim=-1
        )
        origin = displacement.new_zeros(batch, self.trajectory_modes, 1, 2)
        anchors = torch.cat([origin, torch.cumsum(displacement, dim=-2)], dim=-2)
        xy = self._sample_bezier(anchors)
        future_speed = torch.sigmoid(self.speed_anchor_head(hidden)).reshape(
            batch, self.trajectory_modes, self.trajectory_anchor_count
        )
        current_speed = (
            vehicle_state[:, :1].clamp(0.0, 1.0)
            if vehicle_state is not None and vehicle_state.shape[-1] > 0
            else hidden.new_zeros(batch, 1)
        )
        speed_anchors = torch.cat(
            [current_speed.unsqueeze(1).expand(-1, self.trajectory_modes, -1), future_speed], dim=-1
        )
        index = self.sample_segment_index
        u = self.sample_local_time.reshape(1, 1, -1)
        speed = torch.lerp(speed_anchors[..., index], speed_anchors[..., index + 1], u)
        return RsuTrajectoryOutput(
            self.control_head(hidden), torch.cat([xy, speed.unsqueeze(-1)], dim=-1),
            self.mode_head(hidden), gates,
        )

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
    """Encode one LaserScan frame with a compact 1D Conv stack."""

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
            dummy = torch.zeros(1, 1, scan_dim)
            self.net(dummy)

    def forward(self, scan: torch.Tensor) -> torch.Tensor:
        # scan: [B, R]
        x = scan.unsqueeze(1)
        x = self.net(x).flatten(1)
        return F.relu(self.proj(x), inplace=True)


class TemporalScanEncoder(nn.Module):
    """Encode a scan history by applying Conv per frame, then GRU over time."""

    def __init__(self, scan_dim: int, frame_feature_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.frame_encoder = ScanFrameEncoder(scan_dim=scan_dim, feature_dim=frame_feature_dim)
        self.gru = nn.GRU(frame_feature_dim, hidden_dim, batch_first=True)

    def forward(self, scans: torch.Tensor) -> torch.Tensor:
        # scans: [B, T, R]
        batch, history, rays = scans.shape
        encoded = self.frame_encoder(scans.reshape(batch * history, rays))
        encoded = encoded.reshape(batch, history, -1)
        _, hidden = self.gru(encoded)
        return hidden[-1]


class DistanceGatedRsuFusionNet(nn.Module):
    """
    Ego-priority RSU fusion model.

    Inputs:
      ego_scans: [B, T, R]
      rsu_scans: [B, S, T, R]
      rsu_meta: [B, S, M], where meta should include at least distance.
      vehicle_state: optional [B, V]

    Output:
      control: [B, output_dim]
      gates: [B, S]
    """

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

        self.scan_encoder = TemporalScanEncoder(
            scan_dim=scan_dim,
            frame_feature_dim=frame_feature_dim,
            hidden_dim=temporal_hidden_dim,
        )
        gate_input_dim = temporal_hidden_dim * 2 + rsu_meta_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, 1),
        )

        head_input_dim = temporal_hidden_dim * 2 + vehicle_state_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim // 2, output_dim),
            nn.Tanh(),
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
        gate_logits = self.gate(torch.cat([ego_repeated, rsu_features, rsu_meta], dim=-1)).squeeze(-1)
        learned_gate = torch.sigmoid(gate_logits)

        distance = torch.clamp(rsu_meta[..., 0], min=0.0)
        distance_gate = torch.exp(-distance / max(self.distance_decay_m, 1e-6))
        gates = learned_gate * distance_gate

        if rsu_mask is not None:
            gates = gates.masked_fill(~rsu_mask.bool(), 0.0)

        if self.top_k_rsus > 0 and self.top_k_rsus < sensors:
            topk = torch.topk(gates, k=self.top_k_rsus, dim=1).indices
            topk_mask = torch.zeros_like(gates, dtype=torch.bool)
            topk_mask.scatter_(1, topk, True)
            gates = gates.masked_fill(~topk_mask, 0.0)

        normalized_gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        fused_rsu = torch.sum(rsu_features * normalized_gates.unsqueeze(-1), dim=1)

        features = [ego_feature, fused_rsu]
        if self.vehicle_state_dim > 0:
            if vehicle_state is None:
                vehicle_state = ego_feature.new_zeros(batch, self.vehicle_state_dim)
            features.append(vehicle_state)

        control = self.head(torch.cat(features, dim=-1))
        if return_gates:
            return RsuFusionOutput(control=control, gates=gates)
        return control


class RsuTrajectoryFusionNet(nn.Module):
    """Multi-modal future trajectory planner with a learned direct-control head."""

    def __init__(
        self,
        scan_dim: int = 1080,
        rsu_count: int = 6,
        rsu_meta_dim: int = 5,
        vehicle_state_dim: int = 1,
        output_dim: int = 2,
        frame_feature_dim: int = 128,
        temporal_hidden_dim: int = 128,
        fusion_hidden_dim: int = 128,
        distance_decay_m: float = 35.0,
        top_k_rsus: int = 2,
        trajectory_modes: int = 4,
        trajectory_steps: int = 20,
        trajectory_dim: int = 3,
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
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, 1),
        )
        feature_dim = temporal_hidden_dim * 2 + vehicle_state_dim
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.control_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim // 2, output_dim),
            nn.Tanh(),
        )
        self.trajectory_head = nn.Linear(
            fusion_hidden_dim, trajectory_modes * trajectory_steps * trajectory_dim
        )
        self.mode_head = nn.Linear(fusion_hidden_dim, trajectory_modes)

    def _encode_fusion(
        self,
        ego_scans: torch.Tensor,
        rsu_scans: torch.Tensor,
        rsu_meta: torch.Tensor,
        vehicle_state: Optional[torch.Tensor] = None,
        rsu_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sensors, history, rays = rsu_scans.shape
        ego_feature = self.scan_encoder(ego_scans)
        rsu_features = self.scan_encoder(rsu_scans.reshape(batch * sensors, history, rays))
        rsu_features = rsu_features.reshape(batch, sensors, -1)
        ego_repeated = ego_feature.unsqueeze(1).expand(-1, sensors, -1)
        gate_logits = self.gate(
            torch.cat([ego_repeated, rsu_features, rsu_meta], dim=-1)
        ).squeeze(-1)
        gates = torch.sigmoid(gate_logits)
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
        self,
        ego_scans: torch.Tensor,
        rsu_scans: torch.Tensor,
        rsu_meta: torch.Tensor,
        vehicle_state: Optional[torch.Tensor] = None,
        rsu_mask: Optional[torch.Tensor] = None,
        return_gates: bool = True,
    ) -> RsuTrajectoryOutput:
        hidden, gates = self._encode_fusion(
            ego_scans, rsu_scans, rsu_meta, vehicle_state, rsu_mask
        )
        batch = hidden.shape[0]
        raw_trajectory = self.trajectory_head(hidden).reshape(
            batch, self.trajectory_modes, self.trajectory_steps, self.trajectory_dim
        )
        trajectories = torch.cat(
            [torch.tanh(raw_trajectory[..., :2]), torch.sigmoid(raw_trajectory[..., 2:3])],
            dim=-1,
        )
        return RsuTrajectoryOutput(
            control=self.control_head(hidden),
            trajectories=trajectories,
            mode_logits=self.mode_head(hidden),
            gates=gates,
        )


class RsuBezierTrajectoryFusionNet(RsuTrajectoryFusionNet):
    """Predict ordered anchors and sample a C1-continuous piecewise cubic Bezier path."""

    def __init__(
        self,
        *args,
        trajectory_anchor_count: int = 4,
        max_anchor_step_normalized: float = 0.24,
        max_anchor_heading_delta: float = 1.2,
        **kwargs,
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
        local_time = scaled_time - segment_index.to(torch.float32)
        self.register_buffer("sample_segment_index", segment_index, persistent=False)
        self.register_buffer("sample_local_time", local_time, persistent=False)

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
        p0 = anchors[..., index, :]
        p3 = anchors[..., index + 1, :]
        p1 = p0 + tangents[..., index, :] / 3.0
        p2 = p3 - tangents[..., index + 1, :] / 3.0
        u = self.sample_local_time.reshape(1, 1, -1, 1)
        one_minus_u = 1.0 - u
        return (
            one_minus_u.pow(3) * p0
            + 3.0 * one_minus_u.pow(2) * u * p1
            + 3.0 * one_minus_u * u.pow(2) * p2
            + u.pow(3) * p3
        )

    def forward(
        self,
        ego_scans: torch.Tensor,
        rsu_scans: torch.Tensor,
        rsu_meta: torch.Tensor,
        vehicle_state: Optional[torch.Tensor] = None,
        rsu_mask: Optional[torch.Tensor] = None,
        return_gates: bool = True,
    ) -> RsuTrajectoryOutput:
        hidden, gates = self._encode_fusion(
            ego_scans, rsu_scans, rsu_meta, vehicle_state, rsu_mask
        )
        batch = hidden.shape[0]
        raw_anchor = self.anchor_head(hidden).reshape(
            batch, self.trajectory_modes, self.trajectory_anchor_count, 2
        )
        lengths = torch.sigmoid(raw_anchor[..., 0]) * self.max_anchor_step_normalized
        heading_deltas = torch.tanh(raw_anchor[..., 1]) * self.max_anchor_heading_delta
        headings = torch.cumsum(heading_deltas, dim=-1)
        displacements = torch.stack(
            [lengths * torch.cos(headings), lengths * torch.sin(headings)], dim=-1
        )
        origin = displacements.new_zeros(batch, self.trajectory_modes, 1, 2)
        anchors = torch.cat([origin, torch.cumsum(displacements, dim=-2)], dim=-2)
        xy = self._sample_bezier(anchors)

        future_speeds = torch.sigmoid(self.speed_anchor_head(hidden)).reshape(
            batch, self.trajectory_modes, self.trajectory_anchor_count
        )
        if vehicle_state is not None and vehicle_state.shape[-1] > 0:
            current_speed = vehicle_state[:, :1].clamp(0.0, 1.0)
        else:
            current_speed = hidden.new_zeros(batch, 1)
        speed_anchors = torch.cat(
            [current_speed.unsqueeze(1).expand(-1, self.trajectory_modes, -1), future_speeds],
            dim=-1,
        )
        index = self.sample_segment_index
        u = self.sample_local_time.reshape(1, 1, -1)
        speed = torch.lerp(speed_anchors[..., index], speed_anchors[..., index + 1], u)
        trajectories = torch.cat([xy, speed.unsqueeze(-1)], dim=-1)
        return RsuTrajectoryOutput(
            control=self.control_head(hidden), trajectories=trajectories,
            mode_logits=self.mode_head(hidden), gates=gates,
        )


class DepthwiseSeparableBlock(nn.Module):
    """Low-cost spatial block suitable for real-time semantic BEV encoding."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=stride,
                padding=1, groups=in_channels, bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class BevBezierTrajectoryNet(nn.Module):
    """Encode semantic BEV histories and predict controls plus smooth trajectories."""

    def __init__(
        self,
        bev_channels: int = 6,
        vehicle_state_dim: int = 1,
        output_dim: int = 2,
        frame_feature_dim: int = 128,
        temporal_hidden_dim: int = 128,
        fusion_hidden_dim: int = 128,
        trajectory_modes: int = 4,
        trajectory_steps: int = 12,
        trajectory_dim: int = 3,
        trajectory_anchor_count: int = 4,
        max_anchor_step_normalized: float = 0.24,
        max_anchor_heading_delta: float = 1.2,
    ) -> None:
        super().__init__()
        if bev_channels < 1 or trajectory_modes < 1 or trajectory_steps < 1:
            raise ValueError("BEV channels, trajectory modes and steps must be positive")
        if trajectory_dim != 3 or trajectory_anchor_count < 1:
            raise ValueError("trajectory_dim must be 3 and anchor count must be positive")
        self.bev_channels = int(bev_channels)
        self.vehicle_state_dim = int(vehicle_state_dim)
        self.trajectory_modes = int(trajectory_modes)
        self.trajectory_steps = int(trajectory_steps)
        self.trajectory_dim = int(trajectory_dim)
        self.trajectory_anchor_count = int(trajectory_anchor_count)
        self.max_anchor_step_normalized = float(max_anchor_step_normalized)
        self.max_anchor_heading_delta = float(max_anchor_heading_delta)

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(self.bev_channels, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24), nn.SiLU(inplace=True),
            DepthwiseSeparableBlock(24, 32, stride=2),
            DepthwiseSeparableBlock(32, 48, stride=2),
            DepthwiseSeparableBlock(48, 64, stride=2),
            DepthwiseSeparableBlock(64, 96, stride=2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(96, frame_feature_dim), nn.SiLU(inplace=True),
        )
        self.temporal = nn.GRU(frame_feature_dim, temporal_hidden_dim, batch_first=True)
        self.trunk = nn.Sequential(
            nn.Linear(temporal_hidden_dim + self.vehicle_state_dim, fusion_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.control_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2), nn.SiLU(inplace=True),
            nn.Linear(fusion_hidden_dim // 2, output_dim), nn.Tanh(),
        )
        self.mode_head = nn.Linear(fusion_hidden_dim, self.trajectory_modes)
        self.anchor_head = nn.Linear(
            fusion_hidden_dim, self.trajectory_modes * self.trajectory_anchor_count * 2
        )
        self.speed_anchor_head = nn.Linear(
            fusion_hidden_dim, self.trajectory_modes * self.trajectory_anchor_count
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
        value = self.sample_local_time.reshape(1, 1, -1, 1)
        inverse = 1.0 - value
        return (
            inverse.pow(3) * p0 + 3.0 * inverse.pow(2) * value * p1
            + 3.0 * inverse * value.pow(2) * p2 + value.pow(3) * p3
        )

    def forward(
        self, bev: torch.Tensor, vehicle_state: Optional[torch.Tensor] = None,
        return_gates: bool = True,
    ) -> RsuTrajectoryOutput:
        del return_gates
        batch, history, channels, height, width = bev.shape
        if channels != self.bev_channels:
            raise ValueError(f"Expected {self.bev_channels} BEV channels, got {channels}")
        encoded = self.frame_encoder(bev.reshape(batch * history, channels, height, width))
        encoded = encoded.reshape(batch, history, -1)
        _, temporal = self.temporal(encoded)
        features = [temporal[-1]]
        if self.vehicle_state_dim > 0:
            if vehicle_state is None:
                vehicle_state = encoded.new_zeros(batch, self.vehicle_state_dim)
            features.append(vehicle_state)
        hidden = self.trunk(torch.cat(features, dim=-1))

        raw_anchor = self.anchor_head(hidden).reshape(
            batch, self.trajectory_modes, self.trajectory_anchor_count, 2
        )
        lengths = torch.sigmoid(raw_anchor[..., 0]) * self.max_anchor_step_normalized
        heading_deltas = torch.tanh(raw_anchor[..., 1]) * self.max_anchor_heading_delta
        headings = torch.cumsum(heading_deltas, dim=-1)
        displacements = torch.stack(
            [lengths * torch.cos(headings), lengths * torch.sin(headings)], dim=-1
        )
        origin = displacements.new_zeros(batch, self.trajectory_modes, 1, 2)
        anchors = torch.cat([origin, torch.cumsum(displacements, dim=-2)], dim=-2)
        xy = self._sample_bezier(anchors)

        future_speeds = torch.sigmoid(self.speed_anchor_head(hidden)).reshape(
            batch, self.trajectory_modes, self.trajectory_anchor_count
        )
        current_speed = (
            vehicle_state[:, :1].clamp(0.0, 1.0)
            if vehicle_state is not None and vehicle_state.shape[-1] > 0
            else hidden.new_zeros(batch, 1)
        )
        speed_anchors = torch.cat([
            current_speed.unsqueeze(1).expand(-1, self.trajectory_modes, -1), future_speeds
        ], dim=-1)
        index = self.sample_segment_index
        value = self.sample_local_time.reshape(1, 1, -1)
        speed = torch.lerp(speed_anchors[..., index], speed_anchors[..., index + 1], value)
        trajectories = torch.cat([xy, speed.unsqueeze(-1)], dim=-1)
        return RsuTrajectoryOutput(
            control=self.control_head(hidden), trajectories=trajectories,
            mode_logits=self.mode_head(hidden), gates=hidden.new_zeros(batch, 0),
        )

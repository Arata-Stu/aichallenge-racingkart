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

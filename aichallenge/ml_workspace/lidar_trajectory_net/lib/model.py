from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_list(values: Iterable[int]) -> List[int]:
    return [int(v) for v in values]


class ScanEncoder(nn.Module):
    """Encodes one multi-channel 1D scan into a compact feature vector."""

    def __init__(
        self,
        input_channels: int = 3,
        conv_channels: Iterable[int] = (32, 64, 128),
        kernel_sizes: Iterable[int] = (9, 7, 5),
        strides: Iterable[int] = (4, 4, 2),
        embed_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        conv_channels = _as_list(conv_channels)
        kernel_sizes = _as_list(kernel_sizes)
        strides = _as_list(strides)

        if not (len(conv_channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("conv_channels, kernel_sizes, and strides must have the same length.")

        layers = []
        in_channels = input_channels
        for out_channels, kernel_size, stride in zip(conv_channels, kernel_sizes, strides):
            padding = kernel_size // 2
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]
        x = self.conv(x)
        x = self.pool(x)
        return self.proj(x)


class BezierTrajectoryHead(nn.Module):
    """Predicts Bezier control points and samples a smooth ego-frame path."""

    def __init__(
        self,
        embed_dim: int,
        num_control_points: int = 3,
        num_future_points: int = 20,
        output_scale: Tuple[float, float] = (30.0, 10.0),
    ):
        super().__init__()
        if num_control_points < 1:
            raise ValueError("num_control_points must be positive.")
        if num_future_points < 1:
            raise ValueError("num_future_points must be positive.")

        self.num_control_points = num_control_points
        self.num_future_points = num_future_points
        self.register_buffer("output_scale", torch.tensor(output_scale, dtype=torch.float32))

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_control_points * 2),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        num_future_points: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        control_points = self.head(x).view(-1, self.num_control_points, 2)
        control_points = torch.tanh(control_points) * self.output_scale.view(1, 1, 2)
        path = sample_bezier_path(control_points, num_future_points or self.num_future_points)
        return path, control_points


class LidarTrajectoryNet(nn.Module):
    """1D scan encoder + temporal Transformer + Bezier path predictor."""

    def __init__(
        self,
        input_channels: int = 3,
        input_dim: int = 1080,
        history_length: int = 8,
        embed_dim: int = 128,
        conv_channels: Iterable[int] = (32, 64, 128),
        conv_kernel_sizes: Iterable[int] = (9, 7, 5),
        conv_strides: Iterable[int] = (4, 4, 2),
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 512,
        dropout: float = 0.1,
        num_control_points: int = 3,
        num_future_points: int = 20,
        output_scale: Tuple[float, float] = (30.0, 10.0),
    ):
        super().__init__()
        del input_dim  # The encoder is convolutional and accepts variable ray counts.

        self.history_length = history_length
        self.encoder = ScanEncoder(
            input_channels=input_channels,
            conv_channels=conv_channels,
            kernel_sizes=conv_kernel_sizes,
            strides=conv_strides,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, history_length, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.path_head = BezierTrajectoryHead(
            embed_dim=embed_dim,
            num_control_points=num_control_points,
            num_future_points=num_future_points,
            output_scale=output_scale,
        )

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        num_future_points: Optional[int] = None,
        return_control_points: bool = False,
    ):
        # x: [B, T, C, N]
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, T, C, N], got {tuple(x.shape)}")

        batch_size, history_length, channels, num_rays = x.shape
        if history_length > self.pos_embedding.shape[1]:
            raise ValueError(
                f"history_length={history_length} exceeds model maximum "
                f"{self.pos_embedding.shape[1]}."
            )

        encoded = self.encoder(x.reshape(batch_size * history_length, channels, num_rays))
        tokens = encoded.view(batch_size, history_length, -1)
        tokens = tokens + self.pos_embedding[:, :history_length, :]
        tokens = self.temporal_encoder(tokens)
        summary = self.norm(tokens[:, -1, :])

        path, control_points = self.path_head(summary, num_future_points=num_future_points)
        if return_control_points:
            return path, control_points
        return path


def sample_bezier_path(control_points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Samples a Bezier path with P0 fixed at the ego origin."""
    if control_points.ndim != 3 or control_points.shape[-1] != 2:
        raise ValueError("control_points must have shape [B, K, 2].")

    batch_size = control_points.shape[0]
    device = control_points.device
    dtype = control_points.dtype
    origin = torch.zeros(batch_size, 1, 2, device=device, dtype=dtype)
    points = torch.cat([origin, control_points], dim=1)
    degree = points.shape[1] - 1

    t = torch.linspace(1.0 / num_samples, 1.0, num_samples, device=device, dtype=dtype)
    basis_terms = []
    for i in range(degree + 1):
        coefficient = math.comb(degree, i)
        basis = coefficient * torch.pow(1.0 - t, degree - i) * torch.pow(t, i)
        basis_terms.append(basis)
    basis = torch.stack(basis_terms, dim=0)

    return torch.einsum("kn,bkd->bnd", basis, points)

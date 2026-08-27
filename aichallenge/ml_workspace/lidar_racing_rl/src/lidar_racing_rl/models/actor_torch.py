"""PyTorch deployment twin of the Flax LiDAR Actor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TorchActorArchitecture:
    """Shape contract mirrored by the ROS 2 inference package."""

    beam_count: int = 360
    frame_stack: int = 4
    scan_channels: int = 2
    conv_channels: tuple[int, ...] = (32, 64, 64)
    kernel_sizes: tuple[int, ...] = (8, 4, 3)
    strides: tuple[int, ...] = (4, 2, 1)
    hidden_dim: int = 256
    action_dim: int = 2
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @property
    def input_channels(self) -> int:
        return self.frame_stack * self.scan_channels

    def validate(self) -> None:
        if (self.beam_count, self.frame_stack, self.scan_channels) != (360, 4, 2):
            raise ValueError("deployment Actor requires the canonical 4x2x360 input")
        if not (
            len(self.conv_channels) == len(self.kernel_sizes) == len(self.strides) == 3
        ):
            raise ValueError("deployment Actor requires exactly three Conv1D layers")
        if any(
            value <= 0
            for value in (*self.conv_channels, *self.kernel_sizes, *self.strides)
        ):
            raise ValueError("Conv1D dimensions must be positive")
        if self.hidden_dim != 256 or self.action_dim != 2:
            raise ValueError("deployment Actor requires hidden_dim=256 and action_dim=2")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std bounds must be ordered")


class TorchLidarActor(nn.Module):
    """Conv1D Gaussian Actor with deterministic tanh deployment action."""

    def __init__(self, architecture: TorchActorArchitecture | None = None):
        super().__init__()
        self.architecture = architecture or TorchActorArchitecture()
        self.architecture.validate()

        layers: list[nn.Module] = []
        input_channels = self.architecture.input_channels
        output_length = self.architecture.beam_count
        for output_channels, kernel_size, stride in zip(
            self.architecture.conv_channels,
            self.architecture.kernel_sizes,
            self.architecture.strides,
            strict=True,
        ):
            layers.extend(
                (
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                    ),
                    nn.ReLU(),
                )
            )
            output_length = (output_length - kernel_size) // stride + 1
            if output_length <= 0:
                raise ValueError("Conv1D architecture collapses the beam dimension")
            input_channels = output_channels

        self.encoder = nn.Sequential(*layers)
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * output_length, self.architecture.hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(self.architecture.hidden_dim, self.architecture.action_dim)
        self.log_std_head = nn.Linear(
            self.architecture.hidden_dim, self.architecture.action_dim
        )

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected_shape = (
            self.architecture.input_channels,
            self.architecture.beam_count,
        )
        if observation.ndim != 3 or tuple(observation.shape[1:]) != expected_shape:
            raise ValueError(
                f"expected Actor input [batch, {expected_shape[0]}, {expected_shape[1]}]"
            )
        features = self.trunk(self.encoder(observation))
        mean = self.mean_head(features)
        log_std = torch.clamp(
            self.log_std_head(features),
            self.architecture.log_std_min,
            self.architecture.log_std_max,
        )
        return mean, log_std

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the same ``tanh(mean)`` action used by Flax evaluation."""

        mean, _ = self(observation)
        return torch.tanh(mean)


__all__ = ["TorchActorArchitecture", "TorchLidarActor"]

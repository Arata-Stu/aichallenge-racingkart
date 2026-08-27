"""PyTorch model definitions shared by training and ROS 2 inference."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _TinyLidarBase(nn.Module):
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class TinyLidarNet(_TinyLidarBase):
    """Five-layer 1D CNN producing [acceleration, steering]."""

    def __init__(self, input_dim: int = 1080, output_dim: int = 2) -> None:
        super().__init__()
        if input_dim < 214:
            raise ValueError("input_dim is too small for the TinyLiDARNet convolution stack")
        if output_dim < 2:
            raise ValueError("output_dim must be at least 2 ([acceleration, steering])")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_dim)
            features = self._encode(dummy)
            flatten_dim = int(features.flatten(1).shape[1])

        self.fc1 = nn.Linear(flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, self.output_dim)
        self._initialize_weights()

    def _encode(self, value: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.conv1(value))
        value = F.relu(self.conv2(value))
        value = F.relu(self.conv3(value))
        value = F.relu(self.conv4(value))
        return F.relu(self.conv5(value))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self._encode(value).flatten(1)
        value = F.relu(self.fc1(value))
        value = F.relu(self.fc2(value))
        value = F.relu(self.fc3(value))
        return torch.tanh(self.fc4(value))


class TinyLidarNetSmall(_TinyLidarBase):
    """Three-layer lightweight TinyLiDARNet variant."""

    def __init__(self, input_dim: int = 1080, output_dim: int = 2) -> None:
        super().__init__()
        if input_dim < 86:
            raise ValueError("input_dim is too small for the TinyLiDARNetSmall convolution stack")
        if output_dim < 2:
            raise ValueError("output_dim must be at least 2 ([acceleration, steering])")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_dim)
            features = self._encode(dummy)
            flatten_dim = int(features.flatten(1).shape[1])

        self.fc1 = nn.Linear(flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, self.output_dim)
        self._initialize_weights()

    def _encode(self, value: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.conv1(value))
        value = F.relu(self.conv2(value))
        return F.relu(self.conv3(value))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self._encode(value).flatten(1)
        value = F.relu(self.fc1(value))
        value = F.relu(self.fc2(value))
        return torch.tanh(self.fc3(value))


def normalize_architecture(name: str) -> str:
    normalized = name.strip().lower()
    aliases = {"large": "normal", "tinylidarnet": "normal", "tinylidarnetsmall": "small"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"normal", "small"}:
        raise ValueError(f"Unsupported architecture: {name!r}; expected 'normal' or 'small'")
    return normalized


def build_model(architecture: str = "normal", input_dim: int = 1080, output_dim: int = 2) -> nn.Module:
    architecture = normalize_architecture(architecture)
    if architecture == "small":
        return TinyLidarNetSmall(input_dim=input_dim, output_dim=output_dim)
    return TinyLidarNet(input_dim=input_dim, output_dim=output_dim)

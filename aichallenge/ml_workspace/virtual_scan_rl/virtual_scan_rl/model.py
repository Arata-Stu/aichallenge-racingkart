"""TinyLiDARNet-style feature extractor used by the SAC actor and critics."""

from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class TinyLidarFeatureExtractor(BaseFeaturesExtractor):
    """Five-layer 1D CNN with a small vehicle-state branch.

    Temporal scan frames are treated as input channels. This preserves the
    compact TinyLiDARNet representation while exposing obstacle motion to SAC.
    """

    def __init__(self, observation_space, features_dim: int = 128) -> None:
        super().__init__(observation_space, features_dim)
        history_frames, num_rays = observation_space["scan"].shape
        state_dim = observation_space["state"].shape[0]
        if num_rays < 214:
            raise ValueError("TinyLiDAR convolution stack requires at least 214 rays")
        self.scan_encoder = nn.Sequential(
            nn.Conv1d(history_frames, 24, kernel_size=10, stride=4),
            nn.ReLU(),
            nn.Conv1d(24, 36, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(36, 48, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 * 4 + 32, features_dim),
            nn.ReLU(),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        scan = observations["scan"].float()
        state = observations["state"].float()
        scan_features = self.scan_encoder(scan)
        state_features = self.state_encoder(state)
        return self.fusion(torch.cat((scan_features, state_features), dim=1))

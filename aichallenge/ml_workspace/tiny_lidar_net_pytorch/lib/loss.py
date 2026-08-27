"""Loss functions for TinyLiDARNet training."""

import torch
from torch import nn


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, acceleration_weight: float = 1.0, steering_weight: float = 1.0) -> None:
        super().__init__()
        self.acceleration_weight = acceleration_weight
        self.steering_weight = steering_weight
        self.loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        component_loss = self.loss(prediction, target)
        return (
            self.acceleration_weight * component_loss[:, 0].mean()
            + self.steering_weight * component_loss[:, 1].mean()
        )

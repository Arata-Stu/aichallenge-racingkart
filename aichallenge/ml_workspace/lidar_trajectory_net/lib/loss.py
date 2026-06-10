from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryLoss(nn.Module):
    """Loss for ego-frame path prediction."""

    def __init__(
        self,
        point_weight: float = 1.0,
        smoothness_weight: float = 0.05,
        end_point_weight: float = 0.2,
    ):
        super().__init__()
        self.point_weight = point_weight
        self.smoothness_weight = smoothness_weight
        self.end_point_weight = end_point_weight

    def forward(
        self,
        pred_path: torch.Tensor,
        target_path: torch.Tensor,
        control_points: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.point_weight * F.smooth_l1_loss(pred_path, target_path)

        if self.end_point_weight > 0.0:
            loss = loss + self.end_point_weight * F.smooth_l1_loss(
                pred_path[:, -1, :],
                target_path[:, -1, :],
            )

        if self.smoothness_weight > 0.0 and pred_path.shape[1] >= 3:
            second_diff = pred_path[:, 2:, :] - 2.0 * pred_path[:, 1:-1, :] + pred_path[:, :-2, :]
            loss = loss + self.smoothness_weight * second_diff.pow(2).mean()

        if control_points is not None and self.smoothness_weight > 0.0 and control_points.shape[1] >= 3:
            control_second_diff = (
                control_points[:, 2:, :]
                - 2.0 * control_points[:, 1:-1, :]
                + control_points[:, :-2, :]
            )
            loss = loss + 0.25 * self.smoothness_weight * control_second_diff.pow(2).mean()

        return loss

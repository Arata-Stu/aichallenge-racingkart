from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import hydra

from lib.data import MultiSequenceBevTrajectoryDataset, MultiSequenceRsuTrajectoryDataset
from lib.model import (
    BevBezierTrajectoryNet, RsuBezierTrajectoryFusionNet, RsuTrajectoryFusionNet,
    RsuTrajectoryOutput,
)


def clean(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class WeightedControlLoss(nn.Module):
    def __init__(self, acceleration_weight: float, steering_weight: float) -> None:
        super().__init__()
        self.acceleration_weight = float(acceleration_weight)
        self.steering_weight = float(steering_weight)
        if self.acceleration_weight < 0.0 or self.steering_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.acceleration_weight + self.steering_weight <= 0.0:
            raise ValueError("at least one loss weight must be positive")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        acceleration = nn.functional.smooth_l1_loss(prediction[:, 0], target[:, 0])
        steering = nn.functional.smooth_l1_loss(prediction[:, 1], target[:, 1])
        return self.acceleration_weight * acceleration + self.steering_weight * steering


class MultiTaskTrajectoryLoss(nn.Module):
    """Best-of-K trajectory loss plus learned mode selection and direct control."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.control = WeightedControlLoss(cfg.acceleration_weight, cfg.steering_weight)
        self.control_weight = float(cfg.control_weight)
        self.trajectory_weight = float(cfg.trajectory_weight)
        self.average_displacement_weight = float(getattr(cfg, "average_displacement_weight", 1.0))
        self.endpoint_weight = float(getattr(cfg, "endpoint_weight", 1.0))
        self.speed_weight = float(cfg.speed_weight)
        self.mode_weight = float(cfg.mode_weight)
        self.smoothness_weight = float(cfg.smoothness_weight)
        self.diversity_weight = float(cfg.diversity_weight)
        self.diversity_margin = float(cfg.diversity_margin)

    def forward(
        self, prediction: RsuTrajectoryOutput, target_control: torch.Tensor,
        target_trajectory: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = target_trajectory.unsqueeze(1).expand_as(prediction.trajectories)
        displacement = torch.linalg.vector_norm(
            prediction.trajectories[..., :2] - target[..., :2], dim=-1
        )
        ade = displacement.mean(dim=-1)
        fde = displacement[..., -1]
        speed = nn.functional.smooth_l1_loss(
            prediction.trajectories[..., 2], target[..., 2], reduction="none"
        ).mean(dim=-1)
        candidate_loss = (
            self.average_displacement_weight * ade
            + self.endpoint_weight * fde
            + self.speed_weight * speed
        )
        winner = candidate_loss.detach().argmin(dim=1)
        batch_indices = torch.arange(len(winner), device=winner.device)
        chosen = prediction.trajectories[batch_indices, winner]
        trajectory_loss = candidate_loss[batch_indices, winner].mean()
        mode_loss = nn.functional.cross_entropy(prediction.mode_logits, winner)
        control_loss = self.control(prediction.control, target_control)
        if chosen.shape[1] >= 3:
            second_difference = chosen[:, 2:, :2] - 2 * chosen[:, 1:-1, :2] + chosen[:, :-2, :2]
            smoothness = second_difference.abs().mean()
        else:
            smoothness = chosen.new_zeros(())
        endpoints = prediction.trajectories[:, :, -1, :2]
        pairwise = torch.cdist(endpoints, endpoints)
        modes = endpoints.shape[1]
        if modes > 1:
            off_diagonal = ~torch.eye(modes, dtype=torch.bool, device=endpoints.device)
            diversity = nn.functional.relu(self.diversity_margin - pairwise[:, off_diagonal]).mean()
        else:
            diversity = chosen.new_zeros(())
        total = (
            self.control_weight * control_loss
            + self.trajectory_weight * trajectory_loss
            + self.mode_weight * mode_loss
            + self.smoothness_weight * smoothness
            + self.diversity_weight * diversity
        )
        return total, {
            "total": total.detach(), "control": control_loss.detach(),
            "trajectory": trajectory_loss.detach(), "mode": mode_loss.detach(),
            "ade": ade[batch_indices, winner].mean().detach(),
            "fde": fde[batch_indices, winner].mean().detach(),
            "smoothness": smoothness.detach(), "diversity": diversity.detach(),
        }


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in batch.items():
        moved = value.to(device)
        result[key] = clean(moved) if moved.is_floating_point() else moved
    return result


def acceleration_balanced_sampler(
    dataset: MultiSequenceRsuTrajectoryDataset | MultiSequenceBevTrajectoryDataset,
    max_weight_ratio: float = 10.0,
) -> WeightedRandomSampler:
    """Balance brake/coast/partial/full acceleration without discarding frames."""
    categories = []
    for sequence in dataset.datasets:
        raw = sequence.targets[sequence.sample_indices, 0] * sequence.acceleration_scale
        category = np.select(
            [raw < -0.1, raw <= 0.2, raw < 1.8], [0, 1, 2], default=3
        )
        categories.append(category.astype(np.int64))
    merged = np.concatenate(categories)
    counts = np.bincount(merged, minlength=4).astype(np.float64)
    inverse = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
    present = inverse[inverse > 0]
    if len(present):
        inverse = np.minimum(inverse, present.min() * max(1.0, max_weight_ratio))
    weights = inverse[merged]
    print(f"Acceleration sampler bins brake/coast/partial/full={counts.astype(int).tolist()}")
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)


@hydra.main(config_path="./config", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common_dataset_args = dict(
        history_len=cfg.data.history_len,
        acceleration_scale=cfg.data.acceleration_scale,
        trajectory_steps=cfg.data.trajectory_steps,
        trajectory_dt=cfg.data.trajectory_dt,
        trajectory_distance_scale=cfg.data.trajectory_distance_scale,
        max_speed=cfg.data.max_speed,
    )
    use_bev = str(cfg.model.architecture) == "bev_trajectory_bezier_v1"
    if use_bev:
        dataset_args = dict(
            **common_dataset_args,
            bev_channels=tuple(int(value) for value in cfg.data.bev_channels),
            bev_height=int(cfg.data.bev_height),
            bev_width=int(cfg.data.bev_width),
        )
        train_dataset = MultiSequenceBevTrajectoryDataset(cfg.data.train_dir, **dataset_args)
        val_dataset = MultiSequenceBevTrajectoryDataset(cfg.data.val_dir, **dataset_args)
    else:
        dataset_args = dict(**common_dataset_args, max_range=cfg.data.max_range)
        train_dataset = MultiSequenceRsuTrajectoryDataset(cfg.data.train_dir, **dataset_args)
        val_dataset = MultiSequenceRsuTrajectoryDataset(cfg.data.val_dir, **dataset_args)

    sampler = (
        acceleration_balanced_sampler(train_dataset, float(cfg.data.max_balance_ratio))
        if bool(cfg.data.balance_acceleration) else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    common_model_args = dict(
        vehicle_state_dim=cfg.model.vehicle_state_dim,
        output_dim=cfg.model.output_dim,
        frame_feature_dim=cfg.model.frame_feature_dim,
        temporal_hidden_dim=cfg.model.temporal_hidden_dim,
        fusion_hidden_dim=cfg.model.fusion_hidden_dim,
        trajectory_modes=cfg.model.trajectory_modes,
        trajectory_steps=cfg.data.trajectory_steps,
        trajectory_dim=cfg.model.trajectory_dim,
        trajectory_anchor_count=cfg.model.trajectory_anchor_count,
        max_anchor_step_normalized=cfg.model.max_anchor_step_normalized,
        max_anchor_heading_delta=cfg.model.max_anchor_heading_delta,
    )
    if use_bev:
        model_class = BevBezierTrajectoryNet
        model_args = dict(**common_model_args, bev_channels=len(cfg.data.bev_channels))
    else:
        model_class = (
            RsuBezierTrajectoryFusionNet
            if str(cfg.model.architecture) == "trajectory_bezier_v2"
            else RsuTrajectoryFusionNet
        )
        model_args = dict(
            **common_model_args,
            scan_dim=cfg.model.scan_dim,
            rsu_count=cfg.model.rsu_count,
            rsu_meta_dim=cfg.model.rsu_meta_dim,
            distance_decay_m=cfg.model.distance_decay_m,
            top_k_rsus=cfg.model.top_k_rsus,
        )
        if model_class is RsuTrajectoryFusionNet:
            for key in (
                "trajectory_anchor_count", "max_anchor_step_normalized",
                "max_anchor_heading_delta",
            ):
                model_args.pop(key)
    model = model_class(**model_args).to(device)

    pretrained = str(cfg.train.pretrained).strip()
    if pretrained:
        checkpoint = torch.load(Path(pretrained).expanduser(), map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        current = model.state_dict()
        compatible = {key: value for key, value in state_dict.items() if key in current and current[key].shape == value.shape}
        model.load_state_dict(compatible, strict=False)
        print(f"Loaded {len(compatible)}/{len(current)} compatible tensors from: {pretrained}")

    criterion = MultiTaskTrajectoryLoss(cfg.loss)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(cfg.train.run_name).strip() or f"h{cfg.data.history_len}_{timestamp}"
    save_dir = Path(cfg.train.save_dir).expanduser().resolve() / run_name
    log_dir = Path(cfg.train.log_dir).expanduser().resolve() / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    patience = 0
    with SummaryWriter(log_dir) as writer:
        for epoch in range(cfg.train.epochs):
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            val_loss = val_metrics["total"]
            print(
                f"Epoch {epoch + 1:03d}: train={train_metrics['total']:.5f} "
                f"val={val_loss:.5f} ADE={val_metrics['ade'] * cfg.data.trajectory_distance_scale:.3f}m "
                f"FDE={val_metrics['fde'] * cfg.data.trajectory_distance_scale:.3f}m "
                f"control={val_metrics['control']:.5f} mode={val_metrics['mode']:.5f}"
            )
            for name, value in train_metrics.items():
                writer.add_scalar(f"loss/train_{name}", value, epoch + 1)
            for name, value in val_metrics.items():
                writer.add_scalar(f"loss/val_{name}", value, epoch + 1)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "validation_loss": val_loss,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            torch.save(checkpoint, save_dir / "last_model.pth")
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save(checkpoint, save_dir / "best_model.pth")
            else:
                patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"Early stop: no improvement for {patience} epochs")
                break


def run_epoch(
    model: RsuTrajectoryFusionNet | BevBezierTrajectoryNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            if "bev" in batch:
                pred = model(batch["bev"], vehicle_state=batch.get("vehicle_state"))
            else:
                pred = model(
                    batch["ego_scans"], batch["rsu_scans"], batch["rsu_meta"],
                    vehicle_state=batch.get("vehicle_state"), rsu_mask=batch.get("rsu_mask"),
                )
            loss, metrics = criterion(pred, batch["target"], batch["trajectory"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value.item())
        count += 1
    return {name: value / max(1, count) for name, value in totals.items()}


if __name__ == "__main__":
    main()

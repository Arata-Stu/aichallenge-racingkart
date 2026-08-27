#!/usr/bin/env python3
"""Train TinyLiDARNet and save checkpoints consumed directly by the ROS 2 node."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lib.data import MultiSequenceDataset
from lib.loss import WeightedSmoothL1Loss
from lib.model import build_model, normalize_architecture


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, help="Train root or one sequence directory")
    parser.add_argument("--val-dir", required=True, help="Validation root or one sequence directory")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--architecture", choices=("normal", "small"), default="normal")
    parser.add_argument("--input-dim", type=int, default=1080)
    parser.add_argument("--max-range", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--acceleration-weight", type=float, default=1.0)
    parser.add_argument("--steering-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--pretrained", default="", help="Optional .pth checkpoint or state_dict")
    parser.add_argument("--dataset-version", default="default")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    name = requested.strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({requested!r}), but CUDA is unavailable")
    return device


def torch_load(path: str | Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_state_dict(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch_load(path, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError(f"Unsupported pretrained checkpoint: {path}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for scans, targets in loader:
            scans = scans.unsqueeze(1).to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(scans), targets)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = scans.shape[0]
            total_loss += float(loss.detach()) * batch_size
            sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("Dataset is empty")
    return total_loss / sample_count


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    validation_loss: float,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": normalize_architecture(args.architecture),
            "input_dim": args.input_dim,
            "output_dim": 2,
            "max_range": args.max_range,
            "target_names": ["acceleration", "steering"],
            "dataset_version": args.dataset_version,
            "epoch": epoch,
            "validation_loss": validation_loss,
        },
        path,
    )


def main() -> None:
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = MultiSequenceDataset(args.train_dir, args.input_dim, args.max_range)
    validation_dataset = MultiSequenceDataset(args.val_dir, args.input_dim, args.max_range)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    model = build_model(args.architecture, args.input_dim, 2).to(device)
    if args.pretrained:
        model.load_state_dict(load_state_dict(args.pretrained, device), strict=True)
    criterion = WeightedSmoothL1Loss(args.acceleration_weight, args.steering_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"device={device} train_samples={len(train_dataset)} val_samples={len(validation_dataset)}")
    print(f"checkpoints={output_dir} (ROS 2 loads these .pth files directly)")
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss = run_epoch(model, validation_loader, criterion, device, None)
        save_checkpoint(output_dir / "last_model.pth", model, optimizer, args, epoch, validation_loss)
        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, args, epoch, validation_loss)
        else:
            epochs_without_improvement += 1
        marker = " best" if improved else ""
        print(f"epoch={epoch:03d} train={train_loss:.6f} val={validation_loss:.6f}{marker}")
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early stopping: validation did not improve for {args.patience} epochs")
            break


if __name__ == "__main__":
    main()

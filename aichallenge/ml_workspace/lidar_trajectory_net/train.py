from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.data import MultiSeqLidarTrajectoryDataset
from lib.loss import TrajectoryLoss
from lib.runtime import build_model, load_checkpoint, make_checkpoint, resolve_device


def clean_numerical_tensor(x: torch.Tensor) -> torch.Tensor:
    if torch.isnan(x).any() or torch.isinf(x).any():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def make_dataset(cfg: DictConfig, root: str) -> MultiSeqLidarTrajectoryDataset:
    return MultiSeqLidarTrajectoryDataset(
        dataset_root=root,
        history_length=cfg.data.history_length,
        history_stride=cfg.data.history_stride,
        future_num_points=cfg.data.future_num_points,
        future_stride=cfg.data.future_stride,
        max_range=cfg.data.max_range,
        target_max_x=cfg.data.target_max_x,
        target_max_y=cfg.data.target_max_y,
        min_future_forward=cfg.data.min_future_forward,
    )


@hydra.main(config_path="./config", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------")

    device = resolve_device(cfg.train.device)
    print(f"Using device: {device}")

    train_dataset = make_dataset(cfg, cfg.data.train_dir)
    val_dataset = make_dataset(cfg, cfg.data.val_dir)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = build_model(cfg).to(device)

    if cfg.train.pretrained_path:
        state_dict, _, _ = load_checkpoint(cfg.train.pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded pretrained model from {cfg.train.pretrained_path}")

    criterion = TrajectoryLoss(
        point_weight=cfg.loss.point_weight,
        smoothness_weight=cfg.loss.smoothness_weight,
        end_point_weight=cfg.loss.end_point_weight,
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(cfg.train.save_dir).expanduser().resolve()
    log_dir = Path(cfg.train.log_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_dir / "train_config.yaml")

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = cfg.train.early_stop_patience
    best_path = save_dir / "best_model.pth"
    last_path = save_dir / "last_model.pth"

    with SummaryWriter(log_dir / timestamp) as writer:
        for epoch in range(cfg.train.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, cfg)
            val_loss = validate(model, val_loader, criterion, device, cfg)

            print(f"Epoch {epoch + 1:03d}: Train={train_loss:.5f} | Val={val_loss:.5f}")
            writer.add_scalar("Loss/train", train_loss, epoch + 1)
            writer.add_scalar("Loss/val", val_loss, epoch + 1)

            torch.save(make_checkpoint(model, cfg, epoch + 1, val_loss), last_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(make_checkpoint(model, cfg, epoch + 1, val_loss), best_path)
                print(f"[SAVE] Best model updated: {best_path} (val_loss={best_val_loss:.5f})")
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"[EarlyStop] No improvement for {max_patience} epochs.")
                break

    print("Training finished.")


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, cfg) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for scans, targets in tqdm(loader, desc=f"[Train] Epoch {epoch + 1}/{cfg.train.epochs}"):
        scans = clean_numerical_tensor(scans.to(device=device, dtype=torch.float32))
        targets = clean_numerical_tensor(targets.to(device=device, dtype=torch.float32))

        pred_path, control_points = model(scans, return_control_points=True)
        loss = criterion(pred_path, targets, control_points=control_points)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(model, loader, criterion, device, cfg) -> float:
    del cfg
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for scans, targets in tqdm(loader, desc="[Val]", leave=False):
            scans = clean_numerical_tensor(scans.to(device=device, dtype=torch.float32))
            targets = clean_numerical_tensor(targets.to(device=device, dtype=torch.float32))
            pred_path, control_points = model(scans, return_control_points=True)
            loss = criterion(pred_path, targets, control_points=control_points)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    main()

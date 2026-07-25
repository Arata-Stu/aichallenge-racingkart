from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import hydra

from lib.data import MultiSequenceRsuFusionDataset
from lib.model import DistanceGatedRsuFusionNet


def clean(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: clean(value.to(device)) for key, value in batch.items()}


@hydra.main(config_path="./config", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = MultiSequenceRsuFusionDataset(
        cfg.data.train_dir,
        history_len=cfg.data.history_len,
        max_range=cfg.data.max_range,
    )
    val_dataset = MultiSequenceRsuFusionDataset(
        cfg.data.val_dir,
        history_len=cfg.data.history_len,
        max_range=cfg.data.max_range,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DistanceGatedRsuFusionNet(
        scan_dim=cfg.model.scan_dim,
        rsu_count=cfg.model.rsu_count,
        rsu_meta_dim=cfg.model.rsu_meta_dim,
        vehicle_state_dim=cfg.model.vehicle_state_dim,
        output_dim=cfg.model.output_dim,
        frame_feature_dim=cfg.model.frame_feature_dim,
        temporal_hidden_dim=cfg.model.temporal_hidden_dim,
        fusion_hidden_dim=cfg.model.fusion_hidden_dim,
        distance_decay_m=cfg.model.distance_decay_m,
        top_k_rsus=cfg.model.top_k_rsus,
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(cfg.train.save_dir).expanduser().resolve() / f"h{cfg.data.history_len}_{timestamp}"
    log_dir = Path(cfg.train.log_dir).expanduser().resolve() / f"h{cfg.data.history_len}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    patience = 0
    with SummaryWriter(log_dir) as writer:
        for epoch in range(cfg.train.epochs):
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            print(f"Epoch {epoch + 1:03d}: train={train_loss:.5f} val={val_loss:.5f}")
            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("loss/val", val_loss, epoch + 1)

            torch.save(model.state_dict(), save_dir / "last_model.pth")
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save(model.state_dict(), save_dir / "best_model.pth")
            else:
                patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"Early stop: no improvement for {patience} epochs")
                break


def run_epoch(
    model: DistanceGatedRsuFusionNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    train: bool,
) -> float:
    model.train(train)
    total = 0.0
    count = 0
    desc = "train" if train else "val"
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            pred = model(
                batch["ego_scans"],
                batch["rsu_scans"],
                batch["rsu_meta"],
                vehicle_state=batch.get("vehicle_state"),
                rsu_mask=batch.get("rsu_mask"),
            )
            loss = criterion(pred, batch["target"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += float(loss.item())
        count += 1
    return total / max(1, count)


if __name__ == "__main__":
    main()

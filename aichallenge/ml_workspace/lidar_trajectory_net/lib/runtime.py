from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from lib.model import LidarTrajectoryNet


def resolve_device(device_name: str) -> torch.device:
    requested = (device_name or "auto").lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("device is set to 'cuda', but CUDA is not available.")
        return torch.device(requested)

    raise ValueError("device must be one of: 'auto', 'cpu', 'cuda'.")


def build_model(cfg: DictConfig) -> LidarTrajectoryNet:
    return LidarTrajectoryNet(
        input_channels=cfg.model.input_channels,
        input_dim=cfg.model.input_dim,
        history_length=cfg.data.history_length,
        embed_dim=cfg.model.embed_dim,
        conv_channels=list(cfg.model.conv_channels),
        conv_kernel_sizes=list(cfg.model.conv_kernel_sizes),
        conv_strides=list(cfg.model.conv_strides),
        transformer_layers=cfg.model.transformer_layers,
        transformer_heads=cfg.model.transformer_heads,
        transformer_ff_dim=cfg.model.transformer_ff_dim,
        dropout=cfg.model.dropout,
        num_control_points=cfg.model.num_control_points,
        num_future_points=cfg.data.future_num_points,
        output_scale=(cfg.model.output_scale_x, cfg.model.output_scale_y),
    )


def unpack_checkpoint(payload: Any) -> Tuple[Dict[str, torch.Tensor], Optional[DictConfig], Dict[str, Any]]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        checkpoint_cfg = payload.get("config")
        cfg = OmegaConf.create(checkpoint_cfg) if checkpoint_cfg is not None else None
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in ("model_state_dict", "config")
        }
        return payload["model_state_dict"], cfg, metadata

    if not isinstance(payload, dict):
        raise ValueError("Unsupported checkpoint format.")

    # Legacy checkpoints contain only model.state_dict().
    return payload, None, {"format_version": 0}


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: torch.device | str,
) -> Tuple[Dict[str, torch.Tensor], Optional[DictConfig], Dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path).expanduser(), map_location=map_location)
    return unpack_checkpoint(payload)


def make_checkpoint(
    model: torch.nn.Module,
    cfg: DictConfig,
    epoch: int,
    val_loss: float,
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "epoch": int(epoch),
        "val_loss": float(val_loss),
    }

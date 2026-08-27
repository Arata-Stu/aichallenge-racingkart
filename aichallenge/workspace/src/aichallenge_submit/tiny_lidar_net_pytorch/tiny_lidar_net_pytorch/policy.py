"""Checkpoint loading and framework-native PyTorch inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from tiny_lidar_net_pytorch.model import build_model, normalize_architecture


def resolve_device(requested: str) -> torch.device:
    name = requested.strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({requested!r}), but torch.cuda.is_available() is false")
    return device


def _torch_load(path: Path, map_location: torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class TinyLidarTorchPolicy:
    """Load a `.pth` checkpoint and run TinyLiDARNet directly with PyTorch."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        architecture: str | None = None,
        input_dim: int | None = None,
        output_dim: int | None = None,
        max_range: float | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"PyTorch checkpoint not found: {self.checkpoint_path}")
        self.device = resolve_device(device)
        checkpoint = _torch_load(self.checkpoint_path, self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            metadata = checkpoint
        elif isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            state_dict = checkpoint
            metadata = {}
        else:
            raise ValueError(
                "Unsupported checkpoint. Expected a state_dict or a dictionary containing model_state_dict."
            )

        self.architecture = normalize_architecture(
            architecture if architecture is not None else str(metadata.get("architecture", "normal"))
        )
        self.input_dim = int(input_dim if input_dim is not None else metadata.get("input_dim", 1080))
        self.output_dim = int(output_dim if output_dim is not None else metadata.get("output_dim", 2))
        self.max_range = float(max_range if max_range is not None else metadata.get("max_range", 30.0))
        if self.max_range <= 0.0:
            raise ValueError("max_range must be positive")

        self.model = build_model(self.architecture, self.input_dim, self.output_dim).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def preprocess(self, ranges: Sequence[float] | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(ranges):
            values = ranges.detach().to(device="cpu", dtype=torch.float32).clone()
        else:
            # torch.tensor intentionally copies read-only mmap arrays and ROS array fields.
            values = torch.tensor(ranges, dtype=torch.float32)
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError(f"ranges must be a non-empty 1D sequence, got shape {tuple(values.shape)}")
        values = torch.nan_to_num(values, nan=0.0, posinf=self.max_range, neginf=0.0)
        values = values.clamp_(0.0, self.max_range)
        if values.numel() != self.input_dim:
            values = F.interpolate(
                values.reshape(1, 1, -1),
                size=self.input_dim,
                mode="linear",
                align_corners=True,
            ).reshape(-1)
        return (values / self.max_range).reshape(1, 1, self.input_dim).to(self.device)

    def predict_tensor(self, ranges: Sequence[float] | torch.Tensor) -> torch.Tensor:
        inputs = self.preprocess(ranges)
        with torch.inference_mode():
            return self.model(inputs)[0]

    def predict(self, ranges: Sequence[float] | torch.Tensor) -> tuple[float, float]:
        output = self.predict_tensor(ranges).detach().to("cpu", dtype=torch.float32)
        if output.numel() < 2:
            raise RuntimeError(f"Model returned {output.numel()} values; expected acceleration and steering")
        return float(output[0]), float(output[1])

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

from lidar_trajectory_net_controller.model import LidarTrajectoryNet


class LidarTrajectoryNetCore:
    def __init__(
        self,
        ckpt_path: str,
        device: str = "auto",
        use_checkpoint_config: bool = True,
        input_channels: int = 3,
        input_dim: int = 1080,
        history_length: int = 8,
        history_stride: int = 1,
        future_num_points: int = 25,
        embed_dim: int = 128,
        conv_channels: Iterable[int] = (32, 64, 128),
        conv_kernel_sizes: Iterable[int] = (9, 7, 5),
        conv_strides: Iterable[int] = (4, 4, 2),
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 512,
        dropout: float = 0.1,
        num_control_points: int = 3,
        output_scale: Tuple[float, float] = (40.0, 12.0),
        max_range: float = 30.0,
    ):
        if not ckpt_path:
            raise ValueError("model.ckpt_path must be specified.")

        checkpoint_path = Path(ckpt_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.device = self._resolve_device(device)
        checkpoint = self._load_checkpoint(checkpoint_path)
        state_dict, checkpoint_config = self._unpack_checkpoint(checkpoint)

        parameters: Dict[str, Any] = {
            "input_channels": input_channels,
            "input_dim": input_dim,
            "history_length": history_length,
            "history_stride": history_stride,
            "future_num_points": future_num_points,
            "embed_dim": embed_dim,
            "conv_channels": list(conv_channels),
            "conv_kernel_sizes": list(conv_kernel_sizes),
            "conv_strides": list(conv_strides),
            "transformer_layers": transformer_layers,
            "transformer_heads": transformer_heads,
            "transformer_ff_dim": transformer_ff_dim,
            "dropout": dropout,
            "num_control_points": num_control_points,
            "output_scale_x": output_scale[0],
            "output_scale_y": output_scale[1],
            "max_range": max_range,
        }
        if use_checkpoint_config and checkpoint_config:
            self._apply_checkpoint_config(parameters, checkpoint_config)

        self.input_dim = int(parameters["input_dim"])
        self.history_length = int(parameters["history_length"])
        self.history_stride = int(parameters["history_stride"])
        self.future_num_points = int(parameters["future_num_points"])
        self.max_range = float(parameters["max_range"])

        if self.input_dim < 2:
            raise ValueError("model.input_dim must be at least 2.")
        if self.history_length < 1 or self.history_stride < 1:
            raise ValueError("history_length and history_stride must be positive.")
        if self.max_range <= 0.0:
            raise ValueError("max_range must be positive.")

        self.model = LidarTrajectoryNet(
            input_channels=int(parameters["input_channels"]),
            input_dim=self.input_dim,
            history_length=self.history_length,
            embed_dim=int(parameters["embed_dim"]),
            conv_channels=parameters["conv_channels"],
            conv_kernel_sizes=parameters["conv_kernel_sizes"],
            conv_strides=parameters["conv_strides"],
            transformer_layers=int(parameters["transformer_layers"]),
            transformer_heads=int(parameters["transformer_heads"]),
            transformer_ff_dim=int(parameters["transformer_ff_dim"]),
            dropout=float(parameters["dropout"]),
            num_control_points=int(parameters["num_control_points"]),
            num_future_points=self.future_num_points,
            output_scale=(
                float(parameters["output_scale_x"]),
                float(parameters["output_scale_y"]),
            ),
        ).to(self.device)

        normalized_state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
        self.model.load_state_dict(normalized_state_dict, strict=True)
        self.model.eval()

    @property
    def required_history_samples(self) -> int:
        return (self.history_length - 1) * self.history_stride + 1

    def preprocess_pair(
        self,
        free_ranges: np.ndarray,
        obstacle_ranges: np.ndarray,
    ) -> np.ndarray:
        free_scan = self._preprocess_ranges(free_ranges)
        obstacle_scan = self._preprocess_ranges(obstacle_ranges)
        diff = np.clip(free_scan - obstacle_scan, 0.0, self.max_range)
        return (
            np.stack([free_scan, obstacle_scan, diff], axis=0).astype(np.float32)
            / self.max_range
        )

    def predict(self, raw_history: np.ndarray) -> np.ndarray:
        if raw_history.ndim != 3 or raw_history.shape[1] != 3:
            raise ValueError(
                f"Expected raw history shape [N, 3, R], got {raw_history.shape}"
            )
        if len(raw_history) < self.required_history_samples:
            raise ValueError(
                f"Need {self.required_history_samples} history samples, "
                f"got {len(raw_history)}."
            )

        start = len(raw_history) - self.required_history_samples
        model_history = raw_history[start:: self.history_stride]
        if len(model_history) != self.history_length:
            raise RuntimeError(
                f"History selection produced {len(model_history)} samples, "
                f"expected {self.history_length}."
            )

        tensor = torch.from_numpy(model_history).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            prediction = self.model(tensor)[0]
        return prediction.detach().cpu().numpy()

    def _preprocess_ranges(self, ranges: np.ndarray) -> np.ndarray:
        ranges = np.asarray(ranges, dtype=np.float32)
        if ranges.ndim != 1 or len(ranges) < 2:
            raise ValueError(
                f"LaserScan ranges must be a 1D array with at least 2 rays, got {ranges.shape}."
            )
        ranges = np.nan_to_num(
            ranges,
            nan=0.0,
            posinf=self.max_range,
            neginf=0.0,
        )
        ranges = np.clip(ranges, 0.0, self.max_range)

        if len(ranges) != self.input_dim:
            source = np.linspace(0.0, 1.0, len(ranges), dtype=np.float32)
            target = np.linspace(0.0, 1.0, self.input_dim, dtype=np.float32)
            ranges = np.interp(target, source, ranges).astype(np.float32)
        return ranges

    def _load_checkpoint(self, path: Path):
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.device)

    @staticmethod
    def _unpack_checkpoint(checkpoint) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")

        if isinstance(checkpoint.get("model_state_dict"), dict):
            return checkpoint["model_state_dict"], checkpoint.get("config") or {}
        if isinstance(checkpoint.get("state_dict"), dict):
            return checkpoint["state_dict"], checkpoint.get("config") or {}
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint, {}
        raise ValueError("Checkpoint does not contain a PyTorch state_dict.")

    @staticmethod
    def _apply_checkpoint_config(
        parameters: Dict[str, Any],
        checkpoint_config: Dict[str, Any],
    ) -> None:
        data_config = checkpoint_config.get("data", {})
        model_config = checkpoint_config.get("model", {})

        for key in (
            "history_length",
            "history_stride",
            "future_num_points",
            "max_range",
        ):
            if key in data_config:
                parameters[key] = data_config[key]

        for key in (
            "input_channels",
            "input_dim",
            "embed_dim",
            "conv_channels",
            "conv_kernel_sizes",
            "conv_strides",
            "transformer_layers",
            "transformer_heads",
            "transformer_ff_dim",
            "dropout",
            "num_control_points",
            "output_scale_x",
            "output_scale_y",
        ):
            if key in model_config:
                parameters[key] = model_config[key]

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        requested = (device or "auto").lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cpu":
            return torch.device("cpu")
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "model.device is set to 'cuda', but CUDA is not available."
                )
            return torch.device(requested)
        raise ValueError("model.device must be one of: 'auto', 'cpu', 'cuda'.")

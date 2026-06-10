from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

logger = logging.getLogger(__name__)


class LidarTrajectorySequenceDataset(Dataset):
    """Dataset for one sequence of 3-channel virtual scans and ego poses."""

    def __init__(
        self,
        seq_dir: Union[str, Path],
        history_length: int = 8,
        history_stride: int = 1,
        future_num_points: int = 20,
        future_stride: int = 2,
        max_range: float = 30.0,
        target_max_x: float = 40.0,
        target_max_y: float = 20.0,
        min_future_forward: float = 0.0,
    ):
        self.seq_dir = Path(seq_dir)
        self.history_length = history_length
        self.history_stride = history_stride
        self.future_num_points = future_num_points
        self.future_stride = future_stride
        self.max_range = max_range
        self.target_max_x = target_max_x
        self.target_max_y = target_max_y
        self.min_future_forward = min_future_forward

        if history_length < 1 or future_num_points < 1:
            raise ValueError("history_length and future_num_points must be positive.")

        try:
            self.scans = np.load(self.seq_dir / "scan_inputs.npy")
            self.poses = np.load(self.seq_dir / "poses.npy")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing required .npy files in {self.seq_dir}: {e}")

        if self.scans.ndim != 3:
            raise ValueError(f"scan_inputs.npy must have shape [N, C, R], got {self.scans.shape}")
        if self.scans.shape[1] != 3:
            raise ValueError(f"scan_inputs.npy must contain 3 channels, got {self.scans.shape[1]}")
        if self.poses.ndim != 2 or self.poses.shape[1] != 3:
            raise ValueError(f"poses.npy must have shape [N, 3], got {self.poses.shape}")
        if len(self.scans) != len(self.poses):
            raise ValueError(
                f"Data length mismatch in {self.seq_dir}: scans={len(self.scans)}, poses={len(self.poses)}"
            )

        self.scans = np.clip(self.scans.astype(np.float32), 0.0, self.max_range) / self.max_range
        self.poses = self.poses.astype(np.float32)
        self.indices = self._build_valid_indices()

        if not self.indices:
            raise RuntimeError(f"No valid samples in {self.seq_dir}")

    def _build_valid_indices(self) -> List[int]:
        valid = []
        history_span = (self.history_length - 1) * self.history_stride
        future_span = self.future_num_points * self.future_stride

        for idx in range(history_span, len(self.scans) - future_span):
            target = self._build_target_path(idx)
            if self.min_future_forward > 0.0 and target[-1, 0] < self.min_future_forward:
                continue
            if np.any(np.abs(target[:, 0]) > self.target_max_x):
                continue
            if np.any(np.abs(target[:, 1]) > self.target_max_y):
                continue
            valid.append(idx)
        return valid

    def _build_target_path(self, idx: int) -> np.ndarray:
        x0, y0, yaw0 = self.poses[idx]
        cos_yaw = np.cos(yaw0)
        sin_yaw = np.sin(yaw0)

        points = []
        for i in range(1, self.future_num_points + 1):
            future_idx = idx + i * self.future_stride
            xf, yf, _ = self.poses[future_idx]
            dx = xf - x0
            dy = yf - y0
            ego_x = cos_yaw * dx + sin_yaw * dy
            ego_y = -sin_yaw * dx + cos_yaw * dy
            points.append([ego_x, ego_y])

        return np.asarray(points, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.indices[sample_idx]
        history_indices = [
            idx - (self.history_length - 1 - i) * self.history_stride
            for i in range(self.history_length)
        ]
        scan_history = self.scans[history_indices].astype(np.float32)
        target_path = self._build_target_path(idx)
        return scan_history, target_path


class MultiSeqLidarTrajectoryDataset(ConcatDataset):
    """Concatenates all valid lidar trajectory sequence directories."""

    def __init__(
        self,
        dataset_root: Union[str, Path],
        history_length: int = 8,
        history_stride: int = 1,
        future_num_points: int = 20,
        future_stride: int = 2,
        max_range: float = 30.0,
        target_max_x: float = 40.0,
        target_max_y: float = 20.0,
        min_future_forward: float = 0.0,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        dataset_root = Path(dataset_root)
        seq_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
        datasets = []

        for seq_dir in seq_dirs:
            name = seq_dir.name
            if include and not any(token in name for token in include):
                continue
            if exclude and any(token in name for token in exclude):
                continue
            if not ((seq_dir / "scan_inputs.npy").exists() and (seq_dir / "poses.npy").exists()):
                logger.warning("Skipping %s: missing scan_inputs.npy or poses.npy", seq_dir)
                continue

            try:
                datasets.append(
                    LidarTrajectorySequenceDataset(
                        seq_dir=seq_dir,
                        history_length=history_length,
                        history_stride=history_stride,
                        future_num_points=future_num_points,
                        future_stride=future_stride,
                        max_range=max_range,
                        target_max_x=target_max_x,
                        target_max_y=target_max_y,
                        min_future_forward=min_future_forward,
                    )
                )
            except Exception as e:
                logger.warning("Failed to load sequence %s: %s", seq_dir, e)

        if not datasets:
            raise RuntimeError(f"No valid sequences found in {dataset_root}")

        super().__init__(datasets)
        logger.info("Loaded %d sequences from %s. Total samples: %d", len(datasets), dataset_root, len(self))

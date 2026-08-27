"""Datasets for preprocessed TinyLiDARNet sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, Dataset


REQUIRED_FILES = ("scans.npy", "accelerations.npy", "steers.npy")


def find_sequence_directories(root: str | Path) -> list[Path]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root_path}")

    candidates: Iterable[Path]
    if all((root_path / name).is_file() for name in REQUIRED_FILES):
        candidates = [root_path]
    else:
        candidates = (path.parent for path in root_path.rglob("scans.npy"))
    sequence_dirs = sorted(
        {path for path in candidates if all((path / name).is_file() for name in REQUIRED_FILES)}
    )
    if not sequence_dirs:
        raise RuntimeError(f"No valid sequences containing {REQUIRED_FILES} found under {root_path}")
    return sequence_dirs


class ScanControlSequenceDataset(Dataset):
    """One sequence with targets ordered as [acceleration, steering]."""

    def __init__(self, sequence_dir: str | Path, input_dim: int = 1080, max_range: float = 30.0) -> None:
        self.sequence_dir = Path(sequence_dir).expanduser().resolve()
        self.input_dim = int(input_dim)
        self.max_range = float(max_range)
        if self.input_dim <= 0 or self.max_range <= 0.0:
            raise ValueError("input_dim and max_range must be positive")

        self.scans = np.load(self.sequence_dir / "scans.npy", mmap_mode="r")
        self.accelerations = np.load(self.sequence_dir / "accelerations.npy", mmap_mode="r")
        self.steers = np.load(self.sequence_dir / "steers.npy", mmap_mode="r")
        sample_count = len(self.scans)
        if self.scans.ndim != 2:
            raise ValueError(f"Expected scans with shape [N, points], got {self.scans.shape}")
        if len(self.accelerations) != sample_count or len(self.steers) != sample_count:
            raise ValueError(
                f"Length mismatch in {self.sequence_dir}: scans={sample_count}, "
                f"accelerations={len(self.accelerations)}, steers={len(self.steers)}"
            )

    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        scan = torch.from_numpy(np.array(self.scans[index], dtype=np.float32, copy=True))
        scan = torch.nan_to_num(scan, nan=0.0, posinf=self.max_range, neginf=0.0)
        scan = scan.clamp_(0.0, self.max_range)
        if scan.numel() != self.input_dim:
            scan = F.interpolate(
                scan.reshape(1, 1, -1), size=self.input_dim, mode="linear", align_corners=True
            ).reshape(-1)
        scan = scan / self.max_range
        target = torch.tensor(
            [float(self.accelerations[index]), float(self.steers[index])], dtype=torch.float32
        )
        return scan, torch.nan_to_num(target)


class MultiSequenceDataset(ConcatDataset):
    """All valid sequences found below a train or validation directory."""

    def __init__(self, root: str | Path, input_dim: int = 1080, max_range: float = 30.0) -> None:
        self.sequence_directories = find_sequence_directories(root)
        super().__init__(
            [
                ScanControlSequenceDataset(path, input_dim=input_dim, max_range=max_range)
                for path in self.sequence_directories
            ]
        )

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


class RsuFusionSequenceDataset(Dataset):
    """
    Dataset for synchronized ego/RSU LaserScan histories.

    Required files in each sequence directory:
      - ego_scans.npy: [N, R]
      - rsu_scans.npy: [N, S, R]
      - rsu_meta.npy: [N, S, M]
      - targets.npy: [N, C]

    Optional files:
      - vehicle_state.npy: [N, V]
      - rsu_mask.npy: [N, S]
    """

    def __init__(
        self,
        seq_dir: str | Path,
        history_len: int = 5,
        max_range: float = 45.0,
        include_current: bool = True,
    ) -> None:
        self.seq_dir = Path(seq_dir)
        self.history_len = int(history_len)
        self.max_range = float(max_range)
        self.include_current = bool(include_current)

        if self.history_len < 1:
            raise ValueError("history_len must be >= 1")

        self.ego_scans = self._load_scan_array("ego_scans.npy", dims=2)
        self.rsu_scans = self._load_scan_array("rsu_scans.npy", dims=3)
        self.rsu_meta = self._load_array("rsu_meta.npy").astype(np.float32)
        self.targets = self._load_array("targets.npy").astype(np.float32)
        self.vehicle_state = self._load_optional("vehicle_state.npy")
        self.rsu_mask = self._load_optional("rsu_mask.npy")

        n = len(self.ego_scans)
        for name, array in {
            "rsu_scans": self.rsu_scans,
            "rsu_meta": self.rsu_meta,
            "targets": self.targets,
        }.items():
            if len(array) != n:
                raise ValueError(f"{name} length mismatch in {self.seq_dir}: {len(array)} != {n}")

        if self.vehicle_state is not None and len(self.vehicle_state) != n:
            raise ValueError("vehicle_state length mismatch")
        if self.rsu_mask is not None and len(self.rsu_mask) != n:
            raise ValueError("rsu_mask length mismatch")

        self.start_index = self.history_len - 1 if self.include_current else self.history_len
        self.valid_len = max(0, n - self.start_index)

    def _load_array(self, name: str) -> np.ndarray:
        path = self.seq_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        return np.load(path)

    def _load_optional(self, name: str) -> Optional[np.ndarray]:
        path = self.seq_dir / name
        return np.load(path).astype(np.float32) if path.exists() else None

    def _load_scan_array(self, name: str, dims: int) -> np.ndarray:
        array = self._load_array(name).astype(np.float32)
        if array.ndim != dims:
            raise ValueError(f"{name} must have {dims} dims, got {array.shape}")
        array = np.nan_to_num(array, nan=self.max_range, posinf=self.max_range, neginf=self.max_range)
        return np.clip(array, 0.0, self.max_range) / self.max_range

    def __len__(self) -> int:
        return self.valid_len

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        end = index + self.start_index + 1
        start = end - self.history_len

        item = {
            "ego_scans": torch.from_numpy(self.ego_scans[start:end]),
            "rsu_scans": torch.from_numpy(self.rsu_scans[start:end]).permute(1, 0, 2),
            "rsu_meta": torch.from_numpy(self.rsu_meta[end - 1]),
            "target": torch.from_numpy(self.targets[end - 1]),
        }
        if self.vehicle_state is not None:
            item["vehicle_state"] = torch.from_numpy(self.vehicle_state[end - 1])
        if self.rsu_mask is not None:
            item["rsu_mask"] = torch.from_numpy(self.rsu_mask[end - 1].astype(np.bool_))
        return item


class MultiSequenceRsuFusionDataset(ConcatDataset):
    def __init__(self, root_dir: str | Path, history_len: int = 5, max_range: float = 45.0) -> None:
        root = Path(root_dir)
        seq_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        datasets = [
            RsuFusionSequenceDataset(path, history_len=history_len, max_range=max_range)
            for path in seq_dirs
            if (path / "ego_scans.npy").exists()
        ]
        if not datasets:
            raise RuntimeError(f"No valid RSU fusion sequences found in {root}")
        super().__init__(datasets)

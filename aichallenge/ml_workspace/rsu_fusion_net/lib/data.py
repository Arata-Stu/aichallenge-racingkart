from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

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
        acceleration_scale: float = 2.0,
        include_current: bool = True,
    ) -> None:
        self.seq_dir = Path(seq_dir)
        self.history_len = int(history_len)
        self.max_range = float(max_range)
        self.acceleration_scale = float(acceleration_scale)
        self.include_current = bool(include_current)

        if self.history_len < 1:
            raise ValueError("history_len must be >= 1")
        if self.acceleration_scale <= 0.0:
            raise ValueError("acceleration_scale must be positive")

        self.ego_scans = self._load_scan_array("ego_scans.npy", dims=2)
        self.rsu_scans = self._load_scan_array("rsu_scans.npy", dims=3)
        self.rsu_meta = self._load_array("rsu_meta.npy").astype(np.float32)
        self.targets = self._load_array("targets.npy").astype(np.float32)
        self.targets[:, 0] = np.clip(
            self.targets[:, 0] / self.acceleration_scale, -1.0, 1.0
        )
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


class RsuTrajectorySequenceDataset(RsuFusionSequenceDataset):
    """RSU fusion dataset with ego-frame future position and speed targets."""

    def __init__(
        self,
        seq_dir: str | Path,
        history_len: int = 5,
        max_range: float = 45.0,
        acceleration_scale: float = 2.0,
        trajectory_steps: int = 20,
        trajectory_dt: float = 0.25,
        trajectory_distance_scale: float = 50.0,
        max_speed: float = 15.0,
    ) -> None:
        super().__init__(
            seq_dir,
            history_len=history_len,
            max_range=max_range,
            acceleration_scale=acceleration_scale,
        )
        if trajectory_steps < 1 or trajectory_dt <= 0.0:
            raise ValueError("trajectory_steps and trajectory_dt must be positive")
        if trajectory_distance_scale <= 0.0 or max_speed <= 0.0:
            raise ValueError("trajectory_distance_scale and max_speed must be positive")
        self.trajectory_steps = int(trajectory_steps)
        self.trajectory_dt = float(trajectory_dt)
        self.trajectory_distance_scale = float(trajectory_distance_scale)
        self.max_speed = float(max_speed)
        self.ego_poses = self._load_array("ego_poses.npy").astype(np.float64)
        self.timestamps_ns = self._load_array("timestamps_ns.npy").astype(np.int64)
        if self.vehicle_state is None:
            raise FileNotFoundError(f"Missing {self.seq_dir / 'vehicle_state.npy'}")
        n = len(self.ego_scans)
        if len(self.ego_poses) != n or len(self.timestamps_ns) != n:
            raise ValueError("ego pose/timestamp length mismatch")
        if self.ego_poses.shape != (n, 3):
            raise ValueError(f"ego_poses.npy must be [N, 3], got {self.ego_poses.shape}")
        if np.any(np.diff(self.timestamps_ns) < 0):
            raise ValueError("timestamps_ns.npy must be monotonic")
        horizon_ns = int(round(self.trajectory_steps * self.trajectory_dt * 1e9))
        last_time = int(self.timestamps_ns[-1]) if n else 0
        self.sample_indices = np.asarray(
            [
                index
                for index in range(self.start_index, n)
                if int(self.timestamps_ns[index]) + horizon_ns <= last_time
            ],
            dtype=np.int64,
        )
        self.valid_len = int(len(self.sample_indices))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        current = int(self.sample_indices[index])
        base_index = current - self.start_index
        item = super().__getitem__(base_index)
        pose = self.ego_poses[current]
        future_times = self.timestamps_ns[current] + (
            np.arange(1, self.trajectory_steps + 1, dtype=np.float64)
            * self.trajectory_dt * 1e9
        )
        future_indices = np.searchsorted(self.timestamps_ns, future_times, side="left")
        future_indices = np.clip(future_indices, current + 1, len(self.timestamps_ns) - 1)
        future_pose = self.ego_poses[future_indices]
        dx = future_pose[:, 0] - pose[0]
        dy = future_pose[:, 1] - pose[1]
        cos_yaw = np.cos(pose[2])
        sin_yaw = np.sin(pose[2])
        forward = cos_yaw * dx + sin_yaw * dy
        left = -sin_yaw * dx + cos_yaw * dy
        speed = np.asarray(self.vehicle_state[future_indices, 0], dtype=np.float32)
        trajectory = np.stack(
            [
                np.clip(forward / self.trajectory_distance_scale, -1.0, 1.0),
                np.clip(left / self.trajectory_distance_scale, -1.0, 1.0),
                np.clip(speed / self.max_speed, 0.0, 1.0),
            ],
            axis=-1,
        ).astype(np.float32)
        item["vehicle_state"] = torch.from_numpy(
            np.clip(
                np.asarray(self.vehicle_state[current], dtype=np.float32) / self.max_speed,
                -1.0, 1.0,
            )
        )
        item["trajectory"] = torch.from_numpy(trajectory)
        item["sample_index"] = torch.tensor(current, dtype=torch.int64)
        return item


class BevTrajectorySequenceDataset(Dataset):
    """Memory-efficient semantic-BEV history with future trajectory targets."""

    def __init__(
        self,
        seq_dir: str | Path,
        history_len: int = 5,
        acceleration_scale: float = 2.0,
        trajectory_steps: int = 20,
        trajectory_dt: float = 0.25,
        trajectory_distance_scale: float = 50.0,
        max_speed: float = 15.0,
        bev_channels: Sequence[int] = (0, 1, 2, 3, 4, 5),
        bev_height: int | None = None,
        bev_width: int | None = None,
    ) -> None:
        self.seq_dir = Path(seq_dir)
        self.history_len = int(history_len)
        self.acceleration_scale = float(acceleration_scale)
        self.trajectory_steps = int(trajectory_steps)
        self.trajectory_dt = float(trajectory_dt)
        self.trajectory_distance_scale = float(trajectory_distance_scale)
        self.max_speed = float(max_speed)
        self.bev_channels = tuple(int(value) for value in bev_channels)
        self.bev_height = int(bev_height) if bev_height is not None else None
        self.bev_width = int(bev_width) if bev_width is not None else None
        if self.history_len < 1 or self.trajectory_steps < 1 or self.trajectory_dt <= 0.0:
            raise ValueError("history_len, trajectory_steps and trajectory_dt must be positive")
        if self.acceleration_scale <= 0.0 or self.trajectory_distance_scale <= 0.0 or self.max_speed <= 0.0:
            raise ValueError("normalization scales must be positive")
        if not self.bev_channels or len(set(self.bev_channels)) != len(self.bev_channels):
            raise ValueError("bev_channels must be a non-empty list of unique channels")
        if min(self.bev_channels) < 0 or max(self.bev_channels) > 7:
            raise ValueError("BEV channel indices must be between 0 and 7")
        if (self.bev_height is not None and self.bev_height < 1) or (
            self.bev_width is not None and self.bev_width < 1
        ):
            raise ValueError("BEV target height and width must be positive")

        self.bev_frames = np.load(self.seq_dir / "bev_frames.npy", mmap_mode="r")
        if self.bev_frames.ndim != 3 or self.bev_frames.dtype != np.uint8:
            raise ValueError(f"bev_frames.npy must be packed uint8 [N,H,W], got {self.bev_frames.shape}")
        self.targets = np.asarray(np.load(self.seq_dir / "targets.npy"), dtype=np.float32)
        self.targets[:, 0] = np.clip(self.targets[:, 0] / self.acceleration_scale, -1.0, 1.0)
        self.ego_poses = np.asarray(np.load(self.seq_dir / "ego_poses.npy"), dtype=np.float64)
        self.timestamps_ns = np.asarray(np.load(self.seq_dir / "timestamps_ns.npy"), dtype=np.int64)
        self.vehicle_state = np.asarray(np.load(self.seq_dir / "vehicle_state.npy"), dtype=np.float32)
        count = len(self.bev_frames)
        if any(len(array) != count for array in (
            self.targets, self.ego_poses, self.timestamps_ns, self.vehicle_state
        )):
            raise ValueError(f"BEV sequence arrays have different lengths in {self.seq_dir}")
        if self.ego_poses.shape != (count, 3):
            raise ValueError(f"ego_poses.npy must be [N,3], got {self.ego_poses.shape}")
        if np.any(np.diff(self.timestamps_ns) < 0):
            raise ValueError("timestamps_ns.npy must be monotonic")

        self.start_index = self.history_len - 1
        horizon_ns = int(round(self.trajectory_steps * self.trajectory_dt * 1e9))
        last_time = int(self.timestamps_ns[-1]) if count else 0
        self.sample_indices = np.asarray([
            index for index in range(self.start_index, count)
            if int(self.timestamps_ns[index]) + horizon_ns <= last_time
        ], dtype=np.int64)
        self._channel_bits = np.asarray(self.bev_channels, dtype=np.uint8)

    def __len__(self) -> int:
        return int(len(self.sample_indices))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        current = int(self.sample_indices[index])
        packed = np.asarray(
            self.bev_frames[current - self.history_len + 1:current + 1], dtype=np.uint8
        )
        target_height = self.bev_height or packed.shape[1]
        target_width = self.bev_width or packed.shape[2]
        if packed.shape[1:] != (target_height, target_width):
            # Coordinates start at the configured x_min/y_min, so copying from
            # index zero crops only the removed far-forward/far-left extent.
            normalized = np.zeros(
                (packed.shape[0], target_height, target_width), dtype=np.uint8
            )
            copy_height = min(target_height, packed.shape[1])
            copy_width = min(target_width, packed.shape[2])
            normalized[:, :copy_height, :copy_width] = packed[:, :copy_height, :copy_width]
            packed = normalized
        channels = ((packed[..., None] >> self._channel_bits) & 1).astype(np.float32)
        bev = np.ascontiguousarray(channels.transpose(0, 3, 1, 2))

        pose = self.ego_poses[current]
        future_times = self.timestamps_ns[current] + (
            np.arange(1, self.trajectory_steps + 1, dtype=np.float64)
            * self.trajectory_dt * 1e9
        )
        future_indices = np.searchsorted(self.timestamps_ns, future_times, side="left")
        future_indices = np.clip(future_indices, current + 1, len(self.timestamps_ns) - 1)
        future_pose = self.ego_poses[future_indices]
        dx, dy = future_pose[:, 0] - pose[0], future_pose[:, 1] - pose[1]
        cosine, sine = np.cos(pose[2]), np.sin(pose[2])
        forward = cosine * dx + sine * dy
        left = -sine * dx + cosine * dy
        speed = np.asarray(self.vehicle_state[future_indices, 0], dtype=np.float32)
        trajectory = np.stack((
            np.clip(forward / self.trajectory_distance_scale, -1.0, 1.0),
            np.clip(left / self.trajectory_distance_scale, -1.0, 1.0),
            np.clip(speed / self.max_speed, 0.0, 1.0),
        ), axis=-1).astype(np.float32)
        vehicle_state = np.clip(
            np.asarray(self.vehicle_state[current], dtype=np.float32) / self.max_speed,
            -1.0, 1.0,
        )
        return {
            "bev": torch.from_numpy(bev),
            "target": torch.from_numpy(self.targets[current]),
            "vehicle_state": torch.from_numpy(vehicle_state),
            "trajectory": torch.from_numpy(trajectory),
            "sample_index": torch.tensor(current, dtype=torch.int64),
        }


class MultiSequenceRsuFusionDataset(ConcatDataset):
    def __init__(self, root_dir: str | Path, history_len: int = 5, max_range: float = 45.0, acceleration_scale: float = 2.0) -> None:
        root = Path(root_dir)
        seq_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        datasets = [
            RsuFusionSequenceDataset(path, history_len=history_len, max_range=max_range, acceleration_scale=acceleration_scale)
            for path in seq_dirs
            if (path / "ego_scans.npy").exists()
        ]
        if not datasets:
            raise RuntimeError(f"No valid RSU fusion sequences found in {root}")
        super().__init__(datasets)


class MultiSequenceRsuTrajectoryDataset(ConcatDataset):
    def __init__(
        self,
        root_dir: str | Path,
        history_len: int = 5,
        max_range: float = 45.0,
        acceleration_scale: float = 2.0,
        trajectory_steps: int = 20,
        trajectory_dt: float = 0.25,
        trajectory_distance_scale: float = 50.0,
        max_speed: float = 15.0,
    ) -> None:
        root = Path(root_dir)
        seq_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        datasets = []
        missing = []
        trajectory_files = ("ego_poses.npy", "timestamps_ns.npy", "vehicle_state.npy")
        for path in seq_dirs:
            if not (path / "ego_scans.npy").exists():
                continue
            absent = [name for name in trajectory_files if not (path / name).is_file()]
            if absent:
                missing.append(f"{path.name}: {', '.join(absent)}")
                continue
            dataset = RsuTrajectorySequenceDataset(
                path,
                history_len=history_len,
                max_range=max_range,
                acceleration_scale=acceleration_scale,
                trajectory_steps=trajectory_steps,
                trajectory_dt=trajectory_dt,
                trajectory_distance_scale=trajectory_distance_scale,
                max_speed=max_speed,
            )
            if len(dataset):
                datasets.append(dataset)
        if not datasets:
            detail = f" Missing trajectory files: {'; '.join(missing)}" if missing else ""
            raise RuntimeError(f"No trajectory-ready RSU sequences found in {root}.{detail}")
        super().__init__(datasets)


class MultiSequenceBevTrajectoryDataset(ConcatDataset):
    def __init__(self, root_dir: str | Path, **dataset_args: object) -> None:
        root = Path(root_dir)
        datasets = [
            BevTrajectorySequenceDataset(path, **dataset_args)
            for path in sorted(root.iterdir())
            if path.is_dir() and (path / "bev_frames.npy").is_file()
        ]
        if not datasets:
            raise RuntimeError(f"No valid BEV trajectory sequences found in {root}")
        super().__init__(datasets)

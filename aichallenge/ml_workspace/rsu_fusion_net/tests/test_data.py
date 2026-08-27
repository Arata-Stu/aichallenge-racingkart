from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.data import BevTrajectorySequenceDataset, RsuFusionSequenceDataset, RsuTrajectorySequenceDataset  # noqa: E402


def test_acceleration_is_normalized_to_tanh_range(tmp_path: Path) -> None:
    np.save(tmp_path / "ego_scans.npy", np.ones((2, 8), dtype=np.float32))
    np.save(tmp_path / "rsu_scans.npy", np.ones((2, 6, 8), dtype=np.float32))
    np.save(tmp_path / "rsu_meta.npy", np.zeros((2, 6, 5), dtype=np.float32))
    np.save(tmp_path / "targets.npy", np.asarray([[2.0, 0.5], [-2.0, -0.5]], dtype=np.float32))
    dataset = RsuFusionSequenceDataset(tmp_path, history_len=1, acceleration_scale=2.0)
    assert dataset[0]["target"].tolist() == pytest.approx([1.0, 0.5])
    assert dataset[1]["target"].tolist() == pytest.approx([-1.0, -0.5])


def test_future_trajectory_is_transformed_to_current_ego_frame(tmp_path: Path) -> None:
    count = 8
    np.save(tmp_path / "ego_scans.npy", np.ones((count, 64), dtype=np.float32))
    np.save(tmp_path / "rsu_scans.npy", np.ones((count, 2, 64), dtype=np.float32))
    np.save(tmp_path / "rsu_meta.npy", np.zeros((count, 2, 5), dtype=np.float32))
    np.save(tmp_path / "targets.npy", np.zeros((count, 2), dtype=np.float32))
    # Vehicle faces +Y, so future map +Y becomes ego-frame forward +X.
    poses = np.column_stack((np.zeros(count), np.arange(count), np.full(count, np.pi / 2)))
    np.save(tmp_path / "ego_poses.npy", poses)
    np.save(tmp_path / "timestamps_ns.npy", np.arange(count, dtype=np.int64) * 1_000_000_000)
    np.save(tmp_path / "vehicle_state.npy", np.full((count, 1), 2.0, dtype=np.float32))
    dataset = RsuTrajectorySequenceDataset(
        tmp_path, history_len=2, trajectory_steps=2, trajectory_dt=1.0,
        trajectory_distance_scale=10.0, max_speed=10.0,
    )
    sample = dataset[0]
    assert sample["sample_index"].item() == 1
    assert np.allclose(
        sample["trajectory"].numpy(),
        [[0.1, 0.0, 0.2], [0.2, 0.0, 0.2]], atol=1e-5,
    )


def test_bev_dataset_unpacks_selected_channels_and_history(tmp_path: Path) -> None:
    count = 6
    frames = np.zeros((count, 3, 4), dtype=np.uint8)
    frames[:, 1, 2] = (1 << 0) | (1 << 3) | (1 << 7)
    np.save(tmp_path / "bev_frames.npy", frames)
    np.save(tmp_path / "targets.npy", np.tile([2.0, 0.1], (count, 1)).astype(np.float32))
    np.save(tmp_path / "ego_poses.npy", np.column_stack((np.arange(count), np.zeros(count), np.zeros(count))))
    np.save(tmp_path / "timestamps_ns.npy", np.arange(count, dtype=np.int64) * 1_000_000_000)
    np.save(tmp_path / "vehicle_state.npy", np.full((count, 1), 2.0, dtype=np.float32))
    dataset = BevTrajectorySequenceDataset(
        tmp_path, history_len=2, trajectory_steps=2, trajectory_dt=1.0,
        trajectory_distance_scale=10.0, max_speed=10.0, bev_channels=(0, 3),
        bev_height=3, bev_width=3,
    )
    sample = dataset[0]
    assert sample["bev"].shape == (2, 2, 3, 3)
    assert sample["bev"][:, :, 1, 2].tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert sample["target"].tolist() == pytest.approx([1.0, 0.1])
    assert np.allclose(
        sample["trajectory"].numpy(), [[0.1, 0.0, 0.2], [0.2, 0.0, 0.2]]
    )

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess_bag_to_npy import (  # noqa: E402
    dynamic_rsu_meta, pack_bev_image, parse_rsu_poses, pose_speed, synchronize,
)


def test_dynamic_rsu_meta_uses_ego_frame():
    poses = parse_rsu_poses("13,24,90", 1)
    meta = dynamic_rsu_meta(
        np.asarray([10.0, 20.0, math.pi / 2.0]), poses, 0, np.zeros((1, 4), dtype=np.float32)
    )
    assert np.allclose(meta[:3], [5.0, 4.0, -3.0], atol=1e-5)
    assert abs(meta[3]) < 1e-5


def test_dynamic_rsu_meta_falls_back_without_pose():
    fallback = np.asarray([[8.0, 1.0, 2.0, 0.3]], dtype=np.float32)
    assert np.array_equal(dynamic_rsu_meta(None, None, 0, fallback), fallback[0])


def test_pose_speed_uses_timestamp_delta():
    poses = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    timestamps = np.asarray([0, 1_000_000_000, 2_000_000_000], dtype=np.int64)
    assert pose_speed(poses, timestamps).tolist() == [2.0, 2.0, 2.0]


def test_pack_bev_image_preserves_all_semantic_bits():
    class Image:
        encoding = "8UC8"
        height = 1
        width = 2
        step = 18  # Includes two padding bytes.
        data = bytes([255, 0, 255, 0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 255, 9, 9])

    packed = pack_bev_image(Image())
    assert packed.shape == (1, 2)
    assert packed.tolist() == [[0b00000101, 0b10001010]]


def test_synchronize_keeps_bev_aligned_with_training_targets():
    args = SimpleNamespace(
        ego_scan_topic="/scan", control_topic="/control", rsu_scan_topics=["/rsu"],
        rsu_meta="", rsu_poses="", max_sync_dt=0.1, require_bev=True,
        bev_topic="/bev", timestamp_source="bag",
    )
    times = [1_000_000_000, 1_050_000_000]
    streams = {
        "ego": [(stamp, np.ones(4, dtype=np.float32)) for stamp in times],
        "controls": [(stamp, np.asarray([0.5, 0.1], dtype=np.float32)) for stamp in times],
        "rsus": [[(stamp, np.ones(4, dtype=np.float32)) for stamp in times]],
        "poses": [(stamp, np.asarray([index, 0.0, 0.0])) for index, stamp in enumerate(times)],
        "velocities": [(stamp, 1.0) for stamp in times],
        "bev": [(stamp, np.full((2, 3), index + 1, dtype=np.uint8)) for index, stamp in enumerate(times)],
    }
    result = synchronize(args, streams)
    assert result["bev_frames"].shape == (2, 2, 3)
    assert result["bev_frames"][:, 0, 0].tolist() == [1, 2]
    assert result["targets"].shape == (2, 2)

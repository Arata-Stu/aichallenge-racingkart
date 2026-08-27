from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard_backend as dashboard  # noqa: E402


class FakeManager:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run_command(self, job, command, cwd):
        self.commands.append(command)
        return 0


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.config = dashboard.Config(root / "record", root / "datasets", root / "checkpoints")
        for path in (self.config.record_root, self.config.dataset_root, self.config.checkpoint_root):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_sequence(self, split: str, version: str = "default") -> Path:
        path = dashboard.version_root(self.config, version) / split / "seq"
        path.mkdir(parents=True)
        np.save(path / "ego_scans.npy", np.ones((4, 8), dtype=np.float32))
        np.save(path / "rsu_scans.npy", np.ones((4, 6, 8), dtype=np.float32) * 2)
        np.save(path / "rsu_meta.npy", np.zeros((4, 6, 5), dtype=np.float32))
        np.save(path / "targets.npy", np.array([[0.6, -0.2], [0.6, 0], [0.6, 0.1], [0.6, 0.2]], dtype=np.float32))
        np.save(path / "rsu_mask.npy", np.ones((4, 6), dtype=np.bool_))
        np.save(path / "ego_poses.npy", np.column_stack((np.arange(4), np.zeros(4), np.zeros(4))))
        np.save(path / "timestamps_ns.npy", np.arange(4, dtype=np.int64) * 1_000_000_000)
        np.save(path / "vehicle_state.npy", np.ones((4, 1), dtype=np.float32))
        np.save(path / "bev_frames.npy", np.zeros((4, 8, 10), dtype=np.uint8))
        return path

    def test_sequence_detail_and_frame(self) -> None:
        self.make_sequence("train")
        detail = dashboard.sequence_detail(self.config, "default", "train", "seq")
        frame = dashboard.sequence_frame(self.config, "default", "train", "seq", 1)
        self.assertEqual(detail["rsu_count"], 6)
        self.assertEqual(detail["availability"], [1.0] * 6)
        self.assertEqual(len(detail["target"]["steer_histogram"]), 31)
        self.assertEqual(len(detail["target"]["accel_histogram"]), 25)
        self.assertEqual(len(frame["rsus"]), 6)
        self.assertEqual(frame["sensor_fov_deg"], [270.0, 135.0, 135.0, 150.0, 180.0, 150.0, 150.0])
        self.assertAlmostEqual(frame["steering"], 0.0)

    def test_preprocess_supports_both_splits_and_obstacle_free_scan(self) -> None:
        bag = self.config.record_root / "day" / "bag"
        bag.mkdir(parents=True)
        (bag / "metadata.yaml").touch()
        manager = FakeManager()
        worker = dashboard.preprocess_worker(
            self.config,
            {"train": ["day/bag"], "val": ["day/bag"], "ego_scan_topic": "/sensing/lidar/scan_without_obstacles"},
            manager,
        )
        self.assertEqual(worker(dashboard.common.Job("preprocess")), 0)
        self.assertEqual(len(manager.commands), 2)
        self.assertIn("/sensing/lidar/scan_without_obstacles", manager.commands[0])
        self.assertIn("--rsu-poses", manager.commands[0])
        self.assertEqual(manager.commands[0][manager.commands[0].index("--timestamp-source") + 1], "bag")

    def test_training_learns_acceleration_and_steering_and_is_versioned(self) -> None:
        self.make_sequence("train", "rsu-v1")
        self.make_sequence("val", "rsu-v1")
        manager = FakeManager()
        worker = dashboard.training_worker(
            self.config,
            {"dataset_version": "rsu-v1", "history_len": 3, "epochs": 1},
            manager,
        )
        self.assertEqual(worker(dashboard.common.Job("train")), 0)
        command = manager.commands[0]
        self.assertIn("data.history_len=3", command)
        self.assertIn("loss.acceleration_weight=1.0", command)
        self.assertIn("loss.steering_weight=1.0", command)
        self.assertIn("model.trajectory_modes=4", command)
        self.assertIn("model.trajectory_anchor_count=4", command)
        self.assertIn("data.trajectory_steps=12", command)
        self.assertIn("loss.average_displacement_weight=1.0", command)
        self.assertIn("loss.endpoint_weight=1.5", command)
        self.assertTrue(any(arg.startswith("model.max_anchor_step_normalized=") for arg in command))
        self.assertTrue(any("checkpoints/versions/rsu-v1" in arg for arg in command))

    def test_bev_preprocess_and_training_are_selectable(self) -> None:
        bag = self.config.record_root / "bag"
        bag.mkdir(parents=True)
        (bag / "metadata.yaml").touch()
        manager = FakeManager()
        preprocess = dashboard.preprocess_worker(
            self.config, {"train": ["bag"], "input_mode": "bev"}, manager
        )
        self.assertEqual(preprocess(dashboard.common.Job("preprocess")), 0)
        self.assertIn("--require-bev", manager.commands[0])

        self.make_sequence("train")
        self.make_sequence("val")
        training = dashboard.training_worker(
            self.config, {"input_mode": "bev", "epochs": 1}, manager
        )
        self.assertEqual(training(dashboard.common.Job("train")), 0)
        self.assertIn("model.architecture=bev_trajectory_bezier_v1", manager.commands[-1])

    def test_evaluation_prediction_frame(self) -> None:
        self.make_sequence("val")
        evaluation = self.config.evaluation_root / "run"
        sequence_dir = evaluation / "sequences"
        sequence_dir.mkdir(parents=True)
        report = {
            "version": "default", "split": "val", "metrics": {"ade_m": 0.5},
            "trajectory": {"steps": 2, "modes": 2, "dt": 0.25},
            "sequences": [{"id": "seq", "file": "sequence_0000.npz", "metrics": {"ade_m": 0.4}}],
        }
        (evaluation / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
        np.savez_compressed(
            sequence_dir / "sequence_0000.npz", sample_indices=np.asarray([2]),
            trajectories=np.zeros((1, 2, 2, 3)), target_trajectories=np.zeros((1, 2, 3)),
            mode_probabilities=np.asarray([[0.25, 0.75]]), controls=np.asarray([[0.5, 0.1]]),
            target_controls=np.asarray([[0.4, 0.0]]), gates=np.ones((1, 6)),
        )
        result = dashboard.evaluation_frame(self.config, "run", "seq", 2)
        self.assertTrue(result["available"])
        self.assertEqual(result["selected_mode"], 1)
        self.assertEqual(result["ego_pose"], [2.0, 0.0, 0.0])
        self.assertEqual(result["map_trajectories"][0][0][:2], [2.0, 0.0])
        self.assertEqual(len(dashboard.evaluations(self.config)), 1)
        course = dashboard.evaluation_course(self.config, "run", "seq")
        self.assertEqual(course["sample_indices"], [2])
        self.assertEqual(course["sequence"], "seq")

    def test_rejects_path_escape(self) -> None:
        with self.assertRaises(ValueError):
            dashboard.resolved_child(self.config.record_root, "../outside")

    def test_recordings_include_shared_collection_annotation(self) -> None:
        bag = self.config.record_root / "day" / "bag"
        bag.mkdir(parents=True)
        (bag / "metadata.yaml").touch()
        dashboard.common.save_collection_annotation(
            self.config.record_root,
            "day/bag",
            {
                "category": "pass_left",
                "outcome": "success",
                "quality": "accepted",
                "dataset_versions": ["interaction-v1"],
                "notes": "RSU detected the opponent before the corner",
            },
        )
        recording = dashboard.recordings(self.config)[0]
        self.assertEqual(recording["annotation"]["category"], "pass_left")
        self.assertEqual(recording["annotation"]["dataset_versions"], ["interaction-v1"])


if __name__ == "__main__":
    unittest.main()

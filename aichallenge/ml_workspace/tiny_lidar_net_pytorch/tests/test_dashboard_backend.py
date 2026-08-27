from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

import dashboard_backend as dashboard


class FakeManager:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run_command(self, job: dashboard.Job, command: list[str], cwd: Path) -> int:
        self.commands.append(command)
        if "train.py" in command[1]:
            output = Path(command[command.index("--output-dir") + 1])
            (output / "best_model.pth").touch()
        return 0


class DashboardBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.config = dashboard.DashboardConfig(
            record_root=root / "record",
            dataset_root=root / "datasets",
            checkpoint_root=root / "checkpoints",
        )
        for path in (self.config.record_root, self.config.dataset_root, self.config.checkpoint_root):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_sequence(
        self, split: str, name: str = "sequence", version: str = "default"
    ) -> Path:
        path = dashboard.dataset_version_root(self.config, version) / split / name
        path.mkdir(parents=True)
        np.save(path / "scans.npy", np.arange(24, dtype=np.float32).reshape(4, 6))
        np.save(path / "accelerations.npy", np.full(4, 0.6, dtype=np.float32))
        np.save(path / "steers.npy", np.array([-0.2, 0.0, 0.1, 0.3], dtype=np.float32))
        (path / "preprocess_summary.json").write_text(
            json.dumps({"max_range": 30.0, "angle_min": -1.0, "angle_max": 1.0}),
            encoding="utf-8",
        )
        return path

    def test_sequence_detail_and_frame(self) -> None:
        self.make_sequence("train")
        detail = dashboard.sequence_detail(self.config, "train", "sequence")
        frame = dashboard.sequence_frame(self.config, "train", "sequence", 2)
        self.assertEqual(detail["samples"], 4)
        self.assertEqual(detail["scan_points"], 6)
        self.assertAlmostEqual(sum(detail["steering"]["histogram"]), 4)
        self.assertEqual(frame["ranges"], [12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        self.assertAlmostEqual(frame["steering"], 0.1, places=6)

    def test_preprocess_accepts_same_recording_for_train_and_val(self) -> None:
        bag = self.config.record_root / "day" / "bag"
        bag.mkdir(parents=True)
        (bag / "metadata.yaml").touch()
        manager = FakeManager()
        worker = dashboard.preprocess_worker(
            self.config,
            {"train": ["day/bag"], "val": ["day/bag"]},
            manager,  # type: ignore[arg-type]
        )
        self.assertEqual(worker(dashboard.Job("preprocess")), 0)
        self.assertEqual(len(manager.commands), 2)
        outputs = [command[command.index("--output") + 1] for command in manager.commands]
        self.assertIn("/train/", outputs[0])
        self.assertIn("/val/", outputs[1])

    def test_named_version_is_isolated_and_discoverable(self) -> None:
        self.make_sequence("train", version="pretrain-v2")
        self.make_sequence("val", version="pretrain-v2")
        sequences = dashboard.discover_sequences(self.config)
        versions = dashboard.discover_versions(self.config, sequences)
        named = next(item for item in versions if item["id"] == "pretrain-v2")
        detail = dashboard.sequence_detail(
            self.config, "train", "sequence", "pretrain-v2"
        )
        self.assertEqual(named["train_samples"], 4)
        self.assertEqual(named["val_samples"], 4)
        self.assertEqual(detail["version"], "pretrain-v2")
        self.assertFalse((self.config.dataset_root / "train").exists())

    def test_preprocess_targets_named_version(self) -> None:
        bag = self.config.record_root / "day" / "bag"
        bag.mkdir(parents=True)
        (bag / "metadata.yaml").touch()
        manager = FakeManager()
        worker = dashboard.preprocess_worker(
            self.config,
            {"dataset_version": "optimal-v1", "train": ["day/bag"], "val": []},
            manager,  # type: ignore[arg-type]
        )
        self.assertEqual(worker(dashboard.Job("preprocess")), 0)
        output = manager.commands[0][manager.commands[0].index("--output") + 1]
        self.assertIn("/datasets/versions/optimal-v1/train/", output)

    def test_training_defaults_to_steering_only(self) -> None:
        self.make_sequence("train")
        self.make_sequence("val")
        manager = FakeManager()
        worker = dashboard.training_worker(
            self.config,
            {"device": "cpu", "epochs": 1},
            manager,  # type: ignore[arg-type]
        )
        self.assertEqual(worker(dashboard.Job("train")), 0)
        command = manager.commands[0]
        self.assertEqual(command[command.index("--acceleration-weight") + 1], "0.0")
        self.assertTrue((self.config.checkpoint_root / "latest").is_symlink())

    def test_training_uses_named_version_and_versioned_checkpoint(self) -> None:
        self.make_sequence("train", version="optimal-v1")
        self.make_sequence("val", version="optimal-v1")
        manager = FakeManager()
        worker = dashboard.training_worker(
            self.config,
            {"dataset_version": "optimal-v1", "device": "cpu", "epochs": 1},
            manager,  # type: ignore[arg-type]
        )
        self.assertEqual(worker(dashboard.Job("train")), 0)
        command = manager.commands[0]
        train_dir = command[command.index("--train-dir") + 1]
        self.assertTrue(train_dir.endswith("/datasets/versions/optimal-v1/train"))
        self.assertEqual(command[command.index("--dataset-version") + 1], "optimal-v1")
        latest = self.config.checkpoint_root / "latest"
        self.assertTrue(latest.is_symlink())
        self.assertTrue(str(latest.readlink()).startswith("versions/optimal-v1/"))

    def test_rejects_unsafe_dataset_version(self) -> None:
        with self.assertRaises(ValueError):
            dashboard.normalize_dataset_version("../outside")

    def test_rejects_path_escape(self) -> None:
        with self.assertRaises(ValueError):
            dashboard.resolved_child(self.config.record_root, "../outside")

    def test_collection_annotation_is_atomic_and_discoverable(self) -> None:
        bag = self.config.record_root / "day" / "bag"
        bag.mkdir(parents=True)
        (bag / "metadata.yaml").touch()
        annotation = dashboard.save_collection_annotation(
            self.config.record_root,
            "day/bag",
            {
                "category": "follow",
                "outcome": "success",
                "quality": "accepted",
                "dataset_versions": ["interaction-v1", "rsu-ablation-v1"],
                "notes": "kept a safe gap",
            },
        )
        self.assertEqual(annotation["category"], "follow")
        self.assertEqual(annotation["dataset_versions"], ["interaction-v1", "rsu-ablation-v1"])
        self.assertFalse(list(bag.glob(".*collection_annotation*.tmp")))
        recordings = dashboard.discover_recordings(self.config)
        self.assertEqual(recordings[0]["annotation"]["outcome"], "success")
        self.assertEqual(dashboard.latest_recording(self.config.record_root)["id"], "day/bag")

    def test_collection_annotation_rejects_unsafe_input(self) -> None:
        bag = self.config.record_root / "bag"
        bag.mkdir(parents=True)
        (bag / "metadata.yaml").touch()
        with self.assertRaises(ValueError):
            dashboard.save_collection_annotation(
                self.config.record_root,
                "bag",
                {"category": "unknown", "outcome": "success", "quality": "accepted"},
            )
        with self.assertRaises(ValueError):
            dashboard.save_collection_annotation(
                self.config.record_root,
                "bag",
                {
                    "category": "follow",
                    "outcome": "success",
                    "quality": "accepted",
                    "dataset_versions": ["../outside"],
                },
            )

    def test_pid_file_prevents_duplicate_and_recovers_stale_file(self) -> None:
        path = Path(self.temporary.name) / "runtime" / "dashboard.pid"
        first = dashboard.PidFile(path)
        first.acquire()
        with self.assertRaises(RuntimeError):
            dashboard.PidFile(path).acquire()
        first.release()
        path.write_text(
            json.dumps({"pid": 99999999, "start_ticks": 1, "program": "tiny-lidar-dashboard"}),
            encoding="utf-8",
        )
        recovered = dashboard.PidFile(path)
        recovered.acquire()
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["pid"], os.getpid())
        recovered.release()
        self.assertFalse(path.exists())

    def test_shutdown_reaps_running_process_group(self) -> None:
        manager = dashboard.JobManager()

        def worker(job: dashboard.Job) -> int:
            return manager.run_command(
                job,
                [sys.executable, "-c", "import time; time.sleep(60)"],
                Path(self.temporary.name),
            )

        job = manager.start("test", worker)
        deadline = time.monotonic() + 5.0
        while job.process_pid is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(job.process_pid)
        pid = job.process_pid
        manager.shutdown(grace_seconds=0.2)
        self.assertFalse(job.thread and job.thread.is_alive())
        self.assertIsNone(dashboard.process_start_ticks(int(pid)))
        self.assertEqual(job.status, "cancelled")


if __name__ == "__main__":
    unittest.main()

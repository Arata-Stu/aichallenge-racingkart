"""Dependency-light contracts for deterministic evaluation video sampling."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from lidar_racing_rl.evaluation.video import EvaluationTrace, _frame_indices, frame_stride


class EvaluationVideoContractTest(unittest.TestCase):
    def test_trace_selects_environment_and_ego_axes(self) -> None:
        evaluator = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "lidar_racing_rl"
            / "evaluation"
            / "evaluator.py"
        ).read_text(encoding="utf-8")
        self.assertIn("cartesian_states[0, 0]", evaluator)

    def test_four_times_playback_at_twenty_hz_keeps_every_fourth_sample(self) -> None:
        self.assertEqual(frame_stride(control_dt=0.05, fps=20, playback_speed=4.0), 4)
        self.assertEqual(
            _frame_indices(10, control_dt=0.05, fps=20, playback_speed=4.0),
            (0, 4, 8, 9),
        )

    def test_trace_rejects_non_finite_rollout_data(self) -> None:
        trace = EvaluationTrace(
            center_x=(0.0, 1.0, 0.0),
            center_y=(0.0, 0.0, 1.0),
            center_yaw=(0.0, 1.0, 2.0),
            left_widths=(1.0, 1.0, 1.0),
            right_widths=(1.0, 1.0, 1.0),
            poses=((0.0, 0.0, 0.0), (math.nan, 0.0, 0.0)),
            speeds=(0.0, 1.0),
            actions=((0.0, 0.0), (0.0, 0.0)),
            cumulative_progress=(0.0, 1.0),
            race_complete=(False, False),
            collision=(False, False),
            off_track=(False, False),
            truncated=(False, False),
            control_dt=0.05,
            track_length=10.0,
            vehicle_length=0.58,
            vehicle_width=0.31,
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            trace.validate()

    def test_sampling_arguments_are_fail_closed(self) -> None:
        for kwargs in (
            {"control_dt": 0.0, "fps": 20, "playback_speed": 1.0},
            {"control_dt": 0.05, "fps": 0, "playback_speed": 1.0},
            {"control_dt": 0.05, "fps": 20, "playback_speed": 0.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                frame_stride(**kwargs)


if __name__ == "__main__":
    unittest.main()

"""Standard-library tests for the AWSIM vehicle-response bag analyzer."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_awsim_vehicle_response.py"
SPEC = importlib.util.spec_from_file_location("analyze_awsim_vehicle_response", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _timestamp(seconds: float) -> int:
    return round(seconds * MODULE.NANOSECONDS_PER_SECOND)


def _step_experiment(kind: str = "steering_step") -> object:
    settings = {
        "input_start_seconds": 0.95,
        "steady_start_seconds": 8.0,
        "minimum_command_step": 0.5,
        "minimum_response_step": 0.5,
        "minimum_samples_per_window": 10,
    }
    if kind == "acceleration_step":
        settings["maximum_derivative_gap_seconds"] = 0.2
    return MODULE.Experiment("step", kind, 0.0, 10.0, settings)


class ManifestValidationTests(unittest.TestCase):
    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "time_reference": "selected_message_stamp_seconds",
            "experiments": [
                {
                    "id": "steer",
                    "kind": "steering_step",
                    "start_seconds": 0.0,
                    "input_start_seconds": 1.0,
                    "steady_start_seconds": 2.0,
                    "end_seconds": 3.0,
                    "minimum_command_step": 0.1,
                    "minimum_response_step": 0.1,
                    "minimum_samples_per_window": 3,
                },
                {
                    "id": "sweep",
                    "kind": "steering_sine_sweep",
                    "start_seconds": 4.0,
                    "end_seconds": 5.0,
                    "minimum_samples": 3,
                    "minimum_command_peak_to_peak": 0.1,
                    "minimum_response_peak_to_peak": 0.1,
                },
                {
                    "id": "accel",
                    "kind": "acceleration_step",
                    "start_seconds": 6.0,
                    "input_start_seconds": 7.0,
                    "steady_start_seconds": 8.0,
                    "end_seconds": 9.0,
                    "minimum_command_step": 0.1,
                    "minimum_response_step": 0.1,
                    "minimum_samples_per_window": 3,
                    "maximum_derivative_gap_seconds": 0.2,
                },
                {
                    "id": "coast",
                    "kind": "coast",
                    "start_seconds": 10.0,
                    "end_seconds": 11.0,
                    "max_abs_command_acceleration": 0.01,
                    "minimum_speed_mps": 0.1,
                    "minimum_samples": 3,
                    "minimum_r_squared": 0.9,
                },
                {
                    "id": "turn",
                    "kind": "constant_speed_turn",
                    "start_seconds": 12.0,
                    "end_seconds": 13.0,
                    "minimum_speed_mps": 0.1,
                    "minimum_abs_steering_radians": 0.01,
                    "minimum_abs_yaw_rate_radians_per_second": 0.01,
                    "max_alignment_gap_seconds": 0.2,
                    "minimum_samples": 3,
                },
            ],
        }

    def _load(self, payload: dict[str, object]) -> list[object]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiments.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return MODULE.load_experiments(
                path,
                timestamp_source="message",
                acceleration_response="velocity_derivative",
            )

    def test_accepts_explicit_complete_manifest(self) -> None:
        experiments = self._load(self._payload())
        self.assertEqual([item.kind for item in experiments], list(MODULE.EXPERIMENT_KINDS))

    def test_rejects_boolean_schema_version(self) -> None:
        payload = self._payload()
        payload["schema_version"] = True
        with self.assertRaises(MODULE.AnalysisError):
            self._load(payload)

    def test_rejects_missing_derivative_gap_for_default_acceleration_source(self) -> None:
        payload = self._payload()
        acceleration = payload["experiments"][2]
        del acceleration["maximum_derivative_gap_seconds"]
        with self.assertRaises(MODULE.AnalysisError):
            self._load(payload)

    def test_rejects_duplicate_json_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiments.json"
            path.write_text(
                '{"schema_version": 1, "schema_version": 1, "experiments": []}',
                encoding="utf-8",
            )
            with self.assertRaises(MODULE.AnalysisError):
                MODULE.load_experiments(
                    path,
                    timestamp_source="message",
                    acceleration_response="velocity_derivative",
                )


class IdentificationMathTests(unittest.TestCase):
    def test_fopdt_step_recovers_gain_time_constant_and_delay(self) -> None:
        experiment = _step_experiment()
        command = []
        response = []
        gain = 2.0
        delay = 0.2
        time_constant = 0.5
        for index in range(1001):
            time_value = index / 100.0
            command_value = 0.0 if time_value < 1.0 else 1.0
            response_value = 0.0
            if time_value >= 1.0 + delay:
                response_value = gain * (
                    1.0 - math.exp(-(time_value - 1.0 - delay) / time_constant)
                )
            command.append(MODULE.ScalarSample(_timestamp(time_value), command_value))
            response.append(MODULE.ScalarSample(_timestamp(time_value), response_value))

        outcomes = MODULE._step_candidates(
            experiment,
            command,
            response,
            origin_ns=0,
            gain_parameter="steering_gain",
            delay_parameter="steering_delay",
            time_constant_parameter="steering_time_constant",
        )
        self.assertIsInstance(outcomes["steering_gain"], MODULE.Candidate)
        self.assertAlmostEqual(outcomes["steering_gain"].value, gain, delta=0.01)
        self.assertAlmostEqual(
            outcomes["steering_time_constant"].value,
            time_constant,
            delta=0.02,
        )
        self.assertAlmostEqual(outcomes["steering_delay"].value, delay, delta=0.02)

    def test_nonuniform_derivative_uses_only_labelled_interval(self) -> None:
        experiment = MODULE.Experiment(
            "accel",
            "acceleration_step",
            1.0,
            4.0,
            {"maximum_derivative_gap_seconds": 1.0},
        )
        times = (0.0, 1.0, 1.4, 2.2, 3.0, 4.0, 5.0)
        motion = [
            MODULE.MotionSample(_timestamp(time_value), time_value**2, 0.0)
            for time_value in times
        ]
        acceleration = MODULE._derive_acceleration(
            motion,
            experiment,
            origin_ns=0,
        )
        self.assertEqual(
            [sample.timestamp_ns for sample in acceleration],
            [_timestamp(value) for value in (1.4, 2.2, 3.0)],
        )
        for sample in acceleration:
            time_value = sample.timestamp_ns / MODULE.NANOSECONDS_PER_SECOND
            self.assertAlmostEqual(sample.value, 2.0 * time_value, places=10)

    def test_drag_recovers_exponential_decay(self) -> None:
        experiment = MODULE.Experiment(
            "coast",
            "coast",
            0.0,
            5.0,
            {
                "max_abs_command_acceleration": 0.0,
                "minimum_speed_mps": 0.1,
                "minimum_samples": 10,
                "minimum_r_squared": 0.99,
            },
        )
        command = [
            MODULE.ScalarSample(_timestamp(index / 10.0), 0.0)
            for index in range(51)
        ]
        motion = [
            MODULE.MotionSample(
                _timestamp(index / 10.0),
                5.0 * math.exp(-0.2 * index / 10.0),
                0.0,
            )
            for index in range(51)
        ]
        candidate = MODULE._drag_candidate(
            experiment,
            command,
            motion,
            origin_ns=0,
        )
        self.assertIsInstance(candidate, MODULE.Candidate)
        self.assertAlmostEqual(candidate.value, 0.2, places=12)

    def test_wheelbase_uses_actual_steering_and_yaw_rate(self) -> None:
        actual_wheelbase = 1.1
        steering_angle = 0.2
        speed = 4.0
        yaw_rate = speed * math.tan(steering_angle) / actual_wheelbase
        experiment = MODULE.Experiment(
            "turn",
            "constant_speed_turn",
            0.0,
            2.0,
            {
                "minimum_speed_mps": 1.0,
                "minimum_abs_steering_radians": 0.1,
                "minimum_abs_yaw_rate_radians_per_second": 0.1,
                "max_alignment_gap_seconds": 0.2,
                "minimum_samples": 5,
            },
        )
        steering = [
            MODULE.ScalarSample(_timestamp(index / 10.0), steering_angle)
            for index in range(21)
        ]
        motion = [
            MODULE.MotionSample(_timestamp(index / 10.0), speed, yaw_rate)
            for index in range(21)
        ]
        candidate = MODULE._wheelbase_candidate(
            experiment,
            steering,
            motion,
            origin_ns=0,
        )
        self.assertIsInstance(candidate, MODULE.Candidate)
        self.assertAlmostEqual(candidate.value, actual_wheelbase, places=12)


class FailClosedAndOutputTests(unittest.TestCase):
    def test_missing_topics_leave_all_parameters_null_with_reasons(self) -> None:
        topics = {
            role: MODULE.TopicReport(topic=f"/{role}", expected_type="example/Type", status="missing")
            for role in ("control", "steering", "velocity", "odometry", "acceleration")
        }
        data = MODULE.BagData(
            storage_id="mcap",
            bag_start_timestamp_ns=1,
            bag_end_timestamp_ns=2,
            analysis_origin_timestamp_ns=1,
            timestamp_source="message",
            topics=topics,
            series=MODULE.BagSeries(),
        )
        estimates, _ = MODULE.identify(
            data,
            [_step_experiment()],
            motion_source="velocity_status",
            acceleration_response="velocity_derivative",
        )
        for parameter in MODULE.PARAMETERS:
            self.assertIsNone(estimates[parameter]["value"])
            self.assertEqual(estimates[parameter]["status"], "unavailable")
            self.assertTrue(estimates[parameter]["reason"])

    def test_atomic_write_replaces_complete_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vehicle.yaml"
            path.write_text("old\n", encoding="utf-8")
            MODULE._atomic_write(path, "new\ncontent\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "new\ncontent\n")
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_yaml_scientific_notation_remains_numeric_for_yaml_1_1(self) -> None:
        self.assertEqual(MODULE._yaml_scalar(1.0e-6), "1.0e-06")
        self.assertEqual(MODULE._yaml_scalar(1.0e20), "1.0e+20")


if __name__ == "__main__":
    unittest.main()

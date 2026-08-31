#!/usr/bin/env python3
"""Export a verified Flax LiDAR Actor as a PyTorch ROS 2 policy bundle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "deployment" / "awsim.yaml"


def _load_config(path: Path, *, label: str = "configuration") -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError("OmegaConf is not installed; run the project setup first") from error
    return OmegaConf.load(path)


def _select(config: Any, path: str) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.select(config, path)


def _validate_config(config: Any) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    expected_values = {
        "runtime": "pytorch",
        "input.topic": "/sensing/lidar/scan",
        "input.type": "sensor_msgs/msg/LaserScan",
        "input.expected_raw_beams": 750,
        "input.expected_raw_range_min": 0.0,
        "input.expected_raw_range_max": 25.0,
        "input.expected_raw_angle_min": -1.5666074752807617,
        "input.expected_raw_angle_max": 1.5707963705062866,
        "output.topic": "/control/command/control_cmd",
        "output.type": "autoware_auto_control_msgs/msg/AckermannControlCommand",
        "preprocessing.canonical_beams": 360,
        "preprocessing.pooling.type": "angular_minimum",
        "preprocessing.frame_stack": 4,
        "preprocessing.invalid_range_replacement": "range_max",
        "preprocessing.normalization": "range_max",
        "preprocessing.validate_laserscan_metadata": True,
        "control.steering_max_abs": 0.64,
        "control.acceleration_min": -3.2,
        "control.acceleration_max": 3.2,
        "failsafe.enabled": True,
        "failsafe.stop_on_non_finite_actor_output": True,
        "failsafe.stop_on_inference_exception": True,
        "failsafe.stop_on_model_manifest_mismatch": True,
        "model.file": "models/policy_torch.pt",
        "model.manifest": "models/policy_manifest.json",
    }
    for path, expected in expected_values.items():
        actual = _select(config, path)
        if actual != expected:
            errors.append(f"{path} must be {expected!r}; got {actual!r}")

    canonical_profile = (
        _select(config, "preprocessing.field_of_view"),
        _select(config, "preprocessing.expected_range_max"),
    )
    supported_profiles = (
        (1.5 * math.pi, 30.0),
        (math.pi, 25.0),
    )
    if not all(
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in canonical_profile
    ) or not any(
        math.isclose(float(canonical_profile[0]), fov, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(
            float(canonical_profile[1]), maximum, rel_tol=0.0, abs_tol=1.0e-12
        )
        for fov, maximum in supported_profiles
    ):
        errors.append(
            "canonical deployment profile must be legacy 270-degree/30m or "
            "AWSIM e2e 180-degree/25m"
        )

    channels = _select(config, "preprocessing.channels")
    if list(channels or []) != ["range", "validity"]:
        errors.append("preprocessing.channels must be [range, validity]")
    actor_topics = _select(config, "input.actor_sensor_topics")
    if list(actor_topics or []) != ["/sensing/lidar/scan"]:
        errors.append("Actor sensor topics must contain only /sensing/lidar/scan")
    angle_min = _select(config, "preprocessing.expected_angle_min")
    angle_max = _select(config, "preprocessing.expected_angle_max")
    field_of_view = _select(config, "preprocessing.field_of_view")
    if not all(
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in (angle_min, angle_max, field_of_view)
    ):
        errors.append("preprocessing angle bounds and field_of_view must be finite")
    elif not math.isclose(
        float(angle_max) - float(angle_min),
        float(field_of_view),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        errors.append("preprocessing field_of_view must equal angle_max - angle_min")
    tolerance = _select(config, "model.flax_torch_max_absolute_error")
    if (
        not isinstance(tolerance, int | float)
        or isinstance(tolerance, bool)
        or not math.isfinite(float(tolerance))
        or not 0.0 <= float(tolerance) <= 1.0e-5
    ):
        errors.append("Flax/PyTorch parity tolerance must be <= 1e-5")
    if _select(config, "failsafe.scan_timeout_seconds") is None:
        warnings.append("failsafe.scan_timeout_seconds still requires AWSIM calibration")
    if _select(config, "failsafe.minimum_valid_beam_ratio") is None:
        warnings.append("failsafe.minimum_valid_beam_ratio still requires AWSIM calibration")
    return errors, warnings


def _resolved_container(config: Any) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True)


def _resolved_yaml(config: Any) -> str:
    from omegaconf import OmegaConf

    return OmegaConf.to_yaml(config, resolve=True, sort_keys=True)


def _find_training_config(checkpoint: Path) -> Path:
    path = checkpoint.resolve()
    candidates = [path / "resolved_config.yaml"] if path.is_dir() else []
    for parent in (path.parent, *tuple(path.parents)[:4]):
        candidates.append(parent / "resolved_config.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "resolved_config.yaml was not found near the checkpoint; pass --training-config"
    )


def _validate_training_config(training: Any, deployment: Any) -> list[str]:
    errors: list[str] = []
    comparisons = (
        (
            "env.lidar.num_beams",
            "preprocessing.canonical_beams",
        ),
        (
            "env.lidar.frame_stack",
            "preprocessing.frame_stack",
        ),
        (
            "env.lidar.field_of_view",
            "preprocessing.field_of_view",
        ),
        (
            "env.lidar.range_max",
            "preprocessing.expected_range_max",
        ),
    )
    for training_path, deployment_path in comparisons:
        training_value = _select(training, training_path)
        deployment_value = _select(deployment, deployment_path)
        if training_value != deployment_value:
            errors.append(
                f"{training_path}={training_value!r} does not match "
                f"{deployment_path}={deployment_value!r}"
            )
    if list(_select(training, "env.lidar.channels") or []) != ["range", "validity"]:
        errors.append("training env.lidar.channels must be [range, validity]")
    if _select(training, "env.information_boundary.actor_critic_gt_access") is not False:
        errors.append("training Actor/Critic GT access must remain disabled")
    if _select(training, "agent.observation.lidar_only") is not True:
        errors.append("training Actor must remain LiDAR-only")
    if list(_select(training, "agent.observation.input_shape") or []) != [8, 360]:
        errors.append("training Actor input_shape must be [8, 360]")
    if _select(training, "agent.observation.num_beams") != 360:
        errors.append("training Actor must use 360 canonical beams")
    if _select(training, "agent.observation.frame_stack") != 4:
        errors.append("training Actor must use a four-frame stack")
    if _select(training, "agent.observation.channels_per_frame") != 2:
        errors.append("training Actor must use two channels per frame")
    if math.isclose(
        float(_select(deployment, "preprocessing.field_of_view")),
        math.pi,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        action_comparisons = (
            ("vehicle.vehicle.max_steering_angle", "control.steering_max_abs"),
            ("vehicle.vehicle.min_acceleration", "control.acceleration_min"),
            ("vehicle.vehicle.max_acceleration", "control.acceleration_max"),
        )
        for training_path, deployment_path in action_comparisons:
            training_value = float(_select(training, training_path))
            deployment_value = float(_select(deployment, deployment_path))
            if not math.isclose(
                training_value,
                deployment_value,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                errors.append(
                    f"training {training_path}={training_value} does not match "
                    f"{deployment_path}={deployment_value}"
                )
    v1_model_contract: tuple[tuple[str, Any], ...] = (
        ("agent.actor.distribution", "tanh_gaussian"),
        ("agent.actor.encoder.type", "conv1d"),
        ("agent.actor.encoder.channels", [32, 64, 64]),
        ("agent.actor.encoder.kernel_sizes", [8, 4, 3]),
        ("agent.actor.encoder.strides", [4, 2, 1]),
        ("agent.actor.encoder.activation", "relu"),
        ("agent.actor.hidden_sizes", [256]),
        ("agent.actor.action_dim", 2),
        ("agent.actor.log_std_min", -5.0),
        ("agent.actor.log_std_max", 2.0),
        ("agent.critic.count", 2),
        ("agent.critic.share_lidar_encoder", False),
        ("agent.critic.hidden_sizes", [256, 256]),
    )
    for path, expected in v1_model_contract:
        actual = _select(training, path)
        if isinstance(expected, list):
            matches = list(actual or []) == expected
        elif isinstance(expected, bool):
            matches = actual is expected
        else:
            matches = actual == expected
        if not matches:
            errors.append(
                f"training {path} must remain {expected!r} for "
                f"lidar_actor_conv1d_v1; got {actual!r}"
            )
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert and parity-check a Flax checkpoint for ROS 2 deployment.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("exported"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--training-config",
        type=Path,
        help="Resolved training config; auto-detected beside the checkpoint when omitted.",
    )
    parser.add_argument("--parity-seed", type=int, default=0)
    parser.add_argument("--parity-batch-size", type=int, default=16)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the export contract without importing JAX or PyTorch.",
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = _load_config(args.config, label="deployment configuration")
        errors, warnings = _validate_config(config)
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 2

        if args.print_config:
            print(json.dumps(_resolved_container(config), indent=2, sort_keys=True))
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                        "checkpoint_exists": bool(
                            args.checkpoint and args.checkpoint.exists()
                        ),
                        "training_config": (
                            str(args.training_config) if args.training_config else "auto-detect"
                        ),
                        "output": str(args.output),
                        "would_write": [
                            "policy_flax.msgpack",
                            "policy_torch.pt",
                            "policy_manifest.json",
                            "config_snapshot.yaml",
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.checkpoint is None:
            print(
                "ERROR: --checkpoint is required for export. "
                "Use --dry-run to validate the deployment contract only.",
                file=sys.stderr,
            )
            return 2
        if not args.checkpoint.exists():
            print(f"ERROR: checkpoint does not exist: {args.checkpoint}", file=sys.stderr)
            return 2
        training_config_path = args.training_config or _find_training_config(args.checkpoint)
        training_config = _load_config(
            training_config_path,
            label="resolved training configuration",
        )
        training_errors = _validate_training_config(training_config, config)
        if training_errors:
            for error in training_errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 2
        if args.parity_batch_size < 1:
            print("ERROR: --parity-batch-size must be positive", file=sys.stderr)
            return 2

        from lidar_racing_rl.export.export_policy import (
            export_policy_bundle,
            result_as_json,
        )

        result = export_policy_bundle(
            checkpoint=args.checkpoint,
            output_directory=args.output,
            deployment_config=_resolved_container(config),
            training_config=_resolved_container(training_config),
            config_snapshot_yaml=_resolved_yaml(training_config),
            parity_seed=args.parity_seed,
            parity_batch_size=args.parity_batch_size,
        )
        print(result_as_json(result))
        return 0
    except Exception as error:
        if args.debug:
            raise
        print(f"ERROR: policy export failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

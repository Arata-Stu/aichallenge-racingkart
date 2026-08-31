#!/usr/bin/env python3
"""Run deterministic LiDAR-only policy evaluation in F1TENTH Gym JAX."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _normalize_config_name(name: str) -> str:
    normalized = name.removesuffix(".yaml")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("config name must stay within the project config directory")
    return normalized if "/" in normalized else f"train/{normalized}"


def _compose_config(config_name: str, overrides: list[str]) -> Any:
    try:
        from hydra import compose, initialize_config_dir
    except ImportError as error:
        raise RuntimeError("Hydra is not installed; run the project setup first") from error

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        return compose(
            config_name=_normalize_config_name(config_name),
            overrides=overrides,
        )


def _select(config: Any, path: str) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.select(config, path)


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _validate_config(config: Any) -> list[str]:
    errors: list[str] = []
    required_paths = (
        "experiment.name",
        "experiment.stage",
        "env.num_agents",
        "env.lidar.frame_stack",
        "training.ego_agent_index",
        "agent.observation.lidar_only",
        "agent.observation.input_shape",
        "agent.actor.deterministic_evaluation",
        "teacher.reference_line",
        "teacher.base_target_speed",
        "teacher.speed_profile.type",
        "teacher.speed_profile.minimum_corner_speed",
        "teacher.speed_profile.maximum_lateral_acceleration",
        "vehicle.vehicle.width",
    )
    for path in required_paths:
        if _select(config, path) is None:
            errors.append(f"required config value is missing or null: {path}")

    stage = _select(config, "experiment.stage")
    num_agents = _select(config, "env.num_agents")
    expected_agents = {"step1": 1, "step2": 4}.get(stage)
    if expected_agents is None:
        errors.append("experiment.stage must be step1 or step2")
    elif num_agents != expected_agents:
        errors.append(f"{stage} requires env.num_agents={expected_agents}")
    if _select(config, "env.lidar.num_beams") != 360:
        errors.append("evaluation requires the canonical 360-beam observation")
    if _select(config, "env.lidar.frame_stack") != 4:
        errors.append("evaluation requires the four-frame observation contract")
    if list(_select(config, "agent.observation.input_shape") or []) != [8, 360]:
        errors.append("evaluation Actor input_shape must be [8, 360]")
    if _select(config, "agent.observation.lidar_only") is not True:
        errors.append("evaluation Actor must remain LiDAR-only")
    if _select(config, "env.information_boundary.actor_critic_gt_access") is not False:
        errors.append("Actor evaluation must not receive GT state")
    if _select(config, "agent.actor.deterministic_evaluation") is not True:
        errors.append("agent.actor.deterministic_evaluation must be true")
    if _select(config, "training.ego_agent_index") != 0:
        errors.append("evaluation supports learned agent_0 only")
    if _select(config, "teacher.reference_line") != "centerline":
        errors.append("evaluation requires teacher.reference_line=centerline")
    if _select(config, "teacher.speed_profile.type") != "curvature_limited":
        errors.append("teacher.speed_profile.type must be curvature_limited")
    teacher_base_speed = _select(config, "teacher.base_target_speed")
    if not _is_finite_number(teacher_base_speed) or teacher_base_speed <= 0.0:
        errors.append("teacher.base_target_speed must be finite and positive")
    vehicle_width = _select(config, "vehicle.vehicle.width")
    if not _is_finite_number(vehicle_width) or vehicle_width <= 0.0:
        errors.append("vehicle.vehicle.width must be finite and positive")
    if stage == "step2":
        if _select(config, "npc.lateral_controller.reference_line") != "centerline":
            errors.append("NPCs require a centerline reference")
        npc_base_speed = _select(
            config,
            "npc.longitudinal_controller.base_target_speed",
        )
        if not _is_finite_number(npc_base_speed) or npc_base_speed <= 0.0:
            errors.append("NPC base_target_speed must be finite and positive")
        elif npc_base_speed != teacher_base_speed:
            errors.append("teacher and NPC base_target_speed must match")
        if _select(config, "curriculum.enabled") is not False:
            errors.append(
                "Step 2 curriculum scheduling is not integrated; "
                "curriculum.enabled must remain false"
            )
        if _select(config, "curriculum.active_phase") is not None:
            errors.append(
                "Step 2 prototype must not claim an active curriculum phase"
            )
        if _select(config, "training.opponent_pool_enabled") is not False:
            errors.append(
                "past-policy opponents are not integrated into evaluation"
            )
    if _select(config, "env.domain_randomization.enabled") is True:
        errors.append(
            "AWSIM vehicle-response domain randomization is not integrated into "
            "the F1TENTH environment"
        )
    return errors


def _resolved_container(config: Any) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint with deterministic tanh(mean) actions.",
    )
    parser.add_argument("--config-name", default="step1_single_vehicle")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=_positive_int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--video",
        type=Path,
        help="Render the first evaluated episode as a .gif or .mp4 animation.",
    )
    parser.add_argument("--video-fps", type=_positive_int, default=20)
    parser.add_argument("--video-speed", type=_positive_float, default=4.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without importing JAX or loading a checkpoint.",
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("overrides", nargs="*", metavar="HYDRA_OVERRIDE")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = _compose_config(args.config_name, args.overrides)
        errors = _validate_config(config)
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
                        "config_name": str(config.experiment.name),
                        "episodes": args.episodes,
                        "deterministic": True,
                        "actor_observation": "lidar_only",
                        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                        "checkpoint_exists": bool(
                            args.checkpoint and args.checkpoint.exists()
                        ),
                        "output": str(args.output) if args.output else None,
                        "video": str(args.video) if args.video else None,
                        "video_fps": args.video_fps,
                        "video_speed": args.video_speed,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.checkpoint is None:
            print(
                "ERROR: --checkpoint is required for evaluation. "
                "Use --dry-run to validate configuration only.",
                file=sys.stderr,
            )
            return 2
        if not args.checkpoint.exists():
            print(f"ERROR: checkpoint does not exist: {args.checkpoint}", file=sys.stderr)
            return 2

        if args.output is not None and args.output.exists():
            print(f"ERROR: evaluation output already exists: {args.output}", file=sys.stderr)
            return 2
        if args.video is not None:
            if args.video.suffix.lower() not in {".gif", ".mp4"}:
                print("ERROR: --video must end in .gif or .mp4", file=sys.stderr)
                return 2
            if args.video.exists():
                print(f"ERROR: evaluation video already exists: {args.video}", file=sys.stderr)
                return 2

        from lidar_racing_rl.evaluation.evaluator import evaluate_lidar_policy

        result = evaluate_lidar_policy(
            _resolved_container(config),
            checkpoint=args.checkpoint,
            episodes=args.episodes,
            capture_trace=args.video is not None,
        )
        payload = {
            "requested_episodes": result.requested_episodes,
            "completed_episodes": result.completed_episodes,
            "vector_environments": result.vector_environments,
            "environment_steps": result.environment_steps,
            "elapsed_seconds": result.elapsed_seconds,
            "checkpoint_step": result.checkpoint_step,
            "metrics": result.metrics,
            "video": str(args.video) if args.video else None,
        }
        if args.video is not None:
            if result.trace is None:
                raise RuntimeError("evaluation did not return the requested rollout trace")
            from lidar_racing_rl.evaluation.video import render_evaluation_video

            render_evaluation_video(
                result.trace,
                args.video,
                fps=args.video_fps,
                playback_speed=args.video_speed,
            )
        serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
        print(serialized)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
        return 0
    except Exception as error:
        if args.debug:
            raise
        print(f"ERROR: deterministic policy evaluation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

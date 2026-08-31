#!/usr/bin/env python3
"""Run vectorized LiDAR-only Flax Soft Actor-Critic training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
F1TENTH_SUBMODULE = PROJECT_ROOT / "repos" / "f1tenth_gym_jax"
REPOSITORY_ROOT = PROJECT_ROOT.parents[2]


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


def _validate_config(config: Any) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    required_paths = (
        "experiment.name",
        "experiment.stage",
        "env.num_envs",
        "env.num_agents",
        "env.lidar.num_beams",
        "env.lidar.frame_stack",
        "env.episode.max_steps",
        "training.ego_agent_index",
        "training.save_npc_transitions",
        "agent.observation.lidar_only",
        "agent.observation.input_shape",
        "agent.replay_buffer.ego_only",
        "agent.replay_buffer.capacity",
        "agent.update.batch_size",
        "agent.update.target_entropy",
        "agent.update.actor_update_start_step",
        "agent.update.actor_behavior_blend_updates",
        "agent.update.updates_per_collection",
        "agent.optimizer.initial_temperature",
        "agent.checkpoint.save_interval_updates",
        "agent.checkpoint.keep_last",
        "training.total_environment_transitions",
        "training.log_interval_collections",
        "training.require_clean_repositories",
        "teacher.reference_line",
        "teacher.base_target_speed",
        "teacher.speed_profile.type",
        "teacher.speed_profile.minimum_corner_speed",
        "teacher.speed_profile.maximum_lateral_acceleration",
        "teacher.normalized_action_noise_std",
        "reward.trajectory_aided.enabled",
        "reward.trajectory_aided.weight",
        "vehicle.vehicle.width",
        "vehicle.vehicle.max_velocity",
    )
    for path in required_paths:
        if _select(config, path) is None:
            errors.append(f"required config value is missing or null: {path}")

    num_envs = _select(config, "env.num_envs")
    num_agents = _select(config, "env.num_agents")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        errors.append("env.num_envs must be a positive integer")
    if num_agents not in (1, 4):
        errors.append("env.num_agents must be 1 for Step 1 or 4 for Step 2")
    stage = _select(config, "experiment.stage")
    expected_agents = {"step1": 1, "step2": 4}.get(stage)
    if expected_agents is None:
        errors.append("experiment.stage must be step1 or step2")
    elif num_agents != expected_agents:
        errors.append(f"{stage} requires env.num_agents={expected_agents}")
    if _select(config, "env.lidar.num_beams") != 360:
        errors.append("LiDAR-only model contract requires 360 canonical beams")
    if _select(config, "env.lidar.frame_stack") != 4:
        errors.append("initial SAC contract requires a four-frame stack")
    if list(_select(config, "env.lidar.channels") or []) != ["range", "validity"]:
        errors.append("environment LiDAR channels must be [range, validity]")
    if list(_select(config, "agent.observation.input_shape") or []) != [8, 360]:
        errors.append("agent observation input_shape must be [8, 360]")
    if _select(config, "agent.observation.num_beams") != 360:
        errors.append("agent observation must use 360 beams")
    if _select(config, "agent.observation.frame_stack") != 4:
        errors.append("agent observation must use four frames")
    if _select(config, "agent.observation.channels_per_frame") != 2:
        errors.append("agent observation must use two channels per frame")
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
        actual = _select(config, path)
        if isinstance(expected, list):
            matches = list(actual or []) == expected
        elif isinstance(expected, bool):
            matches = actual is expected
        else:
            matches = actual == expected
        if not matches:
            errors.append(
                f"{path} must remain {expected!r} for lidar_actor_conv1d_v1; "
                f"got {actual!r}"
            )
    if _select(config, "env.information_boundary.actor_critic_gt_access") is not False:
        errors.append("Actor/Critic GT access must remain disabled")
    if _select(config, "training.ego_agent_index") != 0:
        errors.append("only agent_0 may be the learned Ego")
    if _select(config, "training.save_npc_transitions") is not False:
        errors.append("NPC transitions must not enter the replay buffer")
    if _select(config, "agent.observation.lidar_only") is not True:
        errors.append("agent.observation.lidar_only must remain true")
    if _select(config, "agent.replay_buffer.ego_only") is not True:
        errors.append("agent.replay_buffer.ego_only must remain true")
    if _select(config, "agent.replay_buffer.implementation") != "jax_ring":
        errors.append("agent.replay_buffer.implementation must be jax_ring")
    if _select(config, "agent.checkpoint.implementation") != "flax_msgpack_atomic":
        errors.append("agent.checkpoint.implementation must be flax_msgpack_atomic")
    if stage == "step2":
        if _select(config, "npc.count") != 3:
            errors.append("Step 2 requires exactly three NPCs")
        if _select(config, "npc.learned") is not False:
            errors.append("Step 2 NPCs must remain fixed policies")
        if _select(config, "npc.save_transitions") is not False:
            errors.append("Step 2 NPC transitions must not be saved")
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
                "past-policy opponents are not integrated; "
                "training.opponent_pool_enabled must remain false"
            )
        reset_spacing = _select(config, "env.reset.longitudinal_spacing")
        safe_distance_max = _select(
            config,
            "npc.longitudinal_controller.safe_following_distance.max",
        )
        vehicle_length = _select(config, "vehicle.vehicle.length")
        if (
            _is_finite_number(reset_spacing)
            and _is_finite_number(safe_distance_max)
            and _is_finite_number(vehicle_length)
            and reset_spacing < safe_distance_max + vehicle_length
        ):
            errors.append(
                "Step 2 reset spacing must be at least the maximum NPC "
                "safe-following distance plus vehicle length"
            )

    if _select(config, "env.domain_randomization.enabled") is True:
        errors.append(
            "AWSIM vehicle-response domain randomization is not integrated into "
            "the F1TENTH environment"
        )

    replay_capacity = _select(config, "agent.replay_buffer.capacity")
    replay_warmup = _select(config, "agent.replay_buffer.warmup_transitions")
    batch_size = _select(config, "agent.update.batch_size")
    for path, value in (
        ("agent.replay_buffer.capacity", replay_capacity),
        ("agent.replay_buffer.warmup_transitions", replay_warmup),
        ("agent.update.batch_size", batch_size),
        (
            "agent.update.updates_per_collection",
            _select(config, "agent.update.updates_per_collection"),
        ),
        (
            "agent.checkpoint.save_interval_updates",
            _select(config, "agent.checkpoint.save_interval_updates"),
        ),
        ("agent.checkpoint.keep_last", _select(config, "agent.checkpoint.keep_last")),
        (
            "training.total_environment_transitions",
            _select(config, "training.total_environment_transitions"),
        ),
        (
            "training.log_interval_collections",
            _select(config, "training.log_interval_collections"),
        ),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            errors.append(f"{path} must be a positive integer")
    actor_update_start_step = _select(
        config,
        "agent.update.actor_update_start_step",
    )
    if (
        isinstance(actor_update_start_step, bool)
        or not isinstance(actor_update_start_step, int)
        or actor_update_start_step < 0
    ):
        errors.append("agent.update.actor_update_start_step must be non-negative")
    actor_behavior_blend_updates = _select(
        config,
        "agent.update.actor_behavior_blend_updates",
    )
    if (
        isinstance(actor_behavior_blend_updates, bool)
        or not isinstance(actor_behavior_blend_updates, int)
        or actor_behavior_blend_updates < 0
    ):
        errors.append("agent.update.actor_behavior_blend_updates must be non-negative")
    if (
        isinstance(replay_capacity, int)
        and isinstance(replay_warmup, int)
        and replay_warmup > replay_capacity
    ):
        errors.append("replay warmup cannot exceed replay capacity")
    if (
        isinstance(replay_warmup, int)
        and not isinstance(replay_warmup, bool)
        and isinstance(batch_size, int)
        and not isinstance(batch_size, bool)
        and replay_warmup < batch_size
    ):
        errors.append("replay warmup must be at least one learner batch")
    if (
        isinstance(replay_capacity, int)
        and not isinstance(replay_capacity, bool)
        and isinstance(num_envs, int)
        and not isinstance(num_envs, bool)
        and replay_capacity < num_envs
    ):
        errors.append("replay capacity must fit one vector-environment collection")

    if _select(config, "training.require_clean_repositories") is not True:
        errors.append("training.require_clean_repositories must remain true for truthful SHAs")
    resume_from = _select(config, "training.resume_from")
    initialize_actor_from = _select(config, "training.initialize_actor_from")
    for path, value in (
        ("training.resume_from", resume_from),
        ("training.initialize_actor_from", initialize_actor_from),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{path} must be null or a non-empty checkpoint path")
    if resume_from is not None and initialize_actor_from is not None:
        errors.append(
            "training.resume_from and training.initialize_actor_from are "
            "mutually exclusive"
        )
    if _select(config, "teacher.reference_line") != "centerline":
        errors.append("training requires teacher.reference_line=centerline")
    teacher_base_speed = _select(config, "teacher.base_target_speed")
    if not _is_finite_number(teacher_base_speed) or teacher_base_speed <= 0.0:
        errors.append("teacher.base_target_speed must be finite and positive")
    corner_speed = _select(config, "teacher.speed_profile.minimum_corner_speed")
    lateral_acceleration = _select(
        config, "teacher.speed_profile.maximum_lateral_acceleration"
    )
    if _select(config, "teacher.speed_profile.type") != "curvature_limited":
        errors.append("teacher.speed_profile.type must be curvature_limited")
    if (
        not _is_finite_number(corner_speed)
        or not _is_finite_number(teacher_base_speed)
        or not 0.0 < corner_speed <= teacher_base_speed
    ):
        errors.append("teacher corner speed must be within (0, base_target_speed]")
    if not _is_finite_number(lateral_acceleration) or lateral_acceleration <= 0.0:
        errors.append("teacher maximum lateral acceleration must be positive")
    maximum_velocity = _select(config, "vehicle.vehicle.max_velocity")
    if not _is_finite_number(maximum_velocity) or maximum_velocity <= 0.0:
        errors.append("vehicle.vehicle.max_velocity must be finite and positive")
    elif _is_finite_number(teacher_base_speed) and teacher_base_speed > maximum_velocity:
        errors.append("teacher.base_target_speed cannot exceed vehicle max_velocity")
    trajectory_aided_enabled = _select(config, "reward.trajectory_aided.enabled")
    trajectory_aided_weight = _select(config, "reward.trajectory_aided.weight")
    if not isinstance(trajectory_aided_enabled, bool):
        errors.append("reward.trajectory_aided.enabled must be boolean")
    if (
        not _is_finite_number(trajectory_aided_weight)
        or trajectory_aided_weight < 0.0
    ):
        errors.append("reward.trajectory_aided.weight must be finite and non-negative")
    elif trajectory_aided_enabled and trajectory_aided_weight <= 0.0:
        errors.append("enabled trajectory-aided reward requires a positive weight")
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
        lateral_min = _select(config, "npc.randomization.lateral_offset.min")
        lateral_max = _select(config, "npc.randomization.lateral_offset.max")
        if (
            not _is_finite_number(lateral_min)
            or not _is_finite_number(lateral_max)
            or lateral_min > lateral_max
        ):
            errors.append("NPC lateral_offset bounds must be finite and ordered")
    vehicle_profile = _select(config, "vehicle.profile")
    if vehicle_profile not in {"f1tenth_nominal", "aichallenge_kart"}:
        errors.append("vehicle.profile must identify the simulator geometry")
    if (
        vehicle_profile == "aichallenge_kart"
        and _select(config, "vehicle.awsim_identification.steering_gain") is None
    ):
        warnings.append("AWSIM vehicle identification values remain uncalibrated")
    if _select(config, "env.domain_randomization.requires_awsim_calibration") is True:
        warnings.append("domain randomization requires measured AWSIM calibration data")
    return errors, warnings


def _resolved_container(config: Any) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True)


def _resolved_yaml(config: Any) -> str:
    from omegaconf import OmegaConf

    return OmegaConf.to_yaml(config, resolve=True, sort_keys=True)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _default_run_directory(config: Any) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    return PROJECT_ROOT / "outputs" / timestamp / str(config.experiment.name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the LiDAR-only SAC policy in vectorized F1TENTH Gym JAX.",
    )
    parser.add_argument("--config-name", default="step1_single_vehicle")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Return success after configuration validation; do not initialize JAX.",
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-transitions",
        type=_positive_int,
        help="Bounded smoke/debug override; full training uses the resolved YAML value.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("overrides", nargs="*", metavar="HYDRA_OVERRIDE")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = _compose_config(args.config_name, args.overrides)
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
            replay_warmup = int(config.agent.replay_buffer.warmup_transitions)
            actor_update_start = int(config.agent.update.actor_update_start_step)
            minimum_smoke_transitions = (
                replay_warmup
                + actor_update_start
                + int(config.agent.update.actor_behavior_blend_updates)
                + int(config.env.num_envs)
            )
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "experiment": str(config.experiment.name),
                        "stage": str(config.experiment.stage),
                        "num_envs": int(config.env.num_envs),
                        "num_agents": int(config.env.num_agents),
                        "jax_initialized": False,
                        "sac_implemented": True,
                        "submodule_initialized": (
                            F1TENTH_SUBMODULE / "f1tenth_gym_jax"
                        ).is_dir(),
                        "output": str(args.output or _default_run_directory(config)),
                        "resume_from": _select(config, "training.resume_from"),
                        "initialize_actor_from": _select(
                            config,
                            "training.initialize_actor_from",
                        ),
                        "sac_smoke_acceptance": {
                            "minimum_environment_transitions": minimum_smoke_transitions,
                            "requires_finite_values": True,
                            "requires_checkpoint_save_and_resume": True,
                            "requires_deterministic_evaluation": True,
                            "requires_replay_sample": True,
                            "requires_actor_and_critic_update": True,
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if not (F1TENTH_SUBMODULE / "f1tenth_gym_jax").is_dir():
            print(
                "ERROR: F1TENTH Gym JAX submodule is not initialized.",
                file=sys.stderr,
            )
            return 2
        run_directory = (args.output or _default_run_directory(config)).resolve()
        if run_directory.exists():
            print(f"ERROR: output directory already exists: {run_directory}", file=sys.stderr)
            return 2

        from lidar_racing_rl.sac.trainer import train_lidar_sac

        result = train_lidar_sac(
            _resolved_container(config),
            resolved_config_yaml=_resolved_yaml(config),
            run_directory=run_directory,
            repository_root=REPOSITORY_ROOT,
            submodule_path=F1TENTH_SUBMODULE,
            max_environment_transitions=args.max_transitions,
        )
        print(
            json.dumps(
                {
                    "run_directory": str(result.run_directory),
                    "environment_transitions": result.environment_transitions,
                    "session_environment_transitions": (
                        result.session_environment_transitions
                    ),
                    "learner_updates": result.learner_updates,
                    "final_checkpoint": str(result.final_checkpoint),
                    "elapsed_seconds": result.elapsed_seconds,
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        if args.debug:
            raise
        print(f"ERROR: LiDAR SAC training failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

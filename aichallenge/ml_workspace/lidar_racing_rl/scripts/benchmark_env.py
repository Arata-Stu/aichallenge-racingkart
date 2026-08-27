#!/usr/bin/env python3
"""Benchmark the vectorized LiDAR racing environment without training SAC."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
F1TENTH_SUBMODULE = PROJECT_ROOT / "repos" / "f1tenth_gym_jax"


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
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


def _validate_benchmark_config(config: Any) -> list[str]:
    errors: list[str] = []
    required_paths = (
        "env.num_envs",
        "env.num_agents",
        "env.simulator.map_name",
        "env.simulator.physics_timestep",
        "env.simulator.timestep_ratio",
        "env.lidar.num_beams",
        "env.lidar.field_of_view",
        "env.lidar.range_min",
        "env.lidar.range_max",
        "env.episode.max_steps",
        "env.episode.max_num_laps",
        "env.reset.longitudinal_spacing",
        "vehicle.vehicle.length",
        "vehicle.vehicle.width",
        "vehicle.vehicle.wheelbase",
        "vehicle.vehicle.min_steering_angle",
        "vehicle.vehicle.max_steering_angle",
        "vehicle.vehicle.min_acceleration",
        "vehicle.vehicle.max_acceleration",
        "vehicle.vehicle.min_velocity",
        "vehicle.vehicle.max_velocity",
        "reward.weights.progress",
        "reward.weights.collision",
        "npc.lateral_controller.reference_line",
        "npc.longitudinal_controller.base_target_speed",
    )
    for path in required_paths:
        if _select(config, path) is None:
            errors.append(f"required config value is missing or null: {path}")

    num_envs = _select(config, "env.num_envs")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        errors.append("env.num_envs must be a positive integer")
    if _select(config, "env.num_agents") != 4:
        errors.append("blueprint benchmark requires env.num_agents=4")
    if _select(config, "env.lidar.num_beams") != 360:
        errors.append("blueprint benchmark requires env.lidar.num_beams=360")
    if _select(config, "env.information_boundary.actor_critic_gt_access") is not False:
        errors.append("Actor/Critic GT access must remain disabled")
    if _select(config, "npc.count") != 3:
        errors.append("four-vehicle benchmark requires npc.count=3")
    if _select(config, "npc.learned") is not False:
        errors.append("benchmark NPCs must remain fixed policies")
    if _select(config, "npc.save_transitions") is not False:
        errors.append("NPC transitions must not enter the replay buffer")
    allowed_gt_consumers = list(
        _select(config, "env.information_boundary.allowed_gt_consumers") or []
    )
    if "npc_controller" not in allowed_gt_consumers:
        errors.append("benchmark NPC control requires npc_controller GT permission")
    if _select(config, "npc.lateral_controller.reference_line") != "centerline":
        errors.append("NPCs require a centerline reference")
    npc_base_speed = _select(config, "npc.longitudinal_controller.base_target_speed")
    if not _is_finite_number(npc_base_speed) or npc_base_speed <= 0.0:
        errors.append("NPC base_target_speed must be finite and positive")
    if _select(config, "curriculum.enabled") is not False:
        errors.append("curriculum scheduling is not integrated into the benchmark")
    if _select(config, "curriculum.active_phase") is not None:
        errors.append("benchmark must not claim an active curriculum phase")
    if _select(config, "training.opponent_pool_enabled") is not False:
        errors.append("past-policy opponents are not integrated into the benchmark")
    if _select(config, "env.domain_randomization.enabled") is True:
        errors.append(
            "AWSIM vehicle-response domain randomization is not integrated into "
            "the F1TENTH environment"
        )

    range_min = _select(config, "env.lidar.range_min")
    range_max = _select(config, "env.lidar.range_max")
    if (
        not _is_finite_number(range_min)
        or not _is_finite_number(range_max)
        or not 0.0 <= range_min < range_max
    ):
        errors.append("LiDAR bounds must satisfy 0 <= range_min < range_max")

    positive_paths = (
        "env.simulator.physics_timestep",
        "env.simulator.timestep_ratio",
        "env.lidar.field_of_view",
        "env.episode.max_steps",
        "env.episode.max_num_laps",
        "vehicle.vehicle.length",
        "vehicle.vehicle.width",
        "vehicle.vehicle.wheelbase",
        "vehicle.vehicle.max_steering_angle",
    )
    for path in positive_paths:
        value = _select(config, path)
        if not _is_finite_number(value) or value <= 0:
            errors.append(f"{path} must be positive")

    minimum_acceleration = _select(config, "vehicle.vehicle.min_acceleration")
    maximum_acceleration = _select(config, "vehicle.vehicle.max_acceleration")
    if (
        not _is_finite_number(minimum_acceleration)
        or not _is_finite_number(maximum_acceleration)
        or minimum_acceleration >= maximum_acceleration
    ):
        errors.append("vehicle acceleration bounds must be ordered")
    minimum_steering = _select(config, "vehicle.vehicle.min_steering_angle")
    maximum_steering = _select(config, "vehicle.vehicle.max_steering_angle")
    if (
        not _is_finite_number(minimum_steering)
        or not _is_finite_number(maximum_steering)
        or minimum_steering >= maximum_steering
    ):
        errors.append("vehicle steering bounds must be ordered")
    minimum_velocity = _select(config, "vehicle.vehicle.min_velocity")
    maximum_velocity = _select(config, "vehicle.vehicle.max_velocity")
    if (
        not _is_finite_number(minimum_velocity)
        or not _is_finite_number(maximum_velocity)
        or minimum_velocity >= maximum_velocity
    ):
        errors.append("vehicle velocity bounds must be ordered")
    return errors


def _require_submodule() -> None:
    package = F1TENTH_SUBMODULE / "f1tenth_gym_jax"
    if not package.is_dir():
        raise RuntimeError(
            "F1TENTH Gym JAX submodule is not initialized. Run "
            "`git submodule update --init --recursive` and sync the f1tenth extra."
        )


def _block_until_ready(jax_module: Any, value: Any) -> None:
    for leaf in jax_module.tree_util.tree_leaves(value):
        blocker = getattr(leaf, "block_until_ready", None)
        if blocker is not None:
            blocker()


def _process_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux (the canonical benchmark host) reports KiB.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _device_peak_memory_bytes(devices: list[Any]) -> int | None:
    peaks: list[int] = []
    for device in devices:
        memory_stats = getattr(device, "memory_stats", None)
        if not callable(memory_stats):
            continue
        stats = memory_stats()
        if not stats:
            continue
        for key in ("peak_bytes_in_use", "peak_pool_bytes", "bytes_in_use"):
            value = stats.get(key)
            if value is not None:
                peaks.append(int(value))
                break
    return max(peaks) if peaks else None


def _write_json(result: dict[str, Any], output: Path | None) -> None:
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{serialized}\n", encoding="utf-8")


def _run_benchmark(config: Any, *, steps: int, seed: int) -> dict[str, Any]:
    _require_submodule()

    # Delay every JAX import until after CLI/config validation and --dry-run.
    import jax
    import jax.numpy as jnp
    import numpy as np

    from lidar_racing_rl.envs.make_env import make_f1tenth_env
    from lidar_racing_rl.envs.scan_corruption import ScanCorruptionConfig
    from lidar_racing_rl.envs.vector_env import LidarRacingEnv, RacingEnvSettings
    from lidar_racing_rl.npc.controller import (
        initialize_npc_controller_state,
        npc_controller_step,
    )
    from lidar_racing_rl.npc.randomization import (
        NpcRandomizationBounds,
        sample_npc_episode_parameters,
    )
    from lidar_racing_rl.npc.reference_line import (
        build_reference_waypoints,
        validate_centerline_clearance,
    )

    settings = RacingEnvSettings.from_config(config.env, config.vehicle, config.reward)
    simulator = make_f1tenth_env(config.env, config.vehicle)
    scan_corruption_config = (
        ScanCorruptionConfig.from_config(config.env)
        if "scan_corruption" in config.env
        else ScanCorruptionConfig()
    )
    environment = LidarRacingEnv(
        simulator,
        settings,
        scan_corruption_config,
    )

    num_envs = int(config.env.num_envs)
    ego_actions = jnp.zeros((num_envs, 2), dtype=jnp.float32)
    npc_count = settings.num_agents - 1
    npc_indices = jnp.arange(1, settings.num_agents, dtype=jnp.int32)
    npc_bounds = NpcRandomizationBounds.from_config(config.npc)
    validate_centerline_clearance(
        simulator,
        vehicle_width=settings.vehicle_width,
        lateral_offset_min=npc_bounds.lateral_offset[0],
        lateral_offset_max=npc_bounds.lateral_offset[1],
    )
    base_controller_state = initialize_npc_controller_state(
        npc_count=npc_count,
        max_control_delay_steps=npc_bounds.control_delay_steps[1],
    )
    batched_reset_controller_state = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (num_envs, *value.shape)),
        base_controller_state,
    )
    waypoints = build_reference_waypoints(
        simulator,
        reference_line=str(config.npc.lateral_controller.reference_line),
        base_target_speed=float(
            config.npc.longitudinal_controller.base_target_speed
        ),
    )

    control_dt = (
        float(config.env.simulator.physics_timestep)
        * int(config.env.simulator.timestep_ratio)
    )
    wheelbase = float(config.vehicle.vehicle.wheelbase)
    steering_min = float(config.vehicle.vehicle.min_steering_angle)
    steering_max = float(config.vehicle.vehicle.max_steering_angle)
    distance_gain = float(config.npc.longitudinal_controller.distance_gain)
    lateral_gate = float(config.npc.longitudinal_controller.lateral_gate)
    minimum_speed = float(config.vehicle.vehicle.min_velocity)

    def sample_parameters(sample_key: Any) -> Any:
        return sample_npc_episode_parameters(
            sample_key,
            npc_count=npc_count,
            bounds=npc_bounds,
        )

    def controller_one(
        cartesian_states: Any,
        parameters: Any,
        controller_state: Any,
        step: Any,
    ) -> Any:
        return npc_controller_step(
            cartesian_states,
            waypoints,
            npc_indices,
            parameters,
            controller_state,
            step,
            wheelbase=wheelbase,
            control_dt=control_dt,
            steering_min=steering_min,
            steering_max=steering_max,
            acceleration_min=settings.min_acceleration,
            acceleration_max=settings.max_acceleration,
            distance_gain=distance_gain,
            lateral_gate=lateral_gate,
            minimum_speed=minimum_speed,
        )

    batched_controller = jax.vmap(controller_one)

    def rollout(root_key: Any) -> Any:
        root_key, reset_key, npc_parameter_key = jax.random.split(root_key, 3)
        reset_keys = jax.random.split(reset_key, num_envs)
        states, _ = environment.reset_batch(reset_keys)
        npc_parameters = jax.vmap(sample_parameters)(
            jax.random.split(npc_parameter_key, num_envs)
        )
        controller_states = batched_reset_controller_state

        def step_once(carry: Any, unused: Any) -> tuple[Any, Any]:
            (
                current_states,
                current_npc_parameters,
                current_controller_states,
                current_key,
            ) = carry
            current_key, step_key, npc_reset_key = jax.random.split(current_key, 3)
            step_keys = jax.random.split(step_key, num_envs)
            npc_actions, next_controller_states = batched_controller(
                current_states.simulator_state.cartesian_states,
                current_npc_parameters,
                current_controller_states,
                current_states.simulator_state.step,
            )
            result = environment.step_batch(
                step_keys,
                current_states,
                ego_actions,
                npc_actions,
            )
            reset_parameters = jax.vmap(sample_parameters)(
                jax.random.split(npc_reset_key, num_envs)
            )
            done = result.terminated | result.truncated

            def select_reset(reset_value: Any, current_value: Any) -> Any:
                mask = done.reshape((num_envs, *(1,) * (current_value.ndim - 1)))
                return jnp.where(mask, reset_value, current_value)

            next_npc_parameters = jax.tree.map(
                select_reset,
                reset_parameters,
                current_npc_parameters,
            )
            next_controller_states = jax.tree.map(
                select_reset,
                batched_reset_controller_state,
                next_controller_states,
            )
            metrics = (result.reward, result.terminated, result.truncated)
            return (
                result.state,
                next_npc_parameters,
                next_controller_states,
                current_key,
            ), metrics

        return jax.lax.scan(
            step_once,
            (states, npc_parameters, controller_states, root_key),
            xs=None,
            length=steps,
        )

    key = jax.random.key(seed)
    compiled_rollout = jax.jit(rollout)
    compile_started = time.perf_counter()
    executable = compiled_rollout.lower(key).compile()
    compile_seconds = time.perf_counter() - compile_started

    rollout_started = time.perf_counter()
    final_carry, metric_history = executable(key)
    _block_until_ready(jax, (final_carry, metric_history))
    rollout_seconds = time.perf_counter() - rollout_started

    rewards, terminated, truncated = jax.device_get(metric_history)
    rewards_array = np.asarray(rewards)
    environment_steps = num_envs * steps
    vehicle_steps = environment_steps * settings.num_agents
    devices = list(jax.devices())
    device_peak = _device_peak_memory_bytes(devices)

    return {
        "schema_version": 1,
        "benchmark": "lidar_racing_env",
        "num_envs": num_envs,
        "num_agents": settings.num_agents,
        "num_beams": settings.num_beams,
        "steps": steps,
        "canonical_blueprint_case": num_envs == 64 and steps == 1000,
        "action_source": "fixed_zero_ego_with_randomized_pure_pursuit_npcs",
        "npc_controller": {
            "episode_randomization": True,
            "gt_safe_following": True,
            "control_delay": True,
            "resample_on_ego_auto_reset": True,
        },
        "scan_corruption_configured": "scan_corruption" in config.env,
        "scan_corruption_enabled": scan_corruption_config.enabled,
        "compile_seconds": compile_seconds,
        "rollout_seconds": rollout_seconds,
        "environment_steps_per_second": environment_steps / rollout_seconds,
        "vehicle_steps_per_second": vehicle_steps / rollout_seconds,
        "peak_memory": _process_peak_rss_bytes(),
        "peak_memory_unit": "bytes",
        "peak_memory_scope": "process_max_rss_including_compile_and_rollout",
        "device_peak_memory_bytes": device_peak,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in devices],
        "non_finite_reward_count": int(np.count_nonzero(~np.isfinite(rewards_array))),
        "terminated_count": int(np.count_nonzero(np.asarray(terminated))),
        "truncated_count": int(np.count_nonzero(np.asarray(truncated))),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark 64x4x360 JAX rollouts through LidarRacingEnv.",
    )
    parser.add_argument("--config-name", default="step2_four_vehicle")
    parser.add_argument("--num-envs", type=_positive_int, default=64)
    parser.add_argument("--steps", type=_positive_int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the benchmark plan without importing JAX.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="HYDRA_OVERRIDE",
        help="Additional Hydra overrides, for example env.episode.max_steps=2000.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        # Hydra's struct composition can omit env.num_envs when a config-group
        # package is selected. ``++`` is intentional: override when present,
        # add when absent, then let the validator enforce the resolved value.
        overrides = [*args.overrides, f"++env.num_envs={args.num_envs}"]
        config = _compose_config(args.config_name, overrides)
        errors = _validate_benchmark_config(config)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 2

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "config_name": _normalize_config_name(args.config_name),
                        "num_envs": int(config.env.num_envs),
                        "num_agents": int(config.env.num_agents),
                        "num_beams": int(config.env.lidar.num_beams),
                        "steps": args.steps,
                        "submodule_initialized": (
                            F1TENTH_SUBMODULE / "f1tenth_gym_jax"
                        ).is_dir(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        result = _run_benchmark(config, steps=args.steps, seed=args.seed)
        _write_json(result, args.output)
        return 0
    except KeyboardInterrupt:
        print("Benchmark interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        if args.debug:
            raise
        print(f"ERROR: benchmark could not start: {error}", file=sys.stderr)
        print("Use --dry-run to validate configuration without initializing JAX.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

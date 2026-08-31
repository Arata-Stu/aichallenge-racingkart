#!/usr/bin/env python3
"""Run one Pure Pursuit teacher or fixed-action LiDAR-only rollout."""

from __future__ import annotations

import argparse
import json
import math
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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return parsed


def _normalized_action(value: str) -> float:
    parsed = float(value)
    if not -1.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("normalized actions must be within [-1, 1]")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("physical actions must be finite")
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


def _validate_config(
    config: Any,
    *,
    action_source: str,
    ego_action_values: tuple[float, float],
    npc_action_values: tuple[float, float],
) -> list[str]:
    errors: list[str] = []
    required_paths = (
        "env.num_agents",
        "env.simulator.map_name",
        "env.simulator.physics_timestep",
        "env.simulator.timestep_ratio",
        "env.lidar.num_beams",
        "env.lidar.frame_stack",
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
        "reward.weights.off_track",
        "reward.weights.smoothness",
        "reward.weights.reverse",
        "teacher.reference_line",
        "teacher.base_target_speed",
        "teacher.lookahead",
        "teacher.speed_multiplier",
        "teacher.speed_profile.type",
        "teacher.speed_profile.minimum_corner_speed",
        "teacher.speed_profile.maximum_lateral_acceleration",
    )
    for path in required_paths:
        if _select(config, path) is None:
            errors.append(f"required config value is missing or null: {path}")

    num_agents = _select(config, "env.num_agents")
    if _select(config, "env.num_envs") != 1:
        errors.append("single rollout requires env.num_envs=1")
    if num_agents not in (1, 4):
        errors.append("single rollout supports one or four configured agents")
    if _select(config, "env.lidar.num_beams") != 360:
        errors.append("single rollout requires canonical 360-beam scans")
    if _select(config, "env.information_boundary.actor_critic_gt_access") is not False:
        errors.append("Actor/Critic GT access must remain disabled")

    range_min = _select(config, "env.lidar.range_min")
    range_max = _select(config, "env.lidar.range_max")
    if (
        not _is_finite_number(range_min)
        or not _is_finite_number(range_max)
        or not 0.0 <= range_min < range_max
    ):
        errors.append("LiDAR bounds must satisfy 0 <= range_min < range_max")
    for path in (
        "env.simulator.physics_timestep",
        "env.simulator.timestep_ratio",
        "env.lidar.frame_stack",
        "env.lidar.field_of_view",
        "env.episode.max_steps",
        "env.episode.max_num_laps",
        "vehicle.vehicle.length",
        "vehicle.vehicle.width",
        "vehicle.vehicle.wheelbase",
    ):
        value = _select(config, path)
        if not _is_finite_number(value) or value <= 0:
            errors.append(f"{path} must be positive")
    allowed_gt_consumers = list(
        _select(config, "env.information_boundary.allowed_gt_consumers") or []
    )
    if action_source == "pure-pursuit" and "teacher_policy" not in allowed_gt_consumers:
        errors.append("Pure Pursuit Ego control requires teacher_policy GT permission")
    if action_source == "pure-pursuit":
        if _select(config, "teacher.reference_line") != "centerline":
            errors.append("teacher requires teacher.reference_line=centerline")
        if _select(config, "teacher.speed_profile.type") != "curvature_limited":
            errors.append("teacher.speed_profile.type must be curvature_limited")
        for path in (
            "teacher.base_target_speed",
            "teacher.lookahead",
            "teacher.speed_multiplier",
        ):
            value = _select(config, path)
            if not _is_finite_number(value) or value <= 0.0:
                errors.append(f"{path} must be finite and positive")
    if num_agents == 4:
        if _select(config, "npc.count") != 3:
            errors.append("four-vehicle rollout requires npc.count=3")
        if _select(config, "npc.learned") is not False:
            errors.append("rollout NPCs must remain fixed policies")
        if _select(config, "npc.lateral_controller.reference_line") != "centerline":
            errors.append("NPCs require a centerline reference")
        npc_base_speed = _select(
            config,
            "npc.longitudinal_controller.base_target_speed",
        )
        if not _is_finite_number(npc_base_speed) or npc_base_speed <= 0.0:
            errors.append("NPC base_target_speed must be finite and positive")
        elif npc_base_speed != _select(config, "teacher.base_target_speed"):
            errors.append("teacher and NPC base_target_speed must match")
        if (
            action_source == "pure-pursuit"
            and "npc_controller" not in allowed_gt_consumers
        ):
            errors.append("Pure Pursuit NPC control requires npc_controller GT permission")
        if _select(config, "curriculum.enabled") is not False:
            errors.append("curriculum scheduling is not integrated into this rollout")
        if _select(config, "curriculum.active_phase") is not None:
            errors.append("diagnostic rollout must not claim an active curriculum phase")
        if _select(config, "training.opponent_pool_enabled") is not False:
            errors.append("past-policy opponents are not integrated into this rollout")
    elif action_source == "fixed" and npc_action_values != (0.0, 0.0):
        errors.append("NPC action options cannot be used in a single-vehicle rollout")

    if action_source == "pure-pursuit" and (
        ego_action_values != (0.0, 0.0) or npc_action_values != (0.0, 0.0)
    ):
        errors.append("fixed action options require --action-source fixed")

    steering, acceleration = npc_action_values
    steering_min = _select(config, "vehicle.vehicle.min_steering_angle")
    steering_max = _select(config, "vehicle.vehicle.max_steering_angle")
    acceleration_min = _select(config, "vehicle.vehicle.min_acceleration")
    acceleration_max = _select(config, "vehicle.vehicle.max_acceleration")
    if (
        not _is_finite_number(steering_min)
        or not _is_finite_number(steering_max)
        or steering_min >= steering_max
    ):
        errors.append("vehicle steering bounds must be finite and ordered")
    if (
        not _is_finite_number(acceleration_min)
        or not _is_finite_number(acceleration_max)
        or acceleration_min >= acceleration_max
    ):
        errors.append("vehicle acceleration bounds must be finite and ordered")
    minimum_velocity = _select(config, "vehicle.vehicle.min_velocity")
    maximum_velocity = _select(config, "vehicle.vehicle.max_velocity")
    if (
        not _is_finite_number(minimum_velocity)
        or not _is_finite_number(maximum_velocity)
        or minimum_velocity >= maximum_velocity
    ):
        errors.append("vehicle velocity bounds must be finite and ordered")
    if action_source == "fixed" and (
        _is_finite_number(steering_min)
        and _is_finite_number(steering_max)
        and not steering_min <= steering <= steering_max
    ):
        errors.append("NPC steering is outside the configured physical bounds")
    if action_source == "fixed" and (
        _is_finite_number(acceleration_min)
        and _is_finite_number(acceleration_max)
        and not acceleration_min <= acceleration <= acceleration_max
    ):
        errors.append("NPC acceleration is outside the configured physical bounds")
    if _select(config, "env.domain_randomization.enabled") is True:
        errors.append(
            "AWSIM vehicle-response domain randomization is not integrated into "
            "the F1TENTH environment"
        )
    return errors


def _require_submodule() -> None:
    if not (F1TENTH_SUBMODULE / "f1tenth_gym_jax").is_dir():
        raise RuntimeError(
            "F1TENTH Gym JAX submodule is not initialized. Run "
            "`git submodule update --init --recursive` and sync the f1tenth extra."
        )


def _block_until_ready(jax_module: Any, value: Any) -> None:
    for leaf in jax_module.tree_util.tree_leaves(value):
        blocker = getattr(leaf, "block_until_ready", None)
        if blocker is not None:
            blocker()


def _write_json(result: dict[str, Any], output: Path | None) -> None:
    serialized = json.dumps(result, allow_nan=False, indent=2, sort_keys=True)
    print(serialized)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{serialized}\n", encoding="utf-8")


def _write_trace_svg(
    output: Path,
    track: Any,
    poses: Any,
    terminated: Any,
    truncated: Any,
    collisions: Any,
    off_tracks: Any,
    vehicle_length: float,
    vehicle_width: float,
) -> None:
    """Render the ordered centerline, boundaries, and rollout trace to SVG."""

    import numpy as np

    centerline = track.centerline
    center_x = np.asarray(centerline.xs, dtype=float)
    center_y = np.asarray(centerline.ys, dtype=float)
    stored_yaw = getattr(centerline, "psis", None)
    if stored_yaw is None:
        delta_x = np.roll(center_x, -1) - np.roll(center_x, 1)
        delta_y = np.roll(center_y, -1) - np.roll(center_y, 1)
        center_yaw = np.arctan2(delta_y, delta_x)
    else:
        center_yaw = np.asarray(stored_yaw, dtype=float)
    left_widths = np.asarray(track.left_widths, dtype=float)
    right_widths = np.asarray(track.right_widths, dtype=float)
    pose_array = np.asarray(poses, dtype=float)
    if (
        center_x.ndim != 1
        or center_y.shape != center_x.shape
        or center_yaw.shape != center_x.shape
        or left_widths.shape != center_x.shape
        or right_widths.shape != center_x.shape
        or pose_array.ndim != 2
        or pose_array.shape[1] < 2
    ):
        raise RuntimeError("track and rollout arrays do not satisfy the SVG trace contract")

    normal_x = -np.sin(center_yaw)
    normal_y = np.cos(center_yaw)
    left_x = center_x + left_widths * normal_x
    left_y = center_y + left_widths * normal_y
    right_x = center_x - right_widths * normal_x
    right_y = center_y - right_widths * normal_y
    all_x = np.concatenate((left_x, right_x, pose_array[:, 0]))
    all_y = np.concatenate((left_y, right_y, pose_array[:, 1]))
    if not np.all(np.isfinite(all_x)) or not np.all(np.isfinite(all_y)):
        raise RuntimeError("cannot render non-finite track or rollout coordinates")

    canvas_width = 1200.0
    canvas_height = 900.0
    margin = 50.0
    span_x = max(float(np.max(all_x) - np.min(all_x)), 1.0e-6)
    span_y = max(float(np.max(all_y) - np.min(all_y)), 1.0e-6)
    scale = min(
        (canvas_width - 2.0 * margin) / span_x,
        (canvas_height - 2.0 * margin) / span_y,
    )
    offset_x = margin + 0.5 * (canvas_width - 2.0 * margin - span_x * scale)
    offset_y = margin + 0.5 * (canvas_height - 2.0 * margin - span_y * scale)
    min_x = float(np.min(all_x))
    max_y = float(np.max(all_y))

    def point(x_value: float, y_value: float) -> str:
        x_pixel = offset_x + (float(x_value) - min_x) * scale
        y_pixel = offset_y + (max_y - float(y_value)) * scale
        return f"{x_pixel:.2f},{y_pixel:.2f}"

    def polyline(xs: Any, ys: Any) -> str:
        return " ".join(point(x_value, y_value) for x_value, y_value in zip(xs, ys))

    done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
    collision = np.asarray(collisions, dtype=bool)
    off_track = np.asarray(off_tracks, dtype=bool)
    segments: list[Any] = []
    start = 0
    for index in np.flatnonzero(done):
        segments.append(pose_array[start : index + 1])
        start = int(index) + 1
    if start < pose_array.shape[0]:
        segments.append(pose_array[start:])

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width:.0f}" '
            f'height="{canvas_height:.0f}" viewBox="0 0 {canvas_width:.0f} '
            f'{canvas_height:.0f}">'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<g fill="none" stroke-linejoin="round" stroke-linecap="round">',
        f'<polyline points="{polyline(left_x, left_y)}" stroke="#111827" stroke-width="2"/>',
        f'<polyline points="{polyline(right_x, right_y)}" stroke="#111827" stroke-width="2"/>',
        (
            f'<polyline points="{polyline(center_x, center_y)}" stroke="#a855f7" '
            'stroke-width="1.5" stroke-dasharray="6 5"/>'
        ),
    ]
    for segment in segments:
        if segment.shape[0] >= 2:
            lines.append(
                f'<polyline points="{polyline(segment[:, 0], segment[:, 1])}" '
                'stroke="#0284c7" stroke-width="2" opacity="0.8"/>'
            )
    lines.append('</g>')
    for index in np.flatnonzero(done):
        color = "#dc2626" if collision[index] else "#f97316"
        cx, cy = point(pose_array[index, 0], pose_array[index, 1]).split(",")
        lines.append(
            f'<circle cx="{cx}" cy="{cy}" r="4" fill="{color}" '
            'stroke="#ffffff" stroke-width="1"/>'
        )
        yaw = pose_array[index, 2]
        local_corners = np.asarray(
            [
                [0.5 * vehicle_length, 0.5 * vehicle_width],
                [0.5 * vehicle_length, -0.5 * vehicle_width],
                [-0.5 * vehicle_length, -0.5 * vehicle_width],
                [-0.5 * vehicle_length, 0.5 * vehicle_width],
            ]
        )
        rotation = np.asarray(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ]
        )
        corners = local_corners @ rotation.T + pose_array[index, 0:2]
        lines.append(
            f'<polygon points="{polyline(corners[:, 0], corners[:, 1])}" '
            f'fill="{color}" fill-opacity="0.12" stroke="{color}" '
            'stroke-width="1"/>'
        )
    for index in np.flatnonzero(off_track & ~collision):
        cx, cy = point(pose_array[index, 0], pose_array[index, 1]).split(",")
        lines.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="#f97316"/>')
    lines.extend(
        (
            '<g font-family="sans-serif" font-size="16" fill="#111827">',
            '<text x="20" y="28">boundary</text>',
            '<text x="120" y="28" fill="#a855f7">centerline order</text>',
            '<text x="290" y="28" fill="#0284c7">rollout</text>',
            '<text x="370" y="28" fill="#dc2626">collision</text>',
            '<text x="455" y="28" fill="#f97316">off-track</text>',
            '</g>',
            '</svg>',
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_rollout(
    config: Any,
    *,
    steps: int,
    seed: int,
    action_source: str,
    lookahead: float,
    speed_multiplier: float,
    ego_action_values: tuple[float, float],
    npc_action_values: tuple[float, float],
    trace_svg: Path | None,
) -> dict[str, Any]:
    _require_submodule()

    # Keep --help and --dry-run usable without importing or initializing JAX.
    import jax
    import jax.numpy as jnp
    import numpy as np

    from lidar_racing_rl.envs.action import normalize_physical_action
    from lidar_racing_rl.envs.make_env import make_f1tenth_env
    from lidar_racing_rl.envs.scan_corruption import ScanCorruptionConfig
    from lidar_racing_rl.envs.vector_env import LidarRacingEnv, RacingEnvSettings
    from lidar_racing_rl.npc.controller import (
        initialize_npc_controller_state,
        npc_controller_step,
    )
    from lidar_racing_rl.npc.pure_pursuit import pure_pursuit_actions
    from lidar_racing_rl.npc.reference_line import (
        build_reference_waypoints,
        validate_centerline_clearance,
    )
    from lidar_racing_rl.npc.randomization import (
        NpcRandomizationBounds,
        sample_npc_episode_parameters,
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
    fixed_ego_action = jnp.asarray(ego_action_values, dtype=jnp.float32)
    if settings.num_agents == 1:
        fixed_npc_action = jnp.empty((0, 2), dtype=jnp.float32)
    else:
        fixed_npc_action = jnp.broadcast_to(
            jnp.asarray(npc_action_values, dtype=jnp.float32),
            (settings.num_agents - 1, 2),
        )

    if action_source == "pure-pursuit":
        waypoints = build_reference_waypoints(
            simulator,
            reference_line=str(config.teacher.reference_line),
            base_target_speed=float(config.teacher.base_target_speed),
            minimum_corner_speed=float(
                config.teacher.speed_profile.minimum_corner_speed
            ),
            maximum_lateral_acceleration=float(
                config.teacher.speed_profile.maximum_lateral_acceleration
            ),
        )
        teacher_lookahead = jnp.asarray([lookahead], dtype=jnp.float32)
        teacher_speed_multiplier = jnp.asarray(
            [speed_multiplier],
            dtype=jnp.float32,
        )
        waypoint_speed_range = (
            float(jnp.min(waypoints[:, 2])),
            float(jnp.max(waypoints[:, 2])),
        )
    else:
        waypoints = None
        teacher_lookahead = None
        teacher_speed_multiplier = None
        waypoint_speed_range = None

    control_dt = (
        float(config.env.simulator.physics_timestep)
        * int(config.env.simulator.timestep_ratio)
    )
    wheelbase = float(config.vehicle.vehicle.wheelbase)
    steering_min = float(config.vehicle.vehicle.min_steering_angle)
    steering_max = float(config.vehicle.vehicle.max_steering_angle)

    def select_ego_action(state: Any) -> Any:
        if action_source == "fixed":
            return fixed_ego_action

        # Simulator pose is GT used only by this diagnostic Ego teacher.  It is
        # never concatenated to or returned as the learned Actor observation.
        ego_physical_action = pure_pursuit_actions(
            state.simulator_state.cartesian_states[0:1],
            waypoints,
            teacher_lookahead,
            teacher_speed_multiplier,
            wheelbase=wheelbase,
            control_dt=control_dt,
            steering_min=steering_min,
            steering_max=steering_max,
            acceleration_min=settings.min_acceleration,
            acceleration_max=settings.max_acceleration,
        )[0]
        return normalize_physical_action(
            ego_physical_action,
            max_steering_angle=settings.max_steering_angle,
            min_acceleration=settings.min_acceleration,
            max_acceleration=settings.max_acceleration,
        )

    use_npc_controller = action_source == "pure-pursuit" and settings.num_agents == 4
    if use_npc_controller:
        npc_count = settings.num_agents - 1
        npc_indices = jnp.arange(1, settings.num_agents, dtype=jnp.int32)
        npc_bounds = NpcRandomizationBounds.from_config(config.npc)
        validate_centerline_clearance(
            simulator,
            vehicle_width=settings.vehicle_width,
            lateral_offset_min=npc_bounds.lateral_offset[0],
            lateral_offset_max=npc_bounds.lateral_offset[1],
        )
        reset_controller_state = initialize_npc_controller_state(
            npc_count=npc_count,
            max_control_delay_steps=npc_bounds.control_delay_steps[1],
        )
        distance_gain = float(config.npc.longitudinal_controller.distance_gain)
        lateral_gate = float(config.npc.longitudinal_controller.lateral_gate)
        minimum_speed = float(config.vehicle.vehicle.min_velocity)

        def sample_parameters(sample_key: Any) -> Any:
            return sample_npc_episode_parameters(
                sample_key,
                npc_count=npc_count,
                bounds=npc_bounds,
            )

        def select_npc_actions(
            state: Any,
            parameters: Any,
            controller_state: Any,
        ) -> Any:
            # All GT accepted here is confined to the fixed NPC policy.
            return npc_controller_step(
                state.simulator_state.cartesian_states,
                waypoints,
                npc_indices,
                parameters,
                controller_state,
                state.simulator_state.step,
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

    def rollout(root_key: Any) -> Any:
        if use_npc_controller:
            root_key, reset_key, npc_parameter_key = jax.random.split(root_key, 3)
        else:
            root_key, reset_key = jax.random.split(root_key)
        state, observation = environment.reset(reset_key)

        if use_npc_controller:
            npc_parameters = sample_parameters(npc_parameter_key)
            controller_state = reset_controller_state

            def step_once_with_npcs(
                carry: Any,
                unused: Any,
            ) -> tuple[Any, Any]:
                (
                    current_state,
                    _,
                    current_parameters,
                    current_controller_state,
                    current_key,
                ) = carry
                current_key, step_key, npc_reset_key = jax.random.split(
                    current_key,
                    3,
                )
                ego_action = select_ego_action(current_state)
                npc_action, next_controller_state = select_npc_actions(
                    current_state,
                    current_parameters,
                    current_controller_state,
                )
                result = environment.step(
                    step_key,
                    current_state,
                    ego_action,
                    npc_action,
                )
                reset_parameters = sample_parameters(npc_reset_key)
                done = result.terminated | result.truncated
                next_parameters = jax.tree.map(
                    lambda reset_value, current_value: jax.lax.select(
                        done,
                        reset_value,
                        current_value,
                    ),
                    reset_parameters,
                    current_parameters,
                )
                next_controller_state = jax.tree.map(
                    lambda reset_value, current_value: jax.lax.select(
                        done,
                        reset_value,
                        current_value,
                    ),
                    reset_controller_state,
                    next_controller_state,
                )
                ego_speed = result.diagnostics.ego_speed
                metrics = (
                    result.reward,
                    result.terminated,
                    result.truncated,
                    ego_speed,
                    ego_action,
                    result.diagnostics.collision,
                    result.diagnostics.collision_with_opponent,
                    result.diagnostics.collision_with_wall,
                    result.diagnostics.unsafe_contact,
                    result.diagnostics.nearest_opponent_distance,
                    result.diagnostics.off_track,
                    result.diagnostics.race_complete,
                    result.diagnostics.unrecoverable,
                    result.diagnostics.terminal_ego_pose,
                    result.diagnostics.terminal_ego_frenet_pose,
                )
                next_carry = (
                    result.state,
                    result.observation,
                    next_parameters,
                    next_controller_state,
                    current_key,
                )
                return next_carry, metrics

            return jax.lax.scan(
                step_once_with_npcs,
                (
                    state,
                    observation,
                    npc_parameters,
                    controller_state,
                    root_key,
                ),
                xs=None,
                length=steps,
            )

        def step_once(carry: Any, unused: Any) -> tuple[Any, Any]:
            current_state, _, current_key = carry
            current_key, step_key = jax.random.split(current_key)
            ego_action = select_ego_action(current_state)
            npc_action = fixed_npc_action
            result = environment.step(
                step_key,
                current_state,
                ego_action,
                npc_action,
            )
            ego_speed = result.diagnostics.ego_speed
            metrics = (
                result.reward,
                result.terminated,
                result.truncated,
                ego_speed,
                ego_action,
                result.diagnostics.collision,
                result.diagnostics.collision_with_opponent,
                result.diagnostics.collision_with_wall,
                result.diagnostics.unsafe_contact,
                result.diagnostics.nearest_opponent_distance,
                result.diagnostics.off_track,
                result.diagnostics.race_complete,
                result.diagnostics.unrecoverable,
                result.diagnostics.terminal_ego_pose,
                result.diagnostics.terminal_ego_frenet_pose,
            )
            return (result.state, result.observation, current_key), metrics

        return jax.lax.scan(
            step_once,
            (state, observation, root_key),
            xs=None,
            length=steps,
        )

    key = jax.random.key(seed)
    started = time.perf_counter()
    final_carry, metric_history = jax.jit(rollout)(key)
    _block_until_ready(jax, (final_carry, metric_history))
    elapsed = time.perf_counter() - started

    if use_npc_controller:
        _, final_observation, _, _, _ = final_carry
    else:
        _, final_observation, _ = final_carry
    (
        rewards,
        terminated,
        truncated,
        ego_speeds,
        ego_actions,
        collisions,
        collisions_with_opponent,
        collisions_with_wall,
        unsafe_contacts,
        nearest_opponent_distances,
        off_tracks,
        race_completes,
        unrecoverables,
        ego_poses,
        ego_frenet_poses,
    ) = jax.device_get(metric_history)
    observation_array = np.asarray(jax.device_get(final_observation))
    rewards_array = np.asarray(rewards)
    ego_speeds_array = np.asarray(ego_speeds)
    ego_actions_array = np.asarray(ego_actions)
    final_observation_finite = bool(np.all(np.isfinite(observation_array)))
    non_finite_reward_count = int(
        np.count_nonzero(~np.isfinite(rewards_array))
    )
    ego_speeds_finite = bool(np.all(np.isfinite(ego_speeds_array)))
    ego_actions_finite = bool(np.all(np.isfinite(ego_actions_array)))
    maximum_ego_speed = (
        float(np.max(ego_speeds_array)) if ego_speeds_finite else None
    )
    reward_sum = (
        float(np.sum(rewards_array)) if non_finite_reward_count == 0 else None
    )
    mean_absolute_ego_action = (
        np.mean(np.abs(ego_actions_array), axis=0).tolist()
        if ego_actions_finite
        else None
    )
    ego_moved = maximum_ego_speed is not None and maximum_ego_speed > 1.0e-3
    collision_count = int(np.count_nonzero(np.asarray(collisions)))
    collision_with_opponent_count = int(
        np.count_nonzero(np.asarray(collisions_with_opponent))
    )
    collision_with_wall_count = int(
        np.count_nonzero(np.asarray(collisions_with_wall))
    )
    unsafe_contact_count = int(
        np.count_nonzero(np.asarray(unsafe_contacts))
    )
    off_track_count = int(np.count_nonzero(np.asarray(off_tracks)))
    race_complete_count = int(np.count_nonzero(np.asarray(race_completes)))
    unrecoverable_count = int(np.count_nonzero(np.asarray(unrecoverables)))
    unexpected_termination_count = int(
        np.count_nonzero(
            np.asarray(collisions)
            | np.asarray(off_tracks)
            | np.asarray(unrecoverables)
        )
    )
    frenet_array = np.asarray(ego_frenet_poses, dtype=float)
    sample_s = np.asarray(simulator.track.centerline.s, dtype=float)
    query_s = np.mod(frenet_array[:, 0], float(simulator.track.s_frame_max))
    left_width = np.interp(query_s, sample_s, simulator.track.left_widths)
    right_width = np.interp(query_s, sample_s, simulator.track.right_widths)
    lateral_error = frenet_array[:, 1]
    heading_error = frenet_array[:, 2]
    body_clearance = (
        0.5 * settings.vehicle_length * np.abs(np.sin(heading_error))
        + 0.5 * settings.vehicle_width * np.abs(np.cos(heading_error))
    )
    boundary_margin = np.where(
        lateral_error >= 0.0,
        left_width - lateral_error - body_clearance,
        right_width + lateral_error - body_clearance,
    )
    unexpected_mask = (
        np.asarray(collisions, dtype=bool)
        | np.asarray(off_tracks, dtype=bool)
        | np.asarray(unrecoverables, dtype=bool)
    )
    teacher_stable = (
        action_source != "pure-pursuit" or unexpected_termination_count == 0
    )
    if trace_svg is not None:
        _write_trace_svg(
            trace_svg,
            simulator.track,
            ego_poses,
            terminated,
            truncated,
            collisions,
            off_tracks,
            settings.vehicle_length,
            settings.vehicle_width,
        )
    return {
        "schema_version": 1,
        "rollout": "single_environment",
        "config_name": str(config.experiment.name),
        "num_agents": settings.num_agents,
        "num_beams": settings.num_beams,
        "frame_stack": settings.frame_stack,
        "steps": steps,
        "action_source": action_source,
        "npc_controller": (
            {
                "episode_randomization": True,
                "gt_safe_following": True,
                "control_delay": True,
                "resample_on_ego_auto_reset": True,
            }
            if use_npc_controller
            else None
        ),
        "scan_corruption_configured": "scan_corruption" in config.env,
        "scan_corruption_enabled": scan_corruption_config.enabled,
        "reset_placement": {
            "all_vehicles_reset_together": True,
            "longitudinal_spacing": settings.reset_longitudinal_spacing,
        },
        "teacher_gt_usage": (
            "pose_for_teacher_and_npc_control_only; Actor observation remains LiDAR-only"
            if action_source == "pure-pursuit"
            else None
        ),
        "elapsed_seconds_including_compile": elapsed,
        "final_observation_shape": list(observation_array.shape),
        "final_observation_finite": final_observation_finite,
        "reward_sum": reward_sum,
        "maximum_ego_speed": maximum_ego_speed,
        "ego_speeds_finite": ego_speeds_finite,
        "ego_moved": ego_moved,
        "mean_absolute_ego_action": mean_absolute_ego_action,
        "ego_actions_finite": ego_actions_finite,
        "non_finite_reward_count": non_finite_reward_count,
        "healthy": (
            final_observation_finite
            and non_finite_reward_count == 0
            and ego_speeds_finite
            and ego_actions_finite
            and (action_source == "fixed" or ego_moved)
            and teacher_stable
        ),
        "terminated_count": int(np.count_nonzero(np.asarray(terminated))),
        "truncated_count": int(np.count_nonzero(np.asarray(truncated))),
        "termination_causes": {
            "collision": collision_count,
            "collision_with_opponent": collision_with_opponent_count,
            "collision_with_wall": collision_with_wall_count,
            "off_track": off_track_count,
            "race_complete": race_complete_count,
            "unrecoverable": unrecoverable_count,
            "unexpected": unexpected_termination_count,
        },
        "interaction_diagnostics": {
            "unsafe_contact_count": unsafe_contact_count,
            "minimum_opponent_distance": (
                float(np.min(np.asarray(nearest_opponent_distances)))
                if settings.num_agents > 1
                else None
            ),
        },
        "track_clearance": {
            "vehicle_length": settings.vehicle_length,
            "vehicle_width": settings.vehicle_width,
            "maximum_absolute_lateral_error": float(
                np.max(np.abs(lateral_error))
            ),
            "maximum_absolute_heading_error": float(
                np.max(np.abs(heading_error))
            ),
            "minimum_body_boundary_margin": float(np.min(boundary_margin)),
            "minimum_unexpected_body_boundary_margin": (
                float(np.min(boundary_margin[unexpected_mask]))
                if np.any(unexpected_mask)
                else None
            ),
        },
        "teacher_stable": (
            teacher_stable if action_source == "pure-pursuit" else None
        ),
        "trace_svg": str(trace_svg) if trace_svg is not None else None,
        "pure_pursuit": (
            {
                "lookahead": lookahead,
                "speed_multiplier": speed_multiplier,
                "waypoint_speed_minimum": waypoint_speed_range[0],
                "waypoint_speed_maximum": waypoint_speed_range[1],
                "maximum_lateral_acceleration": float(
                    config.teacher.speed_profile.maximum_lateral_acceleration
                ),
            }
            if action_source == "pure-pursuit"
            else None
        ),
        "fixed_ego_normalized_action": (
            list(ego_action_values) if action_source == "fixed" else None
        ),
        "fixed_npc_physical_action": (
            list(npc_action_values) if action_source == "fixed" else None
        ),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one JIT-compiled teacher or fixed-action LidarRacingEnv rollout.",
    )
    parser.add_argument("--config-name", default="step1_single_vehicle")
    parser.add_argument("--steps", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-source",
        choices=("pure-pursuit", "fixed"),
        default="pure-pursuit",
        help="Use a GT-permitted Pure Pursuit teacher or fixed CLI actions.",
    )
    parser.add_argument("--lookahead", type=_positive_float)
    parser.add_argument("--speed-multiplier", type=_positive_float)
    parser.add_argument("--ego-steering", type=_normalized_action, default=0.0)
    parser.add_argument("--ego-acceleration", type=_normalized_action, default=0.0)
    parser.add_argument("--npc-steering", type=_finite_float, default=0.0)
    parser.add_argument("--npc-acceleration", type=_finite_float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--trace-svg",
        type=Path,
        help="Render centerline, track boundaries, trajectory, and termination points.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the rollout plan without importing JAX.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("overrides", nargs="*", metavar="HYDRA_OVERRIDE")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = _compose_config(
            args.config_name,
            [*args.overrides, "env.num_envs=1"],
        )
        ego_action_values = (args.ego_steering, args.ego_acceleration)
        npc_action_values = (args.npc_steering, args.npc_acceleration)
        errors = _validate_config(
            config,
            action_source=args.action_source,
            ego_action_values=ego_action_values,
            npc_action_values=npc_action_values,
        )
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 2
        num_agents = int(config.env.num_agents)
        lookahead = (
            args.lookahead
            if args.lookahead is not None
            else float(config.teacher.lookahead)
        )
        speed_multiplier = (
            args.speed_multiplier
            if args.speed_multiplier is not None
            else float(config.teacher.speed_multiplier)
        )

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "config_name": _normalize_config_name(args.config_name),
                        "num_envs": 1,
                        "num_agents": num_agents,
                        "num_beams": int(config.env.lidar.num_beams),
                        "steps": args.steps,
                        "action_source": args.action_source,
                        "teacher_gt_usage": (
                            "pose_for_teacher_and_npc_control_only"
                            if args.action_source == "pure-pursuit"
                            else None
                        ),
                        "actor_observation": "lidar_only",
                        "pure_pursuit": (
                            {
                                "lookahead": lookahead,
                                "speed_multiplier": speed_multiplier,
                            }
                            if args.action_source == "pure-pursuit"
                            else None
                        ),
                        "fixed_ego_normalized_action": (
                            list(ego_action_values)
                            if args.action_source == "fixed"
                            else None
                        ),
                        "fixed_npc_physical_action": (
                            list(npc_action_values)
                            if args.action_source == "fixed"
                            else None
                        ),
                        "submodule_initialized": (
                            F1TENTH_SUBMODULE / "f1tenth_gym_jax"
                        ).is_dir(),
                        "trace_svg": str(args.trace_svg) if args.trace_svg else None,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        result = _run_rollout(
            config,
            steps=args.steps,
            seed=args.seed,
            action_source=args.action_source,
            lookahead=lookahead,
            speed_multiplier=speed_multiplier,
            ego_action_values=ego_action_values,
            npc_action_values=npc_action_values,
            trace_svg=args.trace_svg,
        )
        _write_json(result, args.output)
        if not result["healthy"]:
            print(
                "ERROR: rollout health checks failed; inspect the JSON result.",
                file=sys.stderr,
            )
            return 3
        return 0
    except KeyboardInterrupt:
        print("Rollout interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        if args.debug:
            raise
        print(f"ERROR: rollout could not start: {error}", file=sys.stderr)
        print("Use --dry-run to validate configuration without initializing JAX.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

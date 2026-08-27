"""Deterministic vector policy evaluation with fixed-policy opponents."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class EvaluationResult:
    """Host metrics and checkpoint provenance for one evaluation run."""

    requested_episodes: int
    completed_episodes: int
    vector_environments: int
    environment_steps: int
    elapsed_seconds: float
    checkpoint_step: int
    metrics: dict[str, float | int]


def _value(mapping: Mapping[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"missing evaluation configuration: {'.'.join(path)}")
        current = current[key]
    return current


def _validated_episode_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("episodes must be a positive integer")
    return int(value)


def evaluate_lidar_policy(
    config: Mapping[str, Any],
    *,
    checkpoint: Path,
    episodes: int,
) -> EvaluationResult:
    """Evaluate deterministic ``tanh(mean)`` actions from LiDAR only."""

    import jax
    import jax.numpy as jnp

    from lidar_racing_rl.envs.make_env import make_f1tenth_env
    from lidar_racing_rl.envs.scan_corruption import ScanCorruptionConfig
    from lidar_racing_rl.envs.vector_env import LidarRacingEnv, RacingEnvSettings
    from lidar_racing_rl.evaluation.metrics import (
        evaluation_summary,
        initialize_evaluation_accumulator,
        select_episode_completions,
        update_evaluation_accumulator,
    )
    from lidar_racing_rl.models.actor_flax import TanhGaussianActor
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
    from lidar_racing_rl.sac.checkpoint import (
        checkpoint_config_sha256,
        load_actor_variables,
    )

    episodes = _validated_episode_count(episodes)
    num_envs = min(int(_value(config, "env", "num_envs")), episodes)
    num_agents = int(_value(config, "env", "num_agents"))
    if num_agents == 4:
        curriculum = _value(config, "curriculum")
        if _value(curriculum, "enabled") is not False:
            raise NotImplementedError(
                "Step 2 curriculum scheduling is not integrated into evaluation"
            )
        if _value(curriculum, "active_phase") is not None:
            raise ValueError(
                "uncurried Step 2 evaluation must not claim an active curriculum phase"
            )
        if _value(config, "training", "opponent_pool_enabled") is not False:
            raise NotImplementedError(
                "past-policy opponents are not integrated into evaluation"
            )
    settings = RacingEnvSettings.from_config(
        _value(config, "env"),
        _value(config, "vehicle"),
        _value(config, "reward"),
    )
    simulator = make_f1tenth_env(_value(config, "env"), _value(config, "vehicle"))
    env_config = _value(config, "env")
    corruption = (
        ScanCorruptionConfig.from_config(env_config)
        if "scan_corruption" in env_config
        else ScanCorruptionConfig()
    )
    environment = LidarRacingEnv(simulator, settings, corruption)
    teacher_config = _value(config, "teacher")
    vehicle = _value(config, "vehicle", "vehicle")
    waypoints = build_reference_waypoints(
        simulator,
        reference_line=str(_value(teacher_config, "reference_line")),
        base_target_speed=float(_value(teacher_config, "base_target_speed")),
    )
    validate_centerline_clearance(
        simulator,
        vehicle_width=float(_value(vehicle, "width")),
        lateral_offset_min=0.0,
        lateral_offset_max=0.0,
    )

    actor_variables, checkpoint_metadata = load_actor_variables(checkpoint)
    if checkpoint_metadata.architecture_version != "lidar_actor_conv1d_v1":
        raise ValueError("checkpoint architecture is not supported by this evaluator")
    if checkpoint_metadata.config_sha256 != checkpoint_config_sha256(config):
        raise ValueError("checkpoint config hash does not match the evaluation config")
    actor = TanhGaussianActor(
        log_std_min=float(_value(config, "agent", "actor", "log_std_min")),
        log_std_max=float(_value(config, "agent", "actor", "log_std_max")),
    )

    def deterministic_action(observation: Any) -> Any:
        return actor.apply(
            actor_variables,
            observation,
            method=actor.deterministic_action,
        )

    policy = jax.jit(deterministic_action)
    step_environments = jax.jit(environment.step_batch)
    evaluation_seed = int(_value(config, "seed", "evaluation"))
    key = jax.random.key(evaluation_seed)
    key, reset_key = jax.random.split(key)
    states, observations = jax.jit(environment.reset_batch)(
        jax.random.split(reset_key, num_envs)
    )
    accumulator = initialize_evaluation_accumulator(num_envs)
    update_metrics = jax.jit(update_evaluation_accumulator)

    npc_parameters = None
    npc_controller_states = None
    npc_action_function = None
    npc_reset_function = None
    if num_agents == 4:
        npc_count = 3
        npc_indices = jnp.arange(1, 4, dtype=jnp.int32)
        npc_bounds = NpcRandomizationBounds.from_config(_value(config, "npc"))
        validate_centerline_clearance(
            simulator,
            vehicle_width=float(_value(vehicle, "width")),
            lateral_offset_min=npc_bounds.lateral_offset[0],
            lateral_offset_max=npc_bounds.lateral_offset[1],
        )
        base_controller_state = initialize_npc_controller_state(
            npc_count=npc_count,
            max_control_delay_steps=npc_bounds.control_delay_steps[1],
        )
        reset_controller_states = jax.tree.map(
            lambda value: jnp.broadcast_to(value, (num_envs, *value.shape)),
            base_controller_state,
        )
        key, npc_key = jax.random.split(key)
        npc_parameters = jax.vmap(
            lambda sample_key: sample_npc_episode_parameters(
                sample_key,
                npc_count=npc_count,
                bounds=npc_bounds,
            )
        )(jax.random.split(npc_key, num_envs))
        npc_controller_states = reset_controller_states
        npc_config = _value(config, "npc")
        control_dt = float(_value(config, "env", "simulator", "physics_timestep")) * int(
            _value(config, "env", "simulator", "timestep_ratio")
        )

        def npc_one(cartesian_states: Any, parameters: Any, controller_state: Any, step: Any):
            # GT is consumed only by the fixed evaluation opponents.
            return npc_controller_step(
                cartesian_states,
                waypoints,
                npc_indices,
                parameters,
                controller_state,
                step,
                wheelbase=float(_value(vehicle, "wheelbase")),
                control_dt=control_dt,
                steering_min=float(_value(vehicle, "min_steering_angle")),
                steering_max=float(_value(vehicle, "max_steering_angle")),
                acceleration_min=settings.min_acceleration,
                acceleration_max=settings.max_acceleration,
                distance_gain=float(
                    _value(npc_config, "longitudinal_controller", "distance_gain")
                ),
                lateral_gate=float(
                    _value(npc_config, "longitudinal_controller", "lateral_gate")
                ),
                minimum_speed=float(_value(vehicle, "min_velocity")),
            )

        npc_action_function = jax.jit(jax.vmap(npc_one))

        def reset_npcs(done: Any, current_parameters: Any, next_states: Any, reset_key: Any):
            reset_parameters = jax.vmap(
                lambda sample_key: sample_npc_episode_parameters(
                    sample_key,
                    npc_count=npc_count,
                    bounds=npc_bounds,
                )
            )(jax.random.split(reset_key, num_envs))

            def select(reset_value: Any, current_value: Any) -> Any:
                mask = done.reshape((num_envs, *(1,) * (current_value.ndim - 1)))
                return jnp.where(mask, reset_value, current_value)

            return (
                jax.tree.map(select, reset_parameters, current_parameters),
                jax.tree.map(select, reset_controller_states, next_states),
            )

        npc_reset_function = jax.jit(reset_npcs)

    maximum_steps = int(_value(config, "env", "episode", "max_steps"))
    maximum_collections = maximum_steps * math.ceil(episodes / num_envs) * 2
    collections = 0
    started = time.perf_counter()
    while int(jax.device_get(accumulator.completed_episodes)) < episodes:
        if collections >= maximum_collections:
            raise RuntimeError("evaluation did not finish the requested episodes")
        collections += 1
        ego_actions = policy(observations)
        if num_agents == 4:
            assert npc_action_function is not None
            npc_actions, next_npc_controller_states = npc_action_function(
                states.simulator_state.cartesian_states,
                npc_parameters,
                npc_controller_states,
                states.simulator_state.step,
            )
        else:
            npc_actions = jnp.empty((num_envs, 0, 2), dtype=jnp.float32)
            next_npc_controller_states = None

        key, step_key, npc_reset_key = jax.random.split(key, 3)
        result = step_environments(
            jax.random.split(step_key, num_envs),
            states,
            ego_actions,
            npc_actions,
        )
        diagnostics = result.diagnostics
        done = result.terminated | result.truncated
        remaining_episodes = (
            jnp.asarray(episodes, dtype=jnp.int32) - accumulator.completed_episodes
        )
        accepted_done = select_episode_completions(done, remaining_episodes)
        accumulator = update_metrics(
            accumulator,
            reward=result.reward,
            progress_delta=diagnostics.progress_delta,
            speed=diagnostics.ego_speed,
            normalized_action=ego_actions,
            collision=diagnostics.collision,
            off_track=diagnostics.off_track,
            race_complete=diagnostics.race_complete,
            relative_progress=diagnostics.relative_progress,
            pass_count=diagnostics.pass_count,
            collision_with_opponent=diagnostics.collision_with_opponent,
            collision_with_wall=diagnostics.collision_with_wall,
            unsafe_contact=diagnostics.unsafe_contact,
            following_vehicle=diagnostics.following_vehicle,
            stalled_behind_vehicle=diagnostics.stalled_behind_vehicle,
            nearest_opponent_distance=diagnostics.nearest_opponent_distance,
            ego_rank=diagnostics.ego_rank,
            opponent_present=diagnostics.opponent_present,
            terminated=result.terminated & accepted_done,
            truncated=result.truncated & accepted_done,
        )
        states = result.state
        observations = result.observation
        if num_agents == 4:
            assert npc_reset_function is not None
            npc_parameters, npc_controller_states = npc_reset_function(
                done,
                npc_parameters,
                next_npc_controller_states,
                npc_reset_key,
            )

    elapsed_seconds = time.perf_counter() - started
    summary = jax.device_get(evaluation_summary(accumulator))
    host_metrics: dict[str, float | int] = {}
    for name, value in summary.items():
        scalar = value.item()
        host_metrics[name] = (
            int(scalar)
            if name in ("episodes", "opponent_episodes")
            else float(scalar)
        )
    control_dt = float(_value(config, "env", "simulator", "physics_timestep")) * int(
        _value(config, "env", "simulator", "timestep_ratio")
    )
    host_metrics["mean_episode_seconds"] = (
        float(host_metrics["mean_episode_steps"]) * control_dt
    )
    host_metrics["mean_lap_seconds"] = (
        float(host_metrics["mean_lap_steps"]) * control_dt
    )
    host_metrics["mean_time_to_first_pass_seconds"] = (
        float(host_metrics["mean_steps_to_first_pass"]) * control_dt
    )
    host_metrics["mean_follow_duration_seconds"] = (
        float(host_metrics["mean_follow_duration_steps"]) * control_dt
    )
    return EvaluationResult(
        requested_episodes=episodes,
        completed_episodes=int(host_metrics["episodes"]),
        vector_environments=num_envs,
        environment_steps=collections * num_envs,
        elapsed_seconds=elapsed_seconds,
        checkpoint_step=checkpoint_metadata.step,
        metrics=host_metrics,
    )


__all__ = ["EvaluationResult", "evaluate_lidar_policy"]

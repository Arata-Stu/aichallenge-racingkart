"""End-to-end vectorized LiDAR-only SAC training orchestration."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ARCHITECTURE_VERSION = "lidar_actor_conv1d_v1"


@dataclass(frozen=True)
class TrainingResult:
    """Host-side summary returned after a bounded training run."""

    run_directory: Path
    # Cumulative across warm-restart checkpoints.
    environment_transitions: int
    session_environment_transitions: int
    learner_updates: int
    final_checkpoint: Path
    elapsed_seconds: float


def _value(mapping: Mapping[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"missing training configuration: {'.'.join(path)}")
        current = current[key]
    return current


def _status_is_dirty(status: str) -> bool:
    return any(line and not line.startswith("## ") for line in status.splitlines())


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def train_lidar_sac(
    config: Mapping[str, Any],
    *,
    resolved_config_yaml: str,
    run_directory: Path,
    repository_root: Path,
    submodule_path: Path,
    max_environment_transitions: int | None = None,
) -> TrainingResult:
    """Train one Ego policy; all environment/vehicle/beam axes stay in JAX.

    This function imports and initializes JAX.  CLI validation and dry-run
    paths must therefore call it only after every lightweight preflight check.
    """

    import jax
    import jax.numpy as jnp

    from lidar_racing_rl.envs.action import normalize_physical_action
    from lidar_racing_rl.envs.make_env import make_f1tenth_env
    from lidar_racing_rl.envs.reward import trajectory_aided_action_reward
    from lidar_racing_rl.envs.scan_corruption import ScanCorruptionConfig
    from lidar_racing_rl.envs.vector_env import LidarRacingEnv, RacingEnvSettings
    from lidar_racing_rl.models.actor_flax import TanhGaussianActor
    from lidar_racing_rl.models.critic_flax import TwinQCritic
    from lidar_racing_rl.npc.controller import (
        initialize_npc_controller_state,
        npc_controller_step,
    )
    from lidar_racing_rl.npc.longitudinal_control import (
        limit_speed_for_leading_vehicle,
    )
    from lidar_racing_rl.npc.pure_pursuit import (
        ordered_braking_target_speeds,
        pure_pursuit_actions,
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
        load_checkpoint,
        prune_checkpoints,
        read_checkpoint_metadata,
        save_checkpoint,
    )
    from lidar_racing_rl.sac.collector import transition_from_step
    from lidar_racing_rl.sac.learner import (
        SACLearnerConfig,
        initialize_sac_state,
        make_sac_update,
    )
    from lidar_racing_rl.sac.replay import (
        ReplayBufferConfig,
        initialize_replay_buffer,
        insert_batch,
        replay_memory_bytes,
        sample_batch,
    )
    from lidar_racing_rl.sac.run_artifacts import (
        append_jsonl,
        capture_repository_snapshot,
        write_run_artifacts,
    )
    from lidar_racing_rl.sac.train_state import create_sac_optimizers

    snapshot = capture_repository_snapshot(repository_root, (submodule_path,))
    require_clean = bool(_value(config, "training", "require_clean_repositories"))
    dirty_submodules = [
        path for path, status in snapshot.submodule_status.items() if _status_is_dirty(status)
    ]
    if require_clean and (_status_is_dirty(snapshot.root_status) or dirty_submodules):
        raise RuntimeError(
            "training requires committed source so checkpoint/manifest SHAs are truthful; "
            "commit the reviewed root and fork changes before starting"
        )
    write_run_artifacts(
        run_directory,
        resolved_config_yaml=resolved_config_yaml,
        repository=snapshot,
    )

    num_envs = int(_value(config, "env", "num_envs"))
    num_agents = int(_value(config, "env", "num_agents"))
    if num_agents == 4:
        curriculum = _value(config, "curriculum")
        if _value(curriculum, "enabled") is not False:
            raise NotImplementedError(
                "Step 2 curriculum scheduling is not integrated into the trainer"
            )
        if _value(curriculum, "active_phase") is not None:
            raise ValueError(
                "uncurried Step 2 training must not claim an active curriculum phase"
            )
        if _value(config, "training", "opponent_pool_enabled") is not False:
            raise NotImplementedError(
                "past-policy opponents are not integrated into the trainer"
            )
    configured_transitions = int(
        _value(config, "training", "total_environment_transitions")
    )
    if num_envs < 1 or configured_transitions < 1:
        raise ValueError("num_envs and configured transitions must be positive")
    if max_environment_transitions is not None and (
        isinstance(max_environment_transitions, bool)
        or not isinstance(max_environment_transitions, int)
        or max_environment_transitions < 1
    ):
        raise ValueError("max_environment_transitions must be a positive integer")

    config_hash = checkpoint_config_sha256(config)
    resume_value = _value(config, "training", "resume_from")
    resume_checkpoint = Path(str(resume_value)) if resume_value is not None else None
    resume_metadata = None
    cumulative_environment_transitions = 0
    resume_learner_step = 0
    if resume_checkpoint is not None:
        resume_metadata = read_checkpoint_metadata(resume_checkpoint)
        if resume_metadata.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError("checkpoint architecture version does not match the learner")
        if resume_metadata.config_sha256 != config_hash:
            raise ValueError("checkpoint config hash does not match the resolved run config")
        cumulative_environment_transitions = resume_metadata.environment_transitions
        resume_learner_step = resume_metadata.step

    remaining_configured_transitions = (
        configured_transitions - cumulative_environment_transitions
    )
    if remaining_configured_transitions <= 0:
        raise ValueError(
            "checkpoint already reached the configured total environment transitions"
        )
    session_transition_budget = remaining_configured_transitions
    if max_environment_transitions is not None:
        session_transition_budget = min(
            session_transition_budget,
            max_environment_transitions,
        )

    settings = RacingEnvSettings.from_config(
        _value(config, "env"),
        _value(config, "vehicle"),
        _value(config, "reward"),
    )
    simulator = make_f1tenth_env(_value(config, "env"), _value(config, "vehicle"))
    env_config = _value(config, "env")
    corruption_config = (
        ScanCorruptionConfig.from_config(env_config)
        if "scan_corruption" in env_config
        else ScanCorruptionConfig()
    )
    environment = LidarRacingEnv(simulator, settings, corruption_config)
    teacher_config = _value(config, "teacher")
    speed_profile = _value(teacher_config, "speed_profile")
    vehicle = _value(config, "vehicle", "vehicle")
    waypoints = build_reference_waypoints(
        simulator,
        reference_line=str(_value(teacher_config, "reference_line")),
        base_target_speed=float(_value(teacher_config, "base_target_speed")),
        minimum_corner_speed=float(_value(speed_profile, "minimum_corner_speed")),
        maximum_lateral_acceleration=float(
            _value(speed_profile, "maximum_lateral_acceleration")
        ),
    )
    validate_centerline_clearance(
        simulator,
        vehicle_width=float(_value(vehicle, "width")),
        lateral_offset_min=0.0,
        lateral_offset_max=0.0,
    )

    seeds = _value(config, "seed")

    def continuation_key(seed_name: str) -> Any:
        key = jax.random.key(int(_value(seeds, seed_name)))
        if resume_metadata is None:
            return key
        # fold_in consumes uint32 data. Split both monotonic counters into low
        # and high words so long runs do not silently reuse a truncated prefix.
        for counter in (cumulative_environment_transitions, resume_learner_step):
            low_word = jnp.asarray(counter & 0xFFFF_FFFF, dtype=jnp.uint32)
            high_word = jnp.asarray((counter >> 32) & 0xFFFF_FFFF, dtype=jnp.uint32)
            key = jax.random.fold_in(key, low_word)
            key = jax.random.fold_in(key, high_word)
        return key

    reset_key = continuation_key("reset")
    reset_keys = jax.random.split(reset_key, num_envs)
    states, observations = jax.jit(environment.reset_batch)(reset_keys)

    actor = TanhGaussianActor(
        log_std_min=float(_value(config, "agent", "actor", "log_std_min")),
        log_std_max=float(_value(config, "agent", "actor", "log_std_max")),
    )
    critic = TwinQCritic()
    optimizer_config = _value(config, "agent", "optimizer")
    optimizers = create_sac_optimizers(
        actor_learning_rate=float(_value(optimizer_config, "actor_learning_rate")),
        critic_learning_rate=float(_value(optimizer_config, "critic_learning_rate")),
        alpha_learning_rate=float(_value(optimizer_config, "temperature_learning_rate")),
        gradient_clip_norm=float(_value(config, "agent", "algorithm", "gradient_clip_norm")),
    )
    model_key = jax.random.key(int(_value(seeds, "master")))
    learner_state = initialize_sac_state(
        key=model_key,
        actor=actor,
        critic=critic,
        optimizers=optimizers,
        observation_example=observations[:1],
        normalized_action_example=jnp.zeros((1, 2), dtype=jnp.float32),
        initial_alpha=float(_value(optimizer_config, "initial_temperature")),
    )
    if resume_checkpoint is not None:
        learner_state, loaded_metadata = load_checkpoint(
            resume_checkpoint,
            learner_state,
            expected_architecture_version=ARCHITECTURE_VERSION,
            expected_config_sha256=config_hash,
        )
        if loaded_metadata != resume_metadata:
            raise RuntimeError("checkpoint metadata changed while the run was starting")
        restored_step = int(jax.device_get(learner_state.step))
        if restored_step != loaded_metadata.step:
            raise ValueError("checkpoint metadata step does not match learner state")

    replay_config = _value(config, "agent", "replay_buffer")
    if _value(replay_config, "implementation") != "jax_ring":
        raise ValueError("this trainer requires replay_buffer.implementation=jax_ring")
    storage_dtype_name = str(_value(replay_config, "observation_storage_dtype"))
    storage_dtypes = {"float16": jnp.float16, "float32": jnp.float32}
    if storage_dtype_name not in storage_dtypes:
        raise ValueError("replay observation storage dtype must be float16 or float32")
    replay_capacity = int(_value(replay_config, "capacity"))
    if replay_capacity < num_envs:
        raise ValueError("replay capacity must fit one vector-environment collection")
    replay_state = initialize_replay_buffer(
        ReplayBufferConfig(
            capacity=replay_capacity,
            observation_shape=tuple(observations.shape[1:]),
            action_dim=2,
            observation_storage_dtype=storage_dtypes[storage_dtype_name],
        )
    )
    warmup_transitions = int(_value(replay_config, "warmup_transitions"))
    batch_size = int(_value(config, "agent", "update", "batch_size"))
    updates_per_collection = int(
        _value(config, "agent", "update", "updates_per_collection")
    )
    if warmup_transitions < batch_size:
        raise ValueError("replay warmup must be at least one learner batch")
    if updates_per_collection < 1:
        raise ValueError("updates_per_collection must be positive")

    learner_config = SACLearnerConfig(
        discount=float(_value(config, "agent", "update", "discount")),
        target_smoothing_coefficient=float(
            _value(config, "agent", "update", "target_smoothing_coefficient")
        ),
        target_entropy=float(_value(config, "agent", "update", "target_entropy")),
        detect_non_finite=bool(_value(config, "agent", "algorithm", "detect_non_finite")),
    )
    update_once = jax.jit(
        make_sac_update(
            actor=actor,
            critic=critic,
            optimizers=optimizers,
            config=learner_config,
        )
    )
    step_environments = jax.jit(environment.step_batch)
    insert_transitions = jax.jit(insert_batch)
    sample_replay = jax.jit(lambda key, state: sample_batch(key, state, batch_size))

    def sample_actor_action(params: Any, observation: Any, key: Any) -> Any:
        return actor.apply(
            {"params": params},
            observation,
            key,
            method=actor.sample,
        ).action

    actor_action = jax.jit(sample_actor_action)

    control_dt = float(_value(config, "env", "simulator", "physics_timestep")) * int(
        _value(config, "env", "simulator", "timestep_ratio")
    )
    wheelbase = float(_value(vehicle, "wheelbase"))
    steering_min = float(_value(vehicle, "min_steering_angle"))
    steering_max = float(_value(vehicle, "max_steering_angle"))
    teacher_lookahead = jnp.asarray(
        [float(_value(teacher_config, "lookahead"))], dtype=jnp.float32
    )
    teacher_speed = jnp.asarray(
        [float(_value(teacher_config, "speed_multiplier"))], dtype=jnp.float32
    )
    teacher_vehicle_index = jnp.asarray([0], dtype=jnp.int32)
    if num_agents == 4:
        teacher_safe_distance = jnp.asarray(
            [
                float(
                    _value(
                        config,
                        "npc",
                        "longitudinal_controller",
                        "safe_following_distance",
                        "max",
                    )
                )
            ],
            dtype=jnp.float32,
        )
        teacher_distance_gain = float(
            _value(
                config,
                "npc",
                "longitudinal_controller",
                "distance_gain",
            )
        )
        teacher_lateral_gate = float(
            _value(
                config,
                "npc",
                "longitudinal_controller",
                "lateral_gate",
            )
        )
    else:
        teacher_safe_distance = jnp.asarray([1.0], dtype=jnp.float32)
        teacher_distance_gain = 0.0
        teacher_lateral_gate = 1.0
    teacher_noise_std = _finite_float(
        _value(teacher_config, "normalized_action_noise_std"),
        "teacher.normalized_action_noise_std",
    )
    if teacher_noise_std < 0.0:
        raise ValueError("teacher normalized action noise cannot be negative")
    trajectory_aided_config = _value(config, "reward", "trajectory_aided")
    trajectory_aided_enabled = _value(trajectory_aided_config, "enabled")
    if not isinstance(trajectory_aided_enabled, bool):
        raise ValueError("reward.trajectory_aided.enabled must be boolean")
    trajectory_aided_weight = _finite_float(
        _value(trajectory_aided_config, "weight"),
        "reward.trajectory_aided.weight",
    )
    if trajectory_aided_weight < 0.0:
        raise ValueError("trajectory-aided reward weight cannot be negative")

    def teacher_one(cartesian_states: Any) -> Any:
        # GT is used only by this warmup teacher and never enters replay observations.
        physical_action = pure_pursuit_actions(
            cartesian_states[0:1],
            waypoints,
            teacher_lookahead,
            teacher_speed,
            wheelbase=wheelbase,
            control_dt=control_dt,
            steering_min=steering_min,
            steering_max=steering_max,
            acceleration_min=settings.min_acceleration,
            acceleration_max=settings.max_acceleration,
        )[0]
        if num_agents == 4:
            waypoint_target_speed = ordered_braking_target_speeds(
                cartesian_states[0:1],
                waypoints[jnp.newaxis, ...],
                teacher_speed,
                maximum_deceleration=abs(settings.min_acceleration),
            )
            safe_target_speed = limit_speed_for_leading_vehicle(
                cartesian_states[0:1],
                cartesian_states,
                teacher_vehicle_index,
                waypoint_target_speed,
                teacher_safe_distance,
                distance_gain=teacher_distance_gain,
                lateral_gate=teacher_lateral_gate,
                minimum_speed=0.0,
            )
            safe_acceleration = jnp.clip(
                (safe_target_speed[0] - cartesian_states[0, 3]) / control_dt,
                settings.min_acceleration,
                settings.max_acceleration,
            )
            physical_action = physical_action.at[1].set(safe_acceleration)
        return normalize_physical_action(
            physical_action,
            max_steering_angle=settings.max_steering_angle,
            min_acceleration=settings.min_acceleration,
            max_acceleration=settings.max_acceleration,
        )

    teacher_actions = jax.jit(jax.vmap(teacher_one))

    environment_root_key = continuation_key("environment")
    environment_key, npc_seed_key = jax.random.split(environment_root_key)

    npc_parameters = None
    npc_controller_states = None
    npc_action_function = None
    npc_reset_function = None
    if num_agents == 4:
        npc_count = 3
        npc_indices = jnp.arange(1, 4, dtype=jnp.int32)
        npc_bounds = NpcRandomizationBounds.from_config(_value(config, "npc"))
        npc_bounds.validate_reset_spacing(
            settings.reset_longitudinal_spacing,
            settings.vehicle_length,
        )
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
        npc_parameters = jax.vmap(
            lambda key: sample_npc_episode_parameters(
                key,
                npc_count=npc_count,
                bounds=npc_bounds,
            )
        )(jax.random.split(npc_seed_key, num_envs))
        npc_controller_states = reset_controller_states
        npc_config = _value(config, "npc")

        def npc_one(cartesian_states: Any, parameters: Any, controller_state: Any, step: Any):
            # GT is confined to the fixed, non-learned opponent controller.
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
                distance_gain=float(
                    _value(npc_config, "longitudinal_controller", "distance_gain")
                ),
                lateral_gate=float(
                    _value(npc_config, "longitudinal_controller", "lateral_gate")
                ),
                # Scripted traffic must stop rather than reverse when blocked.
                minimum_speed=0.0,
            )

        npc_action_function = jax.jit(jax.vmap(npc_one))

        def reset_npcs(done: Any, current_parameters: Any, next_states: Any, key: Any):
            reset_parameters = jax.vmap(
                lambda sample_key: sample_npc_episode_parameters(
                    sample_key,
                    npc_count=npc_count,
                    bounds=npc_bounds,
                )
            )(jax.random.split(key, num_envs))

            def select(reset_value: Any, current_value: Any) -> Any:
                mask = done.reshape((num_envs, *(1,) * (current_value.ndim - 1)))
                return jnp.where(mask, reset_value, current_value)

            return (
                jax.tree.map(select, reset_parameters, current_parameters),
                jax.tree.map(select, reset_controller_states, next_states),
            )

        npc_reset_function = jax.jit(reset_npcs)

    action_key = continuation_key("action")
    replay_key = continuation_key("replay")
    session_environment_transitions = 0
    replay_collected_transitions = 0
    learner_updates = int(jax.device_get(learner_state.step))
    collections = 0
    log_interval = int(_value(config, "training", "log_interval_collections"))
    checkpoint_interval = int(
        _value(config, "agent", "checkpoint", "save_interval_updates")
    )
    keep_last_checkpoints = int(_value(config, "agent", "checkpoint", "keep_last"))
    if log_interval < 1 or checkpoint_interval < 1:
        raise ValueError("log and checkpoint intervals must be positive")
    # A resumed learner may already sit exactly on an interval boundary while
    # its new replay buffer is warming up.  Track the boundary separately from
    # checkpoints actually published in this run so we neither re-save one
    # frozen step repeatedly nor miss a boundary when several updates happen
    # in one collection.
    last_checkpoint_bucket = learner_updates // checkpoint_interval
    last_checkpoint_update: int | None = None
    latest_update_metrics = None
    reward_window = jnp.asarray(0.0, dtype=jnp.float32)
    trajectory_aided_reward_window = jnp.asarray(0.0, dtype=jnp.float32)
    progress_window = jnp.asarray(0.0, dtype=jnp.float32)
    collision_window = jnp.asarray(0, dtype=jnp.int32)
    off_track_window = jnp.asarray(0, dtype=jnp.int32)
    race_complete_window = jnp.asarray(0, dtype=jnp.int32)
    unique_pass_window = jnp.asarray(0, dtype=jnp.int32)
    unsafe_contact_window = jnp.asarray(0, dtype=jnp.int32)
    npc_invalid_window = jnp.asarray(0, dtype=jnp.int32)
    terminated_window = jnp.asarray(0, dtype=jnp.int32)
    truncated_window = jnp.asarray(0, dtype=jnp.int32)
    window_transitions = 0
    track_length = float(simulator.track_length)
    control_dt = float(_value(env_config, "simulator", "physics_timestep")) * int(
        _value(env_config, "simulator", "timestep_ratio")
    )
    metrics_path = run_directory / "metrics.jsonl"
    checkpoint_root = run_directory / "checkpoints"
    submodule_commit = next(iter(snapshot.submodule_commits.values()))
    started = time.perf_counter()

    while session_environment_transitions < session_transition_budget:
        collections += 1
        action_key, policy_key, noise_key = jax.random.split(action_key, 3)
        reference_actions = teacher_actions(
            states.simulator_state.cartesian_states
        )
        if replay_collected_transitions < warmup_transitions:
            noise = (
                jax.random.normal(noise_key, reference_actions.shape)
                * teacher_noise_std
            )
            ego_actions = jnp.clip(reference_actions + noise, -1.0, 1.0)
        else:
            ego_actions = actor_action(learner_state.actor_params, observations, policy_key)

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

        environment_key, step_seed, npc_reset_key = jax.random.split(environment_key, 3)
        result = step_environments(
            jax.random.split(step_seed, num_envs),
            states,
            ego_actions,
            npc_actions,
        )
        trajectory_aided_rewards = jnp.where(
            trajectory_aided_enabled,
            trajectory_aided_action_reward(
                ego_actions,
                reference_actions,
                weight=trajectory_aided_weight,
            ),
            jnp.zeros_like(result.reward),
        )
        result = result._replace(reward=result.reward + trajectory_aided_rewards)
        transitions = transition_from_step(observations, ego_actions, result)
        replay_state = insert_transitions(replay_state, transitions)
        session_environment_transitions += num_envs
        replay_collected_transitions += num_envs
        cumulative_environment_transitions += num_envs
        states = result.state
        observations = result.observation
        done = result.terminated | result.truncated

        if num_agents == 4:
            assert npc_reset_function is not None
            npc_parameters, npc_controller_states = npc_reset_function(
                done,
                npc_parameters,
                next_npc_controller_states,
                npc_reset_key,
            )

        reward_window = reward_window + jnp.sum(result.reward)
        trajectory_aided_reward_window = (
            trajectory_aided_reward_window + jnp.sum(trajectory_aided_rewards)
        )
        progress_window = progress_window + jnp.sum(result.diagnostics.progress_delta)
        collision_window = collision_window + jnp.sum(result.diagnostics.collision)
        off_track_window = off_track_window + jnp.sum(result.diagnostics.off_track)
        race_complete_window = race_complete_window + jnp.sum(
            result.diagnostics.race_complete
        )
        unique_pass_window = unique_pass_window + jnp.sum(
            result.diagnostics.pass_count
        )
        unsafe_contact_window = unsafe_contact_window + jnp.sum(
            result.diagnostics.unsafe_contact
        )
        npc_invalid_window = npc_invalid_window + jnp.sum(
            result.diagnostics.npc_collision_without_ego
        )
        terminated_window = terminated_window + jnp.sum(result.terminated)
        truncated_window = truncated_window + jnp.sum(result.truncated)
        window_transitions += num_envs

        if replay_collected_transitions >= warmup_transitions:
            for _ in range(updates_per_collection):
                replay_key, sample_key, update_key = jax.random.split(replay_key, 3)
                batch = sample_replay(sample_key, replay_state)
                learner_state, latest_update_metrics = update_once(
                    learner_state,
                    batch,
                    update_key,
                )
                if not bool(jax.device_get(latest_update_metrics.update_applied)):
                    raise FloatingPointError("SAC update produced a non-finite value")
                learner_updates += 1
                checkpoint_bucket = learner_updates // checkpoint_interval
                if checkpoint_bucket > last_checkpoint_bucket:
                    save_checkpoint(
                        checkpoint_root,
                        learner_state,
                        {"params": learner_state.actor_params},
                        step=learner_updates,
                        environment_transitions=cumulative_environment_transitions,
                        architecture_version=ARCHITECTURE_VERSION,
                        config_sha256=config_hash,
                        root_commit=snapshot.root_commit,
                        submodule_commit=submodule_commit,
                    )
                    prune_checkpoints(checkpoint_root, keep_last_checkpoints)
                    last_checkpoint_update = learner_updates
                    last_checkpoint_bucket = checkpoint_bucket

        if (
            collections % log_interval == 0
            or session_environment_transitions >= session_transition_budget
        ):
            elapsed = time.perf_counter() - started
            progress = float(jax.device_get(progress_window))
            collisions = int(jax.device_get(collision_window))
            off_tracks = int(jax.device_get(off_track_window))
            race_completions = int(jax.device_get(race_complete_window))
            unique_passes = int(jax.device_get(unique_pass_window))
            unsafe_contacts = int(jax.device_get(unsafe_contact_window))
            npc_collision_steps = int(jax.device_get(npc_invalid_window))
            terminations = int(jax.device_get(terminated_window))
            truncations = int(jax.device_get(truncated_window))
            completed_episodes = terminations + truncations
            record: dict[str, Any] = {
                "collection": collections,
                "environment_transitions": cumulative_environment_transitions,
                "session_environment_transitions": session_environment_transitions,
                "replay_collected_transitions": replay_collected_transitions,
                "learner_updates": learner_updates,
                "mean_reward_per_transition": float(
                    jax.device_get(reward_window)
                )
                / window_transitions,
                "mean_trajectory_aided_reward_per_transition": float(
                    jax.device_get(trajectory_aided_reward_window)
                )
                / window_transitions,
                "mean_course_progress_meters_per_transition": (
                    progress / window_transitions
                ),
                "mean_course_progress_fraction_per_transition": (
                    progress / (window_transitions * track_length)
                ),
                "course_progress_meters_per_simulated_second": (
                    progress / (window_transitions * control_dt)
                ),
                "collision_count": collisions,
                "off_track_count": off_tracks,
                "race_complete_count": race_completions,
                "unique_pass_count": unique_passes,
                "unsafe_contact_step_count": unsafe_contacts,
                "npc_collision_without_ego_step_count": npc_collision_steps,
                "collision_rate_per_completed_episode": (
                    collisions / completed_episodes if completed_episodes else 0.0
                ),
                "off_track_rate_per_completed_episode": (
                    off_tracks / completed_episodes if completed_episodes else 0.0
                ),
                "race_completion_rate": (
                    race_completions / completed_episodes
                    if completed_episodes
                    else 0.0
                ),
                "unique_passes_per_transition": (
                    unique_passes / window_transitions
                ),
                "terminated_count": terminations,
                "truncated_count": truncations,
                "environment_transitions_per_second": (
                    session_environment_transitions / elapsed
                ),
                "replay_size": min(
                    replay_collected_transitions,
                    replay_state.reward.shape[0],
                ),
                "replay_allocated_bytes": replay_memory_bytes(replay_state),
            }
            if latest_update_metrics is not None:
                host_metrics = jax.device_get(latest_update_metrics)
                record.update(
                    {
                        "critic_loss": float(host_metrics.critic_loss),
                        "actor_loss": float(host_metrics.actor_loss),
                        "alpha_loss": float(host_metrics.alpha_loss),
                        "alpha": float(host_metrics.alpha),
                        "entropy": float(host_metrics.entropy),
                        "absolute_td_error_mean": float(
                            host_metrics.absolute_td_error_mean
                        ),
                    }
                )
            append_jsonl(metrics_path, record)
            reward_window = jnp.asarray(0.0, dtype=jnp.float32)
            trajectory_aided_reward_window = jnp.asarray(0.0, dtype=jnp.float32)
            progress_window = jnp.asarray(0.0, dtype=jnp.float32)
            collision_window = jnp.asarray(0, dtype=jnp.int32)
            off_track_window = jnp.asarray(0, dtype=jnp.int32)
            race_complete_window = jnp.asarray(0, dtype=jnp.int32)
            unique_pass_window = jnp.asarray(0, dtype=jnp.int32)
            unsafe_contact_window = jnp.asarray(0, dtype=jnp.int32)
            npc_invalid_window = jnp.asarray(0, dtype=jnp.int32)
            terminated_window = jnp.asarray(0, dtype=jnp.int32)
            truncated_window = jnp.asarray(0, dtype=jnp.int32)
            window_transitions = 0

    if learner_updates != last_checkpoint_update:
        final_checkpoint = save_checkpoint(
            checkpoint_root,
            learner_state,
            {"params": learner_state.actor_params},
            step=learner_updates,
            environment_transitions=cumulative_environment_transitions,
            architecture_version=ARCHITECTURE_VERSION,
            config_sha256=config_hash,
            root_commit=snapshot.root_commit,
            submodule_commit=submodule_commit,
        )
        prune_checkpoints(checkpoint_root, keep_last_checkpoints)
    else:
        final_checkpoint = checkpoint_root / f"step_{learner_updates:012d}"
    elapsed_seconds = time.perf_counter() - started
    return TrainingResult(
        run_directory=run_directory,
        environment_transitions=cumulative_environment_transitions,
        session_environment_transitions=session_environment_transitions,
        learner_updates=learner_updates,
        final_checkpoint=final_checkpoint,
        elapsed_seconds=elapsed_seconds,
    )


__all__ = ["ARCHITECTURE_VERSION", "TrainingResult", "train_lidar_sac"]

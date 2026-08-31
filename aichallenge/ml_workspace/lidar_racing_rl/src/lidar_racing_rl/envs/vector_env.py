"""LiDAR-only, ego-reset environment composition for F1TENTH Gym JAX.

The pinned upstream environment intentionally remains the source of vehicle
dynamics, map ray marching, and collision state.  Its public observation is not
used here because it contains ground-truth vehicle state.  The only Actor or
Critic observation returned by this module is built from ``State.scans``.

Ground truth is consumed internally only for the uses allowed by the project
boundary: reset placement, dynamic LiDAR generation, reward, termination, and
fixed NPC control performed by the caller.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from lidar_racing_rl.envs.action import scale_normalized_action
from lidar_racing_rl.envs.observation import (
    CANONICAL_BEAM_COUNT,
    canonicalize_scan,
    initialize_frame_stack,
    update_frame_stack,
)
from lidar_racing_rl.envs.overtaking import (
    OvertakingState,
    ego_opponent_obb_overlaps,
    initialize_overtaking_state,
    minimum_opponent_distance,
    nearest_opponent_relative_progress,
    update_overtaking_state,
)
from lidar_racing_rl.envs.reset_sampler import sample_four_vehicle_frenet
from lidar_racing_rl.envs.reward import (
    step1_reward,
    step2_reward,
    wrapped_progress_delta,
)
from lidar_racing_rl.envs.scan_corruption import (
    ScanCorruptionConfig,
    ScanCorruptionState,
    apply_scan_corruption,
    initialize_scan_corruption_state,
)
from lidar_racing_rl.envs.termination import ego_done_flags


EGO_INDEX = 0
MAX_NPC_COUNT = 3
CANONICAL_FRAME_STACK = 4
CANONICAL_FIELD_OF_VIEW = 1.5 * math.pi
CANONICAL_RANGE_MAX = 30.0


def _required(config: Mapping[str, Any], key: str) -> Any:
    try:
        return config[key]
    except KeyError as exc:
        raise KeyError(f"required LiDAR environment setting is missing: {key}") from exc


def _required_int(config: Mapping[str, Any], key: str) -> int:
    value = _required(config, key)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"required LiDAR environment setting '{key}' must be an integer")
    return int(value)


def _required_bool(config: Mapping[str, Any], key: str) -> bool:
    value = _required(config, key)
    if not isinstance(value, bool):
        raise ValueError(f"required LiDAR environment setting '{key}' must be boolean")
    return value


@dataclass(frozen=True)
class RacingEnvSettings:
    """Static values used to compose the pinned simulator.

    Values are deliberately required rather than guessed in this class.  They
    must come from the resolved Hydra/OmegaConf configuration saved with a run.
    """

    num_agents: int
    num_beams: int
    frame_stack: int
    field_of_view: float
    range_min: float
    range_max: float
    vehicle_length: float
    vehicle_width: float
    max_steering_angle: float
    min_acceleration: float
    max_acceleration: float
    max_steps: int
    max_num_laps: int
    reset_longitudinal_spacing: float
    reset_lateral_jitter: float
    reset_heading_jitter: float
    progress_weight: float
    collision_weight: float
    off_track_weight: float
    smoothness_weight: float
    reverse_weight: float
    relative_progress_enabled: bool
    pass_event_enabled: bool
    unsafe_contact_enabled: bool
    stalled_behind_vehicle_enabled: bool
    relative_progress_weight: float
    pass_weight: float
    unsafe_contact_weight: float
    stalled_behind_weight: float
    pass_behind_distance: float
    pass_ahead_distance: float
    pass_hold_steps: int
    pass_cooldown_steps: int
    unsafe_contact_distance: float
    stalled_max_forward_gap: float
    stalled_speed_threshold: float

    @classmethod
    def from_config(
        cls,
        env_config: Mapping[str, Any],
        vehicle_config: Mapping[str, Any],
        reward_config: Mapping[str, Any],
    ) -> RacingEnvSettings:
        """Build settings from resolved mapping objects without hidden defaults."""

        lidar = _required(env_config, "lidar")
        episode = _required(env_config, "episode")
        reset = _required(env_config, "reset")
        vehicle = vehicle_config.get("vehicle", vehicle_config)
        weights = _required(reward_config, "weights")
        overtaking = _required(reward_config, "overtaking")
        settings = cls(
            num_agents=_required_int(env_config, "num_agents"),
            num_beams=_required_int(lidar, "num_beams"),
            frame_stack=_required_int(lidar, "frame_stack"),
            field_of_view=float(_required(lidar, "field_of_view")),
            range_min=float(_required(lidar, "range_min")),
            range_max=float(_required(lidar, "range_max")),
            vehicle_length=float(_required(vehicle, "length")),
            vehicle_width=float(_required(vehicle, "width")),
            max_steering_angle=float(_required(vehicle, "max_steering_angle")),
            min_acceleration=float(_required(vehicle, "min_acceleration")),
            max_acceleration=float(_required(vehicle, "max_acceleration")),
            max_steps=_required_int(episode, "max_steps"),
            max_num_laps=_required_int(episode, "max_num_laps"),
            reset_longitudinal_spacing=float(_required(reset, "longitudinal_spacing")),
            reset_lateral_jitter=float(_required(reset, "lateral_jitter")),
            reset_heading_jitter=float(_required(reset, "heading_jitter")),
            progress_weight=float(_required(weights, "progress")),
            collision_weight=float(_required(weights, "collision")),
            off_track_weight=float(_required(weights, "off_track")),
            smoothness_weight=float(_required(weights, "smoothness")),
            reverse_weight=float(_required(weights, "reverse")),
            relative_progress_enabled=_required_bool(
                reward_config, "relative_progress"
            ),
            pass_event_enabled=_required_bool(reward_config, "pass_event"),
            unsafe_contact_enabled=_required_bool(reward_config, "unsafe_contact"),
            stalled_behind_vehicle_enabled=_required_bool(
                reward_config, "stalled_behind_vehicle"
            ),
            relative_progress_weight=float(
                _required(weights, "relative_progress")
            ),
            pass_weight=float(_required(weights, "pass")),
            unsafe_contact_weight=float(_required(weights, "unsafe_contact")),
            stalled_behind_weight=float(_required(weights, "stalled_behind")),
            pass_behind_distance=float(
                _required(overtaking, "behind_distance")
            ),
            pass_ahead_distance=float(_required(overtaking, "ahead_distance")),
            pass_hold_steps=_required_int(overtaking, "hold_steps"),
            pass_cooldown_steps=_required_int(overtaking, "cooldown_steps"),
            unsafe_contact_distance=float(
                _required(overtaking, "unsafe_contact_distance")
            ),
            stalled_max_forward_gap=float(
                _required(overtaking, "stalled_max_forward_gap")
            ),
            stalled_speed_threshold=float(
                _required(overtaking, "stalled_speed_threshold")
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Reject configurations that would violate the fixed model contract."""

        finite_values = (
            self.range_min,
            self.range_max,
            self.vehicle_length,
            self.vehicle_width,
            self.max_steering_angle,
            self.min_acceleration,
            self.max_acceleration,
            self.reset_longitudinal_spacing,
            self.reset_lateral_jitter,
            self.reset_heading_jitter,
            self.progress_weight,
            self.collision_weight,
            self.off_track_weight,
            self.smoothness_weight,
            self.reverse_weight,
            self.relative_progress_weight,
            self.pass_weight,
            self.unsafe_contact_weight,
            self.stalled_behind_weight,
            self.pass_behind_distance,
            self.pass_ahead_distance,
            self.unsafe_contact_distance,
            self.stalled_max_forward_gap,
            self.stalled_speed_threshold,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("environment scalar settings must be finite")
        if self.num_agents not in (1, 4):
            raise ValueError("the supported stages require either one or four vehicles")
        if self.num_beams != CANONICAL_BEAM_COUNT:
            raise ValueError("training scans must contain exactly 360 beams")
        if self.frame_stack != CANONICAL_FRAME_STACK:
            raise ValueError("initial Actor/runtime contract requires frame_stack=4")
        if not 0.0 <= self.range_min < self.range_max:
            raise ValueError("LiDAR bounds must satisfy 0 <= range_min < range_max")
        if not math.isclose(
            self.field_of_view,
            CANONICAL_FIELD_OF_VIEW,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("initial AWSIM contract requires a 270-degree field_of_view")
        if not math.isclose(
            self.range_max,
            CANONICAL_RANGE_MAX,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("initial AWSIM normalization contract requires range_max=30")
        if self.vehicle_length <= 0.0 or self.vehicle_width <= 0.0:
            raise ValueError("vehicle dimensions must be positive")
        if self.max_steering_angle <= 0.0:
            raise ValueError("max_steering_angle must be positive")
        if self.min_acceleration >= self.max_acceleration:
            raise ValueError("acceleration bounds must be ordered")
        if self.max_steps < 1 or self.max_num_laps < 1:
            raise ValueError("episode limits must be positive")
        if self.pass_hold_steps < 1 or self.pass_cooldown_steps < 0:
            raise ValueError("pass hold/cooldown steps must be positive/non-negative")
        if self.reset_lateral_jitter < 0.0 or self.reset_heading_jitter < 0.0:
            raise ValueError("reset jitter bounds cannot be negative")
        if (
            self.num_agents == 4
            and self.reset_longitudinal_spacing < self.vehicle_length
        ):
            raise ValueError("reset spacing must be at least one vehicle length")
        if any(
            weight < 0.0
            for weight in (
                self.relative_progress_weight,
                self.pass_weight,
                self.unsafe_contact_weight,
                self.stalled_behind_weight,
            )
        ):
            raise ValueError("Step-2 reward weights cannot be negative")
        if self.pass_behind_distance < 0.0 or self.pass_ahead_distance < 0.0:
            raise ValueError("pass hysteresis distances cannot be negative")
        if (
            self.unsafe_contact_distance <= 0.0
            or self.stalled_max_forward_gap <= 0.0
            or self.stalled_speed_threshold < 0.0
        ):
            raise ValueError("Step-2 proximity thresholds are invalid")

        feature_settings = (
            (
                "relative_progress",
                self.relative_progress_enabled,
                self.relative_progress_weight,
            ),
            ("pass_event", self.pass_event_enabled, self.pass_weight),
            (
                "unsafe_contact",
                self.unsafe_contact_enabled,
                self.unsafe_contact_weight,
            ),
            (
                "stalled_behind_vehicle",
                self.stalled_behind_vehicle_enabled,
                self.stalled_behind_weight,
            ),
        )
        if not all(isinstance(enabled, bool) for _, enabled, _ in feature_settings):
            raise ValueError("Step-2 reward feature flags must be boolean")
        if self.num_agents == 1 and any(
            enabled for _, enabled, _ in feature_settings
        ):
            raise ValueError("single-vehicle stage cannot enable opponent rewards")
        for name, enabled, weight in feature_settings:
            if enabled and weight <= 0.0:
                raise ValueError(f"enabled {name} requires a positive reward weight")
            if not enabled and weight != 0.0:
                raise ValueError(f"disabled {name} requires a zero reward weight")


@struct.dataclass
class LidarRacingState:
    """JAX pytree state kept by the LiDAR-only wrapper."""

    simulator_state: Any
    scan_history: jax.Array
    scan_corruption_state: ScanCorruptionState
    overtaking_state: OvertakingState
    previous_ego_action: jax.Array
    npc_scenario_valid: jax.Array


class StepDiagnostics(NamedTuple):
    """GT-derived evaluation data that never enters Actor/Critic or replay."""

    progress_delta: jax.Array
    ego_speed: jax.Array
    collision: jax.Array
    off_track: jax.Array
    race_complete: jax.Array
    unrecoverable: jax.Array
    relative_progress: jax.Array
    pass_count: jax.Array
    unsafe_contact: jax.Array
    stalled_behind_vehicle: jax.Array
    nearest_opponent_distance: jax.Array
    following_vehicle: jax.Array
    ego_rank: jax.Array
    opponent_present: jax.Array
    collision_with_opponent: jax.Array
    collision_with_wall: jax.Array
    npc_collision_flags: jax.Array
    npc_collision_without_ego: jax.Array
    minimum_npc_speed: jax.Array
    terminal_ego_pose: jax.Array
    terminal_ego_frenet_pose: jax.Array


class StepResult(NamedTuple):
    """One ego transition plus the post-auto-reset collector state."""

    state: LidarRacingState
    observation: jax.Array
    transition_next_observation: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    diagnostics: StepDiagnostics


class LidarRacingEnv:
    """Compose one four-vehicle simulator into a LiDAR-only ego environment."""

    def __init__(
        self,
        simulator: Any,
        settings: RacingEnvSettings,
        scan_corruption_config: ScanCorruptionConfig | None = None,
    ):
        settings.validate()
        expected_agents = tuple(f"agent_{index}" for index in range(settings.num_agents))
        if tuple(simulator.agents) != expected_agents:
            raise ValueError("simulator agent names do not match the configured stage")
        track_length = float(simulator.track_length)
        if not math.isfinite(track_length) or track_length <= 0.0:
            raise ValueError("simulator track_length must be finite and positive")
        if (
            settings.num_agents == 4
            and track_length
            < settings.num_agents * settings.reset_longitudinal_spacing
        ):
            raise ValueError("track is too short for the configured four-vehicle spacing")

        self.simulator = simulator
        self.settings = settings
        self.scan_corruption_config = scan_corruption_config or ScanCorruptionConfig()
        self.scan_corruption_config.validate()
        required_methods = (
            "reset_array",
            "reset_from_frenet_poses",
            "step_env_array",
        )
        missing = [name for name in required_methods if not hasattr(simulator, name)]
        if missing:
            raise ValueError(
                "F1TENTH fork is missing required public array APIs: "
                + ", ".join(missing)
            )
        if not getattr(simulator.track, "has_boundaries", False):
            raise ValueError("track centerline must provide left/right boundary widths")
        try:
            minimum_left_width = min(float(value) for value in simulator.track.left_widths)
            minimum_right_width = min(
                float(value) for value in simulator.track.right_widths
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("track boundary width samples are unavailable") from error
        maximum_heading_error = abs(settings.reset_heading_jitter)
        reset_lateral_footprint = (
            0.5 * settings.vehicle_length * math.sin(maximum_heading_error)
            + 0.5 * settings.vehicle_width * math.cos(maximum_heading_error)
        )
        required_reset_half_width = (
            settings.reset_lateral_jitter + reset_lateral_footprint
        )
        if required_reset_half_width > min(
            minimum_left_width,
            minimum_right_width,
        ):
            raise ValueError(
                "configured reset jitter can place the full vehicle outside track bounds"
            )

    def _reset_one(self, key: jax.Array) -> tuple[LidarRacingState, jax.Array]:
        simulator_key, placement_key, corruption_key = jax.random.split(key, 3)

        if self.settings.num_agents == 4:
            # Frenet GT is permitted here solely for safe initial placement.
            frenet_poses = sample_four_vehicle_frenet(
                placement_key,
                self.simulator.track_length,
                longitudinal_spacing=self.settings.reset_longitudinal_spacing,
                lateral_jitter=self.settings.reset_lateral_jitter,
                heading_jitter=self.settings.reset_heading_jitter,
            )
            _, simulator_state = self.simulator.reset_from_frenet_poses(
                simulator_key,
                frenet_poses,
            )
        else:
            # The upstream single-agent reset widens its configured lateral
            # jitter by 1.5, so it can violate the project's explicit reset
            # envelope even when the nominal vehicle fits the track.
            # Use the explicit fork reset API so the saved project settings are
            # the only source of reset jitter for every training stage.
            anchor_key, lateral_key, heading_key = jax.random.split(
                placement_key, 3
            )
            frenet_poses = jnp.stack(
                (
                    jax.random.uniform(
                        anchor_key,
                        minval=0.0,
                        maxval=self.simulator.track_length,
                    ),
                    jax.random.uniform(
                        lateral_key,
                        minval=-self.settings.reset_lateral_jitter,
                        maxval=self.settings.reset_lateral_jitter,
                    ),
                    jax.random.uniform(
                        heading_key,
                        minval=-self.settings.reset_heading_jitter,
                        maxval=self.settings.reset_heading_jitter,
                    ),
                ),
                axis=0,
            )[jnp.newaxis, :]
            _, simulator_state = self.simulator.reset_from_frenet_poses(
                simulator_key,
                frenet_poses,
            )
        frame = canonicalize_scan(
            simulator_state.scans,
            range_min=self.settings.range_min,
            range_max=self.settings.range_max,
        )
        corruption_state = initialize_scan_corruption_state(
            frame,
            self.scan_corruption_config,
            range_min=0.0,
            range_max=1.0,
        )
        frame, corruption_state = apply_scan_corruption(
            corruption_key,
            frame,
            corruption_state,
            self.scan_corruption_config,
            range_min=0.0,
            range_max=1.0,
            angle_increment=self.settings.field_of_view / (self.settings.num_beams - 1),
        )
        history = initialize_frame_stack(frame, num_frames=self.settings.frame_stack)
        overtaking_state = initialize_overtaking_state(
            simulator_state.frenet_states[EGO_INDEX, 0],
            simulator_state.frenet_states[1:, 0],
            self.simulator.track_length,
            behind_distance=self.settings.pass_behind_distance,
        )
        state = LidarRacingState(
            simulator_state=simulator_state,
            scan_history=history,
            scan_corruption_state=corruption_state,
            overtaking_state=overtaking_state,
            previous_ego_action=jnp.zeros((2,), dtype=jnp.float32),
            npc_scenario_valid=jnp.asarray(True),
        )
        return state, history[EGO_INDEX]

    def reset(self, key: jax.Array) -> tuple[LidarRacingState, jax.Array]:
        """Reset one environment and return only the ego LiDAR observation."""

        return self._reset_one(key)

    def reset_batch(self, keys: jax.Array) -> tuple[LidarRacingState, jax.Array]:
        """Reset ``[num_envs]`` keys with ``jax.vmap``."""

        return jax.vmap(self._reset_one)(keys)

    def _step_one(
        self,
        key: jax.Array,
        state: LidarRacingState,
        ego_normalized_action: jax.Array,
        npc_physical_actions: jax.Array,
    ) -> StepResult:
        if ego_normalized_action.shape != (2,):
            raise ValueError("ego_normalized_action must have shape [2]")
        expected_npc_shape = (self.settings.num_agents - 1, 2)
        if npc_physical_actions.shape != expected_npc_shape:
            raise ValueError("npc_physical_actions must have shape [num_agents - 1, 2]")

        step_key, corruption_key, reset_key = jax.random.split(key, 3)
        ego_physical_action = scale_normalized_action(
            ego_normalized_action,
            max_steering_angle=self.settings.max_steering_angle,
            min_acceleration=self.settings.min_acceleration,
            max_acceleration=self.settings.max_acceleration,
        )
        physical_actions = jnp.concatenate(
            (ego_physical_action[jnp.newaxis, :], npc_physical_actions), axis=0
        )
        simulator_result = self.simulator.step_env_array(
            step_key,
            state.simulator_state,
            physical_actions,
        )
        simulator_state = simulator_result.state

        frame = canonicalize_scan(
            simulator_state.scans,
            range_min=self.settings.range_min,
            range_max=self.settings.range_max,
        )
        frame, corruption_state = apply_scan_corruption(
            corruption_key,
            frame,
            state.scan_corruption_state,
            self.scan_corruption_config,
            range_min=0.0,
            range_max=1.0,
            angle_increment=self.settings.field_of_view / (self.settings.num_beams - 1),
        )
        terminal_history = update_frame_stack(state.scan_history, frame)
        terminal_observation = terminal_history[EGO_INDEX]

        # Simulator GT is permitted only for reward and episode semantics.
        ego_collision = simulator_state.collisions[EGO_INDEX]
        npc_collision_flags = simulator_state.collisions[1:]
        npc_collision_without_ego = (
            jnp.any(npc_collision_flags) & ~ego_collision
        )
        npc_scenario_valid = (
            state.npc_scenario_valid & ~npc_collision_without_ego
        )
        race_complete = (
            simulator_state.num_laps[EGO_INDEX] >= self.settings.max_num_laps
        )
        ego_frenet = simulator_state.frenet_states[EGO_INDEX]
        lateral_clearance = (
            0.5 * self.settings.vehicle_length * jnp.abs(jnp.sin(ego_frenet[2]))
            + 0.5 * self.settings.vehicle_width * jnp.abs(jnp.cos(ego_frenet[2]))
        )
        off_track = self.simulator.track.is_off_track_frenet_jax(
            ego_frenet,
            clearance=lateral_clearance,
        )
        unrecoverable = ~jnp.all(
            jnp.isfinite(simulator_state.cartesian_states[EGO_INDEX])
        )
        terminated, truncated = ego_done_flags(
            collision=ego_collision,
            off_track=off_track,
            race_complete=race_complete,
            unrecoverable=unrecoverable,
            step_count=simulator_state.step,
            max_steps=self.settings.max_steps,
        )
        progress_delta = wrapped_progress_delta(
            state.simulator_state.frenet_states[EGO_INDEX, 0],
            simulator_state.frenet_states[EGO_INDEX, 0],
            self.simulator.track_length,
        )
        progress_delta = jnp.where(jnp.isfinite(progress_delta), progress_delta, 0.0)
        relative_progress, _ = nearest_opponent_relative_progress(
            state.simulator_state.frenet_states[EGO_INDEX, 0],
            simulator_state.frenet_states[EGO_INDEX, 0],
            state.simulator_state.frenet_states[1:, 0],
            simulator_state.frenet_states[1:, 0],
            self.simulator.track_length,
        )
        relative_progress = jnp.where(
            jnp.isfinite(relative_progress), relative_progress, 0.0
        )
        overtaking_state, pass_events, opponent_gaps = update_overtaking_state(
            state.overtaking_state,
            simulator_state.frenet_states[EGO_INDEX, 0],
            simulator_state.frenet_states[1:, 0],
            self.simulator.track_length,
            behind_distance=self.settings.pass_behind_distance,
            ahead_distance=self.settings.pass_ahead_distance,
            hold_steps=self.settings.pass_hold_steps,
            cooldown_steps=self.settings.pass_cooldown_steps,
        )
        nearest_distance = minimum_opponent_distance(
            simulator_state.cartesian_states[EGO_INDEX, 0:2],
            simulator_state.cartesian_states[1:, 0:2],
            no_opponent_value=float("inf"),
        )
        has_opponents = self.settings.num_agents > 1
        opponent_present = jnp.asarray(has_opponents)
        if has_opponents:
            current_forward_gaps = jnp.mod(
                simulator_state.frenet_states[1:, 0]
                - simulator_state.frenet_states[EGO_INDEX, 0],
                self.simulator.track_length,
            )
            nearest_forward_gap = jnp.min(current_forward_gaps)
        else:
            nearest_forward_gap = jnp.asarray(0.0, dtype=relative_progress.dtype)
        unsafe_contact = opponent_present & (
            nearest_distance <= self.settings.unsafe_contact_distance
        )
        opponent_overlaps = ego_opponent_obb_overlaps(
            simulator_state.cartesian_states,
            vehicle_length=self.settings.vehicle_length,
            vehicle_width=self.settings.vehicle_width,
        )
        contact_poses = simulator_state.cartesian_states[
            :,
            jnp.asarray([0, 1, 4]),
        ]
        finite_contact_poses = jnp.all(jnp.isfinite(contact_poses), axis=-1)
        finite_ego_pose = finite_contact_poses[EGO_INDEX]
        all_contact_poses_finite = jnp.all(finite_contact_poses)
        collision_with_opponent = (
            ego_collision & finite_ego_pose & jnp.any(opponent_overlaps)
        )
        # The fork exposes only a combined collision flag. Classify a contact
        # as wall-only when every pose needed to rule out an opponent overlap
        # is finite; otherwise retain only the generic collision diagnostic.
        collision_with_wall = (
            ego_collision & all_contact_poses_finite & ~collision_with_opponent
        )
        following_vehicle = (
            opponent_present
            & jnp.isfinite(nearest_forward_gap)
            & (nearest_forward_gap <= self.settings.stalled_max_forward_gap)
            & ~unsafe_contact
        )
        ego_rank = 1 + jnp.sum(opponent_gaps < 0.0, dtype=jnp.int32)
        nearest_opponent_distance = jnp.where(
            opponent_present,
            nearest_distance,
            jnp.asarray(0.0, dtype=nearest_distance.dtype),
        )
        if has_opponents:
            minimum_npc_speed = jnp.min(
                simulator_state.cartesian_states[1:, 3]
            )
        else:
            minimum_npc_speed = jnp.asarray(
                0.0,
                dtype=nearest_distance.dtype,
            )
        ego_speed = simulator_state.cartesian_states[EGO_INDEX, 3]
        stalled_behind_vehicle = (
            opponent_present
            & jnp.isfinite(nearest_forward_gap)
            & (nearest_forward_gap <= self.settings.stalled_max_forward_gap)
            & (jnp.abs(ego_speed) <= self.settings.stalled_speed_threshold)
        )
        rewardable_pass_events = pass_events & npc_scenario_valid
        reward_relative_progress = jnp.where(
            npc_scenario_valid,
            relative_progress,
            0.0,
        )
        reward_unsafe_contact = unsafe_contact & npc_scenario_valid
        reward_stalled = stalled_behind_vehicle & npc_scenario_valid
        reward = step1_reward(
            state.simulator_state.frenet_states[EGO_INDEX, 0],
            simulator_state.frenet_states[EGO_INDEX, 0],
            state.previous_ego_action,
            ego_normalized_action,
            track_length=self.simulator.track_length,
            collision=ego_collision | unrecoverable,
            off_track=off_track,
            reversing=ego_speed < 0.0,
            progress_weight=self.settings.progress_weight,
            collision_weight=self.settings.collision_weight,
            off_track_weight=self.settings.off_track_weight,
            smoothness_weight=self.settings.smoothness_weight,
            reverse_weight=self.settings.reverse_weight,
        )
        reward = step2_reward(
            reward,
            relative_progress=reward_relative_progress,
            pass_events=rewardable_pass_events,
            unsafe_contact=reward_unsafe_contact,
            stalled_behind_vehicle=reward_stalled,
            relative_progress_weight=(
                self.settings.relative_progress_weight
                if self.settings.relative_progress_enabled
                else 0.0
            ),
            pass_weight=(
                self.settings.pass_weight if self.settings.pass_event_enabled else 0.0
            ),
            unsafe_contact_weight=(
                self.settings.unsafe_contact_weight
                if self.settings.unsafe_contact_enabled
                else 0.0
            ),
            stalled_behind_weight=(
                self.settings.stalled_behind_weight
                if self.settings.stalled_behind_vehicle_enabled
                else 0.0
            ),
        )

        continuing_state = LidarRacingState(
            simulator_state=simulator_state,
            scan_history=terminal_history,
            scan_corruption_state=corruption_state,
            overtaking_state=overtaking_state,
            previous_ego_action=ego_normalized_action,
            npc_scenario_valid=npc_scenario_valid,
        )
        reset_state, reset_observation = self._reset_one(reset_key)
        done = terminated | truncated
        collector_state = jax.tree.map(
            lambda reset_value, current_value: jax.lax.select(
                done, reset_value, current_value
            ),
            reset_state,
            continuing_state,
        )
        collector_observation = jax.lax.select(
            done, reset_observation, terminal_observation
        )

        return StepResult(
            state=collector_state,
            observation=collector_observation,
            transition_next_observation=terminal_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            diagnostics=StepDiagnostics(
                progress_delta=progress_delta,
                ego_speed=ego_speed,
                collision=ego_collision,
                off_track=off_track,
                race_complete=race_complete,
                unrecoverable=unrecoverable,
                relative_progress=relative_progress,
                pass_count=jnp.sum(rewardable_pass_events.astype(jnp.int32)),
                unsafe_contact=unsafe_contact,
                stalled_behind_vehicle=stalled_behind_vehicle,
                nearest_opponent_distance=nearest_opponent_distance,
                following_vehicle=following_vehicle,
                ego_rank=ego_rank,
                opponent_present=opponent_present,
                collision_with_opponent=collision_with_opponent,
                collision_with_wall=collision_with_wall,
                npc_collision_flags=npc_collision_flags,
                npc_collision_without_ego=npc_collision_without_ego,
                minimum_npc_speed=minimum_npc_speed,
                terminal_ego_pose=simulator_state.cartesian_states[
                    EGO_INDEX, jnp.asarray([0, 1, 4])
                ],
                terminal_ego_frenet_pose=ego_frenet,
            ),
        )

    def step(
        self,
        key: jax.Array,
        state: LidarRacingState,
        ego_normalized_action: jax.Array,
        npc_physical_actions: jax.Array,
    ) -> StepResult:
        """Step one environment and auto-reset it when Ego alone is done."""

        return self._step_one(key, state, ego_normalized_action, npc_physical_actions)

    def step_batch(
        self,
        keys: jax.Array,
        states: LidarRacingState,
        ego_normalized_actions: jax.Array,
        npc_physical_actions: jax.Array,
    ) -> StepResult:
        """Vectorize environments; no environment or vehicle Python loop is used."""

        return jax.vmap(self._step_one)(
            keys,
            states,
            ego_normalized_actions,
            npc_physical_actions,
        )


__all__ = [
    "CANONICAL_FIELD_OF_VIEW",
    "CANONICAL_FRAME_STACK",
    "CANONICAL_RANGE_MAX",
    "EGO_INDEX",
    "LidarRacingEnv",
    "LidarRacingState",
    "MAX_NPC_COUNT",
    "RacingEnvSettings",
    "StepResult",
    "StepDiagnostics",
]

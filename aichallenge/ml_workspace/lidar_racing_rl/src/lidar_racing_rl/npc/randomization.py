"""Episode-level diversity for the three fixed Pure Pursuit opponents."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


def _range(config: Mapping[str, Any], key: str) -> tuple[float, float]:
    value = config[key]
    minimum = value["min"]
    maximum = value["max"]
    if minimum is None or maximum is None:
        raise ValueError(f"NPC randomization range '{key}' must be calibrated")
    minimum_float = float(minimum)
    maximum_float = float(maximum)
    if not math.isfinite(minimum_float) or not math.isfinite(maximum_float):
        raise ValueError(f"NPC randomization range '{key}' must be finite")
    if minimum_float > maximum_float:
        raise ValueError(f"NPC randomization range '{key}' is reversed")
    return minimum_float, maximum_float


def _integer_range(config: Mapping[str, Any], key: str) -> tuple[int, int]:
    minimum, maximum = _range(config, key)
    if not minimum.is_integer() or not maximum.is_integer():
        raise ValueError(f"NPC randomization range '{key}' must contain integers")
    return int(minimum), int(maximum)


@dataclass(frozen=True)
class NpcRandomizationBounds:
    """Resolved, static sampling bounds loaded from the NPC YAML."""

    speed_multiplier: tuple[float, float]
    lateral_offset: tuple[float, float]
    lookahead: tuple[float, float]
    steering_gain: tuple[float, float]
    acceleration_gain: tuple[float, float]
    safe_distance: tuple[float, float]
    control_delay_steps: tuple[int, int]
    braking_probability: float
    braking_start_step: tuple[int, int]
    braking_duration_steps: tuple[int, int]
    braking_acceleration: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> NpcRandomizationBounds:
        npc = config.get("npc", config)
        randomization = npc["randomization"]
        longitudinal = npc["longitudinal_controller"]
        braking = randomization["braking_event"]
        bounds = cls(
            speed_multiplier=_range(randomization, "speed_multiplier"),
            lateral_offset=_range(randomization, "lateral_offset"),
            lookahead=_range(randomization, "lookahead"),
            steering_gain=_range(randomization, "steering_gain"),
            acceleration_gain=_range(randomization, "acceleration_gain"),
            safe_distance=_range(longitudinal, "safe_following_distance"),
            control_delay_steps=_integer_range(randomization, "control_delay_steps"),
            braking_probability=float(braking["probability"]),
            braking_start_step=_integer_range(braking, "start_step"),
            braking_duration_steps=_integer_range(braking, "duration_steps"),
            braking_acceleration=float(braking["acceleration"]),
        )
        bounds.validate()
        return bounds

    def validate(self) -> None:
        if not 0.0 <= self.braking_probability <= 1.0:
            raise ValueError("braking probability must be within [0, 1]")
        for name, interval in (
            ("speed_multiplier", self.speed_multiplier),
            ("lateral_offset", self.lateral_offset),
            ("lookahead", self.lookahead),
            ("steering_gain", self.steering_gain),
            ("acceleration_gain", self.acceleration_gain),
            ("safe_distance", self.safe_distance),
        ):
            if not all(math.isfinite(value) for value in interval):
                raise ValueError(f"{name} must be finite")
            if interval[0] > interval[1]:
                raise ValueError(f"{name} range is reversed")
        for name, interval in (
            ("speed_multiplier", self.speed_multiplier),
            ("lookahead", self.lookahead),
            ("steering_gain", self.steering_gain),
            ("acceleration_gain", self.acceleration_gain),
            ("safe_distance", self.safe_distance),
        ):
            if interval[0] <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name, interval in (
            ("control_delay_steps", self.control_delay_steps),
            ("braking_start_step", self.braking_start_step),
            ("braking_duration_steps", self.braking_duration_steps),
        ):
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in interval
            ):
                raise ValueError(f"{name} must contain integers")
            if interval[0] > interval[1]:
                raise ValueError(f"{name} range is reversed")
        if self.control_delay_steps[0] < 0:
            raise ValueError("control delay cannot be negative")
        if self.braking_start_step[0] < 0 or self.braking_duration_steps[0] < 1:
            raise ValueError("braking event steps must be positive")
        if (
            not math.isfinite(self.braking_acceleration)
            or self.braking_acceleration >= 0.0
        ):
            raise ValueError("braking acceleration must be negative")

    def validate_reset_spacing(
        self,
        longitudinal_spacing: float,
        vehicle_length: float,
    ) -> None:
        """Require reset headway beyond the largest sampled safe distance."""

        if (
            not math.isfinite(longitudinal_spacing)
            or not math.isfinite(vehicle_length)
            or vehicle_length <= 0.0
            or longitudinal_spacing < self.safe_distance[1] + vehicle_length
        ):
            raise ValueError(
                "Step 2 reset spacing must be at least the maximum NPC "
                "safe-following distance plus vehicle length"
            )


@struct.dataclass
class NpcEpisodeParameters:
    """Per-NPC arrays sampled once at episode reset."""

    speed_multiplier: jax.Array
    lateral_offset: jax.Array
    lookahead: jax.Array
    steering_gain: jax.Array
    acceleration_gain: jax.Array
    safe_distance: jax.Array
    control_delay_steps: jax.Array
    braking_enabled: jax.Array
    braking_start_step: jax.Array
    braking_duration_steps: jax.Array
    braking_acceleration: jax.Array


def sample_npc_episode_parameters(
    key: jax.Array,
    *,
    npc_count: int,
    bounds: NpcRandomizationBounds,
) -> NpcEpisodeParameters:
    """Sample independent NPC behavior without a Python loop over vehicles."""

    if npc_count < 1:
        raise ValueError("npc_count must be positive")
    bounds.validate()
    keys = jax.random.split(key, 10)
    shape = (npc_count,)

    def uniform(sample_key: jax.Array, interval: tuple[float, float]) -> jax.Array:
        return jax.random.uniform(
            sample_key, shape=shape, minval=interval[0], maxval=interval[1]
        )

    delay = jax.random.randint(
        keys[6],
        shape,
        minval=bounds.control_delay_steps[0],
        maxval=bounds.control_delay_steps[1] + 1,
    )
    brake_start = jax.random.randint(
        keys[8],
        shape,
        minval=bounds.braking_start_step[0],
        maxval=bounds.braking_start_step[1] + 1,
    )
    brake_duration = jax.random.randint(
        keys[9],
        shape,
        minval=bounds.braking_duration_steps[0],
        maxval=bounds.braking_duration_steps[1] + 1,
    )
    return NpcEpisodeParameters(
        # Reset placement is Ego, nearest NPC, ..., farthest NPC.  Sorting the
        # independently sampled speeds in the same order keeps a rear NPC from
        # immediately catching the vehicle ahead while retaining episode-level
        # diversity in every sampled value.
        speed_multiplier=jnp.sort(uniform(keys[0], bounds.speed_multiplier)),
        lateral_offset=uniform(keys[1], bounds.lateral_offset),
        lookahead=uniform(keys[2], bounds.lookahead),
        steering_gain=uniform(keys[3], bounds.steering_gain),
        acceleration_gain=uniform(keys[4], bounds.acceleration_gain),
        safe_distance=uniform(keys[5], bounds.safe_distance),
        control_delay_steps=delay,
        braking_enabled=jax.random.bernoulli(
            keys[7], p=bounds.braking_probability, shape=shape
        ),
        braking_start_step=brake_start,
        braking_duration_steps=brake_duration,
        braking_acceleration=jnp.full(shape, bounds.braking_acceleration),
    )


def offset_waypoint_lines(waypoints: jax.Array, lateral_offsets: jax.Array) -> jax.Array:
    """Create ``[npcs, waypoints, 3+]`` parallel lines from one closed line."""

    if waypoints.ndim != 2 or waypoints.shape[1] < 3:
        raise ValueError("waypoints must have shape [waypoints, 3+]")
    if lateral_offsets.ndim != 1:
        raise ValueError("lateral_offsets must have shape [npcs]")

    xy = waypoints[:, :2]
    tangent = jnp.roll(xy, -1, axis=0) - jnp.roll(xy, 1, axis=0)
    tangent_norm = jnp.maximum(jnp.linalg.norm(tangent, axis=-1, keepdims=True), 1.0e-6)
    normal = jnp.stack((-tangent[:, 1], tangent[:, 0]), axis=-1) / tangent_norm
    offset_xy = xy[jnp.newaxis, ...] + lateral_offsets[:, None, None] * normal[None, ...]
    remaining = jnp.broadcast_to(
        waypoints[jnp.newaxis, :, 2:],
        (lateral_offsets.shape[0], waypoints.shape[0], waypoints.shape[1] - 2),
    )
    return jnp.concatenate((offset_xy, remaining), axis=-1)


def apply_braking_event(
    accelerations: jax.Array,
    step: jax.Array,
    parameters: NpcEpisodeParameters,
) -> jax.Array:
    """Apply each NPC's optional scripted braking window."""

    active = (
        parameters.braking_enabled
        & (step >= parameters.braking_start_step)
        & (step < parameters.braking_start_step + parameters.braking_duration_steps)
    )
    return jnp.where(
        active,
        jnp.minimum(accelerations, parameters.braking_acceleration),
        accelerations,
    )


def select_delayed_actions(
    action_history: jax.Array, delay_steps: jax.Array
) -> jax.Array:
    """Select per-NPC commands from newest-first ``[npcs, history, 2]``."""

    if action_history.ndim != 3 or action_history.shape[-1] != 2:
        raise ValueError("action_history must have shape [npcs, history, 2]")
    if delay_steps.shape != (action_history.shape[0],):
        raise ValueError("delay_steps must have shape [npcs]")
    clipped_delay = jnp.clip(delay_steps, 0, action_history.shape[1] - 1)
    return jnp.take_along_axis(
        action_history,
        clipped_delay[:, None, None],
        axis=1,
    )[:, 0, :]


__all__ = [
    "NpcEpisodeParameters",
    "NpcRandomizationBounds",
    "apply_braking_event",
    "offset_waypoint_lines",
    "sample_npc_episode_parameters",
    "select_delayed_actions",
]

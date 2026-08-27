"""Fail-closed Step 2 curriculum definitions independent of the JAX runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


CURRICULUM_SCHEMA_VERSION = 1
STEP2_PHASES = ("2a", "2b", "2c", "2d", "2e")
FIXED_OPPONENT_SOURCE = "fixed_pure_pursuit"
CHECKPOINT_POOL_SOURCE = "checkpoint_pool"


class CurriculumConfigurationError(ValueError):
    """Raised when a curriculum could weaken the intended stage boundary."""


class UnsupportedCurriculumPhaseError(RuntimeError):
    """Raised when a declared phase has no reviewed training integration."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CurriculumConfigurationError(f"{label} must be a mapping")
    return value


def _exact_keys(
    mapping: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(mapping)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise CurriculumConfigurationError(
            f"{label} schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CurriculumConfigurationError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CurriculumConfigurationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CurriculumConfigurationError(f"{label} must be finite")
    return result


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise CurriculumConfigurationError(f"{label} must be boolean")
    return value


def _strict_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CurriculumConfigurationError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class FloatInterval:
    """Closed, finite floating-point interval."""

    minimum: float
    maximum: float

    @classmethod
    def from_config(cls, value: Any, label: str) -> FloatInterval:
        config = _mapping(value, label)
        _exact_keys(config, {"min", "max"}, label)
        interval = cls(
            minimum=_finite_float(config["min"], f"{label}.min"),
            maximum=_finite_float(config["max"], f"{label}.max"),
        )
        if interval.minimum > interval.maximum:
            raise CurriculumConfigurationError(f"{label} interval is reversed")
        return interval

    def as_config(self) -> dict[str, float]:
        return {"min": self.minimum, "max": self.maximum}


@dataclass(frozen=True)
class IntegerInterval:
    """Closed, non-negative integer interval."""

    minimum: int
    maximum: int

    @classmethod
    def from_config(cls, value: Any, label: str) -> IntegerInterval:
        config = _mapping(value, label)
        _exact_keys(config, {"min", "max"}, label)
        interval = cls(
            minimum=_strict_int(config["min"], f"{label}.min"),
            maximum=_strict_int(config["max"], f"{label}.max"),
        )
        if interval.minimum > interval.maximum:
            raise CurriculumConfigurationError(f"{label} interval is reversed")
        return interval

    def as_config(self) -> dict[str, int]:
        return {"min": self.minimum, "max": self.maximum}


@dataclass(frozen=True)
class BrakingEventDefinition:
    """One phase's scripted NPC braking capability."""

    enabled: bool
    probability: float
    start_step: IntegerInterval
    duration_steps: IntegerInterval
    acceleration: float

    @classmethod
    def from_config(cls, value: Any, label: str) -> BrakingEventDefinition:
        config = _mapping(value, label)
        _exact_keys(
            config,
            {
                "enabled",
                "probability",
                "start_step",
                "duration_steps",
                "acceleration",
            },
            label,
        )
        result = cls(
            enabled=_strict_bool(config["enabled"], f"{label}.enabled"),
            probability=_finite_float(config["probability"], f"{label}.probability"),
            start_step=IntegerInterval.from_config(
                config["start_step"], f"{label}.start_step"
            ),
            duration_steps=IntegerInterval.from_config(
                config["duration_steps"], f"{label}.duration_steps"
            ),
            acceleration=_finite_float(
                config["acceleration"], f"{label}.acceleration"
            ),
        )
        result.validate(label)
        return result

    def validate(self, label: str = "braking_event") -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise CurriculumConfigurationError(f"{label}.probability must be in [0, 1]")
        if self.duration_steps.minimum < 1:
            raise CurriculumConfigurationError(f"{label}.duration_steps must be positive")
        if self.acceleration >= 0.0:
            raise CurriculumConfigurationError(f"{label}.acceleration must be negative")
        if self.enabled and self.probability <= 0.0:
            raise CurriculumConfigurationError(
                f"{label}.probability must be positive when braking is enabled"
            )
        if not self.enabled and self.probability != 0.0:
            raise CurriculumConfigurationError(
                f"{label}.probability must be zero when braking is disabled"
            )

    def as_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "probability": self.probability,
            "start_step": self.start_step.as_config(),
            "duration_steps": self.duration_steps.as_config(),
            "acceleration": self.acceleration,
        }


@dataclass(frozen=True)
class CurriculumPhase:
    """Normalized NPC constraints for one of the blueprint's phases 2a--2e."""

    name: str
    active_npc_count: int
    speed_multiplier: FloatInterval
    line_mode: str
    lateral_offset: FloatInterval
    control_delay_steps: IntegerInterval
    braking_event: BrakingEventDefinition
    opponent_source: str
    training_supported: bool

    @classmethod
    def from_config(cls, name: str, value: Any) -> CurriculumPhase:
        label = f"curriculum.phases.{name}"
        config = _mapping(value, label)
        _exact_keys(
            config,
            {
                "active_npc_count",
                "speed_multiplier",
                "line_mode",
                "lateral_offset",
                "control_delay_steps",
                "braking_event",
                "opponent_source",
                "training_supported",
            },
            label,
        )
        line_mode = _strict_string(config["line_mode"], f"{label}.line_mode")
        opponent_source = _strict_string(
            config["opponent_source"], f"{label}.opponent_source"
        )
        phase = cls(
            name=name,
            active_npc_count=_strict_int(
                config["active_npc_count"],
                f"{label}.active_npc_count",
                minimum=1,
            ),
            speed_multiplier=FloatInterval.from_config(
                config["speed_multiplier"], f"{label}.speed_multiplier"
            ),
            line_mode=line_mode,
            lateral_offset=FloatInterval.from_config(
                config["lateral_offset"], f"{label}.lateral_offset"
            ),
            control_delay_steps=IntegerInterval.from_config(
                config["control_delay_steps"], f"{label}.control_delay_steps"
            ),
            braking_event=BrakingEventDefinition.from_config(
                config["braking_event"], f"{label}.braking_event"
            ),
            opponent_source=opponent_source,
            training_supported=_strict_bool(
                config["training_supported"], f"{label}.training_supported"
            ),
        )
        phase.validate()
        return phase

    def validate(self) -> None:
        label = f"curriculum phase {self.name}"
        if self.name not in STEP2_PHASES:
            raise CurriculumConfigurationError(f"unknown {label}")
        if self.active_npc_count > 3:
            raise CurriculumConfigurationError(f"{label} supports at most three NPCs")
        if self.speed_multiplier.minimum <= 0.0:
            raise CurriculumConfigurationError(
                f"{label} speed multipliers must be positive"
            )
        if self.line_mode not in {"centerline", "random_offset"}:
            raise CurriculumConfigurationError(f"{label} has an unknown line mode")
        if self.line_mode == "centerline" and self.lateral_offset != FloatInterval(0.0, 0.0):
            raise CurriculumConfigurationError(
                f"{label} centerline mode requires zero lateral offset"
            )
        if self.line_mode == "random_offset" and not (
            self.lateral_offset.minimum < 0.0 < self.lateral_offset.maximum
        ):
            raise CurriculumConfigurationError(
                f"{label} random_offset must provide both left and right lines"
            )
        self.braking_event.validate(f"{label} braking_event")
        self._validate_stage_progression()

    def _validate_stage_progression(self) -> None:
        label = f"curriculum phase {self.name}"
        if self.name == "2a":
            if self.active_npc_count != 1:
                raise CurriculumConfigurationError("phase 2a requires exactly one NPC")
            if self.speed_multiplier.maximum >= 1.0:
                raise CurriculumConfigurationError("phase 2a requires a slower NPC")
            self._require_centerline_without_events(label)
        elif self.name == "2b":
            self._require_three_speed_diverse_npcs(label)
            self._require_centerline_without_events(label)
        elif self.name == "2c":
            self._require_three_speed_diverse_npcs(label)
            self._require_multiple_lines_without_events(label)
        elif self.name in {"2d", "2e"}:
            self._require_three_speed_diverse_npcs(label)
            if self.line_mode != "random_offset":
                raise CurriculumConfigurationError(f"{label} requires multiple lines")
            if self.control_delay_steps.maximum < 1:
                raise CurriculumConfigurationError(f"{label} requires control delay")
            if not self.braking_event.enabled:
                raise CurriculumConfigurationError(f"{label} requires braking events")

        if self.name == "2e":
            if self.opponent_source != CHECKPOINT_POOL_SOURCE:
                raise CurriculumConfigurationError("phase 2e requires checkpoint_pool")
            if self.training_supported:
                raise CurriculumConfigurationError(
                    "phase 2e must remain training_supported=false until integration is reviewed"
                )
        elif self.opponent_source != FIXED_OPPONENT_SOURCE:
            raise CurriculumConfigurationError(
                f"{label} requires fixed Pure Pursuit opponents"
            )
        elif not self.training_supported:
            raise CurriculumConfigurationError(
                f"{label} fixed-opponent phase must be marked training_supported"
            )

    def _require_three_speed_diverse_npcs(self, label: str) -> None:
        if self.active_npc_count != 3:
            raise CurriculumConfigurationError(f"{label} requires exactly three NPCs")
        if self.speed_multiplier.minimum >= self.speed_multiplier.maximum:
            raise CurriculumConfigurationError(f"{label} requires speed diversity")

    def _require_centerline_without_events(self, label: str) -> None:
        if self.line_mode != "centerline":
            raise CurriculumConfigurationError(f"{label} must use the centerline")
        self._require_no_events(label)

    def _require_multiple_lines_without_events(self, label: str) -> None:
        if self.line_mode != "random_offset":
            raise CurriculumConfigurationError(f"{label} requires multiple lines")
        self._require_no_events(label)

    def _require_no_events(self, label: str) -> None:
        if self.control_delay_steps != IntegerInterval(0, 0):
            raise CurriculumConfigurationError(f"{label} must not add control delay")
        if self.braking_event.enabled:
            raise CurriculumConfigurationError(f"{label} must not add braking events")

    def npc_constraints(self) -> dict[str, Any]:
        """Return a serialization-safe interface for a future trainer adapter."""

        return {
            "active_npc_count": self.active_npc_count,
            "speed_multiplier": self.speed_multiplier.as_config(),
            "line_mode": self.line_mode,
            "lateral_offset": self.lateral_offset.as_config(),
            "control_delay_steps": self.control_delay_steps.as_config(),
            "braking_event": self.braking_event.as_config(),
            "opponent_source": self.opponent_source,
        }


@dataclass(frozen=True)
class InformationBoundary:
    """Curriculum invariants that prevent GT/NPC data leaking into SAC."""

    learned_ego_agent_index: int
    learned_agent_count: int
    actor_observation: str
    critic_observation: str
    replay_scope: str
    save_npc_transitions: bool

    @classmethod
    def from_config(cls, value: Any) -> InformationBoundary:
        label = "curriculum.information_boundary"
        config = _mapping(value, label)
        _exact_keys(
            config,
            {
                "learned_ego_agent_index",
                "learned_agent_count",
                "actor_observation",
                "critic_observation",
                "replay_scope",
                "save_npc_transitions",
            },
            label,
        )
        result = cls(
            learned_ego_agent_index=_strict_int(
                config["learned_ego_agent_index"],
                f"{label}.learned_ego_agent_index",
            ),
            learned_agent_count=_strict_int(
                config["learned_agent_count"],
                f"{label}.learned_agent_count",
                minimum=1,
            ),
            actor_observation=_strict_string(
                config["actor_observation"], f"{label}.actor_observation"
            ),
            critic_observation=_strict_string(
                config["critic_observation"], f"{label}.critic_observation"
            ),
            replay_scope=_strict_string(config["replay_scope"], f"{label}.replay_scope"),
            save_npc_transitions=_strict_bool(
                config["save_npc_transitions"],
                f"{label}.save_npc_transitions",
            ),
        )
        if result != cls(0, 1, "lidar_only", "lidar_only", "ego_only", False):
            raise CurriculumConfigurationError(
                "curriculum must preserve one LiDAR-only Ego and Ego-only replay"
            )
        return result


@dataclass(frozen=True)
class CurriculumPlan:
    """Validated, complete Step 2 schedule with an explicit active phase."""

    schema_version: int
    active_phase: str
    ordered_phases: tuple[str, ...]
    information_boundary: InformationBoundary
    phases: tuple[CurriculumPhase, ...]

    @classmethod
    def from_config(cls, value: Any) -> CurriculumPlan:
        outer = _mapping(value, "configuration")
        config = _mapping(outer.get("curriculum", outer), "curriculum")
        _exact_keys(
            config,
            {
                "schema_version",
                "active_phase",
                "ordered_phases",
                "information_boundary",
                "phases",
            },
            "curriculum",
        )
        schema_version = _strict_int(config["schema_version"], "curriculum.schema_version")
        if schema_version != CURRICULUM_SCHEMA_VERSION:
            raise CurriculumConfigurationError("unsupported curriculum schema version")
        raw_order = config["ordered_phases"]
        if not isinstance(raw_order, Sequence) or isinstance(raw_order, str | bytes):
            raise CurriculumConfigurationError("curriculum.ordered_phases must be a list")
        ordered_phases = tuple(raw_order)
        if ordered_phases != STEP2_PHASES:
            raise CurriculumConfigurationError(
                f"curriculum phases must be ordered exactly as {STEP2_PHASES}"
            )
        raw_phases = _mapping(config["phases"], "curriculum.phases")
        if set(raw_phases) != set(STEP2_PHASES):
            raise CurriculumConfigurationError(
                "curriculum.phases must define each of 2a, 2b, 2c, 2d, and 2e exactly once"
            )
        phases = tuple(
            CurriculumPhase.from_config(name, raw_phases[name]) for name in STEP2_PHASES
        )
        active_phase = _strict_string(
            config["active_phase"], "curriculum.active_phase"
        )
        if active_phase not in STEP2_PHASES:
            raise CurriculumConfigurationError("curriculum.active_phase is unknown")
        return cls(
            schema_version=schema_version,
            active_phase=active_phase,
            ordered_phases=ordered_phases,
            information_boundary=InformationBoundary.from_config(
                config["information_boundary"]
            ),
            phases=phases,
        )

    def phase(self, name: str | None = None) -> CurriculumPhase:
        selected = self.active_phase if name is None else name
        for phase in self.phases:
            if phase.name == selected:
                return phase
        raise CurriculumConfigurationError(f"unknown curriculum phase: {selected}")

    def training_phase(self, name: str | None = None) -> CurriculumPhase:
        """Return a reviewed phase or fail closed for the unintegrated pool."""

        phase = self.phase(name)
        if not phase.training_supported:
            raise UnsupportedCurriculumPhaseError(
                "phase 2e checkpoint opponents are not integrated into training; "
                "validate the pool offline but keep opponent_pool disabled"
            )
        return phase


__all__ = [
    "CHECKPOINT_POOL_SOURCE",
    "CURRICULUM_SCHEMA_VERSION",
    "FIXED_OPPONENT_SOURCE",
    "STEP2_PHASES",
    "BrakingEventDefinition",
    "CurriculumConfigurationError",
    "CurriculumPhase",
    "CurriculumPlan",
    "FloatInterval",
    "InformationBoundary",
    "IntegerInterval",
    "UnsupportedCurriculumPhaseError",
]

from dataclasses import dataclass
from enum import Enum
from math import hypot
from typing import Optional, Tuple


def is_stuck_candidate(
    *,
    velocity: Optional[float],
    velocity_is_fresh: bool,
    command_speed: Optional[float],
    command_acceleration: Optional[float],
    command_is_fresh: bool,
    mpc_infeasible: bool,
    velocity_threshold: float,
    command_speed_threshold: float,
    command_acceleration_threshold: float,
) -> bool:
    """Return whether current evidence indicates that forward progress is stuck."""
    if velocity is None or not velocity_is_fresh:
        return False
    if abs(velocity) > velocity_threshold:
        return False
    if mpc_infeasible:
        return True
    if (
        not command_is_fresh
        or command_speed is None
        or command_acceleration is None
    ):
        return False
    return (
        command_speed >= command_speed_threshold
        and command_acceleration >= command_acceleration_threshold
    )


class RecoveryState(str, Enum):
    NORMAL = 'NORMAL'
    STOPPING = 'STOPPING'
    SHIFT_REVERSE = 'SHIFT_REVERSE'
    REVERSING = 'REVERSING'
    STOPPING_REVERSE = 'STOPPING_REVERSE'
    SHIFT_DRIVE = 'SHIFT_DRIVE'


@dataclass(frozen=True)
class RecoveryConfig:
    startup_grace_period: float = 5.0
    stuck_timeout: float = 2.5
    stop_velocity_threshold: float = 0.08
    stop_hold_duration: float = 0.30
    gear_settle_duration: float = 0.30
    gear_feedback_timeout: float = 1.0
    reverse_duration: float = 2.5
    reverse_distance: float = 1.5
    cooldown_duration: float = 4.0
    drive_gear: int = 2
    reverse_gear: int = 20


class RecoveryStateMachine:
    """Timing and transition logic independent of ROS message types."""

    def __init__(self, config: RecoveryConfig) -> None:
        self.config = config
        self.state = RecoveryState.NORMAL
        self.state_entered_at = 0.0
        self.last_update_at: Optional[float] = None
        self.stuck_since: Optional[float] = None
        self.stopped_since: Optional[float] = None
        self.reverse_start_position: Optional[Tuple[float, float]] = None
        self.cooldown_until = 0.0
        self.last_transition_reason = 'startup'

    def reset(self, now: float, reason: str) -> None:
        self.state = RecoveryState.NORMAL
        self.state_entered_at = now
        self.last_update_at = now
        self.stuck_since = None
        self.stopped_since = None
        self.reverse_start_position = None
        self.cooldown_until = now + self.config.startup_grace_period
        self.last_transition_reason = reason

    def _transition(self, state: RecoveryState, now: float, reason: str) -> None:
        self.state = state
        self.state_entered_at = now
        self.stopped_since = None
        self.last_transition_reason = reason

    def _stopped_for_hold_time(self, now: float, velocity: Optional[float]) -> bool:
        if velocity is None or abs(velocity) > self.config.stop_velocity_threshold:
            self.stopped_since = None
            return False
        if self.stopped_since is None:
            self.stopped_since = now
        return now - self.stopped_since >= self.config.stop_hold_duration

    def _gear_ready(self, now: float, gear: Optional[int], expected: int) -> bool:
        elapsed = now - self.state_entered_at
        if gear == expected:
            return elapsed >= self.config.gear_settle_duration
        # Some vehicle interfaces do not provide gear feedback. In that case,
        # use a conservative timeout. Explicit feedback for the wrong gear is
        # never ignored.
        return gear is None and elapsed >= self.config.gear_feedback_timeout

    def _reverse_distance(self, position: Optional[Tuple[float, float]]) -> float:
        if position is None or self.reverse_start_position is None:
            return 0.0
        return hypot(
            position[0] - self.reverse_start_position[0],
            position[1] - self.reverse_start_position[1],
        )

    def update(
        self,
        now: float,
        *,
        stuck_candidate: bool,
        velocity: Optional[float],
        position: Optional[Tuple[float, float]],
        gear: Optional[int],
        manual_trigger: bool = False,
    ) -> bool:
        """Advance the state machine and return True on a transition."""
        if self.last_update_at is None:
            self.reset(now, 'initialized')
        elif now < self.last_update_at:
            self.reset(now, 'clock moved backwards')
        self.last_update_at = now
        previous = self.state

        if self.state == RecoveryState.NORMAL:
            if manual_trigger:
                self._transition(RecoveryState.STOPPING, now, 'manual trigger')
                self.stuck_since = None
            elif now < self.cooldown_until:
                self.stuck_since = None
            elif stuck_candidate:
                if self.stuck_since is None:
                    self.stuck_since = now
                elif now - self.stuck_since >= self.config.stuck_timeout:
                    self._transition(RecoveryState.STOPPING, now, 'stuck timeout')
                    self.stuck_since = None
            else:
                self.stuck_since = None

        elif self.state == RecoveryState.STOPPING:
            if self._stopped_for_hold_time(now, velocity):
                self._transition(RecoveryState.SHIFT_REVERSE, now, 'vehicle stopped')

        elif self.state == RecoveryState.SHIFT_REVERSE:
            if self._gear_ready(now, gear, self.config.reverse_gear):
                self.reverse_start_position = position
                self._transition(RecoveryState.REVERSING, now, 'reverse gear ready')

        elif self.state == RecoveryState.REVERSING:
            elapsed = now - self.state_entered_at
            distance_reached = (
                self.config.reverse_distance > 0.0
                and self._reverse_distance(position) >= self.config.reverse_distance
            )
            if elapsed >= self.config.reverse_duration or distance_reached:
                reason = 'reverse distance reached' if distance_reached else 'reverse timeout'
                self._transition(RecoveryState.STOPPING_REVERSE, now, reason)

        elif self.state == RecoveryState.STOPPING_REVERSE:
            if self._stopped_for_hold_time(now, velocity):
                self._transition(RecoveryState.SHIFT_DRIVE, now, 'reverse motion stopped')

        elif self.state == RecoveryState.SHIFT_DRIVE:
            if self._gear_ready(now, gear, self.config.drive_gear):
                self._transition(RecoveryState.NORMAL, now, 'drive gear ready')
                self.reverse_start_position = None
                self.cooldown_until = now + self.config.cooldown_duration

        return self.state != previous

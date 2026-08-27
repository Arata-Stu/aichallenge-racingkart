"""Stage-aware reward and termination logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StepSignals:
    progress_m: float
    track_distance_m: float
    lap_completed: bool
    section_changed: bool
    speed_mps: float
    min_clearance_m: float
    previous_action: np.ndarray
    action: np.ndarray
    collision: bool = False
    off_track: bool = False


class LapReward:
    def __init__(self, config: dict) -> None:
        self.cfg = config

    def compute(self, signals: StepSignals) -> tuple[float, dict[str, float]]:
        forward = max(0.0, signals.progress_m) * float(self.cfg["progress_scale"])
        reverse = min(0.0, signals.progress_m) * float(self.cfg["reverse_progress_scale"])
        section = float(self.cfg["section_bonus"]) if signals.section_changed else 0.0
        lap = float(self.cfg["lap_bonus"]) if signals.lap_completed else 0.0
        steering_delta = abs(float(signals.action[0] - signals.previous_action[0]))
        accel_delta = abs(float(signals.action[1] - signals.previous_action[1]))
        smoothness = -(
            float(self.cfg["steering_delta_penalty"]) * steering_delta
            + float(self.cfg["acceleration_delta_penalty"]) * accel_delta
        )
        clearance_threshold = float(self.cfg["clearance_threshold_m"])
        clearance = -float(self.cfg["clearance_penalty_scale"]) * max(
            0.0, clearance_threshold - signals.min_clearance_m
        ) / max(clearance_threshold, 1e-6)
        collision = -float(self.cfg["collision_penalty"]) if signals.collision else 0.0
        off_track = -float(self.cfg["off_track_penalty"]) if signals.off_track else 0.0
        step = -float(self.cfg["step_penalty"])
        parts = {
            "progress": forward,
            "reverse": reverse,
            "section": section,
            "lap": lap,
            "smoothness": smoothness,
            "clearance": clearance,
            "collision": collision,
            "off_track": off_track,
            "step": step,
        }
        return float(sum(parts.values())), parts


class EpisodeTermination:
    def __init__(self, config: dict, off_track_distance_m: float) -> None:
        self.cfg = config
        self.off_track_distance_m = float(off_track_distance_m)
        self.reset()

    def reset(self) -> None:
        self.collision_steps = 0
        self.stuck_steps = 0

    def update(
        self,
        *,
        step: int,
        lap_delta: int,
        min_clearance_m: float,
        speed_mps: float,
        track_distance_m: float,
    ) -> tuple[bool, bool, str, bool, bool]:
        collision_now = min_clearance_m <= float(self.cfg["collision_distance_m"])
        self.collision_steps = self.collision_steps + 1 if collision_now else 0
        collision = self.collision_steps >= int(self.cfg["collision_patience_steps"])

        if step >= int(self.cfg["stuck_after_steps"]) and speed_mps < float(self.cfg["stuck_speed_mps"]):
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        stuck = self.stuck_steps >= int(self.cfg["stuck_patience_steps"])
        off_track = track_distance_m > self.off_track_distance_m
        lap_done = lap_delta >= int(self.cfg["target_laps"])
        truncated = step >= int(self.cfg["max_episode_steps"])
        terminated = collision or off_track or stuck or lap_done
        reason = ""
        if lap_done:
            reason = "lap_complete"
        elif collision:
            reason = "collision"
        elif off_track:
            reason = "off_track"
        elif stuck:
            reason = "stuck"
        elif truncated:
            reason = "time_limit"
        return terminated, truncated, reason, collision, off_track


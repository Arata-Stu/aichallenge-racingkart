"""Gymnasium environment backed by AWSIM and Virtual Scan."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces

from .intervention import InterventionRecorder
from .reward import EpisodeTermination, LapReward, StepSignals
from .ros_interface import VirtualScanRosInterface
from .scan import ScanHistory, sanitize_scan
from .track import TrackProgress, completed_lap_count


class VirtualScanAWSIMEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        obs_cfg = config["observation"]
        self.num_rays = int(obs_cfg["num_rays"])
        self.history_frames = int(obs_cfg["history_frames"])
        self.max_range_m = float(obs_cfg["max_range_m"])
        self.max_speed_mps = float(obs_cfg["max_speed_mps"])
        self.max_yaw_rate = float(obs_cfg["max_yaw_rate_rad_s"])
        self.action_cfg = config["action"]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "scan": spaces.Box(
                    0.0, 1.0, shape=(self.history_frames, self.num_rays), dtype=np.float32
                ),
                "state": spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
            }
        )
        if not rclpy.ok():
            rclpy.init()
        self.node = VirtualScanRosInterface(config["ros"], config["joy"])
        self.history = ScanHistory(self.history_frames, self.num_rays)
        self.track = TrackProgress(
            config["track"]["raceline_csv"],
            search_back=int(config["track"]["local_search_back"]),
            search_forward=int(config["track"]["local_search_forward"]),
            max_step_m=float(config["track"]["max_progress_per_step_m"]),
        )
        self.reward_function = LapReward(config["reward"])
        self.termination = EpisodeTermination(
            config["termination"], float(config["track"]["off_track_distance_m"])
        )
        joy_cfg = config["joy"]
        self.recorder = (
            InterventionRecorder(joy_cfg["record_dir"], joy_cfg["flush_samples"])
            if joy_cfg["enabled"] and joy_cfg["record_interventions"]
            else None
        )
        self.step_count = 0
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_lap = 0
        self.awsim_lap_transitions = 0
        self.previous_completed_laps = 0
        self.previous_section = 0
        self.last_observation: dict[str, np.ndarray] | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.node.stop()
        vehicle_state_sequence = self.node.awsim_state_sequence
        admin_state_sequence = self.node.admin_state_sequence
        self.node.reset_awsim()
        self.node.get_logger().info("Reset requested; waiting for AWSIM countdown and Start")
        started = self.node.wait_for_episode_start_after(
            vehicle_sequence=vehicle_state_sequence,
            admin_sequence=admin_state_sequence,
            target_state=str(self.config["ros"]["start_state"]),
            vehicle_ready_states=list(self.config["ros"]["vehicle_ready_states"]),
            timeout_sec=float(self.config["ros"]["start_timeout_sec"]),
        )
        if not started:
            raise RuntimeError(
                "Timed out waiting for post-reset AWSIM Start: "
                f"vehicle_state={self.node.awsim_state!r}, "
                f"admin_state={self.node.admin_state!r}. "
                "Verify /awsim/state and the Domain 0 /admin/awsim/state bridge."
            )
        self.node.get_logger().info(
            "AWSIM admin Start and vehicle readiness confirmed; episode step 0 begins"
        )
        scan_sequence = self.node.scan_sequence
        pose_sequence = self.node.pose_sequence
        fresh_observation = self.node.wait_for_observation_after(
            scan_sequence=scan_sequence,
            pose_sequence=pose_sequence,
            timeout_sec=float(self.config["ros"]["sensor_timeout_sec"]),
        )
        if not fresh_observation or self.node.scan_ranges is None or self.node.pose is None:
            raise RuntimeError(
                "A fresh Virtual Scan and localization pose were not received after AWSIM Start. "
                "Verify laserscan_generator, localization, and the configured ROS topics/domain."
            )
        scan = self._normalized_scan()
        self.history.reset(scan)
        x, y, yaw = self.node.pose
        _, track_distance = self.track.reset(x, y, yaw)
        self.termination.reset()
        self.step_count = 0
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_lap = int(self.node.awsim_status.get("lap_count", 0.0))
        self.awsim_lap_transitions = 0
        self.previous_completed_laps = 0
        self.previous_section = int(self.node.awsim_status.get("section", 0.0))
        observation = self._observation()
        self.last_observation = observation
        return observation, self._info(
            reason="", reward_parts={}, progress_m=0.0, track_distance_m=track_distance,
            intervention=False, executed_action=self.previous_action,
        )

    def step(self, action):
        proposed = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        intervention, human_action = self.node.human_action()
        executed = human_action if intervention and human_action is not None else proposed
        executed = np.asarray(executed, dtype=np.float32)
        steering, acceleration = self._physical_action(executed)
        old_sequence = self.node.scan_sequence
        self.node.publish_action(steering, acceleration)
        fresh = self.node.wait_for_scan_after(
            old_sequence, float(self.config["ros"]["step_timeout_sec"])
        )
        if not fresh:
            self.node.get_logger().warn("Virtual Scan timeout; using latest scan")
        if self.node.scan_ranges is None or self.node.pose is None:
            raise RuntimeError("Sensor data disappeared during an RL step")

        self.step_count += 1
        scan = self._normalized_scan()
        self.history.append(scan)
        x, y, _ = self.node.pose
        progress_m, _, track_distance_m = self.track.update(x, y)
        lap = int(self.node.awsim_status.get("lap_count", self.previous_lap))
        section = int(self.node.awsim_status.get("section", self.previous_section))
        if lap > self.previous_lap:
            self.awsim_lap_transitions += 1
        # The first AWSIM lapCount increase occurs when the start line is armed;
        # the next increase represents one completed lap. Raceline progress is a
        # fallback if the counter transition is delayed or missed.
        completed_laps = completed_lap_count(
            self.track.accumulated_progress,
            self.track.total_length,
            self.awsim_lap_transitions,
        )
        lap_delta = completed_laps
        lap_completed = completed_laps > self.previous_completed_laps
        section_changed = section != self.previous_section
        min_clearance_m = float(np.min(scan) * self.max_range_m)
        terminated, truncated, reason, collision, off_track = self.termination.update(
            step=self.step_count,
            lap_delta=lap_delta,
            min_clearance_m=min_clearance_m,
            speed_mps=self.node.speed_mps,
            track_distance_m=track_distance_m,
        )
        signals = StepSignals(
            progress_m=progress_m,
            track_distance_m=track_distance_m,
            lap_completed=lap_completed,
            section_changed=section_changed,
            speed_mps=self.node.speed_mps,
            min_clearance_m=min_clearance_m,
            previous_action=self.previous_action,
            action=executed,
            collision=collision,
            off_track=off_track,
        )
        reward, reward_parts = self.reward_function.compute(signals)
        observation = self._observation()
        info = self._info(
            reason=reason,
            reward_parts=reward_parts,
            progress_m=progress_m,
            track_distance_m=track_distance_m,
            intervention=intervention,
            executed_action=executed,
        )
        if intervention and self.recorder is not None and self.last_observation is not None:
            self.recorder.add(
                scan=self.last_observation["scan"],
                state=self.last_observation["state"],
                proposed_action=proposed,
                executed_action=executed,
                reward=reward,
            )
        self.previous_action = executed.copy()
        self.previous_lap = lap
        self.previous_completed_laps = completed_laps
        self.previous_section = section
        self.last_observation = observation
        if terminated or truncated:
            self.node.get_logger().info(
                "Episode finished: "
                f"reason={reason} steps={self.step_count} "
                f"progress={self.track.accumulated_progress:.2f}m "
                f"speed={self.node.speed_mps:.2f}m/s "
                f"clearance={min_clearance_m:.2f}m "
                f"track_distance={track_distance_m:.2f}m"
            )
            self.node.stop()
            if self.recorder is not None:
                self.recorder.flush()
        return observation, reward, terminated, truncated, info

    def _normalized_scan(self) -> np.ndarray:
        return sanitize_scan(
            self.node.scan_ranges, num_rays=self.num_rays, max_range_m=self.max_range_m
        )

    def _observation(self) -> dict[str, np.ndarray]:
        max_steering = max(float(self.action_cfg["max_steering_rad"]), 1e-6)
        state = np.array(
            [
                np.clip(self.node.speed_mps / self.max_speed_mps, -1.0, 1.0),
                np.clip(self.node.yaw_rate_rad_s / self.max_yaw_rate, -1.0, 1.0),
                np.clip(self.node.steering_rad / max_steering, -1.0, 1.0),
                self.previous_action[0],
                self.previous_action[1],
            ],
            dtype=np.float32,
        )
        return {"scan": self.history.value(), "state": state}

    def _physical_action(self, action: np.ndarray) -> tuple[float, float]:
        steering = float(action[0]) * float(self.action_cfg["max_steering_rad"])
        longitudinal = float(action[1])
        scale = (
            float(self.action_cfg["max_acceleration_mps2"])
            if longitudinal >= 0.0
            else float(self.action_cfg["max_braking_mps2"])
        )
        return steering, longitudinal * scale

    def _info(
        self,
        *,
        reason: str,
        reward_parts: dict[str, float],
        progress_m: float,
        track_distance_m: float,
        intervention: bool,
        executed_action: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "termination_reason": reason,
            "reward_breakdown": reward_parts,
            "progress_m": float(progress_m),
            "total_progress_m": float(self.track.accumulated_progress),
            "track_distance_m": float(track_distance_m),
            "speed_mps": float(self.node.speed_mps),
            "lap_count": int(self.node.awsim_status.get("lap_count", 0.0)),
            "track_lap_count": max(
                0, int(self.track.accumulated_progress / self.track.total_length)
            ),
            "awsim_lap_transitions": self.awsim_lap_transitions,
            "completed_lap_count": completed_lap_count(
                self.track.accumulated_progress,
                self.track.total_length,
                self.awsim_lap_transitions,
            ),
            "lap_time_s": float(self.node.awsim_status.get("lap_time", 0.0)),
            "section": int(self.node.awsim_status.get("section", 0.0)),
            "awsim_state": self.node.awsim_state,
            "awsim_admin_state": self.node.admin_state,
            "human_intervention": bool(intervention),
            "executed_action": np.asarray(executed_action, dtype=np.float32).copy(),
        }

    def close(self) -> None:
        self.node.stop()
        if self.recorder is not None:
            self.recorder.flush()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

"""Configuration loading with paths resolved relative to the selected YAML."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "stage": "lap",
    "ros": {
        "node_name": "virtual_scan_rl_env",
        "scan_topic": "/sensing/lidar/scan",
        "pose_topic": "/localization/pose_with_covariance",
        "velocity_topic": "/vehicle/status/velocity_status",
        "steering_topic": "/vehicle/status/steering_status",
        "status_topic": "/awsim/status",
        "state_topic": "/awsim/state",
        "admin_state_topic": "/awsim/admin_state",
        "joy_topic": "/joy",
        "control_topic": "/control/command/control_cmd",
        "reset_topic": "/awsim/reset",
        "use_sim_time": True,
        "sensor_timeout_sec": 1.0,
        "start_state": "Start",
        "vehicle_ready_states": ["Ready", "Start"],
        "start_timeout_sec": 30.0,
        "step_timeout_sec": 0.25,
    },
    "observation": {
        "num_rays": 1080,
        "history_frames": 4,
        "max_range_m": 30.0,
        "max_speed_mps": 15.0,
        "max_yaw_rate_rad_s": 3.0,
    },
    "action": {
        "max_steering_rad": 1.0,
        "max_acceleration_mps2": 2.0,
        "max_braking_mps2": 2.0,
    },
    "joy": {
        "enabled": True,
        "hold_button_index": 2,
        "steer_axis_index": 0,
        "positive_throttle_axis_index": 5,
        "negative_throttle_axis_index": 2,
        "deadzone": 0.05,
        "timeout_sec": 0.5,
        "record_interventions": True,
        "record_dir": "interventions",
        "flush_samples": 500,
    },
    "track": {
        "raceline_csv": "",
        "local_search_back": 15,
        "local_search_forward": 50,
        "max_progress_per_step_m": 4.0,
        "off_track_distance_m": 5.0,
    },
    "reward": {
        "progress_scale": 2.0,
        "section_bonus": 5.0,
        "lap_bonus": 200.0,
        "reverse_progress_scale": 4.0,
        "step_penalty": 0.02,
        "collision_penalty": 150.0,
        "off_track_penalty": 100.0,
        "steering_delta_penalty": 0.03,
        "acceleration_delta_penalty": 0.01,
        "clearance_threshold_m": 1.0,
        "clearance_penalty_scale": 0.1,
    },
    "termination": {
        "target_laps": 1,
        "max_episode_steps": 15000,
        "collision_distance_m": 0.65,
        "collision_patience_steps": 3,
        "stuck_speed_mps": 0.15,
        "stuck_after_steps": 500,
        "stuck_patience_steps": 500,
    },
    "sac": {
        "total_timesteps": 300000,
        "learning_rate": 0.0003,
        "buffer_size": 100000,
        "learning_starts": 2000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.995,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto_0.1",
        "net_arch": [256, 256],
        "features_dim": 128,
        "device": "auto",
        "seed": 42,
        "checkpoint_freq": 10000,
        "save_replay_buffer": True,
    },
    "output": {
        "checkpoint_dir": "checkpoints/lap",
        "tensorboard_dir": "logs/lap",
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    config = _merge(DEFAULT_CONFIG, loaded)
    config["track"]["raceline_csv"] = _resolve_path(
        str(config["track"].get("raceline_csv", "")), config_path.parent
    )
    for key in ("record_dir",):
        config["joy"][key] = _resolve_path(str(config["joy"][key]), config_path.parent)
    for key in ("checkpoint_dir", "tensorboard_dir"):
        config["output"][key] = _resolve_path(str(config["output"][key]), config_path.parent)
    config["_config_path"] = str(config_path)
    return config

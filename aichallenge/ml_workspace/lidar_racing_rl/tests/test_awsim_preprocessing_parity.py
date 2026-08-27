"""Parity contract between JAX training and NumPy ROS preprocessing."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import jax.numpy as jnp
import numpy as np

from lidar_racing_rl.envs.observation import (
    canonicalize_scan,
    initialize_frame_stack,
    update_frame_stack,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
ROS_PREPROCESSING_PATH = (
    REPOSITORY_ROOT
    / "aichallenge"
    / "workspace"
    / "src"
    / "aichallenge_submit"
    / "lidar_racing_controller"
    / "lidar_racing_controller"
    / "preprocessing.py"
)


def _load_ros_preprocessing() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lidar_racing_ros_preprocessing",
        ROS_PREPROCESSING_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ROS preprocessing from {ROS_PREPROCESSING_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ros_canonicalize(module: ModuleType, ranges: np.ndarray) -> np.ndarray:
    angle_min = -3.0 * math.pi / 4.0
    angle_max = 3.0 * math.pi / 4.0
    return module.canonicalize_laserscan(
        ranges,
        range_min=0.1,
        range_max=30.0,
        angle_min=angle_min,
        angle_max=angle_max,
        angle_increment=(angle_max - angle_min) / 1079.0,
    ).values


def test_awsim_pooling_matches_training_preprocessing() -> None:
    ros_preprocessing = _load_ros_preprocessing()
    ranges = np.linspace(0.1, 30.0, 1080, dtype=np.float32)
    ranges[0:12] = np.array(
        [
            np.nan,
            np.inf,
            31.0,
            0.09,
            0.1,
            2.0,
            30.0,
            30.1,
            -np.inf,
            9.0,
            6.0,
            3.0,
        ],
        dtype=np.float32,
    )

    training = np.asarray(
        canonicalize_scan(jnp.asarray(ranges), range_min=0.1, range_max=30.0)
    )
    deployment = _ros_canonicalize(ros_preprocessing, ranges)

    np.testing.assert_allclose(deployment, training, rtol=0.0, atol=1.0e-7)


def test_actor_channel_order_matches_ros_frame_stack() -> None:
    ros_preprocessing = _load_ros_preprocessing()
    first_ranges = np.full(1080, 12.0, dtype=np.float32)
    second_ranges = np.full(1080, 6.0, dtype=np.float32)
    second_ranges[0:3] = (np.nan, np.inf, 31.0)

    first = _ros_canonicalize(ros_preprocessing, first_ranges)
    second = _ros_canonicalize(ros_preprocessing, second_ranges)

    ros_history = ros_preprocessing.FrameStack(frame_count=4, channels=2, beams=360)
    ros_history.append(first)
    ros_history.append(second)

    training_history = initialize_frame_stack(jnp.asarray(first), num_frames=4)
    training_history = update_frame_stack(training_history, jnp.asarray(second))
    training_actor_input = np.asarray(training_history).reshape(8, 360)

    np.testing.assert_array_equal(ros_history.actor_input(), training_actor_input)

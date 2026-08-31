"""Tests for GT-only reward and termination helpers."""

import jax.numpy as jnp
import numpy as np

from lidar_racing_rl.envs.reward import (
    step1_reward,
    step2_reward,
    trajectory_aided_action_reward,
    wrapped_progress_delta,
)
from lidar_racing_rl.envs.termination import ego_done_flags, update_episode_progress


def test_progress_wraparound() -> None:
    np.testing.assert_allclose(
        wrapped_progress_delta(jnp.asarray(99.0), jnp.asarray(1.0), jnp.asarray(100.0)),
        2.0,
    )
    np.testing.assert_allclose(
        wrapped_progress_delta(jnp.asarray(1.0), jnp.asarray(99.0), jnp.asarray(100.0)),
        -2.0,
    )


def test_step1_reward_penalizes_collision_and_action_jump() -> None:
    reward = step1_reward(
        jnp.asarray(1.0),
        jnp.asarray(2.0),
        jnp.zeros((2,)),
        jnp.ones((2,)),
        track_length=jnp.asarray(100.0),
        collision=jnp.asarray(True),
        off_track=jnp.asarray(False),
        reversing=jnp.asarray(False),
        progress_weight=1.0,
        collision_weight=10.0,
        off_track_weight=10.0,
        smoothness_weight=0.5,
        reverse_weight=1.0,
    )
    np.testing.assert_allclose(reward, -10.0)


def test_step2_reward_composes_relative_pass_and_safety_terms() -> None:
    reward = step2_reward(
        jnp.asarray(1.0),
        relative_progress=jnp.asarray(2.0),
        pass_events=jnp.asarray([True, False, True]),
        unsafe_contact=jnp.asarray(True),
        stalled_behind_vehicle=jnp.asarray(True),
        relative_progress_weight=0.5,
        pass_weight=5.0,
        unsafe_contact_weight=2.0,
        stalled_behind_weight=0.1,
    )

    np.testing.assert_allclose(reward, 9.9)


def test_trajectory_aided_reward_clips_normalized_action_agreement() -> None:
    agent = jnp.asarray([[0.1, 0.2], [1.0, -1.0], [0.4, 0.1]])
    reference = jnp.asarray([[0.1, 0.2], [-1.0, 1.0], [0.0, 0.0]])

    reward = trajectory_aided_action_reward(agent, reference, weight=0.2)

    np.testing.assert_allclose(reward, jnp.asarray([0.2, 0.0, 0.1]))


def test_time_limit_is_truncated_not_terminated() -> None:
    terminated, truncated = ego_done_flags(
        collision=jnp.asarray(False),
        off_track=jnp.asarray(False),
        race_complete=jnp.asarray(False),
        unrecoverable=jnp.asarray(False),
        step_count=jnp.asarray(100),
        max_steps=100,
    )
    assert not bool(terminated)
    assert bool(truncated)


def test_race_completion_is_measured_from_randomized_episode_start() -> None:
    first_delta = wrapped_progress_delta(
        jnp.asarray(90.0), jnp.asarray(10.0), jnp.asarray(100.0)
    )
    progress, race_complete = update_episode_progress(
        jnp.asarray(0.0),
        first_delta,
        track_length=jnp.asarray(100.0),
        max_num_laps=1,
    )
    np.testing.assert_allclose(progress, 20.0)
    assert not bool(race_complete)

    progress, race_complete = update_episode_progress(
        progress,
        jnp.asarray(80.0),
        track_length=jnp.asarray(100.0),
        max_num_laps=1,
    )
    np.testing.assert_allclose(progress, 100.0)
    assert bool(race_complete)

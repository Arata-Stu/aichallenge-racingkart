"""Evaluation aggregation tests for independent vector episode boundaries."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from lidar_racing_rl.evaluation.metrics import (
    EvaluationAccumulator,
    evaluation_summary,
    initialize_evaluation_accumulator,
    select_episode_completions,
    update_evaluation_accumulator,
)


def _step(
    state: EvaluationAccumulator,
    **overrides: Any,
) -> EvaluationAccumulator:
    num_envs = state.episode_return.shape[0]
    zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
    false = jnp.zeros((num_envs,), dtype=jnp.bool_)
    values: dict[str, Any] = {
        "reward": zeros,
        "progress_delta": zeros,
        "speed": zeros,
        "normalized_action": jnp.zeros((num_envs, 2), dtype=jnp.float32),
        "collision": false,
        "off_track": false,
        "race_complete": false,
        "relative_progress": zeros,
        "pass_count": jnp.zeros((num_envs,), dtype=jnp.int32),
        "collision_with_opponent": false,
        "collision_with_wall": false,
        "unsafe_contact": false,
        "following_vehicle": false,
        "stalled_behind_vehicle": false,
        "nearest_opponent_distance": zeros,
        "ego_rank": jnp.ones((num_envs,), dtype=jnp.int32),
        "opponent_present": false,
        "terminated": false,
        "truncated": false,
    }
    values.update(overrides)
    return update_evaluation_accumulator(state, **values)


def test_completed_episode_metrics_and_partial_episode_are_separate() -> None:
    state = initialize_evaluation_accumulator(2)
    state = _step(
        state,
        reward=jnp.asarray([2.0, 3.0]),
        progress_delta=jnp.asarray([1.0, 1.5]),
        speed=jnp.asarray([4.0, 6.0]),
        normalized_action=jnp.asarray([[0.2, -0.1], [0.4, 0.3]]),
        race_complete=jnp.asarray([True, False]),
        terminated=jnp.asarray([True, False]),
    )
    summary = evaluation_summary(state)

    assert int(summary["episodes"]) == 1
    assert float(summary["race_completion_rate"]) == 1.0
    assert float(summary["mean_return"]) == 2.0
    assert float(summary["overtake_success_rate"]) == 0.0
    assert float(summary["minimum_opponent_distance_mean"]) == 0.0
    assert float(state.episode_return[0]) == 0.0
    assert float(state.episode_return[1]) == 3.0


def test_requested_episode_limit_selects_only_remaining_completions() -> None:
    done = jnp.asarray([True, False, True, True])

    selected = select_episode_completions(done, jnp.asarray(2))
    none_remaining = select_episode_completions(done, jnp.asarray(0))

    assert selected.tolist() == [True, False, True, False]
    assert none_remaining.tolist() == [False, False, False, False]


def test_collision_and_time_limit_rates() -> None:
    state = initialize_evaluation_accumulator(2)
    state = _step(
        state,
        speed=jnp.ones((2,)),
        collision=jnp.asarray([True, False]),
        collision_with_wall=jnp.asarray([True, False]),
        opponent_present=jnp.asarray([True, True]),
        nearest_opponent_distance=jnp.asarray([5.0, 6.0]),
        terminated=jnp.asarray([True, False]),
        truncated=jnp.asarray([False, True]),
    )
    summary = evaluation_summary(state)

    assert int(summary["episodes"]) == 2
    assert float(summary["collision_rate"]) == 0.5
    assert float(summary["wall_collision_rate"]) == 0.5
    assert float(summary["truncation_rate"]) == 0.5


def test_overtaking_following_and_recontact_metrics() -> None:
    state = initialize_evaluation_accumulator(1)
    state = _step(
        state,
        relative_progress=jnp.asarray([0.25]),
        pass_count=jnp.asarray([1], dtype=jnp.int32),
        following_vehicle=jnp.asarray([True]),
        nearest_opponent_distance=jnp.asarray([4.0]),
        ego_rank=jnp.asarray([2], dtype=jnp.int32),
        opponent_present=jnp.asarray([True]),
    )
    state = _step(
        state,
        collision=jnp.asarray([True]),
        relative_progress=jnp.asarray([0.5]),
        collision_with_opponent=jnp.asarray([True]),
        unsafe_contact=jnp.asarray([True]),
        nearest_opponent_distance=jnp.asarray([1.0]),
        ego_rank=jnp.asarray([1], dtype=jnp.int32),
        opponent_present=jnp.asarray([True]),
        terminated=jnp.asarray([True]),
    )
    summary = evaluation_summary(state)

    assert int(summary["opponent_episodes"]) == 1
    assert float(summary["overtake_success_rate"]) == 1.0
    assert float(summary["mean_passes_per_episode"]) == 1.0
    assert float(summary["mean_steps_to_first_pass"]) == 1.0
    assert float(summary["mean_relative_progress"]) == 0.75
    assert float(summary["opponent_collision_rate"]) == 1.0
    assert float(summary["unsafe_contact_episode_rate"]) == 1.0
    assert float(summary["mean_follow_duration_steps"]) == 1.0
    assert float(summary["minimum_opponent_distance_mean"]) == 1.0
    assert float(summary["mean_final_rank"]) == 1.0
    assert float(summary["post_pass_recontact_rate"]) == 1.0
    assert float(summary["safe_wait_success_rate"]) == 0.5

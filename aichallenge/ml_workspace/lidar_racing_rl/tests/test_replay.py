"""Replay schema, ring behavior, and time-limit bootstrap tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from lidar_racing_rl.sac.replay import (
    ReplayBufferConfig,
    TransitionBatch,
    can_sample,
    initialize_replay_buffer,
    insert_batch,
    replay_memory_bytes,
    sample_batch,
)


def _batch(start: int, size: int) -> TransitionBatch:
    values = jnp.arange(start, start + size, dtype=jnp.float32)
    observation = jnp.broadcast_to(values[:, None, None], (size, 8, 360))
    return TransitionBatch(
        observation=observation,
        action=jnp.stack((values, -values), axis=-1),
        reward=values,
        terminated=(values % 3) == 0,
        truncated=(values % 3) == 1,
        next_observation=observation + 0.5,
    )


def test_ring_insert_wrap_and_float16_storage() -> None:
    config = ReplayBufferConfig(capacity=5, observation_shape=(8, 360), action_dim=2)
    state = initialize_replay_buffer(config)
    state = insert_batch(state, _batch(0, 3))
    state = insert_batch(state, _batch(3, 3))

    assert int(state.size) == 5
    assert int(state.write_index) == 1
    assert int(state.total_inserted) == 6
    assert state.observation.dtype == jnp.float16
    assert float(state.reward[0]) == 5.0
    assert replay_memory_bytes(state) > 0


def test_sample_is_seed_deterministic_and_restores_float32() -> None:
    state = insert_batch(
        initialize_replay_buffer(
            ReplayBufferConfig(capacity=8, observation_shape=(8, 360), action_dim=2)
        ),
        _batch(0, 8),
    )
    key = jax.random.key(41)
    first = sample_batch(key, state, 4)
    second = sample_batch(key, state, 4)

    assert jnp.array_equal(first.reward, second.reward)
    assert first.observation.dtype == jnp.float32
    assert bool(can_sample(state, 8))
    assert not bool(can_sample(state, 9))


def test_truncation_keeps_bootstrap_but_termination_does_not() -> None:
    transitions = _batch(0, 3)
    assert jnp.array_equal(
        transitions.bootstrap_mask,
        jnp.asarray([0.0, 1.0, 1.0], dtype=jnp.float32),
    )


def test_replay_rejects_non_ego_fields_by_schema() -> None:
    assert "cartesian" not in TransitionBatch.__dataclass_fields__
    assert "frenet" not in TransitionBatch.__dataclass_fields__
    assert "npc" not in TransitionBatch.__dataclass_fields__


def test_insert_rejects_batch_larger_than_capacity() -> None:
    state = initialize_replay_buffer(
        ReplayBufferConfig(capacity=2, observation_shape=(8, 360), action_dim=2)
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        insert_batch(state, _batch(0, 3))

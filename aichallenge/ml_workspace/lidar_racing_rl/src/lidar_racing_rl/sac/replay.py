"""JAX ring replay buffer for ego-only LiDAR transitions.

The buffer deliberately has no fields for simulator ground truth.  Its schema
is the information-boundary contract: observation, normalized action, reward,
termination, time-limit truncation, and next observation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


@dataclass(frozen=True)
class ReplayBufferConfig:
    """Static replay allocation and sampling settings."""

    capacity: int
    observation_shape: tuple[int, ...]
    action_dim: int
    observation_storage_dtype: Any = jnp.float16

    def validate(self) -> None:
        if isinstance(self.capacity, bool) or self.capacity < 1:
            raise ValueError("replay capacity must be positive")
        if not self.observation_shape or any(size < 1 for size in self.observation_shape):
            raise ValueError("observation_shape dimensions must be positive")
        if isinstance(self.action_dim, bool) or self.action_dim < 1:
            raise ValueError("action_dim must be positive")
        dtype = jnp.dtype(self.observation_storage_dtype)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError("observation storage dtype must be floating point")


@struct.dataclass
class TransitionBatch:
    """A batch of Ego SAC transitions; no GT or NPC state is permitted."""

    observation: jax.Array
    action: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    next_observation: jax.Array

    @property
    def bootstrap_mask(self) -> jax.Array:
        """Return 1 for transitions whose target may bootstrap.

        Time-limit truncation intentionally keeps bootstrapping.  Only a true
        MDP termination removes it.
        """

        return (~self.terminated).astype(jnp.float32)


@struct.dataclass
class ReplayBufferState:
    """Fixed-shape storage and scalar ring cursors."""

    observation: jax.Array
    action: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    next_observation: jax.Array
    write_index: jax.Array
    size: jax.Array
    total_inserted: jax.Array


def initialize_replay_buffer(config: ReplayBufferConfig) -> ReplayBufferState:
    """Allocate an empty fixed-shape replay buffer."""

    config.validate()
    observation_shape = (config.capacity, *config.observation_shape)
    return ReplayBufferState(
        observation=jnp.zeros(observation_shape, dtype=config.observation_storage_dtype),
        action=jnp.zeros((config.capacity, config.action_dim), dtype=jnp.float32),
        reward=jnp.zeros((config.capacity,), dtype=jnp.float32),
        terminated=jnp.zeros((config.capacity,), dtype=jnp.bool_),
        truncated=jnp.zeros((config.capacity,), dtype=jnp.bool_),
        next_observation=jnp.zeros(
            observation_shape, dtype=config.observation_storage_dtype
        ),
        write_index=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
        total_inserted=jnp.asarray(0, dtype=jnp.int32),
    )


def _validate_transition_shapes(
    state: ReplayBufferState,
    transitions: TransitionBatch,
) -> int:
    batch_size = transitions.reward.shape[0]
    capacity = state.reward.shape[0]
    expected_observation = (batch_size, *state.observation.shape[1:])
    expected_action = (batch_size, state.action.shape[-1])
    scalar_shape = (batch_size,)
    if batch_size < 1:
        raise ValueError("transition batch must not be empty")
    if batch_size > capacity:
        raise ValueError("transition batch cannot exceed replay capacity")
    if transitions.observation.shape != expected_observation:
        raise ValueError("observation shape does not match replay storage")
    if transitions.next_observation.shape != expected_observation:
        raise ValueError("next_observation shape does not match replay storage")
    if transitions.action.shape != expected_action:
        raise ValueError("action shape does not match replay storage")
    for name, value in (
        ("reward", transitions.reward),
        ("terminated", transitions.terminated),
        ("truncated", transitions.truncated),
    ):
        if value.shape != scalar_shape:
            raise ValueError(f"{name} must have shape [batch]")
    return batch_size


def insert_batch(
    state: ReplayBufferState,
    transitions: TransitionBatch,
) -> ReplayBufferState:
    """Insert one vector-environment batch without a Python environment loop."""

    batch_size = _validate_transition_shapes(state, transitions)
    capacity = state.reward.shape[0]
    indices = (state.write_index + jnp.arange(batch_size, dtype=jnp.int32)) % capacity
    return ReplayBufferState(
        observation=state.observation.at[indices].set(
            transitions.observation.astype(state.observation.dtype)
        ),
        action=state.action.at[indices].set(transitions.action.astype(jnp.float32)),
        reward=state.reward.at[indices].set(transitions.reward.astype(jnp.float32)),
        terminated=state.terminated.at[indices].set(
            transitions.terminated.astype(jnp.bool_)
        ),
        truncated=state.truncated.at[indices].set(
            transitions.truncated.astype(jnp.bool_)
        ),
        next_observation=state.next_observation.at[indices].set(
            transitions.next_observation.astype(state.next_observation.dtype)
        ),
        write_index=(state.write_index + batch_size) % capacity,
        size=jnp.minimum(state.size + batch_size, capacity),
        total_inserted=state.total_inserted + batch_size,
    )


def can_sample(state: ReplayBufferState, batch_size: int) -> jax.Array:
    """Return a JAX boolean indicating whether a full sample is available."""

    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("sample batch_size must be positive")
    return state.size >= batch_size


def sample_batch(
    key: jax.Array,
    state: ReplayBufferState,
    batch_size: int,
) -> TransitionBatch:
    """Sample uniformly with replacement and restore float32 observations.

    The caller must gate this function with :func:`can_sample`.  Keeping that
    condition outside the sampler makes the compiled sample shape static.
    """

    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("sample batch_size must be positive")
    indices = jax.random.randint(
        key,
        shape=(batch_size,),
        minval=0,
        maxval=jnp.maximum(state.size, 1),
    )
    return TransitionBatch(
        observation=state.observation[indices].astype(jnp.float32),
        action=state.action[indices],
        reward=state.reward[indices],
        terminated=state.terminated[indices],
        truncated=state.truncated[indices],
        next_observation=state.next_observation[indices].astype(jnp.float32),
    )


def replay_memory_bytes(state: ReplayBufferState) -> int:
    """Return statically allocated device storage in bytes."""

    arrays = (
        state.observation,
        state.action,
        state.reward,
        state.terminated,
        state.truncated,
        state.next_observation,
        state.write_index,
        state.size,
        state.total_inserted,
    )
    return sum(array.size * array.dtype.itemsize for array in arrays)


__all__ = [
    "ReplayBufferConfig",
    "ReplayBufferState",
    "TransitionBatch",
    "can_sample",
    "initialize_replay_buffer",
    "insert_batch",
    "replay_memory_bytes",
    "sample_batch",
]

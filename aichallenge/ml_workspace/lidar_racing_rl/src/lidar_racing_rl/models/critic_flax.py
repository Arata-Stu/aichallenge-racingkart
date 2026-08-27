"""Independent twin Flax Q networks for LiDAR-only SAC."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from lidar_racing_rl.models.actor_flax import ACTION_DIM
from lidar_racing_rl.models.encoder_flax import LidarEncoder


CRITIC_HIDDEN_SIZES = (256, 256)


class TwinQValues(NamedTuple):
    """Scalar values from the two independent Q networks."""

    q1: jax.Array
    q2: jax.Array


class QNetwork(nn.Module):
    """One LiDAR encoder followed by an action-conditioned scalar Q MLP."""

    action_dim: int = ACTION_DIM
    hidden_sizes: tuple[int, ...] = CRITIC_HIDDEN_SIZES

    def setup(self) -> None:
        if self.action_dim != ACTION_DIM:
            raise ValueError("the racing Critic action_dim must be exactly 2")
        if self.hidden_sizes != CRITIC_HIDDEN_SIZES:
            raise ValueError("the initial Critic requires hidden_sizes=(256, 256)")

        self.encoder = LidarEncoder(name="encoder")
        self.hidden_0 = nn.Dense(
            self.hidden_sizes[0],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            name="hidden_0",
        )
        self.hidden_1 = nn.Dense(
            self.hidden_sizes[1],
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            name="hidden_1",
        )
        self.value_head = nn.Dense(
            1,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            name="value_head",
        )

    def __call__(self, observation: jax.Array, normalized_action: jax.Array) -> jax.Array:
        """Return ``Q(observation, normalized_action)`` for every leading item."""

        features = self.encoder(observation)
        action = jnp.asarray(normalized_action, dtype=jnp.float32)
        if action.ndim < 1 or action.shape[-1] != self.action_dim:
            raise ValueError("normalized_action must have shape [..., 2]")
        if action.shape[:-1] != features.shape[:-1]:
            raise ValueError("observation and action leading dimensions must match")

        value = nn.relu(self.hidden_0(jnp.concatenate((features, action), axis=-1)))
        value = nn.relu(self.hidden_1(value))
        return jnp.squeeze(self.value_head(value), axis=-1)


class TwinQCritic(nn.Module):
    """Twin Q networks with deliberately unshared LiDAR encoders and parameters."""

    action_dim: int = ACTION_DIM
    hidden_sizes: tuple[int, ...] = CRITIC_HIDDEN_SIZES

    def setup(self) -> None:
        # Stable q1/q2 names are part of the checkpoint and conversion contract.
        self.q1 = QNetwork(
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            name="q1",
        )
        self.q2 = QNetwork(
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            name="q2",
        )

    def __call__(
        self,
        observation: jax.Array,
        normalized_action: jax.Array,
    ) -> TwinQValues:
        """Evaluate both independent Q functions without a Python batch loop."""

        return TwinQValues(
            q1=self.q1(observation, normalized_action),
            q2=self.q2(observation, normalized_action),
        )


__all__ = [
    "CRITIC_HIDDEN_SIZES",
    "QNetwork",
    "TwinQCritic",
    "TwinQValues",
]

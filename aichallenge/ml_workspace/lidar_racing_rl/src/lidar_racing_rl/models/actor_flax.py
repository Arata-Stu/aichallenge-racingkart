"""Flax tanh-squashed Gaussian Actor for normalized SAC actions."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from lidar_racing_rl.models.encoder_flax import LidarEncoder


ACTION_DIM = 2
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
_LOG_TWO = math.log(2.0)
_LOG_TWO_PI = math.log(2.0 * math.pi)


class ActorStatistics(NamedTuple):
    """Pre-squash Gaussian parameters returned by ``TanhGaussianActor``."""

    mean: jax.Array
    log_std: jax.Array


class ActorSample(NamedTuple):
    """Reparameterized sample and its corrected squashed log probability."""

    action: jax.Array
    log_probability: jax.Array
    pre_tanh_value: jax.Array
    mean: jax.Array
    log_std: jax.Array


def gaussian_log_probability(
    value: jax.Array,
    mean: jax.Array,
    log_std: jax.Array,
) -> jax.Array:
    """Evaluate a diagonal Gaussian log density and sum over action axes."""

    normalized = (value - mean) * jnp.exp(-log_std)
    elementwise = -0.5 * (jnp.square(normalized) + _LOG_TWO_PI) - log_std
    return jnp.sum(elementwise, axis=-1)


def tanh_log_abs_det_jacobian(pre_tanh_value: jax.Array) -> jax.Array:
    """Compute ``log|d tanh(x) / dx|`` with the stable SAC identity."""

    elementwise = 2.0 * (
        _LOG_TWO - pre_tanh_value - jax.nn.softplus(-2.0 * pre_tanh_value)
    )
    return jnp.sum(elementwise, axis=-1)


def tanh_gaussian_log_probability(
    pre_tanh_value: jax.Array,
    mean: jax.Array,
    log_std: jax.Array,
) -> jax.Array:
    """Evaluate the Gaussian density after the tanh change of variables."""

    return gaussian_log_probability(
        pre_tanh_value,
        mean,
        log_std,
    ) - tanh_log_abs_det_jacobian(pre_tanh_value)


def sample_tanh_gaussian(
    key: jax.Array,
    statistics: ActorStatistics,
) -> ActorSample:
    """Draw a reparameterized normalized action from Actor statistics."""

    noise = jax.random.normal(key, statistics.mean.shape, dtype=statistics.mean.dtype)
    pre_tanh_value = statistics.mean + jnp.exp(statistics.log_std) * noise
    return ActorSample(
        action=jnp.tanh(pre_tanh_value),
        log_probability=tanh_gaussian_log_probability(
            pre_tanh_value,
            statistics.mean,
            statistics.log_std,
        ),
        pre_tanh_value=pre_tanh_value,
        mean=statistics.mean,
        log_std=statistics.log_std,
    )


class TanhGaussianActor(nn.Module):
    """LiDAR-only Actor whose output is normalized to ``[-1, 1]`` by tanh.

    ``mean`` is the unconstrained pre-tanh Gaussian mean.  The deterministic
    normalized action is ``tanh(mean)``; physical steering and acceleration
    scaling deliberately remain in the environment boundary.
    """

    action_dim: int = ACTION_DIM
    log_std_min: float = LOG_STD_MIN
    log_std_max: float = LOG_STD_MAX

    def setup(self) -> None:
        if self.action_dim != ACTION_DIM:
            raise ValueError("the racing Actor action_dim must be exactly 2")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")

        self.encoder = LidarEncoder(name="encoder")
        self.mean_head = nn.Dense(
            self.action_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            name="mean_head",
        )
        self.log_std_head = nn.Dense(
            self.action_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            name="log_std_head",
        )

    def __call__(self, observation: jax.Array) -> ActorStatistics:
        """Return bounded-variance pre-tanh Gaussian parameters."""

        features = self.encoder(observation)
        mean = self.mean_head(features)
        raw_log_std = self.log_std_head(features)
        log_std = jnp.clip(raw_log_std, self.log_std_min, self.log_std_max)
        return ActorStatistics(mean=mean, log_std=log_std)

    def sample(self, observation: jax.Array, key: jax.Array) -> ActorSample:
        """Sample with ``mean + std * noise`` so gradients reach the Actor."""

        return sample_tanh_gaussian(key, self(observation))

    def deterministic_action(self, observation: jax.Array) -> jax.Array:
        """Return the normalized mean action used for evaluation and export."""

        return jnp.tanh(self(observation).mean)


__all__ = [
    "ACTION_DIM",
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "ActorSample",
    "ActorStatistics",
    "TanhGaussianActor",
    "gaussian_log_probability",
    "sample_tanh_gaussian",
    "tanh_gaussian_log_probability",
    "tanh_log_abs_det_jacobian",
]

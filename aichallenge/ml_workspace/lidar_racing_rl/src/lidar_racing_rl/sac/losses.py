"""Pure reparameterized Soft Actor-Critic objectives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from lidar_racing_rl.models.actor_flax import ActorSample
from lidar_racing_rl.models.critic_flax import TwinQValues
from lidar_racing_rl.sac.replay import TransitionBatch


PyTree = Any
ActorSampleFunction = Callable[[PyTree, jax.Array, jax.Array], ActorSample]
CriticFunction = Callable[[PyTree, jax.Array, jax.Array], TwinQValues]


class CriticLossMetrics(NamedTuple):
    """Scalar diagnostics produced with the twin-Q regression loss."""

    target_q_mean: jax.Array
    q1_mean: jax.Array
    q2_mean: jax.Array
    absolute_td_error_mean: jax.Array


class ActorLossMetrics(NamedTuple):
    """Scalar diagnostics produced with the reparameterized Actor loss."""

    q_mean: jax.Array
    entropy: jax.Array
    log_probability_mean: jax.Array


class AlphaLossMetrics(NamedTuple):
    """Scalar diagnostics for automatic entropy-temperature tuning."""

    alpha: jax.Array
    entropy_error: jax.Array


def bootstrap_mask(terminated: jax.Array) -> jax.Array:
    """Return zero only for real MDP termination.

    ``truncated`` is intentionally absent: a time-limit transition keeps its
    target bootstrap as required by the environment contract.
    """

    return 1.0 - jnp.asarray(terminated, dtype=jnp.float32)


def soft_critic_target(
    *,
    actor_sample_fn: ActorSampleFunction,
    critic_fn: CriticFunction,
    actor_params: PyTree,
    target_critic_params: PyTree,
    log_alpha: jax.Array,
    batch: TransitionBatch,
    key: jax.Array,
    discount: float,
) -> jax.Array:
    """Build the entropy-regularized target from the target twin Critic."""

    next_sample = actor_sample_fn(actor_params, batch.next_observation, key)
    next_q = critic_fn(
        target_critic_params,
        batch.next_observation,
        next_sample.action,
    )
    next_soft_value = jnp.minimum(next_q.q1, next_q.q2) - jnp.exp(
        log_alpha
    ) * next_sample.log_probability
    target = jnp.asarray(batch.reward, dtype=jnp.float32) + (
        jnp.asarray(discount, dtype=jnp.float32)
        * bootstrap_mask(batch.terminated)
        * next_soft_value
    )
    return jax.lax.stop_gradient(target)


def critic_loss(
    *,
    actor_sample_fn: ActorSampleFunction,
    critic_fn: CriticFunction,
    critic_params: PyTree,
    target_critic_params: PyTree,
    actor_params: PyTree,
    log_alpha: jax.Array,
    batch: TransitionBatch,
    key: jax.Array,
    discount: float,
) -> tuple[jax.Array, CriticLossMetrics]:
    """Return the averaged independent twin-Q squared Bellman error."""

    target_q = soft_critic_target(
        actor_sample_fn=actor_sample_fn,
        critic_fn=critic_fn,
        actor_params=actor_params,
        target_critic_params=target_critic_params,
        log_alpha=log_alpha,
        batch=batch,
        key=key,
        discount=discount,
    )
    predicted_q = critic_fn(critic_params, batch.observation, batch.action)
    q1_error = predicted_q.q1 - target_q
    q2_error = predicted_q.q2 - target_q
    loss = 0.5 * jnp.mean(jnp.square(q1_error) + jnp.square(q2_error))
    metrics = CriticLossMetrics(
        target_q_mean=jnp.mean(target_q),
        q1_mean=jnp.mean(predicted_q.q1),
        q2_mean=jnp.mean(predicted_q.q2),
        absolute_td_error_mean=0.5
        * jnp.mean(jnp.abs(q1_error) + jnp.abs(q2_error)),
    )
    return loss, metrics


def actor_loss(
    *,
    actor_sample_fn: ActorSampleFunction,
    critic_fn: CriticFunction,
    actor_params: PyTree,
    critic_params: PyTree,
    log_alpha: jax.Array,
    observation: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, ActorLossMetrics]:
    """Return ``E[alpha * log pi(a|s) - min(Q1, Q2)]``.

    ``actor_sample_fn`` must use ``mean + std * noise``.  Therefore the Q term
    remains differentiable with respect to Actor parameters through action.
    """

    sample = actor_sample_fn(actor_params, observation, key)
    predicted_q = critic_fn(critic_params, observation, sample.action)
    minimum_q = jnp.minimum(predicted_q.q1, predicted_q.q2)
    loss = jnp.mean(jnp.exp(log_alpha) * sample.log_probability - minimum_q)
    metrics = ActorLossMetrics(
        q_mean=jnp.mean(minimum_q),
        entropy=-jnp.mean(sample.log_probability),
        log_probability_mean=jnp.mean(sample.log_probability),
    )
    return loss, metrics


def alpha_loss(
    log_alpha: jax.Array,
    log_probability: jax.Array,
    *,
    target_entropy: float,
) -> tuple[jax.Array, AlphaLossMetrics]:
    """Tune log-alpha toward the configured target policy entropy."""

    entropy_residual = jax.lax.stop_gradient(
        jnp.asarray(log_probability, dtype=jnp.float32)
        + jnp.asarray(target_entropy, dtype=jnp.float32)
    )
    loss = -jnp.mean(log_alpha * entropy_residual)
    return loss, AlphaLossMetrics(
        alpha=jnp.exp(log_alpha),
        entropy_error=-jnp.mean(entropy_residual),
    )


__all__ = [
    "ActorLossMetrics",
    "ActorSampleFunction",
    "AlphaLossMetrics",
    "CriticFunction",
    "CriticLossMetrics",
    "actor_loss",
    "alpha_loss",
    "bootstrap_mask",
    "critic_loss",
    "soft_critic_target",
]

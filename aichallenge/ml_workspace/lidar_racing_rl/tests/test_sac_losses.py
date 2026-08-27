"""Executable SAC objective tests for the dependency-enabled container."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lidar_racing_rl.models.actor_flax import ActorSample
from lidar_racing_rl.models.critic_flax import TwinQValues
from lidar_racing_rl.sac.losses import actor_loss, alpha_loss, soft_critic_target
from lidar_racing_rl.sac.replay import TransitionBatch
from lidar_racing_rl.sac.train_state import polyak_update


def _batch() -> TransitionBatch:
    observation = jnp.zeros((2, 8, 360), dtype=jnp.float32)
    return TransitionBatch(
        observation=observation,
        action=jnp.zeros((2, 2), dtype=jnp.float32),
        reward=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        terminated=jnp.asarray([True, False]),
        truncated=jnp.asarray([False, True]),
        next_observation=observation,
    )


def _constant_actor_sample(params, observation, key) -> ActorSample:
    del key
    batch_shape = observation.shape[:-2]
    mean = jnp.broadcast_to(params["mean"], (*batch_shape, 2))
    log_std = jnp.zeros_like(mean)
    action = jnp.tanh(mean)
    return ActorSample(
        action=action,
        log_probability=jnp.full(batch_shape, -2.0, dtype=jnp.float32),
        pre_tanh_value=mean,
        mean=mean,
        log_std=log_std,
    )


def _constant_critic(params, observation, action) -> TwinQValues:
    del action
    shape = observation.shape[:-2]
    return TwinQValues(
        q1=jnp.full(shape, params["q1"], dtype=jnp.float32),
        q2=jnp.full(shape, params["q2"], dtype=jnp.float32),
    )


def test_truncation_keeps_target_bootstrap_and_termination_removes_it() -> None:
    target = soft_critic_target(
        actor_sample_fn=_constant_actor_sample,
        critic_fn=_constant_critic,
        actor_params={"mean": jnp.zeros((2,), dtype=jnp.float32)},
        target_critic_params={"q1": 10.0, "q2": 12.0},
        log_alpha=jnp.asarray(0.0),
        batch=_batch(),
        key=jax.random.key(1),
        discount=0.5,
    )

    # min target Q is 10 and -alpha*log_pi adds 2: 1 + 0.5*12 = 7.
    assert jnp.allclose(target, jnp.asarray([1.0, 7.0], dtype=jnp.float32))


def test_actor_q_term_is_reparameterized_through_sampled_action() -> None:
    def action_sensitive_critic(params, observation, action) -> TwinQValues:
        del params, observation
        value = jnp.sum(action, axis=-1)
        return TwinQValues(q1=value, q2=value)

    def objective(mean: jax.Array) -> jax.Array:
        loss, _ = actor_loss(
            actor_sample_fn=_constant_actor_sample,
            critic_fn=action_sensitive_critic,
            actor_params={"mean": mean},
            critic_params={},
            log_alpha=jnp.asarray(0.0),
            observation=_batch().observation,
            key=jax.random.key(2),
        )
        return loss

    gradient = jax.grad(objective)(jnp.asarray([0.1, -0.2], dtype=jnp.float32))
    assert gradient.shape == (2,)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0.0)


def test_alpha_loss_uses_stopped_policy_entropy_residual() -> None:
    log_alpha = jnp.asarray(0.3, dtype=jnp.float32)
    log_probability = jnp.asarray([-1.0, -3.0], dtype=jnp.float32)
    loss, metrics = alpha_loss(log_alpha, log_probability, target_entropy=-2.0)
    gradient = jax.grad(
        lambda value: alpha_loss(
            value,
            log_probability,
            target_entropy=-2.0,
        )[0]
    )(log_alpha)

    assert jnp.isclose(loss, 1.2)
    assert jnp.isclose(gradient, 4.0)
    assert jnp.isclose(metrics.alpha, jnp.exp(log_alpha))


def test_polyak_update_uses_online_tau_weight() -> None:
    target = {"weight": jnp.asarray([0.0, 4.0], dtype=jnp.float32)}
    online = {"weight": jnp.asarray([10.0, 0.0], dtype=jnp.float32)}

    updated = polyak_update(target, online, 0.25)

    assert jnp.allclose(updated["weight"], jnp.asarray([2.5, 3.0]))

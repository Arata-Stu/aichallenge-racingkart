"""Executable one-update SAC test for the dependency-enabled container."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lidar_racing_rl.models.actor_flax import TanhGaussianActor
from lidar_racing_rl.models.critic_flax import TwinQCritic
from lidar_racing_rl.sac.learner import (
    SACLearnerConfig,
    initialize_sac_state,
    make_sac_update,
)
from lidar_racing_rl.sac.replay import TransitionBatch
from lidar_racing_rl.sac.train_state import create_sac_optimizers


def _trees_allclose(left: object, right: object) -> bool:
    return all(
        bool(jnp.allclose(left_leaf, right_leaf))
        for left_leaf, right_leaf in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
            strict=True,
        )
    )


def test_compiled_update_changes_online_and_polyak_target_state() -> None:
    actor = TanhGaussianActor()
    critic = TwinQCritic()
    optimizers = create_sac_optimizers()
    observation = jnp.zeros((2, 8, 360), dtype=jnp.float32)
    action = jnp.zeros((2, 2), dtype=jnp.float32)
    state = initialize_sac_state(
        key=jax.random.key(1),
        actor=actor,
        critic=critic,
        optimizers=optimizers,
        observation_example=observation,
        normalized_action_example=action,
    )
    batch = TransitionBatch(
        observation=observation,
        action=action,
        reward=jnp.ones((2,), dtype=jnp.float32),
        terminated=jnp.asarray([False, True]),
        truncated=jnp.asarray([True, False]),
        next_observation=observation,
    )
    update = jax.jit(
        make_sac_update(
            actor=actor,
            critic=critic,
            optimizers=optimizers,
        )
    )

    updated, metrics = update(state, batch, jax.random.key(2))

    assert int(updated.step) == 1
    assert bool(metrics.all_finite)
    assert bool(metrics.update_applied)
    old_target_bias = state.target_critic_params["q1"]["value_head"]["bias"]
    new_online_bias = updated.critic_params["q1"]["value_head"]["bias"]
    new_target_bias = updated.target_critic_params["q1"]["value_head"]["bias"]
    expected_target_bias = old_target_bias + 0.005 * (
        new_online_bias - old_target_bias
    )
    assert jnp.allclose(new_target_bias, expected_target_bias)


def test_non_finite_update_is_reported_and_rejected() -> None:
    actor = TanhGaussianActor()
    critic = TwinQCritic()
    optimizers = create_sac_optimizers()
    observation = jnp.zeros((1, 8, 360), dtype=jnp.float32)
    action = jnp.zeros((1, 2), dtype=jnp.float32)
    state = initialize_sac_state(
        key=jax.random.key(3),
        actor=actor,
        critic=critic,
        optimizers=optimizers,
        observation_example=observation,
        normalized_action_example=action,
    )
    batch = TransitionBatch(
        observation=observation,
        action=action,
        reward=jnp.asarray([jnp.nan], dtype=jnp.float32),
        terminated=jnp.asarray([False]),
        truncated=jnp.asarray([False]),
        next_observation=observation,
    )
    update = jax.jit(
        make_sac_update(
            actor=actor,
            critic=critic,
            optimizers=optimizers,
        )
    )

    rejected, metrics = update(state, batch, jax.random.key(4))

    assert int(rejected.step) == 0
    assert not bool(metrics.all_finite)
    assert not bool(metrics.update_applied)


def test_critic_only_transfer_phase_freezes_actor_and_temperature() -> None:
    actor = TanhGaussianActor()
    critic = TwinQCritic()
    optimizers = create_sac_optimizers()
    observation = jnp.zeros((2, 8, 360), dtype=jnp.float32)
    action = jnp.zeros((2, 2), dtype=jnp.float32)
    state = initialize_sac_state(
        key=jax.random.key(5),
        actor=actor,
        critic=critic,
        optimizers=optimizers,
        observation_example=observation,
        normalized_action_example=action,
    )
    batch = TransitionBatch(
        observation=observation,
        action=action,
        reward=jnp.ones((2,), dtype=jnp.float32),
        terminated=jnp.asarray([False, False]),
        truncated=jnp.asarray([False, False]),
        next_observation=observation,
    )
    update = jax.jit(
        make_sac_update(
            actor=actor,
            critic=critic,
            optimizers=optimizers,
            config=SACLearnerConfig(actor_update_start_step=10),
        )
    )

    updated, metrics = update(state, batch, jax.random.key(6))

    assert int(updated.step) == 1
    assert bool(metrics.update_applied)
    assert _trees_allclose(updated.actor_params, state.actor_params)
    assert _trees_allclose(updated.actor_opt_state, state.actor_opt_state)
    assert bool(jnp.allclose(updated.log_alpha, state.log_alpha))
    assert _trees_allclose(updated.alpha_opt_state, state.alpha_opt_state)
    assert not _trees_allclose(updated.critic_params, state.critic_params)

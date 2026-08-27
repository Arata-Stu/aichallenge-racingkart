"""Executable Flax model tests for the dependency-enabled training container."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lidar_racing_rl.models.actor_flax import (
    ActorStatistics,
    TanhGaussianActor,
    sample_tanh_gaussian,
)
from lidar_racing_rl.models.critic_flax import TwinQCritic
from lidar_racing_rl.models.encoder_flax import prepare_lidar_observation


def test_frame_and_channel_axes_fold_without_reordering_beams() -> None:
    canonical = jnp.arange(4 * 2 * 360, dtype=jnp.float32).reshape(4, 2, 360)
    folded = prepare_lidar_observation(canonical)

    assert folded.shape == (8, 360)
    assert jnp.array_equal(folded, canonical.reshape(8, 360))
    assert jnp.array_equal(prepare_lidar_observation(folded), folded)


def test_actor_shapes_bounds_and_stable_parameter_tree() -> None:
    actor = TanhGaussianActor()
    observation = jnp.zeros((3, 4, 2, 360), dtype=jnp.float32)
    variables = actor.init(jax.random.key(1), observation)
    statistics = actor.apply(variables, observation)
    sample = actor.apply(
        variables,
        observation,
        jax.random.key(2),
        method=actor.sample,
    )
    deterministic = actor.apply(
        variables,
        observation,
        method=actor.deterministic_action,
    )

    assert statistics.mean.shape == (3, 2)
    assert statistics.log_std.shape == (3, 2)
    assert jnp.all(statistics.log_std >= -5.0)
    assert jnp.all(statistics.log_std <= 2.0)
    assert sample.action.shape == (3, 2)
    assert sample.log_probability.shape == (3,)
    assert deterministic.shape == (3, 2)
    assert jnp.all(jnp.abs(sample.action) <= 1.0)
    assert jnp.all(jnp.abs(deterministic) <= 1.0)

    params = variables["params"]
    assert params["encoder"]["conv_0"]["kernel"].shape == (8, 8, 32)
    assert params["encoder"]["conv_1"]["kernel"].shape == (4, 32, 64)
    assert params["encoder"]["conv_2"]["kernel"].shape == (3, 64, 64)
    assert params["encoder"]["dense"]["kernel"].shape == (2624, 256)
    assert params["mean_head"]["kernel"].shape == (256, 2)
    assert params["log_std_head"]["kernel"].shape == (256, 2)


def test_tanh_log_probability_is_finite_for_saturated_actions() -> None:
    statistics = ActorStatistics(
        mean=jnp.asarray([[50.0, -50.0]], dtype=jnp.float32),
        log_std=jnp.asarray([[-5.0, -5.0]], dtype=jnp.float32),
    )
    sample = sample_tanh_gaussian(jax.random.key(3), statistics)

    assert jnp.all(jnp.isfinite(sample.log_probability))
    assert jnp.all(jnp.isfinite(sample.action))


def test_twin_critic_has_independent_encoder_trees_and_scalar_outputs() -> None:
    critic = TwinQCritic()
    observation = jnp.zeros((2, 8, 360), dtype=jnp.float32)
    action = jnp.zeros((2, 2), dtype=jnp.float32)
    variables = critic.init(jax.random.key(4), observation, action)
    values = critic.apply(variables, observation, action)

    assert values.q1.shape == (2,)
    assert values.q2.shape == (2,)
    params = variables["params"]
    assert set(params) == {"q1", "q2"}
    for name in ("q1", "q2"):
        assert params[name]["encoder"]["conv_0"]["kernel"].shape == (8, 8, 32)
        assert params[name]["encoder"]["dense"]["kernel"].shape == (2624, 256)
        assert params[name]["hidden_0"]["kernel"].shape == (258, 256)
        assert params[name]["hidden_1"]["kernel"].shape == (256, 256)
        assert params[name]["value_head"]["kernel"].shape == (256, 1)
    assert not jnp.array_equal(
        params["q1"]["encoder"]["conv_0"]["kernel"],
        params["q2"]["encoder"]["conv_0"]["kernel"],
    )


def test_deterministic_actor_apply_can_be_jitted() -> None:
    actor = TanhGaussianActor()
    observation = jnp.zeros((1, 8, 360), dtype=jnp.float32)
    variables = actor.init(jax.random.key(5), observation)

    apply_actor = jax.jit(
        lambda lidar: actor.apply(
            variables,
            lidar,
            method=actor.deterministic_action,
        )
    )
    assert apply_actor(observation).shape == (1, 2)


def test_deterministic_actor_apply_can_be_vmapped() -> None:
    actor = TanhGaussianActor()
    observation = jnp.zeros((3, 8, 360), dtype=jnp.float32)
    variables = actor.init(jax.random.key(6), observation[0])

    def apply_one(lidar):
        return actor.apply(
            variables,
            lidar,
            method=actor.deterministic_action,
        )

    assert jax.vmap(apply_one)(observation).shape == (3, 2)

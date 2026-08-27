"""Flax/PyTorch Actor conversion contract tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lidar_racing_rl.models.actor_flax import TanhGaussianActor
from lidar_racing_rl.models.parameter_conversion import (
    build_torch_actor_from_flax,
    deterministic_parity_error,
    flax_actor_to_torch_state_dict,
)


def test_flax_to_torch_state_dict_names_and_shapes() -> None:
    actor = TanhGaussianActor()
    observation = jnp.zeros((2, 8, 360), dtype=jnp.float32)
    variables = actor.init(jax.random.key(0), observation)
    state_dict = flax_actor_to_torch_state_dict(variables)

    assert tuple(state_dict) == (
        "encoder.0.weight",
        "encoder.0.bias",
        "encoder.2.weight",
        "encoder.2.bias",
        "encoder.4.weight",
        "encoder.4.bias",
        "trunk.1.weight",
        "trunk.1.bias",
        "mean_head.weight",
        "mean_head.bias",
        "log_std_head.weight",
        "log_std_head.bias",
    )
    assert tuple(state_dict["encoder.0.weight"].shape) == (32, 8, 8)
    assert tuple(state_dict["trunk.1.weight"].shape) == (256, 2624)


def test_deterministic_flax_torch_parity_is_within_blueprint_tolerance() -> None:
    actor = TanhGaussianActor()
    observation = jax.random.uniform(jax.random.key(1), (4, 8, 360))
    variables = actor.init(jax.random.key(2), observation)
    torch_actor = build_torch_actor_from_flax(variables)

    error = deterministic_parity_error(actor, variables, torch_actor, observation)
    assert error <= 1.0e-5

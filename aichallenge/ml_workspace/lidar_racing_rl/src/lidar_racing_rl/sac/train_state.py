"""Immutable optimizer and target-network state for Flax SAC."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct


PyTree = Any


@dataclass(frozen=True)
class SACOptimizers:
    """Static Optax transformations captured outside the jitted update."""

    actor: optax.GradientTransformation
    critic: optax.GradientTransformation
    alpha: optax.GradientTransformation


@struct.dataclass
class SACTrainState:
    """All dynamic SAC learner state as a JAX pytree.

    The models use no mutable Flax collections, so only each ``params``
    collection is stored.  The target Critic has the same pytree structure as
    ``critic_params`` and is updated only through Polyak averaging.
    """

    step: jax.Array
    actor_params: PyTree
    critic_params: PyTree
    target_critic_params: PyTree
    log_alpha: jax.Array
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    alpha_opt_state: optax.OptState

    @property
    def alpha(self) -> jax.Array:
        """Return the positive entropy temperature."""

        return jnp.exp(self.log_alpha)


def _validate_positive_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")


def create_sac_optimizers(
    *,
    actor_learning_rate: float = 3.0e-4,
    critic_learning_rate: float = 3.0e-4,
    alpha_learning_rate: float = 3.0e-4,
    gradient_clip_norm: float = 10.0,
) -> SACOptimizers:
    """Create Adam optimizers with global-norm clipping from ``sac.yaml``."""

    _validate_positive_float("actor_learning_rate", actor_learning_rate)
    _validate_positive_float("critic_learning_rate", critic_learning_rate)
    _validate_positive_float("alpha_learning_rate", alpha_learning_rate)
    _validate_positive_float("gradient_clip_norm", gradient_clip_norm)

    def clipped_adam(learning_rate: float) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(gradient_clip_norm),
            optax.adam(learning_rate),
        )

    return SACOptimizers(
        actor=clipped_adam(actor_learning_rate),
        critic=clipped_adam(critic_learning_rate),
        alpha=clipped_adam(alpha_learning_rate),
    )


def create_sac_train_state(
    *,
    actor_params: PyTree,
    critic_params: PyTree,
    optimizers: SACOptimizers,
    initial_alpha: float = 1.0,
) -> SACTrainState:
    """Initialize optimizer state and an exact target-Critic parameter copy.

    ``actor_params`` and ``critic_params`` are the values under the Flax
    ``variables["params"]`` key, not full variables dictionaries.
    """

    _validate_positive_float("initial_alpha", initial_alpha)
    target_critic_params = jax.tree_util.tree_map(lambda value: value, critic_params)
    log_alpha = jnp.asarray(math.log(initial_alpha), dtype=jnp.float32)
    return SACTrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        actor_params=actor_params,
        critic_params=critic_params,
        target_critic_params=target_critic_params,
        log_alpha=log_alpha,
        actor_opt_state=optimizers.actor.init(actor_params),
        critic_opt_state=optimizers.critic.init(critic_params),
        alpha_opt_state=optimizers.alpha.init(log_alpha),
    )


def polyak_update(
    target_params: PyTree,
    online_params: PyTree,
    tau: float,
) -> PyTree:
    """Return ``(1 - tau) * target + tau * online`` for every parameter."""

    if isinstance(tau, bool) or not math.isfinite(tau) or not 0.0 < tau <= 1.0:
        raise ValueError("tau must be finite and in (0, 1]")

    def update_leaf(target: jax.Array, online: jax.Array) -> jax.Array:
        coefficient = jnp.asarray(tau, dtype=online.dtype)
        return target + coefficient * (online - target)

    return jax.tree_util.tree_map(update_leaf, target_params, online_params)


__all__ = [
    "PyTree",
    "SACOptimizers",
    "SACTrainState",
    "create_sac_optimizers",
    "create_sac_train_state",
    "polyak_update",
]

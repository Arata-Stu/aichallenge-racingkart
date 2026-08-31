"""JIT-ready Flax/Optax orchestration for one Soft Actor-Critic update."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct

from lidar_racing_rl.models.actor_flax import TanhGaussianActor
from lidar_racing_rl.models.critic_flax import TwinQCritic
from lidar_racing_rl.sac.losses import actor_loss, alpha_loss, critic_loss
from lidar_racing_rl.sac.replay import TransitionBatch
from lidar_racing_rl.sac.train_state import (
    SACOptimizers,
    SACTrainState,
    create_sac_train_state,
    polyak_update,
)


@dataclass(frozen=True)
class SACLearnerConfig:
    """Static update settings captured by :func:`make_sac_update`."""

    discount: float = 0.99
    target_smoothing_coefficient: float = 0.005
    target_entropy: float = -2.0
    actor_update_start_step: int = 0
    detect_non_finite: bool = True

    def validate(self) -> None:
        if (
            isinstance(self.discount, bool)
            or not math.isfinite(self.discount)
            or not 0.0 <= self.discount <= 1.0
        ):
            raise ValueError("discount must be finite and in [0, 1]")
        tau = self.target_smoothing_coefficient
        if isinstance(tau, bool) or not math.isfinite(tau) or not 0.0 < tau <= 1.0:
            raise ValueError("target_smoothing_coefficient must be finite and in (0, 1]")
        if isinstance(self.target_entropy, bool) or not math.isfinite(
            self.target_entropy
        ):
            raise ValueError("target_entropy must be finite")
        if (
            isinstance(self.actor_update_start_step, bool)
            or not isinstance(self.actor_update_start_step, int)
            or self.actor_update_start_step < 0
        ):
            raise ValueError("actor_update_start_step must be a non-negative integer")
        if not isinstance(self.detect_non_finite, bool):
            raise ValueError("detect_non_finite must be boolean")


@struct.dataclass
class SACUpdateMetrics:
    """Fixed-structure scalar diagnostics returned by a compiled update."""

    critic_loss: jax.Array
    actor_loss: jax.Array
    alpha_loss: jax.Array
    alpha: jax.Array
    entropy: jax.Array
    target_q_mean: jax.Array
    q1_mean: jax.Array
    q2_mean: jax.Array
    absolute_td_error_mean: jax.Array
    all_finite: jax.Array
    update_applied: jax.Array


SACUpdateFunction = Callable[
    [SACTrainState, TransitionBatch, jax.Array],
    tuple[SACTrainState, SACUpdateMetrics],
]


def initialize_sac_state(
    *,
    key: jax.Array,
    actor: TanhGaussianActor,
    critic: TwinQCritic,
    optimizers: SACOptimizers,
    observation_example: jax.Array,
    normalized_action_example: jax.Array,
    initial_alpha: float = 1.0,
) -> SACTrainState:
    """Initialize Flax ``params`` and optimizer/target state.

    The returned state intentionally contains parameter collections rather
    than full Flax variables dictionaries.  These models have no BatchNorm,
    cache, or other mutable collection; adding one is therefore an explicit
    learner-state API change rather than silent state loss.
    """

    actor_key, critic_key = jax.random.split(key)
    actor_variables = actor.init(actor_key, observation_example)
    critic_variables = critic.init(
        critic_key,
        observation_example,
        normalized_action_example,
    )
    if set(actor_variables) != {"params"} or set(critic_variables) != {"params"}:
        raise ValueError("SAC models must expose only the immutable params collection")
    return create_sac_train_state(
        actor_params=actor_variables["params"],
        critic_params=critic_variables["params"],
        optimizers=optimizers,
        initial_alpha=initial_alpha,
    )


def _tree_all_finite(*trees: Any) -> jax.Array:
    checks = [
        jnp.all(jnp.isfinite(jnp.asarray(leaf)))
        for tree in trees
        for leaf in jax.tree_util.tree_leaves(tree)
    ]
    if not checks:
        return jnp.asarray(True)
    return jnp.all(jnp.stack(checks))


def make_sac_update(
    *,
    actor: TanhGaussianActor,
    critic: TwinQCritic,
    optimizers: SACOptimizers,
    config: SACLearnerConfig | None = None,
) -> SACUpdateFunction:
    """Return a pure ``(state, batch, key) -> (state, metrics)`` update.

    Models, optimizer transformations, and scalar settings are captured as
    static Python objects.  The returned function can therefore be wrapped as
    ``jax.jit(make_sac_update(...))`` without marking them static manually.
    """

    if config is None:
        config = SACLearnerConfig()
    config.validate()

    def actor_sample_fn(params: Any, observation: jax.Array, key: jax.Array):
        return actor.apply(
            {"params": params},
            observation,
            key,
            method=actor.sample,
        )

    def critic_fn(
        params: Any,
        observation: jax.Array,
        normalized_action: jax.Array,
    ):
        return critic.apply(
            {"params": params},
            observation,
            normalized_action,
        )

    def update(
        state: SACTrainState,
        batch: TransitionBatch,
        key: jax.Array,
    ) -> tuple[SACTrainState, SACUpdateMetrics]:
        critic_key, actor_key = jax.random.split(key)

        def critic_objective(critic_params: Any):
            return critic_loss(
                actor_sample_fn=actor_sample_fn,
                critic_fn=critic_fn,
                critic_params=critic_params,
                target_critic_params=state.target_critic_params,
                actor_params=state.actor_params,
                log_alpha=state.log_alpha,
                batch=batch,
                key=critic_key,
                discount=config.discount,
            )

        (critic_loss_value, critic_metrics), critic_gradients = jax.value_and_grad(
            critic_objective,
            has_aux=True,
        )(state.critic_params)
        critic_updates, critic_opt_state = optimizers.critic.update(
            critic_gradients,
            state.critic_opt_state,
            state.critic_params,
        )
        proposed_critic_params = optax.apply_updates(
            state.critic_params,
            critic_updates,
        )

        def actor_objective(actor_params: Any):
            return actor_loss(
                actor_sample_fn=actor_sample_fn,
                critic_fn=critic_fn,
                actor_params=actor_params,
                critic_params=proposed_critic_params,
                log_alpha=state.log_alpha,
                observation=batch.observation,
                key=actor_key,
            )

        (actor_loss_value, actor_metrics), actor_gradients = jax.value_and_grad(
            actor_objective,
            has_aux=True,
        )(state.actor_params)
        actor_updates, actor_opt_state = optimizers.actor.update(
            actor_gradients,
            state.actor_opt_state,
            state.actor_params,
        )
        proposed_actor_params = optax.apply_updates(state.actor_params, actor_updates)

        def alpha_objective(log_alpha: jax.Array):
            return alpha_loss(
                log_alpha,
                actor_metrics.log_probability_mean,
                target_entropy=config.target_entropy,
            )

        (alpha_loss_value, alpha_metrics), alpha_gradient = jax.value_and_grad(
            alpha_objective,
            has_aux=True,
        )(state.log_alpha)
        alpha_updates, alpha_opt_state = optimizers.alpha.update(
            alpha_gradient,
            state.alpha_opt_state,
            state.log_alpha,
        )
        proposed_log_alpha = optax.apply_updates(state.log_alpha, alpha_updates)

        actor_updates_enabled = (
            state.step
            >= jnp.asarray(config.actor_update_start_step, dtype=state.step.dtype)
        )

        def select_actor_update(proposed: Any, frozen: Any) -> Any:
            return jax.tree_util.tree_map(
                lambda proposed_leaf, frozen_leaf: jnp.where(
                    actor_updates_enabled,
                    proposed_leaf,
                    frozen_leaf,
                ),
                proposed,
                frozen,
            )

        selected_actor_params = select_actor_update(
            proposed_actor_params,
            state.actor_params,
        )
        selected_actor_opt_state = select_actor_update(
            actor_opt_state,
            state.actor_opt_state,
        )
        selected_log_alpha = jnp.where(
            actor_updates_enabled,
            proposed_log_alpha,
            state.log_alpha,
        )
        selected_alpha_opt_state = select_actor_update(
            alpha_opt_state,
            state.alpha_opt_state,
        )

        proposed_state = SACTrainState(
            step=state.step + jnp.asarray(1, dtype=state.step.dtype),
            actor_params=selected_actor_params,
            critic_params=proposed_critic_params,
            target_critic_params=polyak_update(
                state.target_critic_params,
                proposed_critic_params,
                config.target_smoothing_coefficient,
            ),
            log_alpha=selected_log_alpha,
            actor_opt_state=selected_actor_opt_state,
            critic_opt_state=critic_opt_state,
            alpha_opt_state=selected_alpha_opt_state,
        )
        all_finite = _tree_all_finite(
            critic_loss_value,
            actor_loss_value,
            alpha_loss_value,
            critic_gradients,
            actor_gradients,
            alpha_gradient,
            critic_metrics,
            actor_metrics,
            alpha_metrics,
            proposed_state,
        )
        update_applied = (
            all_finite if config.detect_non_finite else jnp.asarray(True)
        )
        selected_state = jax.lax.cond(
            update_applied,
            lambda _: proposed_state,
            lambda _: state,
            operand=None,
        )
        metrics = SACUpdateMetrics(
            critic_loss=critic_loss_value,
            actor_loss=actor_loss_value,
            alpha_loss=alpha_loss_value,
            alpha=selected_state.alpha,
            entropy=actor_metrics.entropy,
            target_q_mean=critic_metrics.target_q_mean,
            q1_mean=critic_metrics.q1_mean,
            q2_mean=critic_metrics.q2_mean,
            absolute_td_error_mean=critic_metrics.absolute_td_error_mean,
            all_finite=all_finite,
            update_applied=update_applied,
        )
        return selected_state, metrics

    return update


__all__ = [
    "SACLearnerConfig",
    "SACUpdateFunction",
    "SACUpdateMetrics",
    "initialize_sac_state",
    "make_sac_update",
]

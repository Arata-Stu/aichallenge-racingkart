"""JAX episode aggregation for LiDAR racing evaluation.

All opponent-related inputs are ground-truth diagnostics used only for
evaluation. They never become Actor/Critic observations or replay fields.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class EvaluationAccumulator:
    """Per-environment partial episodes and completed-episode totals."""

    episode_return: jax.Array
    progress: jax.Array
    speed_sum: jax.Array
    episode_steps: jax.Array
    previous_action: jax.Array
    steering_variation: jax.Array
    acceleration_variation: jax.Array
    collision_seen: jax.Array
    off_track_seen: jax.Array
    relative_progress: jax.Array
    pass_count: jax.Array
    first_pass_step: jax.Array
    opponent_collision_seen: jax.Array
    wall_collision_seen: jax.Array
    unsafe_contact_seen: jax.Array
    unsafe_contact_steps: jax.Array
    following_steps: jax.Array
    stalled_steps: jax.Array
    safe_wait_steps: jax.Array
    wait_opportunity_steps: jax.Array
    minimum_opponent_distance: jax.Array
    opponent_present_seen: jax.Array
    post_pass_recontact_seen: jax.Array
    completed_episodes: jax.Array
    completed_races: jax.Array
    collision_episodes: jax.Array
    off_track_episodes: jax.Array
    truncated_episodes: jax.Array
    return_sum: jax.Array
    progress_sum: jax.Array
    speed_sum_completed: jax.Array
    step_sum_completed: jax.Array
    race_step_sum: jax.Array
    steering_variation_sum: jax.Array
    acceleration_variation_sum: jax.Array
    completed_opponent_episodes: jax.Array
    overtake_success_episodes: jax.Array
    pass_count_sum: jax.Array
    first_pass_step_sum: jax.Array
    relative_progress_sum: jax.Array
    opponent_collision_episodes: jax.Array
    wall_collision_episodes: jax.Array
    unsafe_contact_episodes: jax.Array
    unsafe_contact_step_sum: jax.Array
    following_step_sum: jax.Array
    stalled_step_sum: jax.Array
    safe_wait_step_sum: jax.Array
    wait_opportunity_step_sum: jax.Array
    minimum_opponent_distance_sum: jax.Array
    final_rank_sum: jax.Array
    post_pass_recontact_episodes: jax.Array


def initialize_evaluation_accumulator(
    num_envs: int,
    action_dim: int = 2,
) -> EvaluationAccumulator:
    """Create zeroed metrics for a fixed vector-environment width."""

    if isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("num_envs must be positive")
    if action_dim != 2:
        raise ValueError("LiDAR racing evaluation requires two-dimensional actions")
    zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
    zero_ints = jnp.zeros((num_envs,), dtype=jnp.int32)
    false = jnp.zeros((num_envs,), dtype=jnp.bool_)
    scalar_int = jnp.asarray(0, dtype=jnp.int32)
    scalar_float = jnp.asarray(0.0, dtype=jnp.float32)
    return EvaluationAccumulator(
        episode_return=zeros,
        progress=zeros,
        speed_sum=zeros,
        episode_steps=zero_ints,
        previous_action=jnp.zeros((num_envs, action_dim), dtype=jnp.float32),
        steering_variation=zeros,
        acceleration_variation=zeros,
        collision_seen=false,
        off_track_seen=false,
        relative_progress=zeros,
        pass_count=zero_ints,
        first_pass_step=zero_ints,
        opponent_collision_seen=false,
        wall_collision_seen=false,
        unsafe_contact_seen=false,
        unsafe_contact_steps=zero_ints,
        following_steps=zero_ints,
        stalled_steps=zero_ints,
        safe_wait_steps=zero_ints,
        wait_opportunity_steps=zero_ints,
        minimum_opponent_distance=jnp.full(
            (num_envs,),
            jnp.inf,
            dtype=jnp.float32,
        ),
        opponent_present_seen=false,
        post_pass_recontact_seen=false,
        completed_episodes=scalar_int,
        completed_races=scalar_int,
        collision_episodes=scalar_int,
        off_track_episodes=scalar_int,
        truncated_episodes=scalar_int,
        return_sum=scalar_float,
        progress_sum=scalar_float,
        speed_sum_completed=scalar_float,
        step_sum_completed=scalar_int,
        race_step_sum=scalar_int,
        steering_variation_sum=scalar_float,
        acceleration_variation_sum=scalar_float,
        completed_opponent_episodes=scalar_int,
        overtake_success_episodes=scalar_int,
        pass_count_sum=scalar_int,
        first_pass_step_sum=scalar_int,
        relative_progress_sum=scalar_float,
        opponent_collision_episodes=scalar_int,
        wall_collision_episodes=scalar_int,
        unsafe_contact_episodes=scalar_int,
        unsafe_contact_step_sum=scalar_int,
        following_step_sum=scalar_int,
        stalled_step_sum=scalar_int,
        safe_wait_step_sum=scalar_int,
        wait_opportunity_step_sum=scalar_int,
        minimum_opponent_distance_sum=scalar_float,
        final_rank_sum=scalar_int,
        post_pass_recontact_episodes=scalar_int,
    )


def select_episode_completions(
    done: jax.Array,
    remaining_episodes: jax.Array,
) -> jax.Array:
    """Select at most ``remaining_episodes`` terminal environments.

    Selection is deterministic in environment-index order and preserves the
    input ``[num_envs]`` shape.  This lets a final vector step finish exactly
    the requested number of evaluation episodes without a Python loop over
    environments.
    """

    done_array = jnp.asarray(done, dtype=jnp.bool_)
    remaining = jnp.asarray(remaining_episodes, dtype=jnp.int32)
    if done_array.ndim != 1:
        raise ValueError("done must have shape [num_envs]")
    if remaining.shape != ():
        raise ValueError("remaining_episodes must be scalar")
    completion_order = jnp.cumsum(done_array.astype(jnp.int32))
    return done_array & (completion_order <= jnp.maximum(remaining, 0))


def update_evaluation_accumulator(
    state: EvaluationAccumulator,
    *,
    reward: jax.Array,
    progress_delta: jax.Array,
    speed: jax.Array,
    normalized_action: jax.Array,
    collision: jax.Array,
    off_track: jax.Array,
    race_complete: jax.Array,
    relative_progress: jax.Array,
    pass_count: jax.Array,
    collision_with_opponent: jax.Array,
    collision_with_wall: jax.Array,
    unsafe_contact: jax.Array,
    following_vehicle: jax.Array,
    stalled_behind_vehicle: jax.Array,
    nearest_opponent_distance: jax.Array,
    ego_rank: jax.Array,
    opponent_present: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
) -> EvaluationAccumulator:
    """Accumulate a vector step and finalize each Ego episode independently."""

    num_envs = state.episode_return.shape[0]
    vector_shape = (num_envs,)
    if normalized_action.shape != (num_envs, 2):
        raise ValueError("normalized_action must have shape [num_envs, 2]")
    for name, value in (
        ("reward", reward),
        ("progress_delta", progress_delta),
        ("speed", speed),
        ("collision", collision),
        ("off_track", off_track),
        ("race_complete", race_complete),
        ("relative_progress", relative_progress),
        ("pass_count", pass_count),
        ("collision_with_opponent", collision_with_opponent),
        ("collision_with_wall", collision_with_wall),
        ("unsafe_contact", unsafe_contact),
        ("following_vehicle", following_vehicle),
        ("stalled_behind_vehicle", stalled_behind_vehicle),
        ("nearest_opponent_distance", nearest_opponent_distance),
        ("ego_rank", ego_rank),
        ("opponent_present", opponent_present),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        if value.shape != vector_shape:
            raise ValueError(f"{name} must have shape [num_envs]")

    next_return = state.episode_return + reward
    next_progress = state.progress + progress_delta
    next_speed_sum = state.speed_sum + speed
    next_steps = state.episode_steps + 1
    action_delta = jnp.abs(normalized_action - state.previous_action)
    next_steering_variation = state.steering_variation + action_delta[:, 0]
    next_acceleration_variation = state.acceleration_variation + action_delta[:, 1]
    next_collision_seen = state.collision_seen | collision
    next_off_track_seen = state.off_track_seen | off_track
    next_relative_progress = state.relative_progress + relative_progress
    next_pass_count = state.pass_count + pass_count.astype(jnp.int32)
    first_pass_now = (state.first_pass_step == 0) & (pass_count > 0)
    next_first_pass_step = jnp.where(
        first_pass_now,
        next_steps,
        state.first_pass_step,
    )
    next_opponent_collision_seen = (
        state.opponent_collision_seen | collision_with_opponent
    )
    next_wall_collision_seen = state.wall_collision_seen | collision_with_wall
    next_unsafe_contact_seen = state.unsafe_contact_seen | unsafe_contact
    next_unsafe_contact_steps = (
        state.unsafe_contact_steps + unsafe_contact.astype(jnp.int32)
    )
    next_following_steps = (
        state.following_steps + following_vehicle.astype(jnp.int32)
    )
    next_stalled_steps = (
        state.stalled_steps + stalled_behind_vehicle.astype(jnp.int32)
    )
    wait_opportunity = following_vehicle | stalled_behind_vehicle | unsafe_contact
    safe_wait = following_vehicle & ~stalled_behind_vehicle & ~unsafe_contact
    next_safe_wait_steps = state.safe_wait_steps + safe_wait.astype(jnp.int32)
    next_wait_opportunity_steps = (
        state.wait_opportunity_steps + wait_opportunity.astype(jnp.int32)
    )
    finite_opponent_distance = opponent_present & jnp.isfinite(
        nearest_opponent_distance
    )
    next_minimum_opponent_distance = jnp.where(
        finite_opponent_distance,
        jnp.minimum(state.minimum_opponent_distance, nearest_opponent_distance),
        state.minimum_opponent_distance,
    )
    next_opponent_present_seen = state.opponent_present_seen | opponent_present
    next_post_pass_recontact_seen = state.post_pass_recontact_seen | (
        (state.pass_count > 0) & collision_with_opponent
    )
    done = terminated | truncated
    done_float = done.astype(jnp.float32)
    done_int = done.astype(jnp.int32)
    opponent_done = done & next_opponent_present_seen
    overtake_success = opponent_done & (next_pass_count > 0)
    completed_minimum_distance = jnp.where(
        opponent_done & jnp.isfinite(next_minimum_opponent_distance),
        next_minimum_opponent_distance,
        0.0,
    )

    return EvaluationAccumulator(
        episode_return=jnp.where(done, 0.0, next_return),
        progress=jnp.where(done, 0.0, next_progress),
        speed_sum=jnp.where(done, 0.0, next_speed_sum),
        episode_steps=jnp.where(done, 0, next_steps),
        previous_action=jnp.where(done[:, None], 0.0, normalized_action),
        steering_variation=jnp.where(done, 0.0, next_steering_variation),
        acceleration_variation=jnp.where(done, 0.0, next_acceleration_variation),
        collision_seen=jnp.where(done, False, next_collision_seen),
        off_track_seen=jnp.where(done, False, next_off_track_seen),
        relative_progress=jnp.where(done, 0.0, next_relative_progress),
        pass_count=jnp.where(done, 0, next_pass_count),
        first_pass_step=jnp.where(done, 0, next_first_pass_step),
        opponent_collision_seen=jnp.where(
            done,
            False,
            next_opponent_collision_seen,
        ),
        wall_collision_seen=jnp.where(done, False, next_wall_collision_seen),
        unsafe_contact_seen=jnp.where(done, False, next_unsafe_contact_seen),
        unsafe_contact_steps=jnp.where(done, 0, next_unsafe_contact_steps),
        following_steps=jnp.where(done, 0, next_following_steps),
        stalled_steps=jnp.where(done, 0, next_stalled_steps),
        safe_wait_steps=jnp.where(done, 0, next_safe_wait_steps),
        wait_opportunity_steps=jnp.where(done, 0, next_wait_opportunity_steps),
        minimum_opponent_distance=jnp.where(
            done,
            jnp.inf,
            next_minimum_opponent_distance,
        ),
        opponent_present_seen=jnp.where(done, False, next_opponent_present_seen),
        post_pass_recontact_seen=jnp.where(
            done,
            False,
            next_post_pass_recontact_seen,
        ),
        completed_episodes=state.completed_episodes + jnp.sum(done_int),
        completed_races=state.completed_races + jnp.sum(done & race_complete),
        collision_episodes=(
            state.collision_episodes + jnp.sum(done & next_collision_seen)
        ),
        off_track_episodes=(
            state.off_track_episodes + jnp.sum(done & next_off_track_seen)
        ),
        truncated_episodes=state.truncated_episodes + jnp.sum(done & truncated),
        return_sum=state.return_sum + jnp.sum(done_float * next_return),
        progress_sum=state.progress_sum + jnp.sum(done_float * next_progress),
        speed_sum_completed=(
            state.speed_sum_completed + jnp.sum(done_float * next_speed_sum)
        ),
        step_sum_completed=(
            state.step_sum_completed + jnp.sum(jnp.where(done, next_steps, 0))
        ),
        race_step_sum=(
            state.race_step_sum
            + jnp.sum(jnp.where(done & race_complete, next_steps, 0))
        ),
        steering_variation_sum=(
            state.steering_variation_sum
            + jnp.sum(done_float * next_steering_variation)
        ),
        acceleration_variation_sum=(
            state.acceleration_variation_sum
            + jnp.sum(done_float * next_acceleration_variation)
        ),
        completed_opponent_episodes=(
            state.completed_opponent_episodes + jnp.sum(opponent_done)
        ),
        overtake_success_episodes=(
            state.overtake_success_episodes + jnp.sum(overtake_success)
        ),
        pass_count_sum=(
            state.pass_count_sum
            + jnp.sum(jnp.where(opponent_done, next_pass_count, 0))
        ),
        first_pass_step_sum=(
            state.first_pass_step_sum
            + jnp.sum(jnp.where(overtake_success, next_first_pass_step, 0))
        ),
        relative_progress_sum=(
            state.relative_progress_sum
            + jnp.sum(jnp.where(opponent_done, next_relative_progress, 0.0))
        ),
        opponent_collision_episodes=(
            state.opponent_collision_episodes
            + jnp.sum(opponent_done & next_opponent_collision_seen)
        ),
        wall_collision_episodes=(
            state.wall_collision_episodes
            + jnp.sum(opponent_done & next_wall_collision_seen)
        ),
        unsafe_contact_episodes=(
            state.unsafe_contact_episodes
            + jnp.sum(opponent_done & next_unsafe_contact_seen)
        ),
        unsafe_contact_step_sum=(
            state.unsafe_contact_step_sum
            + jnp.sum(jnp.where(opponent_done, next_unsafe_contact_steps, 0))
        ),
        following_step_sum=(
            state.following_step_sum
            + jnp.sum(jnp.where(opponent_done, next_following_steps, 0))
        ),
        stalled_step_sum=(
            state.stalled_step_sum
            + jnp.sum(jnp.where(opponent_done, next_stalled_steps, 0))
        ),
        safe_wait_step_sum=(
            state.safe_wait_step_sum
            + jnp.sum(jnp.where(opponent_done, next_safe_wait_steps, 0))
        ),
        wait_opportunity_step_sum=(
            state.wait_opportunity_step_sum
            + jnp.sum(jnp.where(opponent_done, next_wait_opportunity_steps, 0))
        ),
        minimum_opponent_distance_sum=(
            state.minimum_opponent_distance_sum
            + jnp.sum(completed_minimum_distance)
        ),
        final_rank_sum=(
            state.final_rank_sum + jnp.sum(jnp.where(opponent_done, ego_rank, 0))
        ),
        post_pass_recontact_episodes=(
            state.post_pass_recontact_episodes
            + jnp.sum(overtake_success & next_post_pass_recontact_seen)
        ),
    )


def evaluation_summary(state: EvaluationAccumulator) -> dict[str, jax.Array]:
    """Return finite scalar means/rates over completed episodes only."""

    episodes = jnp.maximum(state.completed_episodes, 1)
    steps = jnp.maximum(state.step_sum_completed, 1)
    races = jnp.maximum(state.completed_races, 1)
    opponent_episodes = jnp.maximum(state.completed_opponent_episodes, 1)
    successful_overtakes = jnp.maximum(state.overtake_success_episodes, 1)
    wait_opportunities = jnp.maximum(state.wait_opportunity_step_sum, 1)
    return {
        "episodes": state.completed_episodes,
        "opponent_episodes": state.completed_opponent_episodes,
        "race_completion_rate": state.completed_races / episodes,
        "collision_rate": state.collision_episodes / episodes,
        "off_track_rate": state.off_track_episodes / episodes,
        "truncation_rate": state.truncated_episodes / episodes,
        "mean_return": state.return_sum / episodes,
        "mean_progress": state.progress_sum / episodes,
        "mean_speed": state.speed_sum_completed / steps,
        "mean_episode_steps": state.step_sum_completed / episodes,
        "mean_lap_steps": state.race_step_sum / races,
        "mean_steering_variation": state.steering_variation_sum / episodes,
        "mean_acceleration_variation": state.acceleration_variation_sum / episodes,
        "overtake_success_rate": (
            state.overtake_success_episodes / opponent_episodes
        ),
        "mean_passes_per_episode": state.pass_count_sum / opponent_episodes,
        "mean_steps_to_first_pass": (
            state.first_pass_step_sum / successful_overtakes
        ),
        "mean_relative_progress": state.relative_progress_sum / opponent_episodes,
        "opponent_collision_rate": (
            state.opponent_collision_episodes / opponent_episodes
        ),
        "wall_collision_rate": state.wall_collision_episodes / opponent_episodes,
        "unsafe_contact_episode_rate": (
            state.unsafe_contact_episodes / opponent_episodes
        ),
        "unsafe_contact_step_rate": state.unsafe_contact_step_sum / steps,
        "mean_follow_duration_steps": (
            state.following_step_sum / opponent_episodes
        ),
        "stalled_behind_step_rate": state.stalled_step_sum / steps,
        "minimum_opponent_distance_mean": (
            state.minimum_opponent_distance_sum / opponent_episodes
        ),
        "mean_final_rank": state.final_rank_sum / opponent_episodes,
        "post_pass_recontact_rate": (
            state.post_pass_recontact_episodes / successful_overtakes
        ),
        # Configured-distance proxy for blocked sections: near-opponent steps
        # without an unsafe contact or a stalled Ego.
        "safe_wait_success_rate": state.safe_wait_step_sum / wait_opportunities,
    }


__all__ = [
    "EvaluationAccumulator",
    "evaluation_summary",
    "initialize_evaluation_accumulator",
    "select_episode_completions",
    "update_evaluation_accumulator",
]

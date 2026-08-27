"""Dynamic vehicle LiDAR scan generation for one vectorizable environment."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from lidar_racing_rl.geometry.ray_obb import ray_obb_distance


def pairwise_dynamic_scan(
    vehicle_poses: Array,
    vehicle_dimensions: Array,
    beam_angles: Array,
    max_range: float | Array,
    *,
    parallel_epsilon: float = 1.0e-7,
) -> Array:
    """Compute every observer-to-target OBB intersection in one environment.

    Args:
        vehicle_poses: Vehicle center poses ``[agents, 3]`` as ``x, y, yaw``.
        vehicle_dimensions: Full ``length, width`` values with shape
            ``[agents, 2]`` or ``[2]``. A single pair is broadcast to all agents.
        beam_angles: LiDAR beam angles relative to each observer's yaw with
            shape ``[beams]``.
        max_range: Scalar range used for misses and invalid geometry.
        parallel_epsilon: Epsilon forwarded to the slab intersection primitive.

    Returns:
        Pairwise ranges with shape ``[observer, target, beam]``. The diagonal
        ``observer == target`` is explicitly masked to ``max_range``.

    Notes:
        This is a single-environment function. Apply ``jax.vmap`` to the whole
        function to obtain ``[env, observer, target, beam]`` without Python
        loops over environments, observers, targets, or beams.
    """
    dtype = jnp.result_type(
        vehicle_poses, vehicle_dimensions, beam_angles, max_range, jnp.float32
    )
    poses = jnp.asarray(vehicle_poses, dtype=dtype)
    dimensions = jnp.asarray(vehicle_dimensions, dtype=dtype)
    relative_angles = jnp.asarray(beam_angles, dtype=dtype)
    agent_count = poses.shape[0]

    dimensions = jnp.broadcast_to(dimensions, (agent_count, 2))
    observer_yaw = poses[:, 2]
    world_beam_angles = observer_yaw[:, None] + relative_angles[None, :]
    ray_directions = jnp.stack(
        (jnp.cos(world_beam_angles), jnp.sin(world_beam_angles)), axis=-1
    )

    ray_origins = poses[:, None, None, :2]
    ray_directions = ray_directions[:, None, :, :]
    target_centers = poses[None, :, None, :2]
    target_yaws = poses[None, :, None, 2]
    target_half_extents = dimensions[None, :, None, :] * 0.5

    pairwise_ranges = ray_obb_distance(
        ray_origin=ray_origins,
        ray_direction=ray_directions,
        obb_center=target_centers,
        obb_yaw=target_yaws,
        obb_half_extents=target_half_extents,
        max_range=max_range,
        parallel_epsilon=parallel_epsilon,
    )

    self_mask = jnp.eye(agent_count, dtype=bool)[:, :, None]
    safe_max_range = _safe_max_range(max_range, pairwise_ranges.dtype)
    return jnp.where(self_mask, safe_max_range, pairwise_ranges)


def dynamic_vehicle_scan(
    vehicle_poses: Array,
    vehicle_dimensions: Array,
    beam_angles: Array,
    max_range: float | Array,
    *,
    parallel_epsilon: float = 1.0e-7,
) -> Array:
    """Reduce target OBB intersections to one scan per observer.

    Args:
        vehicle_poses: Vehicle center poses with shape ``[agents, 3]``.
        vehicle_dimensions: Full length and width with shape ``[agents, 2]`` or
            ``[2]``.
        beam_angles: Relative LiDAR beam angles with shape ``[beams]``.
        max_range: Scalar range used for misses and invalid geometry.
        parallel_epsilon: Epsilon forwarded to the slab intersection primitive.

    Returns:
        Dynamic-only ranges with shape ``[agents, beams]``.
    """
    pairwise_ranges = pairwise_dynamic_scan(
        vehicle_poses,
        vehicle_dimensions,
        beam_angles,
        max_range,
        parallel_epsilon=parallel_epsilon,
    )
    return jnp.min(pairwise_ranges, axis=1)


def combine_static_and_dynamic_scan(
    static_scan: Array,
    dynamic_scan: Array,
    max_range: float | Array,
) -> Array:
    """Combine map and vehicle ranges while preserving invalid-map semantics.

    Args:
        static_scan: Static map ranges with shape ``[agents, beams]``.
        dynamic_scan: Dynamic OBB ranges with shape ``[agents, beams]``.
        max_range: Scalar upper bound and invalid-value replacement.

    Returns:
        NaN-free ranges with shape ``[agents, beams]``.  If neither the static
        scan nor a dynamic OBB provides a valid return, the output is ``-1`` so
        downstream canonicalization retains ``validity=0``.  A dynamic value
        equal to ``max_range`` is the ray-caster's miss sentinel, not a hit.
    """
    dtype = jnp.result_type(static_scan, dynamic_scan, max_range, jnp.float32)
    static_ranges = jnp.asarray(static_scan, dtype=dtype)
    dynamic_ranges = jnp.asarray(dynamic_scan, dtype=dtype)
    safe_max_range = _safe_max_range(max_range, dtype)
    static_valid = (
        jnp.isfinite(static_ranges)
        & (static_ranges >= 0.0)
        & (static_ranges <= safe_max_range)
    )
    dynamic_hit = (
        jnp.isfinite(dynamic_ranges)
        & (dynamic_ranges >= 0.0)
        & (dynamic_ranges < safe_max_range)
    )
    static_candidate = jnp.where(static_valid, static_ranges, safe_max_range)
    dynamic_candidate = jnp.where(dynamic_hit, dynamic_ranges, safe_max_range)
    nearest = jnp.minimum(static_candidate, dynamic_candidate)
    has_valid_return = static_valid | dynamic_hit
    invalid_sentinel = jnp.asarray(-1.0, dtype=dtype)
    return jnp.where(has_valid_return, nearest, invalid_sentinel)


def dynamic_lidar_scan(
    static_scan: Array,
    vehicle_poses: Array,
    vehicle_dimensions: Array,
    beam_angles: Array,
    max_range: float | Array,
    *,
    parallel_epsilon: float = 1.0e-7,
) -> Array:
    """Add dynamic vehicle OBBs to a static scan for one environment.

    Args:
        static_scan: Static map ranges with shape ``[agents, beams]``.
        vehicle_poses: Vehicle center poses ``[agents, 3]`` as ``x, y, yaw``.
        vehicle_dimensions: Full length and width with shape ``[agents, 2]`` or
            ``[2]``.
        beam_angles: Relative LiDAR beam angles with shape ``[beams]``.
        max_range: Scalar range used for misses and invalid values.
        parallel_epsilon: Epsilon forwarded to the slab intersection primitive.

    Returns:
        Static-plus-dynamic ranges with shape ``[agents, beams]``.

    Notes:
        The function has no Python loop over agents or beams. It can be lifted
        to ``[env, agents, beams]`` with ``jax.vmap``.
    """
    vehicle_ranges = dynamic_vehicle_scan(
        vehicle_poses,
        vehicle_dimensions,
        beam_angles,
        max_range,
        parallel_epsilon=parallel_epsilon,
    )
    return combine_static_and_dynamic_scan(static_scan, vehicle_ranges, max_range)


def _safe_max_range(max_range: float | Array, dtype: jnp.dtype) -> Array:
    requested = jnp.asarray(max_range, dtype=dtype)
    return jnp.where(jnp.isfinite(requested) & (requested >= 0.0), requested, 0.0)

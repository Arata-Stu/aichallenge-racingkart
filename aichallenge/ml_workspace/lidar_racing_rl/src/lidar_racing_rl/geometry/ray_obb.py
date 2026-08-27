"""Vectorized ray--oriented-bounding-box intersection primitives."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def ray_obb_distance(
    ray_origin: Array,
    ray_direction: Array,
    obb_center: Array,
    obb_yaw: Array,
    obb_half_extents: Array,
    max_range: float | Array,
    *,
    parallel_epsilon: float = 1.0e-7,
) -> Array:
    """Return the first non-negative ray entry distance for an oriented box.

    The leading dimensions of every input follow NumPy broadcasting rules. This
    makes the primitive usable for a single ray as well as an entire
    ``[observer, target, beam]`` tensor without Python loops.

    Args:
        ray_origin: Ray origins with shape ``[..., 2]`` in world coordinates.
        ray_direction: Unit ray directions with shape ``[..., 2]`` in world
            coordinates. A zero-length or non-finite direction is treated as a
            miss.
        obb_center: OBB centers with shape ``[..., 2]`` in world coordinates.
        obb_yaw: Counter-clockwise OBB yaw angles with shape ``[...]`` in
            radians.
        obb_half_extents: Positive half length and half width with shape
            ``[..., 2]`` in the OBB-local x and y axes.
        max_range: Scalar distance returned for misses and invalid input.
        parallel_epsilon: Direction components with a smaller absolute value
            are handled as parallel to the corresponding slab.

    Returns:
        Broadcast entry distances with shape ``[...]``. Misses, intersections
        behind the ray origin, intersections farther than ``max_range``, and
        invalid inputs are represented by ``max_range``. The result never
        contains NaN or infinity when ``max_range`` is finite and non-negative.

    Notes:
        The returned parameter is a metric distance only when ``ray_direction``
        is a unit vector. A ray whose origin is already inside the OBB has a
        negative entry distance and is deliberately treated as a miss. This is
        useful for rejecting invalid overlapping vehicle placements.
    """
    dtype = jnp.result_type(
        ray_origin,
        ray_direction,
        obb_center,
        obb_yaw,
        obb_half_extents,
        max_range,
        jnp.float32,
    )
    origin = jnp.asarray(ray_origin, dtype=dtype)
    direction = jnp.asarray(ray_direction, dtype=dtype)
    center = jnp.asarray(obb_center, dtype=dtype)
    yaw = jnp.asarray(obb_yaw, dtype=dtype)
    half_extents = jnp.asarray(obb_half_extents, dtype=dtype)
    requested_max_range = jnp.asarray(max_range, dtype=dtype)

    max_range_is_valid = jnp.isfinite(requested_max_range) & (requested_max_range >= 0.0)
    safe_max_range = jnp.where(max_range_is_valid, requested_max_range, 0.0)
    epsilon = jnp.asarray(abs(parallel_epsilon), dtype=dtype)

    # Rotate the ray by -yaw so that the oriented box becomes axis aligned.
    relative_origin = origin - center
    cosine = jnp.cos(yaw)
    sine = jnp.sin(yaw)
    local_origin = jnp.stack(
        (
            cosine * relative_origin[..., 0] + sine * relative_origin[..., 1],
            -sine * relative_origin[..., 0] + cosine * relative_origin[..., 1],
        ),
        axis=-1,
    )
    local_direction = jnp.stack(
        (
            cosine * direction[..., 0] + sine * direction[..., 1],
            -sine * direction[..., 0] + cosine * direction[..., 1],
        ),
        axis=-1,
    )

    # The explicit safe divisor prevents 0/0 and inf-inf NaNs in parallel
    # slabs. Parallel axes are then represented by unbounded intervals.
    parallel = jnp.abs(local_direction) <= epsilon
    safe_direction = jnp.where(parallel, jnp.ones_like(local_direction), local_direction)
    first_plane = (-half_extents - local_origin) / safe_direction
    second_plane = (half_extents - local_origin) / safe_direction
    axis_entry = jnp.minimum(first_plane, second_plane)
    axis_exit = jnp.maximum(first_plane, second_plane)
    axis_entry = jnp.where(parallel, -jnp.inf, axis_entry)
    axis_exit = jnp.where(parallel, jnp.inf, axis_exit)

    inside_parallel_slab = jnp.abs(local_origin) <= half_extents + epsilon
    parallel_miss = jnp.any(parallel & ~inside_parallel_slab, axis=-1)
    entry_distance = jnp.max(axis_entry, axis=-1)
    exit_distance = jnp.min(axis_exit, axis=-1)

    finite_input = (
        jnp.all(jnp.isfinite(origin), axis=-1)
        & jnp.all(jnp.isfinite(direction), axis=-1)
        & jnp.all(jnp.isfinite(center), axis=-1)
        & jnp.isfinite(yaw)
        & jnp.all(jnp.isfinite(half_extents), axis=-1)
        & jnp.all(half_extents > 0.0, axis=-1)
        & max_range_is_valid
    )
    nonzero_direction = jnp.linalg.norm(direction, axis=-1) > epsilon
    hit = (
        finite_input
        & nonzero_direction
        & ~parallel_miss
        & (exit_distance >= entry_distance)
        & (exit_distance >= 0.0)
        & (entry_distance >= 0.0)
        & (entry_distance <= safe_max_range)
    )

    distance = jnp.where(hit, entry_distance, safe_max_range)
    # Keep invalid arithmetic from escaping even if this function is reused
    # with traced or adversarial inputs.
    return jnp.where(jnp.isfinite(distance), distance, safe_max_range)

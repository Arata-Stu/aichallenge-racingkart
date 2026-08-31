"""LiDAR-only observation preprocessing shared by training and AWSIM export.

This module deliberately accepts only range samples and range bounds.  Beam/FOV
metadata is a caller-side contract: before calling, the adapter must verify the
expected angular coverage, beam ordering, and angle increment. ROS deployment
performs raw AWSIM angular pooling before this canonical representation; the
F1TENTH environment already generates the same 360 angular bins.

``range_min`` and ``range_max`` must be finite scalar values satisfying
``0 <= range_min < range_max``.  They are configuration/metadata values rather
than learned observations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


CANONICAL_BEAM_COUNT = 360
AWSIM_BEAM_COUNT = 1080
SCAN_CHANNEL_COUNT = 2
RANGE_CHANNEL = 0
VALIDITY_CHANNEL = 1


def canonicalize_scan(
    ranges: jax.Array,
    *,
    range_min: float | jax.Array,
    range_max: float | jax.Array,
) -> jax.Array:
    """Convert a 360- or 1080-beam scan to ``[..., 2, 360]``.

    Channel 0 contains distance divided by ``range_max`` and clipped to
    ``[0, 1]``.  Channel 1 contains a float validity mask.  NaN, infinity, and
    samples outside the inclusive ``[range_min, range_max]`` interval are
    invalid.

    For 1080-beam input, contiguous groups of three are reduced using the
    minimum valid distance.  A group is valid when at least one member is
    valid; a wholly invalid group is represented by ``range_max`` with a zero
    validity value.  A 360-beam input follows the same rules with identity
    pooling.

    Leading batch dimensions are preserved.  The beam-count branch uses the
    static array shape, so the function can be composed with ``jax.jit`` and
    ``jax.vmap`` without a Python loop over beams.
    """

    scan = jnp.asarray(ranges, dtype=jnp.float32)
    if scan.ndim < 1:
        raise ValueError("ranges must have a beam axis")

    beam_count = scan.shape[-1]
    if beam_count == AWSIM_BEAM_COUNT:
        grouped_scan = scan.reshape(
            (
                *scan.shape[:-1],
                CANONICAL_BEAM_COUNT,
                AWSIM_BEAM_COUNT // CANONICAL_BEAM_COUNT,
            )
        )
    elif beam_count == CANONICAL_BEAM_COUNT:
        grouped_scan = scan[..., :, jnp.newaxis]
    else:
        raise ValueError(
            "ranges must contain either "
            f"{CANONICAL_BEAM_COUNT} or {AWSIM_BEAM_COUNT} beams; got {beam_count}"
        )

    minimum = jnp.asarray(range_min, dtype=scan.dtype)
    maximum = jnp.asarray(range_max, dtype=scan.dtype)
    valid_samples = (
        jnp.isfinite(grouped_scan)
        & (grouped_scan >= minimum)
        & (grouped_scan <= maximum)
    )

    group_is_valid = jnp.any(valid_samples, axis=-1)
    valid_minimum = jnp.min(
        jnp.where(valid_samples, grouped_scan, maximum),
        axis=-1,
    )
    canonical_range = jnp.where(group_is_valid, valid_minimum, maximum)
    normalized_range = jnp.clip(canonical_range / maximum, 0.0, 1.0)
    validity = group_is_valid.astype(scan.dtype)

    return jnp.stack((normalized_range, validity), axis=-2)


def initialize_frame_stack(frame: jax.Array, *, num_frames: int = 4) -> jax.Array:
    """Initialize ``[..., frames, 2, 360]`` by repeating the first frame.

    ``num_frames`` is a static model configuration value.  Repeating the first
    observation prevents reset-time history from containing artificial zeros
    or maximum-range scans.
    """

    canonical_frame = jnp.asarray(frame, dtype=jnp.float32)
    _validate_canonical_frame_shape(canonical_frame)
    if num_frames < 1:
        raise ValueError("num_frames must be positive")

    output_shape = (
        *canonical_frame.shape[:-2],
        num_frames,
        SCAN_CHANNEL_COUNT,
        CANONICAL_BEAM_COUNT,
    )
    return jnp.broadcast_to(canonical_frame[..., jnp.newaxis, :, :], output_shape)


def update_frame_stack(frame_stack: jax.Array, frame: jax.Array) -> jax.Array:
    """Drop the oldest frame and append ``frame`` without Python iteration."""

    history = jnp.asarray(frame_stack, dtype=jnp.float32)
    canonical_frame = jnp.asarray(frame, dtype=jnp.float32)
    _validate_canonical_frame_shape(canonical_frame)

    if history.ndim < 3 or history.shape[-2:] != (
        SCAN_CHANNEL_COUNT,
        CANONICAL_BEAM_COUNT,
    ):
        raise ValueError("frame_stack must have shape [..., frames, 2, 360]")
    if history.shape[-3] < 1:
        raise ValueError("frame_stack must contain at least one frame")
    if history.shape[:-3] != canonical_frame.shape[:-2]:
        raise ValueError("frame_stack and frame batch dimensions must match")

    return jnp.concatenate(
        (history[..., 1:, :, :], canonical_frame[..., jnp.newaxis, :, :]),
        axis=-3,
    )


def _validate_canonical_frame_shape(frame: jax.Array) -> None:
    if frame.ndim < 2 or frame.shape[-2:] != (
        SCAN_CHANNEL_COUNT,
        CANONICAL_BEAM_COUNT,
    ):
        raise ValueError("frame must have shape [..., 2, 360]")


__all__ = [
    "AWSIM_BEAM_COUNT",
    "CANONICAL_BEAM_COUNT",
    "RANGE_CHANNEL",
    "SCAN_CHANNEL_COUNT",
    "VALIDITY_CHANNEL",
    "canonicalize_scan",
    "initialize_frame_stack",
    "update_frame_stack",
]

"""Flax LiDAR encoder shared by the SAC Actor and each Q network.

Only the canonical LiDAR tensor is accepted.  In particular, this API has no
generic observation dictionary through which simulator ground truth could be
added later.  A frame stack ``[..., 4, 2, 360]`` is folded to
``[..., 8, 360]`` before the Conv1D stack.  The already-folded form is also
accepted as the stable training/export boundary.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn


FRAME_STACK = 4
CHANNELS_PER_FRAME = 2
CANONICAL_BEAMS = 360
FOLDED_CHANNELS = FRAME_STACK * CHANNELS_PER_FRAME

LIDAR_ENCODER_CHANNELS = (32, 64, 64)
LIDAR_ENCODER_KERNEL_SIZES = (8, 4, 3)
LIDAR_ENCODER_STRIDES = (4, 2, 1)
LIDAR_FEATURE_DIM = 256


def prepare_lidar_observation(observation: jax.Array) -> jax.Array:
    """Return a float32 ``[..., 8, 360]`` LiDAR-only tensor.

    The canonical environment boundary is ``[..., 4, 2, 360]``.  Folding the
    two history axes here gives the Conv1D encoder a fixed channels-first
    representation.  Accepting ``[..., 8, 360]`` as an identity case keeps the
    learner and exported Actor interfaces identical.
    """

    lidar = jnp.asarray(observation, dtype=jnp.float32)
    canonical_shape = (FRAME_STACK, CHANNELS_PER_FRAME, CANONICAL_BEAMS)
    folded_shape = (FOLDED_CHANNELS, CANONICAL_BEAMS)

    if lidar.ndim >= 3 and lidar.shape[-3:] == canonical_shape:
        return lidar.reshape((*lidar.shape[:-3], *folded_shape))
    if lidar.ndim >= 2 and lidar.shape[-2:] == folded_shape:
        return lidar
    raise ValueError(
        "LiDAR observation must have shape [..., 4, 2, 360] or [..., 8, 360]"
    )


class LidarEncoder(nn.Module):
    """Three-layer Conv1D encoder with stable export-oriented parameter names."""

    channels: tuple[int, ...] = LIDAR_ENCODER_CHANNELS
    kernel_sizes: tuple[int, ...] = LIDAR_ENCODER_KERNEL_SIZES
    strides: tuple[int, ...] = LIDAR_ENCODER_STRIDES
    feature_dim: int = LIDAR_FEATURE_DIM

    def setup(self) -> None:
        if not (
            len(self.channels) == len(self.kernel_sizes) == len(self.strides) == 3
        ):
            raise ValueError("LiDAR encoder requires exactly three convolution layers")
        if any(value <= 0 for value in (*self.channels, *self.kernel_sizes, *self.strides)):
            raise ValueError("LiDAR encoder dimensions and strides must be positive")
        if self.feature_dim <= 0:
            raise ValueError("LiDAR feature_dim must be positive")

        self.convolutions = tuple(
            nn.Conv(
                features=channels,
                kernel_size=(kernel_size,),
                strides=(stride,),
                padding="VALID",
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                name=f"conv_{index}",
            )
            for index, (channels, kernel_size, stride) in enumerate(
                zip(self.channels, self.kernel_sizes, self.strides, strict=True)
            )
        )
        self.dense = nn.Dense(
            self.feature_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            name="dense",
        )

    def __call__(self, observation: jax.Array) -> jax.Array:
        """Encode canonical LiDAR while preserving all leading batch axes."""

        lidar = prepare_lidar_observation(observation)
        leading_shape = lidar.shape[:-2]
        flat_batch_size = math.prod(leading_shape) if leading_shape else 1
        # Flax Conv uses channels-last input: [..., spatial, channels].
        encoded = jnp.swapaxes(lidar, -2, -1).reshape(
            (flat_batch_size, CANONICAL_BEAMS, FOLDED_CHANNELS)
        )
        for convolution in self.convolutions:
            encoded = nn.relu(convolution(encoded))
        # PyTorch Conv1d emits [..., channels, spatial].  Restore that layout
        # before Flatten so a direct Dense-kernel transpose preserves parity.
        encoded = jnp.swapaxes(encoded, -2, -1)
        encoded = encoded.reshape((flat_batch_size, -1))
        features = nn.relu(self.dense(encoded))
        return features.reshape((*leading_shape, self.feature_dim))


__all__ = [
    "CANONICAL_BEAMS",
    "CHANNELS_PER_FRAME",
    "FOLDED_CHANNELS",
    "FRAME_STACK",
    "LIDAR_ENCODER_CHANNELS",
    "LIDAR_ENCODER_KERNEL_SIZES",
    "LIDAR_ENCODER_STRIDES",
    "LIDAR_FEATURE_DIM",
    "LidarEncoder",
    "prepare_lidar_observation",
]

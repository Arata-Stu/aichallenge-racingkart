"""JAX-composable LiDAR corruption with explicit calibration parameters.

The public functions operate on canonical scans shaped ``[..., 2, beams]``.
Channel 0 is range and channel 1 is validity. All configured distance values
must use the same units as the range channel (physical or normalized).

Every corruption parameter defaults to ``None``. Consequently, setting the
global ``enabled`` flag cannot silently activate guessed noise values before
AWSIM calibration statistics are available.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


RANGE_CHANNEL = 0
VALIDITY_CHANNEL = 1
SCAN_CHANNEL_COUNT = 2


@dataclass(frozen=True)
class ScanCorruptionConfig:
    """Static, hashable corruption settings populated from AWSIM statistics.

    A field group is disabled when all of its values are ``None``. Supplying
    only part of a group is rejected by :meth:`validate`; the implementation
    never fills a missing calibration value with a guessed default.

    ``frame_delay_probabilities`` is an explicit categorical distribution in
    increasing delay order: index 0 means the current frame, index 1 means one
    step old, and so on. This avoids assuming a uniform delay distribution.
    Far-leak probability is evaluated per valid beam. Single-beam and sector
    dropout probabilities are evaluated once per leading scan item.
    """

    enabled: bool = False

    far_leak_probability: float | None = None
    far_leak_extra_min: float | None = None
    far_leak_extra_max: float | None = None

    single_beam_dropout_probability: float | None = None

    sector_dropout_probability: float | None = None
    sector_dropout_width_beams: int | None = None

    frame_hold_probability: float | None = None
    frame_delay_probabilities: tuple[float, ...] | None = None

    gaussian_noise_base_std: float | None = None
    gaussian_noise_range_scale: float | None = None

    angle_bias_mean_radians: float | None = None
    angle_bias_std_radians: float | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ScanCorruptionConfig:
        """Build from the nested ``configs/env/dynamic_lidar.yaml`` mapping."""

        root = config.get("scan_corruption", config)
        far_leak = root["far_leak"]
        single_dropout = root["single_beam_dropout"]
        sector_dropout = root["sector_dropout"]
        frame_hold = root["frame_hold"]
        frame_delay = root["frame_delay"]
        gaussian_noise = root["gaussian_noise"]
        angle_bias = root["angle_bias"]
        delay_probabilities = frame_delay["probabilities"]
        instance = cls(
            enabled=bool(root["enabled"]),
            far_leak_probability=far_leak["probability"],
            far_leak_extra_min=far_leak["extra_min"],
            far_leak_extra_max=far_leak["extra_max"],
            single_beam_dropout_probability=single_dropout["probability"],
            sector_dropout_probability=sector_dropout["probability"],
            sector_dropout_width_beams=sector_dropout["width_beams"],
            frame_hold_probability=frame_hold["probability"],
            frame_delay_probabilities=(
                None
                if delay_probabilities is None
                else tuple(float(value) for value in delay_probabilities)
            ),
            gaussian_noise_base_std=gaussian_noise["base_std"],
            gaussian_noise_range_scale=gaussian_noise["range_scale"],
            angle_bias_mean_radians=angle_bias["mean_radians"],
            angle_bias_std_radians=angle_bias["std_radians"],
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        """Raise ``ValueError`` for partial, non-finite, or unsafe settings."""
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")

        _validate_complete_group(
            "far_leak",
            (
                self.far_leak_probability,
                self.far_leak_extra_min,
                self.far_leak_extra_max,
            ),
        )
        _validate_probability("far_leak_probability", self.far_leak_probability)
        if self.far_leak_extra_min is not None:
            _validate_finite_nonnegative("far_leak_extra_min", self.far_leak_extra_min)
            _validate_finite_nonnegative("far_leak_extra_max", self.far_leak_extra_max)
            if self.far_leak_extra_min <= 0.0:
                raise ValueError("far_leak_extra_min must be greater than zero")
            if self.far_leak_extra_max < self.far_leak_extra_min:
                raise ValueError("far_leak_extra_max must be >= far_leak_extra_min")

        _validate_probability(
            "single_beam_dropout_probability",
            self.single_beam_dropout_probability,
        )

        _validate_complete_group(
            "sector_dropout",
            (self.sector_dropout_probability, self.sector_dropout_width_beams),
        )
        _validate_probability(
            "sector_dropout_probability", self.sector_dropout_probability
        )
        if self.sector_dropout_width_beams is not None and (
            isinstance(self.sector_dropout_width_beams, bool)
            or not isinstance(self.sector_dropout_width_beams, int)
            or self.sector_dropout_width_beams < 1
        ):
            raise ValueError("sector_dropout_width_beams must be a positive integer")

        _validate_probability("frame_hold_probability", self.frame_hold_probability)
        if self.frame_delay_probabilities is not None:
            if not isinstance(self.frame_delay_probabilities, tuple):
                raise ValueError("frame_delay_probabilities must be a tuple")
            if not self.frame_delay_probabilities:
                raise ValueError("frame_delay_probabilities must not be empty")
            for probability in self.frame_delay_probabilities:
                _validate_probability("frame_delay probability", probability)
            if not math.isclose(
                sum(self.frame_delay_probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-6
            ):
                raise ValueError("frame_delay_probabilities must sum to one")

        _validate_complete_group(
            "gaussian_noise",
            (self.gaussian_noise_base_std, self.gaussian_noise_range_scale),
        )
        if self.gaussian_noise_base_std is not None:
            _validate_finite_nonnegative(
                "gaussian_noise_base_std", self.gaussian_noise_base_std
            )
            _validate_finite_nonnegative(
                "gaussian_noise_range_scale", self.gaussian_noise_range_scale
            )

        _validate_complete_group(
            "angle_bias",
            (self.angle_bias_mean_radians, self.angle_bias_std_radians),
        )
        if self.angle_bias_mean_radians is not None:
            _validate_finite("angle_bias_mean_radians", self.angle_bias_mean_radians)
            _validate_finite_nonnegative(
                "angle_bias_std_radians", self.angle_bias_std_radians
            )

    @property
    def history_length(self) -> int:
        """Number of frames required by the configured delay distribution."""
        if self.frame_delay_probabilities is None:
            return 1
        return len(self.frame_delay_probabilities)


class ScanCorruptionState(NamedTuple):
    """Temporal corruption state represented entirely by JAX arrays.

    Attributes:
        history: Spatially corrupted frames shaped
            ``[..., history, 2, beams]``, oldest first.
        last_output: Previously emitted frame shaped ``[..., 2, beams]``.
    """

    history: jax.Array
    last_output: jax.Array


def initialize_scan_corruption_state(
    scan: jax.Array,
    config: ScanCorruptionConfig,
    *,
    range_min: float,
    range_max: float,
) -> ScanCorruptionState:
    """Initialize temporal state by repeating one sanitized canonical scan.

    Args:
        scan: Initial canonical scan shaped ``[..., 2, beams]``.
        config: Static corruption configuration.
        range_min: Smallest valid range in the range channel's units.
        range_max: Largest valid range and invalid-range replacement.

    Returns:
        A state with history shape ``[..., config.history_length, 2, beams]``.
    """
    config.validate()
    _validate_range_bounds(range_min, range_max)
    canonical_scan = _sanitize_scan(scan, range_min, range_max)
    history_shape = (
        *canonical_scan.shape[:-2],
        config.history_length,
        SCAN_CHANNEL_COUNT,
        canonical_scan.shape[-1],
    )
    history = jnp.broadcast_to(canonical_scan[..., None, :, :], history_shape)
    return ScanCorruptionState(history=history, last_output=canonical_scan)


def apply_scan_corruption(
    key: jax.Array,
    scan: jax.Array,
    state: ScanCorruptionState,
    config: ScanCorruptionConfig,
    *,
    range_min: float,
    range_max: float,
    angle_increment: float | None = None,
) -> tuple[jax.Array, ScanCorruptionState]:
    """Corrupt one canonical scan and advance its temporal state.

    Args:
        key: JAX PRNG key. With leading environment or vehicle dimensions in
            ``scan``, independent events are sampled for every leading item.
        scan: Canonical input shaped ``[..., 2, beams]``.
        state: History initialized with :func:`initialize_scan_corruption_state`.
        config: Static, frozen configuration. Close over it and the scalar
            metadata when using ``jax.jit``/``jax.vmap``, or mark them as static
            JIT arguments.
        range_min: Smallest valid range in the range channel's units.
        range_max: Largest valid range and invalid-range replacement.
        angle_increment: Beam angular increment in radians. Required only when
            angle bias parameters are configured.

    Returns:
        ``(corrupted_scan, next_state)``. The scan has shape
        ``[..., 2, beams]``; invalid beams always contain ``range_max`` and a
        zero validity value. Far leak and finite noise remain valid samples.

    Notes:
        The implementation uses array broadcasting and gathering only. It has
        no Python loop over environment, vehicle, or beam axes.
    """
    config.validate()
    _validate_range_bounds(range_min, range_max)
    canonical_scan = _sanitize_scan(scan, range_min, range_max)
    _validate_state_shape(state, canonical_scan, config.history_length)

    (
        far_event_key,
        far_distance_key,
        single_dropout_event_key,
        single_dropout_index_key,
        sector_event_key,
        sector_start_key,
        hold_key,
        delay_key,
        gaussian_key,
        angle_key,
    ) = jax.random.split(key, 10)

    ranges = canonical_scan[..., RANGE_CHANNEL, :]
    validity = canonical_scan[..., VALIDITY_CHANNEL, :]

    if config.enabled and config.angle_bias_mean_radians is not None:
        if angle_increment is None:
            raise ValueError("angle_increment is required when angle bias is configured")
        _validate_angle_increment(angle_increment)
        ranges, validity = _apply_angle_bias(
            angle_key,
            ranges,
            validity,
            mean_radians=config.angle_bias_mean_radians,
            std_radians=config.angle_bias_std_radians,
            angle_increment=angle_increment,
            range_max=range_max,
        )

    if config.enabled and config.gaussian_noise_base_std is not None:
        standard_deviation = (
            config.gaussian_noise_base_std
            + config.gaussian_noise_range_scale * ranges
        )
        noise = jax.random.normal(gaussian_key, ranges.shape, dtype=ranges.dtype)
        noisy_ranges = jnp.clip(
            ranges + noise * standard_deviation,
            range_min,
            range_max,
        )
        ranges = jnp.where(validity > 0.5, noisy_ranges, range_max)

    if config.enabled and config.far_leak_probability is not None:
        leak_event = jax.random.bernoulli(
            far_event_key,
            p=config.far_leak_probability,
            shape=ranges.shape,
        )
        unit_extra = jax.random.uniform(
            far_distance_key,
            shape=ranges.shape,
            minval=0.0,
            maxval=1.0,
            dtype=ranges.dtype,
        )
        leak_extra = config.far_leak_extra_min + unit_extra * (
            config.far_leak_extra_max - config.far_leak_extra_min
        )
        leaked_ranges = jnp.minimum(ranges + leak_extra, range_max)
        leak_event &= (validity > 0.5) & (ranges < range_max)
        ranges = jnp.where(leak_event, leaked_ranges, ranges)

    dropout = jnp.zeros_like(validity, dtype=bool)
    if config.enabled and config.single_beam_dropout_probability is not None:
        leading_event_shape = (*ranges.shape[:-1], 1)
        single_dropout_event = jax.random.bernoulli(
            single_dropout_event_key,
            p=config.single_beam_dropout_probability,
            shape=leading_event_shape,
        )
        single_dropout_index = jax.random.randint(
            single_dropout_index_key,
            shape=leading_event_shape,
            minval=0,
            maxval=ranges.shape[-1],
        )
        beam_index = jnp.arange(ranges.shape[-1])
        dropout |= single_dropout_event & (beam_index == single_dropout_index)

    if config.enabled and config.sector_dropout_probability is not None:
        sector_width = config.sector_dropout_width_beams
        beam_count = ranges.shape[-1]
        if sector_width > beam_count:
            raise ValueError("sector_dropout_width_beams must not exceed beam count")
        leading_event_shape = (*ranges.shape[:-1], 1)
        sector_event = jax.random.bernoulli(
            sector_event_key,
            p=config.sector_dropout_probability,
            shape=leading_event_shape,
        )
        sector_start = jax.random.randint(
            sector_start_key,
            shape=leading_event_shape,
            minval=0,
            maxval=beam_count - sector_width + 1,
        )
        beam_index = jnp.arange(beam_count)
        in_sector = (beam_index >= sector_start) & (
            beam_index < sector_start + sector_width
        )
        dropout |= sector_event & in_sector

    ranges = jnp.where(dropout, range_max, ranges)
    validity = jnp.where(dropout, 0.0, validity)
    spatial_scan = jnp.stack((ranges, validity), axis=-2)
    spatial_scan = _sanitize_scan(spatial_scan, range_min, range_max)

    next_history = jnp.concatenate(
        (state.history[..., 1:, :, :], spatial_scan[..., None, :, :]),
        axis=-3,
    )
    delayed_scan = spatial_scan
    if config.enabled and config.frame_delay_probabilities is not None:
        probabilities = jnp.asarray(
            config.frame_delay_probabilities, dtype=spatial_scan.dtype
        )
        delays = jax.random.categorical(
            delay_key,
            jnp.log(probabilities),
            shape=spatial_scan.shape[:-2],
        )
        history_index = config.history_length - 1 - delays
        gather_index = history_index[..., None, None, None]
        gather_shape = (*next_history.shape[:-3], 1, *next_history.shape[-2:])
        gather_index = jnp.broadcast_to(gather_index, gather_shape)
        delayed_scan = jnp.take_along_axis(next_history, gather_index, axis=-3)[
            ..., 0, :, :
        ]

    output_scan = delayed_scan
    if config.enabled and config.frame_hold_probability is not None:
        hold_event = jax.random.bernoulli(
            hold_key,
            p=config.frame_hold_probability,
            shape=(*spatial_scan.shape[:-2], 1, 1),
        )
        output_scan = jnp.where(hold_event, state.last_output, delayed_scan)

    output_scan = _sanitize_scan(output_scan, range_min, range_max)
    next_state = ScanCorruptionState(history=next_history, last_output=output_scan)
    return output_scan, next_state


def _apply_angle_bias(
    key: jax.Array,
    ranges: jax.Array,
    validity: jax.Array,
    *,
    mean_radians: float,
    std_radians: float,
    angle_increment: float,
    range_max: float,
) -> tuple[jax.Array, jax.Array]:
    """Shift ranges and validity together using linear angular resampling."""
    bias = mean_radians + std_radians * jax.random.normal(
        key, ranges.shape[:-1], dtype=ranges.dtype
    )
    beam_count = ranges.shape[-1]
    source_position = jnp.arange(beam_count, dtype=ranges.dtype) - (
        bias[..., None] / angle_increment
    )
    lower_float = jnp.floor(source_position)
    upper_float = lower_float + 1.0
    interpolation_weight = source_position - lower_float
    exact_lower = interpolation_weight <= 1.0e-6

    lower_in_bounds = (lower_float >= 0.0) & (lower_float < beam_count)
    upper_in_bounds = upper_float < beam_count
    sample_in_bounds = lower_in_bounds & (exact_lower | upper_in_bounds)
    lower_index = jnp.clip(lower_float, 0.0, beam_count - 1).astype(jnp.int32)
    upper_index = jnp.clip(upper_float, 0.0, beam_count - 1).astype(jnp.int32)

    lower_range = jnp.take_along_axis(ranges, lower_index, axis=-1)
    upper_range = jnp.take_along_axis(ranges, upper_index, axis=-1)
    lower_validity = jnp.take_along_axis(validity, lower_index, axis=-1) > 0.5
    upper_validity = jnp.take_along_axis(validity, upper_index, axis=-1) > 0.5
    sample_is_valid = sample_in_bounds & lower_validity & (
        exact_lower | upper_validity
    )

    shifted_range = lower_range + interpolation_weight * (upper_range - lower_range)
    shifted_range = jnp.where(sample_is_valid, shifted_range, range_max)
    shifted_validity = sample_is_valid.astype(validity.dtype)
    return shifted_range, shifted_validity


def _sanitize_scan(scan: jax.Array, range_min: float, range_max: float) -> jax.Array:
    canonical_scan = jnp.asarray(scan, dtype=jnp.float32)
    if canonical_scan.ndim < 2 or canonical_scan.shape[-2] != SCAN_CHANNEL_COUNT:
        raise ValueError("scan must have shape [..., 2, beams]")
    if canonical_scan.shape[-1] < 1:
        raise ValueError("scan must contain at least one beam")
    ranges = canonical_scan[..., RANGE_CHANNEL, :]
    declared_validity = canonical_scan[..., VALIDITY_CHANNEL, :] > 0.5
    valid = (
        declared_validity
        & jnp.isfinite(ranges)
        & (ranges >= range_min)
        & (ranges <= range_max)
    )
    safe_ranges = jnp.where(valid, ranges, range_max)
    return jnp.stack((safe_ranges, valid.astype(canonical_scan.dtype)), axis=-2)


def _validate_state_shape(
    state: ScanCorruptionState,
    scan: jax.Array,
    history_length: int,
) -> None:
    expected_history_shape = (
        *scan.shape[:-2],
        history_length,
        SCAN_CHANNEL_COUNT,
        scan.shape[-1],
    )
    if state.history.shape != expected_history_shape:
        raise ValueError(
            f"state.history must have shape {expected_history_shape}; "
            f"got {state.history.shape}"
        )
    if state.last_output.shape != scan.shape:
        raise ValueError(
            f"state.last_output must have shape {scan.shape}; "
            f"got {state.last_output.shape}"
        )


def _validate_complete_group(name: str, values: tuple[object | None, ...]) -> None:
    configured = tuple(value is not None for value in values)
    if any(configured) and not all(configured):
        raise ValueError(f"{name} parameters must be all configured or all None")


def _validate_probability(name: str, value: float | None) -> None:
    if value is None:
        return
    _validate_finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_finite_nonnegative(name: str, value: float) -> None:
    _validate_finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _validate_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_range_bounds(range_min: float, range_max: float) -> None:
    _validate_finite("range_min", range_min)
    _validate_finite("range_max", range_max)
    if range_min < 0.0 or range_max <= range_min:
        raise ValueError("range bounds must satisfy 0 <= range_min < range_max")


def _validate_angle_increment(angle_increment: float) -> None:
    _validate_finite("angle_increment", angle_increment)
    if angle_increment <= 0.0:
        raise ValueError("angle_increment must be positive")


__all__ = [
    "RANGE_CHANNEL",
    "SCAN_CHANNEL_COUNT",
    "VALIDITY_CHANNEL",
    "ScanCorruptionConfig",
    "ScanCorruptionState",
    "apply_scan_corruption",
    "initialize_scan_corruption_state",
]

"""Unit tests for calibrated LiDAR corruption and temporal state."""

import jax
import jax.numpy as jnp
import pytest

from lidar_racing_rl.envs.scan_corruption import (
    ScanCorruptionConfig,
    ScanCorruptionState,
    apply_scan_corruption,
    initialize_scan_corruption_state,
)


RANGE_MIN = 0.0
RANGE_MAX = 10.0


def _scan(ranges: jax.Array, validity: jax.Array | None = None) -> jax.Array:
    ranges = jnp.asarray(ranges, dtype=jnp.float32)
    if validity is None:
        validity = jnp.ones_like(ranges)
    return jnp.stack((ranges, validity), axis=-2)


def _apply_once(
    scan: jax.Array,
    config: ScanCorruptionConfig,
    *,
    initial_scan: jax.Array | None = None,
    angle_increment: float | None = None,
) -> tuple[jax.Array, ScanCorruptionState]:
    state = initialize_scan_corruption_state(
        scan if initial_scan is None else initial_scan,
        config,
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
    )
    return apply_scan_corruption(
        jax.random.PRNGKey(0),
        scan,
        state,
        config,
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
        angle_increment=angle_increment,
    )


def test_unmeasured_default_configuration_is_no_op() -> None:
    clean_scan = _scan(jnp.array([1.0, 2.0, 3.0, 4.0]))
    config = ScanCorruptionConfig(enabled=True)

    output, _ = _apply_once(clean_scan, config)

    assert bool(jnp.array_equal(output, clean_scan))


def test_nested_unmeasured_yaml_mapping_builds_no_op_config() -> None:
    config = ScanCorruptionConfig.from_config(
        {
            "scan_corruption": {
                "enabled": False,
                "far_leak": {"probability": None, "extra_min": None, "extra_max": None},
                "single_beam_dropout": {"probability": None},
                "sector_dropout": {"probability": None, "width_beams": None},
                "frame_hold": {"probability": None},
                "frame_delay": {"probabilities": None},
                "gaussian_noise": {"base_std": None, "range_scale": None},
                "angle_bias": {"mean_radians": None, "std_radians": None},
            }
        }
    )

    assert not config.enabled
    assert config.history_length == 1


def test_global_disable_overrides_populated_calibration() -> None:
    clean_scan = _scan(jnp.array([1.0, 2.0, 3.0, 4.0]))
    config = ScanCorruptionConfig(
        enabled=False,
        far_leak_probability=1.0,
        far_leak_extra_min=2.0,
        far_leak_extra_max=2.0,
        single_beam_dropout_probability=1.0,
    )

    output, _ = _apply_once(clean_scan, config)

    assert bool(jnp.array_equal(output, clean_scan))


def test_partial_calibration_group_is_rejected() -> None:
    config = ScanCorruptionConfig(enabled=True, far_leak_probability=0.1)

    with pytest.raises(ValueError, match="far_leak parameters"):
        config.validate()


def test_delay_distribution_must_be_explicit_and_normalized() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        ScanCorruptionConfig(
            enabled=True,
            frame_delay_probabilities=(0.4, 0.4),
        ).validate()


def test_far_leak_moves_valid_ranges_farther_without_invalidating_them() -> None:
    clean_scan = _scan(jnp.array([1.0, 5.0, 9.0]))
    config = ScanCorruptionConfig(
        enabled=True,
        far_leak_probability=1.0,
        far_leak_extra_min=2.0,
        far_leak_extra_max=2.0,
    )

    output, _ = _apply_once(clean_scan, config)

    assert bool(jnp.allclose(output[0], jnp.array([3.0, 7.0, 10.0])))
    assert bool(jnp.array_equal(output[1], jnp.ones(3)))


def test_single_beam_dropout_sets_invalid_range_and_validity_together() -> None:
    clean_scan = _scan(jnp.array([1.0, 2.0, 3.0, 4.0]))
    config = ScanCorruptionConfig(
        enabled=True,
        single_beam_dropout_probability=1.0,
    )

    output, _ = _apply_once(clean_scan, config)

    dropped = output[1] == 0.0
    assert int(jnp.count_nonzero(dropped)) == 1
    assert bool(jnp.all(output[0][dropped] == RANGE_MAX))
    assert bool(jnp.array_equal(output[1][~dropped], jnp.ones(3)))


def test_sector_dropout_is_contiguous_and_has_configured_width() -> None:
    clean_scan = _scan(jnp.arange(1.0, 9.0))
    config = ScanCorruptionConfig(
        enabled=True,
        sector_dropout_probability=1.0,
        sector_dropout_width_beams=3,
    )

    output, _ = _apply_once(clean_scan, config)
    dropped = output[1] == 0.0
    transitions = jnp.count_nonzero(jnp.diff(dropped.astype(jnp.int32)))

    assert int(jnp.count_nonzero(dropped)) == 3
    assert int(transitions) <= 2
    assert bool(jnp.all(output[0][dropped] == RANGE_MAX))


def test_frame_hold_reuses_previous_output_but_advances_history() -> None:
    first_scan = _scan(jnp.array([1.0, 2.0, 3.0]))
    second_scan = _scan(jnp.array([4.0, 5.0, 6.0]))
    config = ScanCorruptionConfig(enabled=True, frame_hold_probability=1.0)

    output, next_state = _apply_once(
        second_scan,
        config,
        initial_scan=first_scan,
    )

    assert bool(jnp.array_equal(output, first_scan))
    assert bool(jnp.array_equal(next_state.history[..., -1, :, :], second_scan))


def test_frame_delay_uses_explicit_categorical_history() -> None:
    first_scan = _scan(jnp.array([1.0, 2.0]))
    second_scan = _scan(jnp.array([3.0, 4.0]))
    third_scan = _scan(jnp.array([5.0, 6.0]))
    config = ScanCorruptionConfig(
        enabled=True,
        frame_delay_probabilities=(0.0, 1.0),
    )
    state = initialize_scan_corruption_state(
        first_scan,
        config,
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
    )

    second_output, state = apply_scan_corruption(
        jax.random.PRNGKey(1),
        second_scan,
        state,
        config,
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
    )
    third_output, _ = apply_scan_corruption(
        jax.random.PRNGKey(2),
        third_scan,
        state,
        config,
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
    )

    assert bool(jnp.array_equal(second_output, first_scan))
    assert bool(jnp.array_equal(third_output, second_scan))


def test_distance_dependent_gaussian_noise_preserves_validity_and_bounds() -> None:
    clean_scan = _scan(jnp.linspace(1.0, 9.0, 32))
    config = ScanCorruptionConfig(
        enabled=True,
        gaussian_noise_base_std=0.1,
        gaussian_noise_range_scale=0.02,
    )

    output, _ = _apply_once(clean_scan, config)

    assert not bool(jnp.allclose(output[0], clean_scan[0]))
    assert bool(jnp.all((output[0] >= RANGE_MIN) & (output[0] <= RANGE_MAX)))
    assert bool(jnp.array_equal(output[1], clean_scan[1]))


def test_angle_bias_shifts_ranges_and_validity_together() -> None:
    clean_scan = _scan(jnp.array([1.0, 2.0, 3.0, 4.0]))
    config = ScanCorruptionConfig(
        enabled=True,
        angle_bias_mean_radians=0.1,
        angle_bias_std_radians=0.0,
    )

    output, _ = _apply_once(clean_scan, config, angle_increment=0.1)

    assert bool(jnp.allclose(output[0], jnp.array([RANGE_MAX, 1.0, 2.0, 3.0])))
    assert bool(jnp.array_equal(output[1], jnp.array([0.0, 1.0, 1.0, 1.0])))


def test_invalid_input_is_canonicalized_to_max_range_and_zero_validity() -> None:
    dirty_scan = _scan(
        jnp.array([jnp.nan, jnp.inf, -1.0, 11.0, 5.0]),
        jnp.ones(5),
    )

    output, _ = _apply_once(dirty_scan, ScanCorruptionConfig())

    assert bool(jnp.all(jnp.isfinite(output)))
    assert bool(
        jnp.array_equal(
            output[0],
            jnp.array([RANGE_MAX, RANGE_MAX, RANGE_MAX, RANGE_MAX, 5.0]),
        )
    )
    assert bool(jnp.array_equal(output[1], jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])))


def test_jit_vmap_composition_preserves_environment_and_vehicle_axes() -> None:
    batch_scan = jnp.broadcast_to(
        _scan(jnp.array([1.0, 2.0, 3.0, 4.0])),
        (2, 3, 2, 4),
    )
    config = ScanCorruptionConfig(
        enabled=True,
        far_leak_probability=1.0,
        far_leak_extra_min=1.0,
        far_leak_extra_max=1.0,
    )
    batched_state = initialize_scan_corruption_state(
        batch_scan,
        config,
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
    )

    def corrupt_environment(
        key: jax.Array,
        environment_scan: jax.Array,
        history: jax.Array,
        last_output: jax.Array,
    ) -> tuple[jax.Array, ScanCorruptionState]:
        return apply_scan_corruption(
            key,
            environment_scan,
            ScanCorruptionState(history, last_output),
            config,
            range_min=RANGE_MIN,
            range_max=RANGE_MAX,
        )

    output, next_state = jax.jit(jax.vmap(corrupt_environment))(
        jax.random.split(jax.random.PRNGKey(3), 2),
        batch_scan,
        batched_state.history,
        batched_state.last_output,
    )

    assert output.shape == (2, 3, 2, 4)
    assert next_state.history.shape == (2, 3, 1, 2, 4)
    assert bool(jnp.allclose(output[..., 0, :], batch_scan[..., 0, :] + 1.0))
    assert bool(jnp.array_equal(output[..., 1, :], batch_scan[..., 1, :]))

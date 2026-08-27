import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lidar_racing_rl.envs.observation import (
    AWSIM_BEAM_COUNT,
    CANONICAL_BEAM_COUNT,
    RANGE_CHANNEL,
    VALIDITY_CHANNEL,
    canonicalize_scan,
    initialize_frame_stack,
    update_frame_stack,
)


RANGE_MIN = 0.1
RANGE_MAX = 30.0


def test_awsim_scan_uses_minimum_of_valid_samples_in_each_group() -> None:
    scan = jnp.full((AWSIM_BEAM_COUNT,), RANGE_MAX, dtype=jnp.float32)
    scan = scan.at[0:3].set(jnp.array([jnp.nan, 9.0, 3.0]))
    scan = scan.at[3:6].set(jnp.array([jnp.inf, RANGE_MIN - 0.01, RANGE_MAX + 1.0]))
    scan = scan.at[6:9].set(jnp.array([jnp.nan, RANGE_MAX, jnp.inf]))

    canonical = canonicalize_scan(
        scan,
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
    )

    assert canonical.shape == (2, CANONICAL_BEAM_COUNT)
    np.testing.assert_allclose(canonical[RANGE_CHANNEL, 0], 3.0 / RANGE_MAX)
    np.testing.assert_allclose(canonical[VALIDITY_CHANNEL, 0], 1.0)
    np.testing.assert_allclose(canonical[RANGE_CHANNEL, 1], 1.0)
    np.testing.assert_allclose(canonical[VALIDITY_CHANNEL, 1], 0.0)
    # A valid maximum-range return remains distinguishable from an invalid bin.
    np.testing.assert_allclose(canonical[RANGE_CHANNEL, 2], 1.0)
    np.testing.assert_allclose(canonical[VALIDITY_CHANNEL, 2], 1.0)


def test_awsim_pooling_preserves_leading_batch_dimensions() -> None:
    scan = jnp.full((2, 4, AWSIM_BEAM_COUNT), 6.0, dtype=jnp.float32)

    canonical = jax.jit(canonicalize_scan)(
        scan,
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
    )

    assert canonical.shape == (2, 4, 2, CANONICAL_BEAM_COUNT)
    np.testing.assert_allclose(canonical[..., RANGE_CHANNEL, :], 0.2)
    np.testing.assert_allclose(canonical[..., VALIDITY_CHANNEL, :], 1.0)


def test_canonical_input_uses_identity_pooling_and_marks_boundaries() -> None:
    scan = jnp.full((CANONICAL_BEAM_COUNT,), 15.0, dtype=jnp.float32)
    boundary_values = jnp.array(
        [
            RANGE_MIN,
            RANGE_MAX,
            RANGE_MIN - 0.01,
            RANGE_MAX + 0.01,
            jnp.nan,
            jnp.inf,
            -jnp.inf,
        ],
        dtype=jnp.float32,
    )
    scan = scan.at[: boundary_values.size].set(boundary_values)

    canonical = canonicalize_scan(
        scan,
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
    )

    expected_validity = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected_range = np.array(
        [RANGE_MIN / RANGE_MAX, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(
        canonical[VALIDITY_CHANNEL, : boundary_values.size],
        expected_validity,
    )
    np.testing.assert_allclose(
        canonical[RANGE_CHANNEL, : boundary_values.size],
        expected_range,
    )
    np.testing.assert_allclose(
        canonical[RANGE_CHANNEL, boundary_values.size :],
        0.5,
    )
    np.testing.assert_allclose(
        canonical[VALIDITY_CHANNEL, boundary_values.size :],
        1.0,
    )


@pytest.mark.parametrize("beam_count", [0, 1, 359, 361, 1079, 1081])
def test_rejects_noncanonical_beam_counts(beam_count: int) -> None:
    with pytest.raises(ValueError, match="360 or 1080"):
        canonicalize_scan(
            jnp.zeros((beam_count,), dtype=jnp.float32),
            range_min=RANGE_MIN,
            range_max=RANGE_MAX,
        )


def test_frame_stack_initialization_repeats_first_frame() -> None:
    scan = jnp.linspace(RANGE_MIN, RANGE_MAX, CANONICAL_BEAM_COUNT)
    frame = canonicalize_scan(
        scan,
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
    )

    history = jax.jit(lambda value: initialize_frame_stack(value, num_frames=4))(frame)

    assert history.shape == (4, 2, CANONICAL_BEAM_COUNT)
    expected = np.broadcast_to(np.asarray(frame), history.shape)
    np.testing.assert_allclose(history, expected)


def test_frame_stack_update_appends_newest_frame() -> None:
    first = jnp.stack(
        (
            jnp.zeros((CANONICAL_BEAM_COUNT,), dtype=jnp.float32),
            jnp.ones((CANONICAL_BEAM_COUNT,), dtype=jnp.float32),
        )
    )
    newest = first.at[RANGE_CHANNEL].set(0.75)
    history = initialize_frame_stack(first, num_frames=4)

    updated = jax.jit(update_frame_stack)(history, newest)

    assert updated.shape == history.shape
    np.testing.assert_allclose(updated[:-1], history[1:])
    np.testing.assert_allclose(updated[-1], newest)


def test_frame_stack_rejects_noncanonical_frame_shape() -> None:
    with pytest.raises(ValueError, match=r"\[\.\.\., 2, 360\]"):
        initialize_frame_stack(jnp.zeros((CANONICAL_BEAM_COUNT,)), num_frames=4)

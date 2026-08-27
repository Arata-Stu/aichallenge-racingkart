"""Unit tests for four-vehicle reset sampling."""

import jax
import jax.numpy as jnp
import numpy as np

from lidar_racing_rl.envs.reset_sampler import sample_four_vehicle_frenet


def test_sample_four_vehicle_frenet_shape_and_bounds() -> None:
    track_length = 100.0
    sampled = sample_four_vehicle_frenet(
        jax.random.key(7),
        jnp.asarray(track_length),
        longitudinal_spacing=4.0,
        lateral_jitter=0.25,
        heading_jitter=0.02,
    )

    assert sampled.shape == (4, 3)
    assert bool(jnp.all((sampled[:, 0] >= 0.0) & (sampled[:, 0] < track_length)))
    assert bool(jnp.all(jnp.abs(sampled[:, 1]) <= 0.25))
    assert bool(jnp.all(jnp.abs(sampled[:, 2]) <= 0.02))

    ordered = jnp.sort(sampled[:, 0])
    circular_gaps = jnp.diff(jnp.concatenate((ordered, ordered[:1] + track_length)))
    np.testing.assert_allclose(jnp.sort(circular_gaps)[:3], 4.0, atol=1.0e-5)

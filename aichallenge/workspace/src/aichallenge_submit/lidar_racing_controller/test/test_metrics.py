import math

from lidar_racing_controller.metrics import percentile


def test_percentile_interpolates_p50_and_p95() -> None:
    samples = [1.0, 2.0, 3.0, 4.0]

    assert percentile(samples, 50.0) == 2.5
    assert math.isclose(percentile(samples, 95.0), 3.85)


def test_percentile_accepts_a_single_sample() -> None:
    assert percentile([7.5], 50.0) == 7.5
    assert percentile([7.5], 95.0) == 7.5

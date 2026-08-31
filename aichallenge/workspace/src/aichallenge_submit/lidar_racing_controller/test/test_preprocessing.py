import math

import numpy as np
import pytest

from lidar_racing_controller.preprocessing import (
    FrameStack,
    ScanValidationError,
    canonicalize_laserscan,
)


def _canonicalize(ranges: np.ndarray):
    angle_min = -3.0 * math.pi / 4.0
    angle_max = 3.0 * math.pi / 4.0
    angle_increment = (angle_max - angle_min) / 1079.0
    return canonicalize_laserscan(
        ranges,
        range_min=0.1,
        range_max=30.0,
        angle_min=angle_min,
        angle_max=angle_max,
        angle_increment=angle_increment,
    )


def test_minimum_pooling_keeps_nearest_valid_sample() -> None:
    ranges = np.full(1080, 30.0, dtype=np.float32)
    ranges[3:6] = (9.0, 6.0, 3.0)

    result = _canonicalize(ranges)

    assert result.values.shape == (2, 360)
    assert result.values.dtype == np.float32
    assert result.values[0, 1] == pytest.approx(0.1)
    assert result.values[1, 1] == 1.0


def test_all_invalid_group_preserves_missing_observation_semantics() -> None:
    ranges = np.full(1080, 30.0, dtype=np.float32)
    ranges[0:3] = (np.nan, np.inf, 31.0)

    result = _canonicalize(ranges)

    assert result.values[0, 0] == 1.0
    assert result.values[1, 0] == 0.0
    assert result.valid_ratio == pytest.approx(359.0 / 360.0)


def test_invalid_metadata_is_rejected() -> None:
    ranges = np.full(1080, 10.0, dtype=np.float32)

    with pytest.raises(ScanValidationError, match="angle metadata"):
        canonicalize_laserscan(
            ranges,
            range_min=0.1,
            range_max=30.0,
            angle_min=-3.0 * math.pi / 4.0,
            angle_max=3.0 * math.pi / 4.0,
            angle_increment=0.001,
        )


def test_self_consistent_scan_with_wrong_field_of_view_is_rejected() -> None:
    ranges = np.full(1080, 10.0, dtype=np.float32)
    angle_increment = 2.0 * math.pi / 1079.0

    with pytest.raises(ScanValidationError, match="angle_min mismatch"):
        canonicalize_laserscan(
            ranges,
            range_min=0.1,
            range_max=30.0,
            angle_min=-math.pi,
            angle_max=math.pi,
            angle_increment=angle_increment,
        )


def test_expected_field_of_view_accepts_metadata_within_tolerance() -> None:
    ranges = np.full(1080, 10.0, dtype=np.float32)
    offset = 0.005
    angle_min = -3.0 * math.pi / 4.0 + offset
    angle_max = 3.0 * math.pi / 4.0 + offset

    result = canonicalize_laserscan(
        ranges,
        range_min=0.1,
        range_max=30.0,
        angle_min=angle_min,
        angle_max=angle_max,
        angle_increment=(angle_max - angle_min) / 1079.0,
    )

    assert result.values.shape == (2, 360)


def test_awsim_750_beams_are_angularly_pooled_to_360() -> None:
    raw_min = -1.5666074752807617
    raw_max = 1.5707963705062866
    ranges = np.full(750, 25.0, dtype=np.float32)
    center = int(round((0.0 - raw_min) / ((raw_max - raw_min) / 749.0)))
    ranges[center] = 2.5

    result = canonicalize_laserscan(
        ranges,
        range_min=0.0,
        range_max=25.0,
        angle_min=raw_min,
        angle_max=raw_max,
        angle_increment=(raw_max - raw_min) / 749.0,
        expected_raw_beams=750,
        expected_range_min=0.0,
        expected_range_max=25.0,
        expected_angle_min=raw_min,
        expected_angle_max=raw_max,
        canonical_range_max=25.0,
        canonical_angle_min=-math.pi / 2.0,
        canonical_angle_max=math.pi / 2.0,
    )

    assert result.values.shape == (2, 360)
    assert np.min(result.values[0]) == pytest.approx(0.1)
    assert np.all(result.values[1] == 1.0)


def test_awsim_range_min_mismatch_is_rejected() -> None:
    raw_min = -1.5666074752807617
    raw_max = 1.5707963705062866

    with pytest.raises(ScanValidationError, match="range_min mismatch"):
        canonicalize_laserscan(
            np.full(750, 10.0, dtype=np.float32),
            range_min=0.1,
            range_max=25.0,
            angle_min=raw_min,
            angle_max=raw_max,
            angle_increment=(raw_max - raw_min) / 749.0,
            expected_raw_beams=750,
            expected_range_min=0.0,
            expected_range_max=25.0,
            expected_angle_min=raw_min,
            expected_angle_max=raw_max,
        )


def test_narrow_raw_fov_pads_legacy_actor_outer_sectors_as_invalid() -> None:
    raw_min = -1.5666074752807617
    raw_max = 1.5707963705062866
    result = canonicalize_laserscan(
        np.full(750, 12.5, dtype=np.float32),
        range_min=0.0,
        range_max=25.0,
        angle_min=raw_min,
        angle_max=raw_max,
        angle_increment=(raw_max - raw_min) / 749.0,
        expected_raw_beams=750,
        expected_range_min=0.0,
        expected_range_max=25.0,
        expected_angle_min=raw_min,
        expected_angle_max=raw_max,
        canonical_range_max=30.0,
        canonical_angle_min=-3.0 * math.pi / 4.0,
        canonical_angle_max=3.0 * math.pi / 4.0,
    )

    valid_indices = np.flatnonzero(result.values[1] > 0.5)
    assert valid_indices.size == 240
    assert valid_indices[0] == 60
    assert valid_indices[-1] == 299
    assert np.all(result.values[0, :60] == 1.0)
    assert np.all(result.values[1, :60] == 0.0)
    assert np.all(result.values[0, 300:] == 1.0)
    assert np.all(result.values[1, 300:] == 0.0)
    assert result.values[0, 180] == pytest.approx(12.5 / 30.0)


def test_frame_stack_repeats_first_frame_then_updates_chronologically() -> None:
    history = FrameStack(frame_count=4, channels=2, beams=360)
    first = np.stack(
        (
            np.full(360, 0.25, dtype=np.float32),
            np.ones(360, dtype=np.float32),
        )
    )
    second = first.copy()
    second[0] = 0.5

    history.append(first)
    assert history.stacked().shape == (4, 2, 360)
    np.testing.assert_array_equal(history.stacked()[0], first)
    np.testing.assert_array_equal(history.stacked()[-1], first)

    history.append(second)
    np.testing.assert_array_equal(history.stacked()[-2], first)
    np.testing.assert_array_equal(history.stacked()[-1], second)
    assert history.actor_input().shape == (8, 360)

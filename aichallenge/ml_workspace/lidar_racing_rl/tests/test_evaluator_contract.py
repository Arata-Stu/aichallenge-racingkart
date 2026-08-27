"""Host-only evaluator input contract tests."""

import pytest

from lidar_racing_rl.evaluation.evaluator import _validated_episode_count


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_episode_count_requires_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _validated_episode_count(value)


def test_episode_count_accepts_positive_integer() -> None:
    assert _validated_episode_count(3) == 3

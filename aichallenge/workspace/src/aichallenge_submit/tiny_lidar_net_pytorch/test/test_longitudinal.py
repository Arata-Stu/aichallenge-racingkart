import pytest

from tiny_lidar_net_pytorch.longitudinal import RuleBasedAccelerationController


def test_launches_at_full_acceleration_then_limits_at_upper_threshold():
    controller = RuleBasedAccelerationController()

    assert controller.command(0.0) == pytest.approx(1.0)
    assert controller.command(15.0 / 3.6) == pytest.approx(1.0)
    assert controller.command(16.0 / 3.6) == pytest.approx(0.7)


def test_hysteresis_prevents_switching_near_threshold():
    controller = RuleBasedAccelerationController()

    controller.command(17.0 / 3.6)
    assert controller.command(15.0 / 3.6) == pytest.approx(0.7)
    assert controller.command(14.0 / 3.6) == pytest.approx(1.0)


def test_uses_absolute_speed_for_reverse_motion():
    controller = RuleBasedAccelerationController()

    assert controller.command(-17.0 / 3.6) == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("threshold", "hysteresis"),
    [(0.0, 0.0), (15.0, -1.0), (15.0, 15.0)],
)
def test_rejects_invalid_speed_configuration(threshold, hysteresis):
    with pytest.raises(ValueError):
        RuleBasedAccelerationController(
            speed_threshold_kmh=threshold,
            speed_hysteresis_kmh=hysteresis,
        )

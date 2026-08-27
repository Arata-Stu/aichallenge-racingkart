import math

import pytest

from rsu_fusion_net_pytorch.geometry import quaternion_yaw, relative_rsu_meta


def test_relative_meta_is_in_ego_frame():
    result = relative_rsu_meta((10.0, 20.0, math.pi / 2), (13.0, 24.0, math.pi / 2), 0.05)
    assert result == pytest.approx([5.0, 4.0, -3.0, 0.0, 0.05])


def test_quaternion_yaw():
    assert quaternion_yaw(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)) == pytest.approx(math.pi / 2)

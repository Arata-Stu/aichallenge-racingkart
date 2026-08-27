"""Geometry shared by ROS callbacks and unit tests."""

from __future__ import annotations

import math


def relative_rsu_meta(
    ego_pose: tuple[float, float, float] | None,
    rsu_pose: tuple[float, float, float],
    age_sec: float,
) -> list[float]:
    if ego_pose is None:
        return [0.0, 0.0, 0.0, 0.0, float(age_sec)]
    dx = rsu_pose[0] - ego_pose[0]
    dy = rsu_pose[1] - ego_pose[1]
    yaw = ego_pose[2]
    relative_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    relative_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    relative_yaw = math.atan2(
        math.sin(rsu_pose[2] - yaw), math.cos(rsu_pose[2] - yaw)
    )
    return [math.hypot(dx, dy), relative_x, relative_y, relative_yaw, float(age_sec)]


def quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
import rclpy
from autoware_auto_planning_msgs.msg import PathPointWithLaneId, PathWithLaneId
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from lidar_trajectory_net_controller.lidar_trajectory_net_core import (
    LidarTrajectoryNetCore,
)


def stamp_to_nanoseconds(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def quaternion_to_yaw(quaternion) -> float:
    siny_cosp = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y
    )
    cosy_cosp = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z
    )
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    quaternion = Quaternion()
    quaternion.z = math.sin(yaw * 0.5)
    quaternion.w = math.cos(yaw * 0.5)
    return quaternion


class LidarTrajectoryNetNode(Node):
    def __init__(self):
        super().__init__("lidar_trajectory_net_controller")
        self._declare_parameters()

        self.debug = bool(self.get_parameter("debug").value)
        self.log_interval_sec = float(
            self.get_parameter("log_interval_sec").value
        )
        self.sync_tolerance_ns = int(
            float(self.get_parameter("scan_sync_tolerance_sec").value) * 1e9
        )
        self.max_history_gap_ns = int(
            float(self.get_parameter("max_history_gap_sec").value) * 1e9
        )
        self.max_odometry_age_ns = int(
            float(self.get_parameter("max_odometry_age_sec").value) * 1e9
        )
        self.path_frame_id = str(self.get_parameter("path.frame_id").value)
        self.target_velocity_mps = float(
            self.get_parameter("path.target_velocity_mps").value
        )
        self.include_current_pose = bool(
            self.get_parameter("path.include_current_pose").value
        )
        self.publish_debug_path = bool(
            self.get_parameter("path.publish_debug_path").value
        )
        self.lane_id = int(self.get_parameter("path.lane_id").value)

        self.core = LidarTrajectoryNetCore(
            ckpt_path=str(self.get_parameter("model.ckpt_path").value),
            device=str(self.get_parameter("model.device").value),
            use_checkpoint_config=bool(
                self.get_parameter("model.use_checkpoint_config").value
            ),
            input_channels=int(self.get_parameter("model.input_channels").value),
            input_dim=int(self.get_parameter("model.input_dim").value),
            history_length=int(self.get_parameter("model.history_length").value),
            history_stride=int(self.get_parameter("model.history_stride").value),
            future_num_points=int(
                self.get_parameter("model.future_num_points").value
            ),
            embed_dim=int(self.get_parameter("model.embed_dim").value),
            conv_channels=list(
                self.get_parameter("model.conv_channels").value
            ),
            conv_kernel_sizes=list(
                self.get_parameter("model.conv_kernel_sizes").value
            ),
            conv_strides=list(
                self.get_parameter("model.conv_strides").value
            ),
            transformer_layers=int(
                self.get_parameter("model.transformer_layers").value
            ),
            transformer_heads=int(
                self.get_parameter("model.transformer_heads").value
            ),
            transformer_ff_dim=int(
                self.get_parameter("model.transformer_ff_dim").value
            ),
            dropout=float(self.get_parameter("model.dropout").value),
            num_control_points=int(
                self.get_parameter("model.num_control_points").value
            ),
            output_scale=(
                float(self.get_parameter("model.output_scale_x").value),
                float(self.get_parameter("model.output_scale_y").value),
            ),
            max_range=float(self.get_parameter("max_range").value),
        )

        self.history = deque(maxlen=self.core.required_history_samples)
        self.latest_free_scan: Optional[LaserScan] = None
        self.latest_obstacle_scan: Optional[LaserScan] = None
        self.latest_odometry: Optional[Odometry] = None
        self.last_processed_scan_stamp_ns: Optional[int] = None
        self.last_history_stamp_ns: Optional[int] = None
        self.inference_times_ms = []
        self.last_log_time = time.monotonic()

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            LaserScan,
            "input/scan",
            self._free_scan_callback,
            scan_qos,
        )
        self.create_subscription(
            LaserScan,
            "input/scan_with_obstacles",
            self._obstacle_scan_callback,
            scan_qos,
        )
        self.create_subscription(
            Odometry,
            "input/odometry",
            self._odometry_callback,
            scan_qos,
        )
        self.path_publisher = self.create_publisher(
            PathWithLaneId,
            "output/path",
            path_qos,
        )
        self.debug_path_publisher = self.create_publisher(
            Path,
            "output/debug_path",
            path_qos,
        )

        self.get_logger().info(
            "LiDAR Trajectory Net ready: "
            f"device={self.core.device}, history={self.core.history_length}, "
            f"history_stride={self.core.history_stride}, "
            f"future_points={self.core.future_num_points}, "
            f"input_dim={self.core.input_dim}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("model.ckpt_path", "")
        self.declare_parameter("model.device", "auto")
        self.declare_parameter("model.use_checkpoint_config", True)
        self.declare_parameter("model.input_channels", 3)
        self.declare_parameter("model.input_dim", 1080)
        self.declare_parameter("model.history_length", 8)
        self.declare_parameter("model.history_stride", 1)
        self.declare_parameter("model.future_num_points", 25)
        self.declare_parameter("model.embed_dim", 128)
        self.declare_parameter("model.conv_channels", [32, 64, 128])
        self.declare_parameter("model.conv_kernel_sizes", [9, 7, 5])
        self.declare_parameter("model.conv_strides", [4, 4, 2])
        self.declare_parameter("model.transformer_layers", 2)
        self.declare_parameter("model.transformer_heads", 4)
        self.declare_parameter("model.transformer_ff_dim", 512)
        self.declare_parameter("model.dropout", 0.1)
        self.declare_parameter("model.num_control_points", 3)
        self.declare_parameter("model.output_scale_x", 40.0)
        self.declare_parameter("model.output_scale_y", 12.0)
        self.declare_parameter("max_range", 30.0)
        self.declare_parameter("scan_sync_tolerance_sec", 0.01)
        self.declare_parameter("max_history_gap_sec", 0.2)
        self.declare_parameter("max_odometry_age_sec", 0.2)
        self.declare_parameter("path.frame_id", "map")
        self.declare_parameter("path.target_velocity_mps", 3.0)
        self.declare_parameter("path.include_current_pose", True)
        self.declare_parameter("path.publish_debug_path", True)
        self.declare_parameter("path.lane_id", -1)
        self.declare_parameter("log_interval_sec", 5.0)
        self.declare_parameter("debug", False)

    def _free_scan_callback(self, message: LaserScan) -> None:
        self.latest_free_scan = message
        self._try_inference()

    def _obstacle_scan_callback(self, message: LaserScan) -> None:
        self.latest_obstacle_scan = message
        self._try_inference()

    def _odometry_callback(self, message: Odometry) -> None:
        self.latest_odometry = message
        self._try_inference()

    def _try_inference(self) -> None:
        if (
            self.latest_free_scan is None
            or self.latest_obstacle_scan is None
            or self.latest_odometry is None
        ):
            return

        free_stamp_ns = stamp_to_nanoseconds(self.latest_free_scan.header.stamp)
        obstacle_stamp_ns = stamp_to_nanoseconds(
            self.latest_obstacle_scan.header.stamp
        )
        if abs(free_stamp_ns - obstacle_stamp_ns) > self.sync_tolerance_ns:
            return
        if free_stamp_ns == self.last_processed_scan_stamp_ns:
            return

        odometry_stamp_ns = stamp_to_nanoseconds(
            self.latest_odometry.header.stamp
        )
        if (
            self.max_odometry_age_ns > 0
            and free_stamp_ns > 0
            and odometry_stamp_ns > 0
            and abs(free_stamp_ns - odometry_stamp_ns)
            > self.max_odometry_age_ns
        ):
            return

        if (
            self.last_history_stamp_ns is not None
            and (
                free_stamp_ns <= self.last_history_stamp_ns
                or free_stamp_ns - self.last_history_stamp_ns
                > self.max_history_gap_ns
            )
        ):
            self.history.clear()

        frame = self.core.preprocess_pair(
            np.asarray(self.latest_free_scan.ranges, dtype=np.float32),
            np.asarray(self.latest_obstacle_scan.ranges, dtype=np.float32),
        )
        self.history.append(frame)
        self.last_processed_scan_stamp_ns = free_stamp_ns
        self.last_history_stamp_ns = free_stamp_ns

        if len(self.history) < self.core.required_history_samples:
            return

        started = time.perf_counter()
        prediction = self.core.predict(np.asarray(self.history, dtype=np.float32))
        path, debug_path = self._make_path_messages(
            prediction,
            self.latest_odometry,
            self.latest_free_scan.header.stamp,
        )
        self.path_publisher.publish(path)
        if self.publish_debug_path:
            self.debug_path_publisher.publish(debug_path)

        if self.debug:
            self.inference_times_ms.append(
                (time.perf_counter() - started) * 1000.0
            )
            self._log_performance()

    def _make_path_messages(
        self,
        ego_prediction: np.ndarray,
        odometry: Odometry,
        stamp,
    ) -> Tuple[PathWithLaneId, Path]:
        origin_pose = odometry.pose.pose
        origin_yaw = quaternion_to_yaw(origin_pose.orientation)
        cos_yaw = math.cos(origin_yaw)
        sin_yaw = math.sin(origin_yaw)

        ego_points = np.asarray(ego_prediction, dtype=np.float64)
        if self.include_current_pose:
            ego_points = np.concatenate(
                [np.zeros((1, 2), dtype=np.float64), ego_points],
                axis=0,
            )

        map_points = np.empty_like(ego_points)
        map_points[:, 0] = (
            origin_pose.position.x
            + cos_yaw * ego_points[:, 0]
            - sin_yaw * ego_points[:, 1]
        )
        map_points[:, 1] = (
            origin_pose.position.y
            + sin_yaw * ego_points[:, 0]
            + cos_yaw * ego_points[:, 1]
        )
        path_yaws = self._calculate_path_yaws(map_points, origin_yaw)

        path = PathWithLaneId()
        path.header.stamp = stamp
        path.header.frame_id = self.path_frame_id

        debug_path = Path()
        debug_path.header.stamp = stamp
        debug_path.header.frame_id = self.path_frame_id

        for (x, y), yaw in zip(map_points, path_yaws):
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = float(origin_pose.position.z)
            pose.orientation = yaw_to_quaternion(float(yaw))

            path_point = PathPointWithLaneId()
            path_point.point.pose = pose
            path_point.point.longitudinal_velocity_mps = float(
                self.target_velocity_mps
            )
            if self.lane_id >= 0:
                path_point.lane_ids = [self.lane_id]
            path.points.append(path_point)

            pose_stamped = PoseStamped()
            pose_stamped.header = debug_path.header
            pose_stamped.pose = pose
            debug_path.poses.append(pose_stamped)

        return path, debug_path

    @staticmethod
    def _calculate_path_yaws(
        points: np.ndarray,
        fallback_yaw: float,
    ) -> np.ndarray:
        yaws = np.full(len(points), fallback_yaw, dtype=np.float64)
        if len(points) < 2:
            return yaws

        for index in range(len(points)):
            if index + 1 < len(points):
                delta = points[index + 1] - points[index]
            else:
                delta = points[index] - points[index - 1]
            if np.hypot(delta[0], delta[1]) > 1e-6:
                yaws[index] = math.atan2(delta[1], delta[0])
        return yaws

    def _log_performance(self) -> None:
        now = time.monotonic()
        if now - self.last_log_time < self.log_interval_sec:
            return
        if self.inference_times_ms:
            average_ms = float(np.mean(self.inference_times_ms))
            maximum_ms = float(np.max(self.inference_times_ms))
            self.get_logger().info(
                f"Inference avg={average_ms:.2f} ms, "
                f"max={maximum_ms:.2f} ms, "
                f"rate={1000.0 / average_ms:.2f} Hz"
            )
            self.inference_times_ms.clear()
        self.last_log_time = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarTrajectoryNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

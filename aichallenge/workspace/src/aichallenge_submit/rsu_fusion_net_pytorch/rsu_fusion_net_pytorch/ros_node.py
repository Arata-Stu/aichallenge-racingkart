#!/usr/bin/env python3
"""Run a learned acceleration/steering RSU fusion policy in ROS 2."""

from __future__ import annotations

from collections import deque
import math
import time
from typing import Any

import rclpy
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import VelocityReport
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import torch

from rsu_fusion_net_pytorch.geometry import quaternion_yaw, relative_rsu_meta
from rsu_fusion_net_pytorch.policy import RsuFusionTorchPolicy


DEFAULT_RSU_TOPICS = [f"/rsu/curve_{index:02d}/scan" for index in range(1, 7)]
DEFAULT_RSU_POSES = [
    89620.233296, 43157.347478, 111.0,
    89639.094992, 43147.310706, -60.0,
    89629.923709, 43179.151500, 110.0,
    89655.187999, 43180.362834, 30.0,
    89655.014956, 43167.730346, -130.0,
    89665.916671, 43154.751761, 5.0,
]


class RsuFusionNetPyTorchNode(Node):
    def __init__(self) -> None:
        super().__init__("rsu_fusion_net_pytorch_node")
        self.declare_parameter("model.checkpoint_path", "")
        self.declare_parameter("device", "auto")
        self.declare_parameter("ego_scan_topic", "/sensing/lidar/scan")
        self.declare_parameter("rsu_topics", DEFAULT_RSU_TOPICS)
        self.declare_parameter("rsu_poses", DEFAULT_RSU_POSES)
        self.declare_parameter("pose_topic", "/localization/pose_with_covariance")
        self.declare_parameter("velocity_topic", "/vehicle/status/velocity_status")
        self.declare_parameter("control_cmd_topic", "/control/command/control_cmd")
        self.declare_parameter("rsu_timeout_sec", 0.2)
        self.declare_parameter("control_mode", "ai")
        self.declare_parameter("acceleration_scale", 1.0)
        self.declare_parameter("steering_scale", 1.0)
        self.declare_parameter("debug", False)
        self.declare_parameter("log_interval_sec", 5.0)
        self.declare_parameter("trajectory_frame", "base_link")
        self.declare_parameter("selected_trajectory_topic", "~/selected_trajectory")
        self.declare_parameter("candidate_trajectories_topic", "~/candidate_trajectories")
        self.declare_parameter("mode_probabilities_topic", "~/mode_probabilities")

        self.policy = RsuFusionTorchPolicy(
            str(self.get_parameter("model.checkpoint_path").value),
            device=str(self.get_parameter("device").value),
        )
        requested_control_mode = str(self.get_parameter("control_mode").value).lower()
        if requested_control_mode not in {"auto", "ai"}:
            raise ValueError("control_mode must be 'auto' or 'ai'; fixed acceleration is prohibited")
        # auto intentionally no longer falls back to fixed acceleration.
        self.control_mode = "ai" if requested_control_mode == "auto" else requested_control_mode
        if self.control_mode == "ai" and not self.policy.learns_acceleration:
            raise ValueError(
                "This checkpoint was trained with acceleration_weight=0. "
                "Retrain with acceleration loss; fixed acceleration is not supported."
            )
        self.acceleration_scale = float(self.get_parameter("acceleration_scale").value)
        self.steering_scale = float(self.get_parameter("steering_scale").value)
        self.rsu_timeout_sec = float(self.get_parameter("rsu_timeout_sec").value)
        if self.rsu_timeout_sec <= 0.0:
            raise ValueError("rsu_timeout_sec must be positive")

        self.rsu_topics = [str(value) for value in self.get_parameter("rsu_topics").value]
        if len(self.rsu_topics) != self.policy.rsu_count:
            raise ValueError(
                f"Checkpoint expects {self.policy.rsu_count} RSUs, got {len(self.rsu_topics)} topics"
            )
        flat_poses = [float(value) for value in self.get_parameter("rsu_poses").value]
        if len(flat_poses) != len(self.rsu_topics) * 3:
            raise ValueError("rsu_poses must contain x,y,yaw_deg for every RSU topic")
        self.rsu_poses = [
            (flat_poses[i], flat_poses[i + 1], math.radians(flat_poses[i + 2]))
            for i in range(0, len(flat_poses), 3)
        ]

        self.ego_history: deque[torch.Tensor] = deque(maxlen=self.policy.history_len)
        self.rsu_history: deque[list[torch.Tensor]] = deque(maxlen=self.policy.history_len)
        self.latest_rsu: list[tuple[float, torch.Tensor] | None] = [None] * len(self.rsu_topics)
        self.latest_pose: tuple[float, float, float] | None = None
        self.latest_velocity_mps = 0.0
        self.empty_scan = torch.ones(self.policy.scan_dim, dtype=torch.float32)
        self.debug = bool(self.get_parameter("debug").value)
        self.log_interval_sec = float(self.get_parameter("log_interval_sec").value)
        self.last_log_time = time.monotonic()
        self.inference_times_ms: list[float] = []

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        ego_topic = str(self.get_parameter("ego_scan_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        velocity_topic = str(self.get_parameter("velocity_topic").value)
        command_topic = str(self.get_parameter("control_cmd_topic").value)
        self.ego_subscription = self.create_subscription(LaserScan, ego_topic, self.on_ego_scan, scan_qos)
        self.rsu_subscriptions = [
            self.create_subscription(
                LaserScan, topic, self.make_rsu_callback(index), scan_qos
            )
            for index, topic in enumerate(self.rsu_topics)
        ]
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped, pose_topic, self.on_pose, 1
        )
        self.velocity_subscription = self.create_subscription(
            VelocityReport, velocity_topic, self.on_velocity, 1
        )
        self.control_publisher = self.create_publisher(AckermannControlCommand, command_topic, 1)
        self.trajectory_publisher = self.create_publisher(
            Path, str(self.get_parameter("selected_trajectory_topic").value), 1
        )
        self.candidates_publisher = self.create_publisher(
            MarkerArray, str(self.get_parameter("candidate_trajectories_topic").value), 1
        )
        self.mode_publisher = self.create_publisher(
            Float32MultiArray, str(self.get_parameter("mode_probabilities_topic").value), 1
        )
        self.trajectory_frame = str(self.get_parameter("trajectory_frame").value)
        self.get_logger().info(
            f"Ready: checkpoint={self.policy.checkpoint_path}, device={self.policy.device}, "
            f"history={self.policy.history_len}, control_mode={self.control_mode}, ego={ego_topic}"
        )

    def make_rsu_callback(self, index: int) -> Any:
        def callback(message: LaserScan) -> None:
            self.latest_rsu[index] = (
                time.monotonic(), self.policy.preprocess_scan(message.ranges)
            )
        return callback

    def on_pose(self, message: PoseWithCovarianceStamped) -> None:
        pose = message.pose.pose
        orientation = pose.orientation
        self.latest_pose = (
            float(pose.position.x),
            float(pose.position.y),
            quaternion_yaw(orientation.x, orientation.y, orientation.z, orientation.w),
        )

    def on_velocity(self, message: VelocityReport) -> None:
        self.latest_velocity_mps = abs(float(message.longitudinal_velocity))

    def on_ego_scan(self, message: LaserScan) -> None:
        start = time.perf_counter()
        now = time.monotonic()
        rsu_frame: list[torch.Tensor] = []
        rsu_meta: list[list[float]] = []
        rsu_mask: list[bool] = []
        for index, latest in enumerate(self.latest_rsu):
            age = self.rsu_timeout_sec + 1.0 if latest is None else max(0.0, now - latest[0])
            valid = latest is not None and age <= self.rsu_timeout_sec
            rsu_frame.append(latest[1] if valid and latest is not None else self.empty_scan)
            rsu_meta.append(relative_rsu_meta(self.latest_pose, self.rsu_poses[index], age))
            rsu_mask.append(valid)

        self.ego_history.append(self.policy.preprocess_scan(message.ranges))
        self.rsu_history.append(rsu_frame)
        if len(self.ego_history) < self.policy.history_len:
            return

        prediction = self.policy.predict_full(
            self.ego_history, self.rsu_history, rsu_meta, rsu_mask,
            ego_speed=self.latest_velocity_mps,
        )
        acceleration, steering, gates = (
            prediction.acceleration, prediction.steering, prediction.gates
        )
        acceleration *= self.acceleration_scale
        steering *= self.steering_scale

        command = AckermannControlCommand()
        command.stamp = self.get_clock().now().to_msg()
        command.longitudinal.acceleration = float(acceleration)
        command.lateral.steering_tire_angle = float(steering)
        self.control_publisher.publish(command)
        if prediction.trajectories:
            self.publish_trajectories(prediction, command.stamp)

        if self.debug:
            self.inference_times_ms.append((time.perf_counter() - start) * 1000.0)
            if now - self.last_log_time >= self.log_interval_sec:
                average = sum(self.inference_times_ms) / len(self.inference_times_ms)
                active = sum(rsu_mask)
                self.get_logger().info(
                    f"inference={average:.2f}ms active_rsu={active}/{len(rsu_mask)} "
                    f"accel={acceleration:.3f} steer={steering:.3f} gates={gates}"
                )
                self.inference_times_ms.clear()
                self.last_log_time = now

    def publish_trajectories(self, prediction: Any, stamp: Any) -> None:
        selected = prediction.trajectories[prediction.selected_mode]
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.trajectory_frame
        for x, y, _speed in selected:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.trajectory_publisher.publish(path)

        markers = MarkerArray()
        colors = ((0.33, 0.90, 0.82), (0.79, 0.98, 0.25), (1.0, 0.61, 0.32), (0.71, 0.55, 1.0))
        for mode, trajectory in enumerate(prediction.trajectories):
            marker = Marker()
            marker.header = path.header
            marker.ns = "rsu_trajectory_candidates"
            marker.id = mode
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.18 if mode == prediction.selected_mode else 0.06
            red, green, blue = colors[mode % len(colors)]
            marker.color.r, marker.color.g, marker.color.b = red, green, blue
            marker.color.a = 1.0 if mode == prediction.selected_mode else 0.35
            marker.points = [Point(x=float(x), y=float(y), z=0.08) for x, y, _ in trajectory]
            markers.markers.append(marker)
        self.candidates_publisher.publish(markers)
        probability_message = Float32MultiArray()
        probability_message.data = [float(value) for value in prediction.mode_probabilities]
        self.mode_publisher.publish(probability_message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: RsuFusionNetPyTorchNode | None = None
    try:
        node = RsuFusionNetPyTorchNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

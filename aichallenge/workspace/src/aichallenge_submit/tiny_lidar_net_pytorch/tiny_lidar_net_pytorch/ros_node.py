#!/usr/bin/env python3
"""ROS 2 node running TinyLiDARNet directly with PyTorch."""

from __future__ import annotations

import time

import rclpy
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import VelocityReport
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from tiny_lidar_net_pytorch.longitudinal import RuleBasedAccelerationController
from tiny_lidar_net_pytorch.policy import TinyLidarTorchPolicy


class TinyLidarNetPyTorchNode(Node):
    def __init__(self) -> None:
        super().__init__("tiny_lidar_net_pytorch_node")
        self.declare_parameter("model.checkpoint_path", "")
        self.declare_parameter("model.architecture", "")
        self.declare_parameter("model.input_dim", 0)
        self.declare_parameter("model.output_dim", 0)
        self.declare_parameter("device", "auto")
        self.declare_parameter("max_range", 0.0)
        self.declare_parameter("control_mode", "rule_based")
        self.declare_parameter("fixed_acceleration", 0.7)
        self.declare_parameter("startup_acceleration", 1.0)
        self.declare_parameter("cruise_acceleration", 0.7)
        self.declare_parameter("startup_speed_threshold_kmh", 15.0)
        self.declare_parameter("speed_hysteresis_kmh", 1.0)
        self.declare_parameter("velocity_timeout_sec", 0.5)
        self.declare_parameter("acceleration_scale", 1.0)
        self.declare_parameter("steering_scale", 1.0)
        self.declare_parameter("debug", False)
        self.declare_parameter("log_interval_sec", 5.0)

        checkpoint_path = str(self.get_parameter("model.checkpoint_path").value)
        architecture_value = str(self.get_parameter("model.architecture").value)
        input_dim_value = int(self.get_parameter("model.input_dim").value)
        output_dim_value = int(self.get_parameter("model.output_dim").value)
        max_range_value = float(self.get_parameter("max_range").value)
        self.policy = TinyLidarTorchPolicy(
            checkpoint_path,
            device=str(self.get_parameter("device").value),
            architecture=architecture_value or None,
            input_dim=input_dim_value or None,
            output_dim=output_dim_value or None,
            max_range=max_range_value or None,
        )
        self.control_mode = str(self.get_parameter("control_mode").value).lower()
        if self.control_mode not in {"ai", "fixed", "rule_based"}:
            raise ValueError("control_mode must be 'ai', 'fixed', or 'rule_based'")
        self.fixed_acceleration = float(self.get_parameter("fixed_acceleration").value)
        self.cruise_acceleration = float(self.get_parameter("cruise_acceleration").value)
        self.velocity_timeout_sec = float(self.get_parameter("velocity_timeout_sec").value)
        if self.velocity_timeout_sec < 0.0:
            raise ValueError("velocity_timeout_sec must be non-negative")
        self.longitudinal_controller = RuleBasedAccelerationController(
            startup_acceleration=float(self.get_parameter("startup_acceleration").value),
            cruise_acceleration=self.cruise_acceleration,
            speed_threshold_kmh=float(
                self.get_parameter("startup_speed_threshold_kmh").value
            ),
            speed_hysteresis_kmh=float(self.get_parameter("speed_hysteresis_kmh").value),
        )
        self.latest_velocity_mps: float | None = None
        self.latest_velocity_time: float | None = None
        self.acceleration_scale = float(self.get_parameter("acceleration_scale").value)
        self.steering_scale = float(self.get_parameter("steering_scale").value)
        self.debug = bool(self.get_parameter("debug").value)
        self.log_interval_sec = float(self.get_parameter("log_interval_sec").value)
        self.inference_times_ms: list[float] = []
        self.last_log_time = time.monotonic()

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.scan_subscription = self.create_subscription(LaserScan, "/scan", self.on_scan, scan_qos)
        self.velocity_subscription = self.create_subscription(
            VelocityReport, "/velocity", self.on_velocity, 1
        )
        self.control_publisher = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1
        )
        self.get_logger().info(
            f"Ready: checkpoint={self.policy.checkpoint_path}, device={self.policy.device}, "
            f"architecture={self.policy.architecture}, input_dim={self.policy.input_dim}, "
            f"control_mode={self.control_mode}"
        )

    def on_velocity(self, message: VelocityReport) -> None:
        self.latest_velocity_mps = float(message.longitudinal_velocity)
        self.latest_velocity_time = time.monotonic()

    def rule_based_acceleration(self) -> float:
        now = time.monotonic()
        velocity_is_fresh = (
            self.latest_velocity_mps is not None
            and self.latest_velocity_time is not None
            and now - self.latest_velocity_time <= self.velocity_timeout_sec
        )
        if not velocity_is_fresh:
            # Still allow the vehicle to start if velocity feedback is temporarily absent.
            return self.cruise_acceleration
        return self.longitudinal_controller.command(self.latest_velocity_mps)

    def on_scan(self, message: LaserScan) -> None:
        start_time = time.perf_counter()
        acceleration, steering = self.policy.predict(message.ranges)
        if self.control_mode == "fixed":
            acceleration = self.fixed_acceleration
        elif self.control_mode == "rule_based":
            acceleration = self.rule_based_acceleration()
        else:
            acceleration *= self.acceleration_scale
        steering *= self.steering_scale

        command = AckermannControlCommand()
        command.stamp = self.get_clock().now().to_msg()
        command.longitudinal.acceleration = float(acceleration)
        command.lateral.steering_tire_angle = float(steering)
        self.control_publisher.publish(command)

        if self.debug:
            self.inference_times_ms.append((time.perf_counter() - start_time) * 1000.0)
            self.log_performance_if_due()

    def log_performance_if_due(self) -> None:
        now = time.monotonic()
        if now - self.last_log_time < self.log_interval_sec:
            return
        if self.inference_times_ms:
            average = sum(self.inference_times_ms) / len(self.inference_times_ms)
            frequency = 1000.0 / average if average > 0.0 else 0.0
            self.get_logger().info(
                f"PyTorch inference avg={average:.2f} ms ({frequency:.1f} Hz), "
                f"max={max(self.inference_times_ms):.2f} ms"
            )
            self.inference_times_ms.clear()
        self.last_log_time = now


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: TinyLidarNetPyTorchNode | None = None
    try:
        node = TinyLidarNetPyTorchNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

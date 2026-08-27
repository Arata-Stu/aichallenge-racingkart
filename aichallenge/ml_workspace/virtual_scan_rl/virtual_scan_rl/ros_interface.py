"""ROS 2 I/O for the Gymnasium environment."""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import SteeringReport, VelocityReport
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import Empty, Float32MultiArray, String

from .intervention import joy_action


class VirtualScanRosInterface(Node):
    def __init__(self, ros_cfg: dict, joy_cfg: dict) -> None:
        super().__init__(
            str(ros_cfg["node_name"]),
            parameter_overrides=[Parameter("use_sim_time", value=bool(ros_cfg["use_sim_time"]))],
        )
        self.ros_cfg = ros_cfg
        self.joy_cfg = joy_cfg
        self.scan_ranges: tuple[float, ...] | None = None
        self.scan_sequence = 0
        self.pose: tuple[float, float, float] | None = None
        self.pose_sequence = 0
        self.speed_mps = 0.0
        self.yaw_rate_rad_s = 0.0
        self.steering_rad = 0.0
        self.awsim_status: dict[str, float] = {}
        self.awsim_state = ""
        self.awsim_state_sequence = 0
        self.admin_state = ""
        self.admin_state_sequence = 0
        self.joy_axes: tuple[float, ...] = ()
        self.joy_buttons: tuple[int, ...] = ()
        self.joy_received_at = 0.0

        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(LaserScan, ros_cfg["scan_topic"], self._scan_cb, best_effort)
        self.create_subscription(
            PoseWithCovarianceStamped, ros_cfg["pose_topic"], self._pose_cb, reliable
        )
        self.create_subscription(VelocityReport, ros_cfg["velocity_topic"], self._velocity_cb, reliable)
        self.create_subscription(SteeringReport, ros_cfg["steering_topic"], self._steering_cb, reliable)
        self.create_subscription(Float32MultiArray, ros_cfg["status_topic"], self._status_cb, reliable)
        self.create_subscription(String, ros_cfg["state_topic"], self._state_cb, reliable)
        self.create_subscription(
            String, ros_cfg["admin_state_topic"], self._admin_state_cb, reliable
        )
        if bool(joy_cfg["enabled"]):
            self.create_subscription(Joy, ros_cfg["joy_topic"], self._joy_cb, reliable)
        self.control_publisher = self.create_publisher(
            AckermannControlCommand, ros_cfg["control_topic"], reliable
        )
        self.reset_publisher = self.create_publisher(Empty, ros_cfg["reset_topic"], reliable)

    def _scan_cb(self, message: LaserScan) -> None:
        self.scan_ranges = tuple(message.ranges)
        self.scan_sequence += 1

    def _pose_cb(self, message: PoseWithCovarianceStamped) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        siny = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.pose = (float(position.x), float(position.y), float(math.atan2(siny, cosy)))
        self.pose_sequence += 1

    def _velocity_cb(self, message: VelocityReport) -> None:
        self.speed_mps = float(message.longitudinal_velocity)
        self.yaw_rate_rad_s = float(message.heading_rate)

    def _steering_cb(self, message: SteeringReport) -> None:
        self.steering_rad = float(message.steering_tire_angle)

    def _status_cb(self, message: Float32MultiArray) -> None:
        keys = ("session_time", "lap_count", "lap_time", "section", "time_scale")
        self.awsim_status = {
            key: float(message.data[index]) if index < len(message.data) else 0.0
            for index, key in enumerate(keys)
        }

    def _state_cb(self, message: String) -> None:
        state = str(message.data)
        if state != self.awsim_state:
            self.get_logger().info(f"AWSIM vehicle state: {state}")
        self.awsim_state = state
        self.awsim_state_sequence += 1

    def _admin_state_cb(self, message: String) -> None:
        state = str(message.data)
        if state != self.admin_state:
            self.get_logger().info(f"AWSIM admin state: {state}")
        self.admin_state = state
        self.admin_state_sequence += 1

    def _joy_cb(self, message: Joy) -> None:
        self.joy_axes = tuple(message.axes)
        self.joy_buttons = tuple(message.buttons)
        self.joy_received_at = time.monotonic()

    def human_action(self) -> tuple[bool, np.ndarray | None]:
        if not bool(self.joy_cfg["enabled"]):
            return False, None
        if time.monotonic() - self.joy_received_at > float(self.joy_cfg["timeout_sec"]):
            return False, None
        button = int(self.joy_cfg["hold_button_index"])
        active = 0 <= button < len(self.joy_buttons) and self.joy_buttons[button] == 1
        if not active:
            return False, None
        action = joy_action(
            self.joy_axes,
            steer_axis=int(self.joy_cfg["steer_axis_index"]),
            positive_axis=int(self.joy_cfg["positive_throttle_axis_index"]),
            negative_axis=int(self.joy_cfg["negative_throttle_axis_index"]),
            deadzone=float(self.joy_cfg["deadzone"]),
        )
        return True, action

    def publish_action(self, steering_rad: float, acceleration_mps2: float) -> None:
        message = AckermannControlCommand()
        stamp = self.get_clock().now().to_msg()
        message.stamp = stamp
        message.lateral.stamp = stamp
        message.longitudinal.stamp = stamp
        message.lateral.steering_tire_angle = float(steering_rad)
        message.lateral.steering_tire_rotation_rate = 1.0
        message.longitudinal.acceleration = float(acceleration_mps2)
        self.control_publisher.publish(message)

    def stop(self) -> None:
        self.publish_action(0.0, 0.0)

    def reset_awsim(self) -> None:
        self.reset_publisher.publish(Empty())

    def spin_for(self, duration_sec: float) -> None:
        deadline = time.monotonic() + max(0.0, duration_sec)
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, deadline - time.monotonic()))

    def wait_for_scan_after(self, sequence: int, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while self.scan_sequence <= sequence and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.02, deadline - time.monotonic()))
        return self.scan_sequence > sequence

    def wait_for_episode_start_after(
        self,
        *,
        vehicle_sequence: int,
        admin_sequence: int,
        target_state: str,
        vehicle_ready_states: list[str],
        timeout_sec: float,
    ) -> bool:
        """Wait for global Start and a post-reset ready vehicle state."""
        target = target_state.strip().casefold()
        ready_states = {state.strip().casefold() for state in vehicle_ready_states}
        deadline = time.monotonic() + timeout_sec
        saw_admin_pre_start = False
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, deadline - time.monotonic()))
            if self.admin_state_sequence > admin_sequence:
                saw_admin_pre_start |= self.admin_state.strip().casefold() != target
            vehicle_ready = (
                self.awsim_state_sequence > vehicle_sequence
                and self.awsim_state.strip().casefold() in ready_states
            )
            admin_started = (
                saw_admin_pre_start
                and self.admin_state_sequence > admin_sequence
                and self.admin_state.strip().casefold() == target
            )
            if vehicle_ready and admin_started:
                return True
        return False

    def wait_for_observation_after(
        self, *, scan_sequence: int, pose_sequence: int, timeout_sec: float
    ) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self.scan_sequence > scan_sequence and self.pose_sequence > pose_sequence:
                return True
            rclpy.spin_once(self, timeout_sec=min(0.02, deadline - time.monotonic()))
        return self.scan_sequence > scan_sequence and self.pose_sequence > pose_sequence

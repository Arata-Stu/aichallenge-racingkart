import copy
from math import isfinite
from typing import Optional, Tuple

from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import GearCommand, GearReport, VelocityReport
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Empty, String

from .state_machine import (
    RecoveryConfig,
    RecoveryState,
    RecoveryStateMachine,
    is_stuck_candidate,
)


class StuckRecoveryController(Node):
    """Gate MPC output and temporarily override it with a reverse maneuver."""

    def __init__(self) -> None:
        super().__init__('stuck_recovery_controller')
        self._declare_parameters()

        config = RecoveryConfig(
            startup_grace_period=self._float_param('startup_grace_period'),
            stuck_timeout=self._float_param('stuck_timeout'),
            stop_velocity_threshold=self._float_param('stop_velocity_threshold'),
            stop_hold_duration=self._float_param('stop_hold_duration'),
            gear_settle_duration=self._float_param('gear_settle_duration'),
            gear_feedback_timeout=self._float_param('gear_feedback_timeout'),
            reverse_duration=self._float_param('reverse_duration'),
            reverse_distance=self._float_param('reverse_distance'),
            cooldown_duration=self._float_param('cooldown_duration'),
            drive_gear=GearCommand.DRIVE,
            reverse_gear=GearCommand.REVERSE,
        )
        self._machine = RecoveryStateMachine(config)

        self._command_timeout = self._float_param('command_timeout')
        self._sensor_timeout = self._float_param('sensor_timeout')
        self._infeasible_timeout = self._float_param('infeasible_timeout')
        self._stuck_velocity_threshold = self._float_param('stuck_velocity_threshold')
        self._stuck_command_speed_threshold = self._float_param('stuck_command_speed_threshold')
        self._stuck_command_acceleration_threshold = self._float_param(
            'stuck_command_acceleration_threshold'
        )
        self._reverse_speed = self._float_param('reverse_speed')
        self._reverse_acceleration = self._float_param('reverse_acceleration')
        self._reverse_steering_scale = self._float_param('reverse_steering_scale')
        self._reverse_max_steering = self._float_param('reverse_max_steering')
        self._braking_acceleration = self._float_param('braking_acceleration')

        self._nominal_command: Optional[AckermannControlCommand] = None
        self._nominal_received_at: Optional[float] = None
        self._velocity: Optional[float] = None
        self._velocity_received_at: Optional[float] = None
        self._position: Optional[Tuple[float, float]] = None
        self._position_received_at: Optional[float] = None
        self._gear: Optional[int] = None
        self._gear_received_at: Optional[float] = None
        self._infeasible_received_at: Optional[float] = None
        self._manual_trigger = False
        self._recovery_steering = 0.0
        self._stuck_candidate_source = 'none'

        self._control_pub = self.create_publisher(
            AckermannControlCommand, '/control/command/control_cmd', 1
        )
        self._gear_pub = self.create_publisher(
            GearCommand, '/control/command/gear_cmd', 1
        )
        state_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._state_pub = self.create_publisher(
            String, '/control/recovery/state', state_qos
        )

        self.create_subscription(
            AckermannControlCommand,
            '/control/command/control_cmd_mpc',
            self._on_nominal_command,
            1,
        )
        self.create_subscription(
            Bool,
            '/control/mpc/infeasible',
            self._on_infeasible,
            1,
        )
        self.create_subscription(
            VelocityReport,
            '/vehicle/status/velocity_status',
            self._on_velocity,
            1,
        )
        self.create_subscription(
            Odometry,
            '/localization/kinematic_state',
            self._on_odometry,
            1,
        )
        self.create_subscription(
            GearReport,
            '/vehicle/status/gear_status',
            self._on_gear,
            1,
        )
        self.create_subscription(
            Empty,
            '/control/recovery/trigger',
            self._on_manual_trigger,
            1,
        )

        publish_rate = self._float_param('publish_rate')
        if publish_rate <= 0.0:
            raise ValueError('publish_rate must be positive')
        self.create_timer(1.0 / publish_rate, self._on_timer)
        self.get_logger().info(
            'Stuck recovery ready: MPC is gated through /control/command/control_cmd_mpc'
        )

    def _declare_parameters(self) -> None:
        defaults = {
            'publish_rate': 40.0,
            'command_timeout': 0.5,
            'sensor_timeout': 0.5,
            'infeasible_timeout': 0.75,
            'startup_grace_period': 5.0,
            'stuck_timeout': 2.5,
            'stuck_velocity_threshold': 0.20,
            'stuck_command_speed_threshold': 1.0,
            'stuck_command_acceleration_threshold': 0.30,
            'stop_velocity_threshold': 0.08,
            'stop_hold_duration': 0.30,
            'gear_settle_duration': 0.30,
            'gear_feedback_timeout': 1.0,
            'reverse_speed': -2.0,
            'reverse_acceleration': 1.0,
            'reverse_duration': 2.5,
            'reverse_distance': 1.5,
            'reverse_steering_scale': 1.0,
            'reverse_max_steering': 0.50,
            'braking_acceleration': -2.5,
            'cooldown_duration': 4.0,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

    def _float_param(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not isfinite(value):
            raise ValueError(f'{name} must be finite')
        return value

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_nominal_command(self, msg: AckermannControlCommand) -> None:
        self._nominal_command = msg
        self._nominal_received_at = self._now_seconds()

    def _on_velocity(self, msg: VelocityReport) -> None:
        self._velocity = float(msg.longitudinal_velocity)
        self._velocity_received_at = self._now_seconds()

    def _on_infeasible(self, msg: Bool) -> None:
        # Keep only the timestamp of a positive observation.  A short hold
        # prevents one feasible solver retry from resetting the stuck timer.
        if msg.data:
            self._infeasible_received_at = self._now_seconds()

    def _on_odometry(self, msg: Odometry) -> None:
        self._position = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))
        self._position_received_at = self._now_seconds()

    def _on_gear(self, msg: GearReport) -> None:
        self._gear = int(msg.report)
        self._gear_received_at = self._now_seconds()

    def _on_manual_trigger(self, _msg: Empty) -> None:
        self._manual_trigger = True

    @staticmethod
    def _fresh(now: float, received_at: Optional[float], timeout: float) -> bool:
        return received_at is not None and 0.0 <= now - received_at <= timeout

    def _stuck_candidate(self, now: float) -> bool:
        command_is_fresh = self._fresh(
            now, self._nominal_received_at, self._command_timeout
        )
        infeasible_is_recent = self._fresh(
            now, self._infeasible_received_at, self._infeasible_timeout
        )
        command_speed = None
        command_acceleration = None
        if self._nominal_command is not None:
            command_speed = float(self._nominal_command.longitudinal.speed)
            command_acceleration = float(
                self._nominal_command.longitudinal.acceleration
            )

        candidate = is_stuck_candidate(
            velocity=self._velocity,
            velocity_is_fresh=self._fresh(
                now, self._velocity_received_at, self._sensor_timeout
            ),
            command_speed=command_speed,
            command_acceleration=command_acceleration,
            command_is_fresh=command_is_fresh,
            mpc_infeasible=infeasible_is_recent,
            velocity_threshold=self._stuck_velocity_threshold,
            command_speed_threshold=self._stuck_command_speed_threshold,
            command_acceleration_threshold=(
                self._stuck_command_acceleration_threshold
            ),
        )
        if candidate:
            self._stuck_candidate_source = (
                'mpc infeasible' if infeasible_is_recent else 'forward command'
            )
        else:
            self._stuck_candidate_source = 'none'
        return candidate

    def _fresh_position(self, now: float) -> Optional[Tuple[float, float]]:
        if self._fresh(now, self._position_received_at, self._sensor_timeout):
            return self._position
        return None

    def _fresh_gear(self, now: float) -> Optional[int]:
        if self._fresh(now, self._gear_received_at, self._sensor_timeout):
            return self._gear
        return None

    def _make_stop_command(self) -> AckermannControlCommand:
        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.speed = 0.0
        cmd.longitudinal.acceleration = self._braking_acceleration
        cmd.lateral.steering_tire_angle = 0.0
        return cmd

    def _make_reverse_command(self) -> AckermannControlCommand:
        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.speed = self._reverse_speed
        cmd.longitudinal.acceleration = self._reverse_acceleration
        cmd.lateral.steering_tire_angle = self._recovery_steering
        return cmd

    def _publish_gear(self, command: int) -> None:
        msg = GearCommand()
        msg.stamp = self.get_clock().now().to_msg()
        msg.command = command
        self._gear_pub.publish(msg)

    def _capture_recovery_steering(self) -> None:
        nominal_steering = 0.0
        if self._nominal_command is not None:
            nominal_steering = float(self._nominal_command.lateral.steering_tire_angle)
        requested = nominal_steering * self._reverse_steering_scale
        limit = abs(self._reverse_max_steering)
        self._recovery_steering = max(-limit, min(limit, requested))

    def _on_timer(self) -> None:
        now = self._now_seconds()
        previous = self._machine.state
        previous_stuck_since = self._machine.stuck_since
        stuck_candidate = self._stuck_candidate(now)
        transitioned = self._machine.update(
            now,
            stuck_candidate=stuck_candidate,
            velocity=self._velocity
            if self._fresh(now, self._velocity_received_at, self._sensor_timeout)
            else None,
            position=self._fresh_position(now),
            gear=self._fresh_gear(now),
            manual_trigger=self._manual_trigger,
        )
        self._manual_trigger = False

        if (
            previous == RecoveryState.NORMAL
            and previous_stuck_since is None
            and self._machine.stuck_since is not None
        ):
            self.get_logger().warn(
                f'Stuck candidate started ({self._stuck_candidate_source})'
            )

        if previous == RecoveryState.NORMAL and self._machine.state == RecoveryState.STOPPING:
            self._capture_recovery_steering()

        if transitioned:
            self.get_logger().warn(
                f'Recovery state: {previous.value} -> {self._machine.state.value} '
                f'({self._machine.last_transition_reason})'
            )

        state_msg = String()
        state_msg.data = self._machine.state.value
        self._state_pub.publish(state_msg)

        if self._machine.state in (RecoveryState.SHIFT_REVERSE, RecoveryState.REVERSING,
                                   RecoveryState.STOPPING_REVERSE):
            self._publish_gear(GearCommand.REVERSE)
        else:
            self._publish_gear(GearCommand.DRIVE)

        if self._machine.state == RecoveryState.NORMAL:
            if (
                self._nominal_command is not None
                and self._fresh(now, self._nominal_received_at, self._command_timeout)
            ):
                self._control_pub.publish(copy.deepcopy(self._nominal_command))
            else:
                self._control_pub.publish(self._make_stop_command())
        elif self._machine.state == RecoveryState.REVERSING:
            self._control_pub.publish(self._make_reverse_command())
        else:
            self._control_pub.publish(self._make_stop_command())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StuckRecoveryController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

"""ROS 2 node for fail-closed LiDAR-only policy inference."""

from __future__ import annotations

import math
import threading
import time

from autoware_auto_control_msgs.msg import AckermannControlCommand
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from lidar_racing_controller.metrics import percentile
from lidar_racing_controller.policy import PolicyLoadError, PolicyRuntime
from lidar_racing_controller.preprocessing import (
    FrameStack,
    ScanValidationError,
    canonicalize_laserscan,
)
from lidar_racing_controller.safety import (
    CommandRateLimiter,
    ControlTarget,
    UnsafeControlError,
    safe_stop_target,
    scale_normalized_action,
    scan_timed_out,
)


class LidarRacingControllerNode(Node):
    """Convert LaserScan history into Ackermann commands, failing closed on any fault."""

    def __init__(self) -> None:
        super().__init__("lidar_racing_controller")
        self._declare_parameters()

        self._expected_raw_beams = int(self.get_parameter("preprocessing.raw_beams").value)
        self._canonical_beams = int(
            self.get_parameter("preprocessing.canonical_beams").value
        )
        self._frame_count = int(self.get_parameter("preprocessing.frame_stack").value)
        self._scan_channels = 2
        self._expected_range_max = float(
            self.get_parameter("preprocessing.expected_range_max").value
        )
        self._expected_angle_min = float(
            self.get_parameter("preprocessing.expected_angle_min").value
        )
        self._expected_angle_max = float(
            self.get_parameter("preprocessing.expected_angle_max").value
        )
        self._metadata_tolerance = float(
            self.get_parameter("preprocessing.metadata_tolerance").value
        )
        self._minimum_valid_ratio = float(
            self.get_parameter("failsafe.minimum_valid_beam_ratio").value
        )
        self._scan_timeout_seconds = float(
            self.get_parameter("failsafe.scan_timeout_seconds").value
        )
        self._safe_acceleration = float(
            self.get_parameter("failsafe.safe_acceleration").value
        )
        self._steering_max_abs = float(
            self.get_parameter("action.steering_max_abs").value
        )
        self._acceleration_min = float(
            self.get_parameter("action.acceleration_min").value
        )
        self._acceleration_max = float(
            self.get_parameter("action.acceleration_max").value
        )
        self._control_rate_hz = float(self.get_parameter("control.rate_hz").value)
        self._debug = bool(self.get_parameter("debug.enabled").value)
        self._log_interval_seconds = float(
            self.get_parameter("debug.log_interval_seconds").value
        )
        self._validate_configuration()

        self._frames = FrameStack(
            frame_count=self._frame_count,
            channels=self._scan_channels,
            beams=self._canonical_beams,
        )
        self._safe_target = safe_stop_target(acceleration=self._safe_acceleration)
        self._rate_limiter = CommandRateLimiter(
            steering_rate=float(self.get_parameter("control.steering_rate_limit").value),
            acceleration_rate=float(
                self.get_parameter("control.acceleration_rate_limit").value
            ),
        )

        self._state_lock = threading.Lock()
        self._last_scan_monotonic: float | None = None
        self._latest_target: ControlTarget | None = None
        self._scan_fault: str | None = None
        self._last_safety_reason: str | None = None
        self._last_publish_monotonic = time.monotonic()
        self._inference_times_ms: list[float] = []
        self._last_metrics_log_monotonic = time.monotonic()

        self._policy: PolicyRuntime | None = None
        self._policy_error: str | None = None
        self._load_policy()

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._control_publisher = self.create_publisher(
            AckermannControlCommand,
            "output/control_cmd",
            1,
        )
        self._scan_subscription = self.create_subscription(
            LaserScan,
            "input/scan",
            self._on_scan,
            scan_qos,
        )

        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._control_timer = self.create_timer(
            1.0 / self._control_rate_hz,
            self._on_control_timer,
            clock=self._steady_clock,
        )
        self.get_logger().info(
            "LiDAR racing controller started; fail-safe output remains active until a "
            "verified model and valid LaserScan are available"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("model.path", "")
        self.declare_parameter("model.manifest_path", "")
        self.declare_parameter("model.device", "cpu")
        self.declare_parameter("preprocessing.raw_beams", 1080)
        self.declare_parameter("preprocessing.canonical_beams", 360)
        self.declare_parameter("preprocessing.frame_stack", 4)
        self.declare_parameter("preprocessing.expected_range_max", 30.0)
        self.declare_parameter("preprocessing.expected_angle_min", -3.0 * math.pi / 4.0)
        self.declare_parameter("preprocessing.expected_angle_max", 3.0 * math.pi / 4.0)
        self.declare_parameter("preprocessing.metadata_tolerance", 0.01)
        self.declare_parameter("failsafe.scan_timeout_seconds", 0.25)
        self.declare_parameter("failsafe.minimum_valid_beam_ratio", 0.5)
        self.declare_parameter("failsafe.safe_acceleration", -1.0)
        self.declare_parameter("action.steering_max_abs", 0.64)
        self.declare_parameter("action.acceleration_min", -3.2)
        self.declare_parameter("action.acceleration_max", 3.2)
        self.declare_parameter("control.rate_hz", 20.0)
        self.declare_parameter("control.steering_rate_limit", 1.5)
        self.declare_parameter("control.acceleration_rate_limit", 6.0)
        self.declare_parameter("debug.enabled", True)
        self.declare_parameter("debug.log_interval_seconds", 5.0)

    def _validate_configuration(self) -> None:
        if self._expected_raw_beams <= 0 or self._canonical_beams <= 0:
            raise ValueError("configured beam counts must be positive")
        if self._expected_raw_beams % self._canonical_beams != 0:
            raise ValueError("raw_beams must be divisible by canonical_beams")
        if self._frame_count <= 0:
            raise ValueError("frame_stack must be positive")
        if not math.isfinite(self._expected_range_max) or self._expected_range_max <= 0.0:
            raise ValueError("expected_range_max must be finite and positive")
        if not math.isfinite(self._expected_angle_min) or not math.isfinite(
            self._expected_angle_max
        ):
            raise ValueError("expected LiDAR angle bounds must be finite")
        if self._expected_angle_max <= self._expected_angle_min:
            raise ValueError("expected_angle_max must exceed expected_angle_min")
        if not math.isfinite(self._metadata_tolerance) or self._metadata_tolerance < 0.0:
            raise ValueError("metadata_tolerance must be finite and non-negative")
        if not 0.0 <= self._minimum_valid_ratio <= 1.0:
            raise ValueError("minimum_valid_beam_ratio must be in [0, 1]")
        if self._control_rate_hz <= 0.0:
            raise ValueError("control.rate_hz must be positive")
        if self._log_interval_seconds <= 0.0:
            raise ValueError("debug.log_interval_seconds must be positive")

        # Exercise pure validators at startup so unsafe parameter combinations fail early.
        safe_stop_target(acceleration=self._safe_acceleration)
        scale_normalized_action(
            (0.0, 0.0),
            steering_max_abs=self._steering_max_abs,
            acceleration_min=self._acceleration_min,
            acceleration_max=self._acceleration_max,
        )
        scan_timed_out(
            now_monotonic=0.0,
            last_scan_monotonic=None,
            timeout_seconds=self._scan_timeout_seconds,
        )

    def _load_policy(self) -> None:
        model_path = str(self.get_parameter("model.path").value)
        manifest_path = str(self.get_parameter("model.manifest_path").value)
        device = str(self.get_parameter("model.device").value)
        if not model_path or not manifest_path:
            self._policy_error = "model.path and model.manifest_path must both be configured"
            self.get_logger().error(self._policy_error)
            return

        try:
            self._policy = PolicyRuntime.load(
                model_path=model_path,
                manifest_path=manifest_path,
                device=device,
                expected_beam_count=self._canonical_beams,
                expected_frame_stack=self._frame_count,
                expected_scan_channels=self._scan_channels,
                expected_range_max=self._expected_range_max,
                expected_angle_min=self._expected_angle_min,
                expected_angle_max=self._expected_angle_max,
                expected_steering_max_abs=self._steering_max_abs,
                expected_acceleration_min=self._acceleration_min,
                expected_acceleration_max=self._acceleration_max,
            )
        except (PolicyLoadError, OSError, RuntimeError, ValueError) as error:
            self._policy_error = f"policy unavailable: {error}"
            self.get_logger().error(self._policy_error)
            return

        self.get_logger().info(
            "Verified policy loaded: "
            f"architecture={self._policy.manifest.architecture_version}, device={device}"
        )

    def _record_scan_fault(self, *, received_at: float, reason: str) -> None:
        with self._state_lock:
            self._last_scan_monotonic = received_at
            self._latest_target = None
            self._scan_fault = reason
            self._frames.reset()

    def _on_scan(self, message: LaserScan) -> None:
        received_at = time.monotonic()
        try:
            canonical = canonicalize_laserscan(
                message.ranges,
                range_min=message.range_min,
                range_max=message.range_max,
                angle_min=message.angle_min,
                angle_max=message.angle_max,
                angle_increment=message.angle_increment,
                expected_raw_beams=self._expected_raw_beams,
                canonical_beams=self._canonical_beams,
                expected_range_max=self._expected_range_max,
                expected_angle_min=self._expected_angle_min,
                expected_angle_max=self._expected_angle_max,
                metadata_tolerance=self._metadata_tolerance,
            )
        except (ScanValidationError, TypeError, ValueError) as error:
            self._record_scan_fault(received_at=received_at, reason=f"invalid LaserScan: {error}")
            return

        if canonical.valid_ratio < self._minimum_valid_ratio:
            self._record_scan_fault(
                received_at=received_at,
                reason=(
                    f"valid beam ratio {canonical.valid_ratio:.3f} is below "
                    f"{self._minimum_valid_ratio:.3f}"
                ),
            )
            return

        with self._state_lock:
            self._last_scan_monotonic = received_at
            self._scan_fault = None
            self._frames.append(canonical.values)
            observation = self._frames.actor_input()

        if self._policy is None:
            return

        inference_started = time.monotonic()
        try:
            action = self._policy.predict(observation)
            target = scale_normalized_action(
                action,
                steering_max_abs=self._steering_max_abs,
                acceleration_min=self._acceleration_min,
                acceleration_max=self._acceleration_max,
            )
        except (RuntimeError, TypeError, ValueError, UnsafeControlError) as error:
            self._record_scan_fault(
                received_at=received_at,
                reason=f"policy inference failed: {error}",
            )
            return

        elapsed_ms = (time.monotonic() - inference_started) * 1000.0
        with self._state_lock:
            self._latest_target = target
            self._scan_fault = None
            if self._debug:
                self._inference_times_ms.append(elapsed_ms)

    def _safety_reason(self, now_monotonic: float) -> str | None:
        with self._state_lock:
            last_scan = self._last_scan_monotonic
            scan_fault = self._scan_fault
            target = self._latest_target

        if self._policy_error is not None:
            return self._policy_error
        if scan_timed_out(
            now_monotonic=now_monotonic,
            last_scan_monotonic=last_scan,
            timeout_seconds=self._scan_timeout_seconds,
        ):
            with self._state_lock:
                self._latest_target = None
                self._frames.reset()
            return "LaserScan timeout"
        if scan_fault is not None:
            return scan_fault
        if target is None:
            return "waiting for a valid policy inference"
        return None

    def _on_control_timer(self) -> None:
        now_monotonic = time.monotonic()
        reason = self._safety_reason(now_monotonic)
        elapsed = max(0.0, now_monotonic - self._last_publish_monotonic)
        self._last_publish_monotonic = now_monotonic

        if reason is not None:
            target = self._safe_target
            # Safety commands bypass rate limiting and become the recovery baseline.
            self._rate_limiter.reset(target)
        else:
            with self._state_lock:
                latest_target = self._latest_target
            if latest_target is None:
                target = self._safe_target
                reason = "control target disappeared"
                self._rate_limiter.reset(target)
            else:
                target = self._rate_limiter.step(latest_target, elapsed_seconds=elapsed)

        self._publish_target(target)
        self._log_safety_transition(reason)
        self._log_inference_metrics(now_monotonic)

    def _publish_target(self, target: ControlTarget) -> None:
        stamp = self.get_clock().now().to_msg()
        command = AckermannControlCommand()
        command.stamp = stamp
        command.lateral.stamp = stamp
        command.lateral.steering_tire_angle = float(target.steering_angle)
        command.longitudinal.stamp = stamp
        command.longitudinal.speed = 0.0
        command.longitudinal.acceleration = float(target.acceleration)
        self._control_publisher.publish(command)

    def _log_safety_transition(self, reason: str | None) -> None:
        if reason == self._last_safety_reason:
            return
        if reason is None:
            self.get_logger().info("Fail-safe cleared; publishing policy control")
        else:
            self.get_logger().warning(f"Fail-safe active: {reason}")
        self._last_safety_reason = reason

    def _log_inference_metrics(self, now_monotonic: float) -> None:
        if not self._debug:
            return
        if now_monotonic - self._last_metrics_log_monotonic < self._log_interval_seconds:
            return
        with self._state_lock:
            timings = self._inference_times_ms
            self._inference_times_ms = []
        if timings:
            self.get_logger().info(
                "Inference latency: "
                f"p50={percentile(timings, 50.0):.2f} ms, "
                f"p95={percentile(timings, 95.0):.2f} ms, "
                f"max={max(timings):.2f} ms, "
                f"samples={len(timings)}"
            )
        self._last_metrics_log_monotonic = now_monotonic


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LidarRacingControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

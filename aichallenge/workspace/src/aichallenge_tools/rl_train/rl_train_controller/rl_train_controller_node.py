#!/usr/bin/env python3
"""Bridge AWSIM reset and global state between a player domain and domain 0."""

import os
import threading

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Empty, String


class DomainBridgeNode(Node):
    """Forward reset commands between two explicitly initialized contexts."""

    def __init__(
        self,
        destination_context: Context,
        source_context: Context,
        source_domain_id: int,
        destination_domain_id: int,
    ):
        super().__init__('awsim_reset_domain_bridge', context=source_context)

        self.declare_parameter('src_topic',  '/awsim/reset')
        self.declare_parameter('dst_topic',  '/admin/awsim/reset')
        self.declare_parameter('admin_state_topic', '/admin/awsim/state')
        self.declare_parameter('forwarded_state_topic', '/awsim/admin_state')

        src_topic = self.get_parameter('src_topic').value
        dst_topic = self.get_parameter('dst_topic').value
        admin_state_topic = self.get_parameter('admin_state_topic').value
        forwarded_state_topic = self.get_parameter('forwarded_state_topic').value

        self._state_pub = self.create_publisher(String, forwarded_state_topic, 10)
        self._last_admin_state = None

        self._pub_node = rclpy.create_node(
            'awsim_reset_domain_bridge_publisher',
            context=destination_context,
            use_global_arguments=False,
        )
        self._pub = self._pub_node.create_publisher(Empty, dst_topic, 10)
        self._admin_state_sub = self._pub_node.create_subscription(
            String,
            admin_state_topic,
            self._admin_state_cb,
            10,
        )

        self.create_subscription(Empty, src_topic, self._reset_cb, 10)

        self.get_logger().info(
            f"AWSIM reset bridge ready: DOMAIN={source_domain_id}:{src_topic} "
            f"-> DOMAIN={destination_domain_id}:{dst_topic}"
        )
        self.get_logger().info(
            f"AWSIM state bridge ready: DOMAIN={destination_domain_id}:{admin_state_topic} "
            f"-> DOMAIN={source_domain_id}:{forwarded_state_topic}"
        )

    def _reset_cb(self, _msg: Empty):
        self._pub.publish(Empty())
        self.get_logger().info("Forwarded AWSIM reset request")

    def _admin_state_cb(self, msg: String):
        forwarded = String()
        forwarded.data = msg.data
        self._state_pub.publish(forwarded)
        if msg.data != self._last_admin_state:
            self.get_logger().info(f"Forwarded AWSIM admin state: {msg.data}")
            self._last_admin_state = msg.data

    def destroy_node(self):
        self._pub_node.destroy_node()
        super().destroy_node()


def _domain_id_from_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        domain_id = int(raw_value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got: {raw_value!r}") from exc
    if not 0 <= domain_id <= 232:
        raise SystemExit(f"{name} must be between 0 and 232, got: {domain_id}")
    return domain_id


def main(args=None):
    source_domain_id = _domain_id_from_env('ROS_DOMAIN_ID', 1)
    destination_domain_id = _domain_id_from_env('AWSIM_DOMAIN_ID', 0)

    destination_context = Context()
    rclpy.init(
        context=destination_context,
        domain_id=destination_domain_id,
        args=args,
    )

    source_context = Context()
    rclpy.init(context=source_context, domain_id=source_domain_id, args=args)

    node = DomainBridgeNode(
        destination_context=destination_context,
        source_context=source_context,
        source_domain_id=source_domain_id,
        destination_domain_id=destination_domain_id,
    )

    destination_executor = SingleThreadedExecutor(context=destination_context)
    destination_executor.add_node(node._pub_node)
    destination_thread = threading.Thread(
        target=destination_executor.spin,
        name='awsim-reset-domain-publisher',
        daemon=True,
    )
    destination_thread.start()

    source_executor = SingleThreadedExecutor(context=source_context)
    source_executor.add_node(node)

    try:
        source_executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        source_executor.shutdown()
        destination_executor.shutdown()
        destination_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown(context=source_context)
        rclpy.shutdown(context=destination_context)


if __name__ == '__main__':
    main()

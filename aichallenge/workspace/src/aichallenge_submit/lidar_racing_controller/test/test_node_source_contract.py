"""Dependency-free source checks for the LiDAR-only ROS information boundary."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


NODE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "lidar_racing_controller"
    / "node.py"
)


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class NodeSourceContractTest(unittest.TestCase):
    def test_node_subscribes_only_to_the_internal_lidar_topic(self) -> None:
        tree = ast.parse(
            NODE_SOURCE.read_text(encoding="utf-8"),
            filename=str(NODE_SOURCE),
        )
        subscriptions: list[tuple[str | None, str | None]] = []
        publishers: list[tuple[str | None, str | None]] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func,
                ast.Attribute,
            ):
                continue
            if len(node.args) < 2:
                continue
            message_type = (
                node.args[0].id if isinstance(node.args[0], ast.Name) else None
            )
            topic = _literal_string(node.args[1])
            if node.func.attr == "create_subscription":
                subscriptions.append((message_type, topic))
            elif node.func.attr == "create_publisher":
                publishers.append((message_type, topic))

        self.assertEqual(subscriptions, [("LaserScan", "input/scan")])
        self.assertEqual(
            publishers,
            [("AckermannControlCommand", "output/control_cmd")],
        )

    def test_node_source_contains_no_forbidden_gt_or_localization_topics(self) -> None:
        source = NODE_SOURCE.read_text(encoding="utf-8")
        forbidden_topics = (
            "/awsim",
            "/ground_truth",
            "/localization",
            "/odom",
            "/tf",
            "/vehicle/status",
        )

        for topic in forbidden_topics:
            with self.subTest(topic=topic):
                self.assertNotIn(topic, source)

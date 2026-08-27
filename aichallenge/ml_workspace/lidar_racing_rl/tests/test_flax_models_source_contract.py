"""Dependency-free structural checks for the Flax SAC model contract."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "src" / "lidar_racing_rl" / "models"


def _tree(name: str) -> ast.Module:
    return ast.parse((MODELS_ROOT / name).read_text(encoding="utf-8"))


def _literal_assignments(tree: ast.Module) -> dict[str, object]:
    assignments: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    assignments[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return assignments


def _string_literals(tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


class FlaxModelSourceContractTest(unittest.TestCase):
    def test_models_initializer_keeps_torch_deployment_jax_free(self) -> None:
        tree = _tree("__init__.py")
        imports = [
            node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)
        ]
        self.assertEqual(imports, [])

    def test_encoder_topology_matches_blueprint(self) -> None:
        tree = _tree("encoder_flax.py")
        values = _literal_assignments(tree)

        self.assertEqual(values["FRAME_STACK"], 4)
        self.assertEqual(values["CHANNELS_PER_FRAME"], 2)
        self.assertEqual(values["CANONICAL_BEAMS"], 360)
        self.assertEqual(values["LIDAR_ENCODER_CHANNELS"], (32, 64, 64))
        self.assertEqual(values["LIDAR_ENCODER_KERNEL_SIZES"], (8, 4, 3))
        self.assertEqual(values["LIDAR_ENCODER_STRIDES"], (4, 2, 1))
        self.assertEqual(values["LIDAR_FEATURE_DIM"], 256)

        strings = _string_literals(tree)
        called_attributes = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertIn("VALID", strings)
        self.assertIn("dense", strings)
        self.assertGreaterEqual(called_attributes.count("swapaxes"), 2)

    def test_actor_heads_and_bounds_have_stable_names(self) -> None:
        tree = _tree("actor_flax.py")
        values = _literal_assignments(tree)
        strings = _string_literals(tree)
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        self.assertEqual(values["ACTION_DIM"], 2)
        self.assertEqual(values["LOG_STD_MIN"], -5.0)
        self.assertEqual(values["LOG_STD_MAX"], 2.0)
        self.assertIn("encoder", strings)
        self.assertIn("mean_head", strings)
        self.assertIn("log_std_head", strings)
        self.assertIn("clip", called_attributes)
        self.assertIn("tanh", called_attributes)
        self.assertIn("normal", called_attributes)

    def test_exported_actor_uses_highest_gpu_matmul_precision(self) -> None:
        for filename in ("encoder_flax.py", "actor_flax.py"):
            tree = _tree(filename)
            layer_calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"Conv", "Dense"}
            ]
            self.assertGreater(len(layer_calls), 0)
            for call in layer_calls:
                precision = next(
                    (keyword.value for keyword in call.keywords if keyword.arg == "precision"),
                    None,
                )
                self.assertIsNotNone(precision)
                self.assertEqual(
                    ast.unparse(precision),
                    "jax.lax.Precision.HIGHEST",
                )

    def test_twin_q_encoders_are_independent_and_named(self) -> None:
        tree = _tree("critic_flax.py")
        values = _literal_assignments(tree)
        strings = _string_literals(tree)

        self.assertEqual(values["CRITIC_HIDDEN_SIZES"], (256, 256))
        self.assertIn("q1", strings)
        self.assertIn("q2", strings)
        self.assertIn("hidden_0", strings)
        self.assertIn("value_head", strings)

        assignments = {
            node.targets[0].attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
        }
        self.assertIn("q1", assignments)
        self.assertIn("q2", assignments)

    def test_expected_export_kernel_shapes_are_fixed(self) -> None:
        length = 360
        input_channels = 8
        convolution_shapes: list[tuple[int, int, int]] = []
        for output_channels, kernel_size, stride in zip(
            (32, 64, 64),
            (8, 4, 3),
            (4, 2, 1),
            strict=True,
        ):
            convolution_shapes.append((kernel_size, input_channels, output_channels))
            length = (length - kernel_size) // stride + 1
            input_channels = output_channels

        self.assertEqual(
            convolution_shapes,
            [(8, 8, 32), (4, 32, 64), (3, 64, 64)],
        )
        self.assertEqual(length, 41)
        self.assertEqual((length * 64, 256), (2624, 256))
        self.assertEqual((256 + 2, 256), (258, 256))

    def test_models_do_not_import_environment_or_ground_truth(self) -> None:
        forbidden_fragments = ("envs", "frenet", "waypoint", "ground_truth")
        for path in MODELS_ROOT.glob("*_flax.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported_modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    imported_modules.append(node.module)
            joined = " ".join(imported_modules).lower()
            for fragment in forbidden_fragments:
                self.assertNotIn(fragment, joined)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Run dependency-free structural checks from macOS or a minimal Python image."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import tomllib
import unittest
import xml.etree.ElementTree as element_tree
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[2]
SUBMODULE_ROOT = PROJECT_ROOT / "repos" / "f1tenth_gym_jax"
ROS_CONTROLLER_ROOT = (
    REPOSITORY_ROOT
    / "aichallenge"
    / "workspace"
    / "src"
    / "aichallenge_submit"
    / "lidar_racing_controller"
)


def _python_sources(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if not any(part in {".git", ".venv", "__pycache__"} for part in path.parts)
    )


def _parse_python_sources() -> int:
    roots = (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", PROJECT_ROOT / "tests")
    files = [path for root in roots for path in _python_sources(root)]
    files.extend(_python_sources(ROS_CONTROLLER_ROOT))
    if not (SUBMODULE_ROOT / "f1tenth_gym_jax").is_dir():
        raise RuntimeError(
            "F1TENTH Gym JAX submodule is not initialized; run git submodule update first"
        )
    files.extend(_python_sources(SUBMODULE_ROOT / "f1tenth_gym_jax"))
    files.extend(_python_sources(SUBMODULE_ROOT / "tests"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return len(files)


def _parse_data_files() -> dict[str, int]:
    toml_files = [PROJECT_ROOT / "pyproject.toml", SUBMODULE_ROOT / "pyproject.toml"]
    for path in toml_files:
        with path.open("rb") as stream:
            tomllib.load(stream)

    json_files = sorted(
        path
        for path in PROJECT_ROOT.rglob("*.json")
        if "repos" not in path.relative_to(PROJECT_ROOT).parts
        and "outputs" not in path.relative_to(PROJECT_ROOT).parts
    )
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))

    xml_roots = (
        ROS_CONTROLLER_ROOT,
        REPOSITORY_ROOT
        / "aichallenge"
        / "workspace"
        / "src"
        / "aichallenge_submit"
        / "aichallenge_submit_launch",
        REPOSITORY_ROOT
        / "aichallenge"
        / "workspace"
        / "src"
        / "aichallenge_system"
        / "aichallenge_system_launch",
    )
    xml_files = [path for root in xml_roots for path in sorted(root.rglob("*.xml"))]
    for path in xml_files:
        element_tree.parse(path)
    return {"toml": len(toml_files), "json": len(json_files), "xml": len(xml_files)}


def _load_test_module(path: Path, index: int) -> object:
    name = f"lidar_racing_source_contract_{index}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load source contract test: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_standard_library_tests() -> int:
    test_paths = (
        PROJECT_ROOT / "tests" / "test_analyze_awsim_vehicle_response.py",
        PROJECT_ROOT / "tests" / "test_sac_source_contract.py",
        PROJECT_ROOT / "tests" / "test_flax_models_source_contract.py",
        PROJECT_ROOT / "tests" / "test_curriculum_contract.py",
        PROJECT_ROOT / "tests" / "test_opponent_pool_contract.py",
        ROS_CONTROLLER_ROOT / "test" / "test_node_source_contract.py",
    )
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    suite = unittest.TestSuite()
    for index, path in enumerate(test_paths):
        suite.addTests(
            unittest.defaultTestLoader.loadTestsFromModule(
                _load_test_module(path, index)
            )
        )
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("one or more dependency-free source contract tests failed")
    return result.testsRun


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse source/config files and run standard-library contracts."
    )
    parser.parse_args()
    python_count = _parse_python_sources()
    data_counts = _parse_data_files()
    test_count = _run_standard_library_tests()
    print(
        json.dumps(
            {
                "status": "ok",
                "python_ast_files": python_count,
                "standard_library_tests": test_count,
                **{f"{kind}_files": count for kind, count in data_counts.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

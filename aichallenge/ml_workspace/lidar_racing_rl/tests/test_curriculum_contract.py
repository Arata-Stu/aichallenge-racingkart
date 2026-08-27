"""Standard-library tests for the fail-closed Step 2 curriculum interface."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from lidar_racing_rl.curriculum.opponent_pool import (
    ARCHITECTURE_VERSION,
    OpponentPoolConfigurationError,
    OpponentPoolNotIntegratedError,
    OpponentPoolSettings,
)
from lidar_racing_rl.curriculum.schedule import (
    STEP2_PHASES,
    CurriculumConfigurationError,
    CurriculumPlan,
    UnsupportedCurriculumPhaseError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _braking(enabled: bool) -> dict[str, object]:
    return {
        "enabled": enabled,
        "probability": 0.15 if enabled else 0.0,
        "start_step": {"min": 100, "max": 1200},
        "duration_steps": {"min": 10, "max": 60},
        "acceleration": -2.0,
    }


def _phase(
    *,
    count: int,
    speed: tuple[float, float],
    line_mode: str,
    lateral: tuple[float, float],
    delay: tuple[int, int],
    braking: bool,
    source: str = "fixed_pure_pursuit",
    supported: bool = True,
) -> dict[str, object]:
    return {
        "active_npc_count": count,
        "speed_multiplier": {"min": speed[0], "max": speed[1]},
        "line_mode": line_mode,
        "lateral_offset": {"min": lateral[0], "max": lateral[1]},
        "control_delay_steps": {"min": delay[0], "max": delay[1]},
        "braking_event": _braking(braking),
        "opponent_source": source,
        "training_supported": supported,
    }


def _valid_plan() -> dict[str, object]:
    return {
        "schema_version": 1,
        "active_phase": "2a",
        "ordered_phases": list(STEP2_PHASES),
        "information_boundary": {
            "learned_ego_agent_index": 0,
            "learned_agent_count": 1,
            "actor_observation": "lidar_only",
            "critic_observation": "lidar_only",
            "replay_scope": "ego_only",
            "save_npc_transitions": False,
        },
        "phases": {
            "2a": _phase(
                count=1,
                speed=(0.65, 0.80),
                line_mode="centerline",
                lateral=(0.0, 0.0),
                delay=(0, 0),
                braking=False,
            ),
            "2b": _phase(
                count=3,
                speed=(0.65, 1.05),
                line_mode="centerline",
                lateral=(0.0, 0.0),
                delay=(0, 0),
                braking=False,
            ),
            "2c": _phase(
                count=3,
                speed=(0.65, 1.05),
                line_mode="random_offset",
                lateral=(-0.2, 0.2),
                delay=(0, 0),
                braking=False,
            ),
            "2d": _phase(
                count=3,
                speed=(0.65, 1.05),
                line_mode="random_offset",
                lateral=(-0.2, 0.2),
                delay=(0, 2),
                braking=True,
            ),
            "2e": _phase(
                count=3,
                speed=(0.65, 1.05),
                line_mode="random_offset",
                lateral=(-0.2, 0.2),
                delay=(0, 2),
                braking=True,
                source="checkpoint_pool",
                supported=False,
            ),
        },
    }


def _disabled_pool_settings() -> dict[str, object]:
    return {
        "schema_version": 1,
        "enabled": False,
        "training_integration": False,
        "manifest": None,
        "sampling_algorithm": "sha256_context_v1",
        "seed": 6,
        "validation": {
            "expected_architecture_version": ARCHITECTURE_VERSION,
            "allowed_training_config_sha256": [],
            "allowed_root_commits": [],
            "allowed_submodule_commits": [],
        },
    }


class CurriculumContractTest(unittest.TestCase):
    def test_complete_plan_normalizes_all_blueprint_phases(self) -> None:
        plan = CurriculumPlan.from_config(_valid_plan())

        self.assertEqual(tuple(phase.name for phase in plan.phases), STEP2_PHASES)
        self.assertEqual(plan.training_phase("2a").active_npc_count, 1)
        self.assertEqual(plan.training_phase("2b").active_npc_count, 3)
        self.assertEqual(plan.training_phase("2c").line_mode, "random_offset")
        self.assertEqual(plan.training_phase("2d").control_delay_steps.maximum, 2)
        self.assertTrue(plan.training_phase("2d").braking_event.enabled)

    def test_phase_2e_is_declared_but_cannot_enter_training(self) -> None:
        plan = CurriculumPlan.from_config(_valid_plan())

        self.assertEqual(plan.phase("2e").opponent_source, "checkpoint_pool")
        with self.assertRaises(UnsupportedCurriculumPhaseError):
            plan.training_phase("2e")

    def test_information_boundary_must_remain_lidar_and_ego_only(self) -> None:
        config = _valid_plan()
        config["information_boundary"]["actor_observation"] = "gt_pose"

        with self.assertRaises(CurriculumConfigurationError):
            CurriculumPlan.from_config(config)

    def test_phase_2a_requires_one_strictly_slower_npc(self) -> None:
        for key, value in (
            ("active_npc_count", 3),
            ("speed_multiplier", {"min": 0.65, "max": 1.0}),
        ):
            with self.subTest(key=key):
                config = _valid_plan()
                config["phases"]["2a"][key] = value
                with self.assertRaises(CurriculumConfigurationError):
                    CurriculumPlan.from_config(config)

    def test_phase_2b_requires_speed_diversity(self) -> None:
        config = _valid_plan()
        config["phases"]["2b"]["speed_multiplier"] = {"min": 0.8, "max": 0.8}

        with self.assertRaises(CurriculumConfigurationError):
            CurriculumPlan.from_config(config)

    def test_phase_2c_requires_left_and_right_lines(self) -> None:
        config = _valid_plan()
        config["phases"]["2c"]["lateral_offset"] = {"min": 0.0, "max": 0.2}

        with self.assertRaises(CurriculumConfigurationError):
            CurriculumPlan.from_config(config)

    def test_phase_2d_requires_delay_and_braking(self) -> None:
        variants = (
            ("control_delay_steps", {"min": 0, "max": 0}),
            ("braking_event", _braking(False)),
        )
        for key, value in variants:
            with self.subTest(key=key):
                config = _valid_plan()
                config["phases"]["2d"][key] = value
                with self.assertRaises(CurriculumConfigurationError):
                    CurriculumPlan.from_config(config)

    def test_phase_2e_cannot_claim_reviewed_training_support(self) -> None:
        config = _valid_plan()
        config["phases"]["2e"]["training_supported"] = True

        with self.assertRaises(CurriculumConfigurationError):
            CurriculumPlan.from_config(config)

    def test_disabled_pool_settings_are_explicitly_offline_only(self) -> None:
        settings = OpponentPoolSettings.from_config(_disabled_pool_settings())

        self.assertFalse(settings.enabled)
        self.assertIsNone(settings.validation_policy)
        with self.assertRaises(OpponentPoolNotIntegratedError):
            settings.require_training_ready()

    def test_pool_settings_reject_live_training_integration(self) -> None:
        config = _disabled_pool_settings()
        config["training_integration"] = True

        with self.assertRaises(OpponentPoolNotIntegratedError):
            OpponentPoolSettings.from_config(config)

    def test_source_configs_name_every_stage_and_seal_the_pool(self) -> None:
        curriculum = (
            PROJECT_ROOT / "configs" / "curriculum" / "step2.yaml"
        ).read_text(encoding="utf-8")
        disabled_pool = (
            PROJECT_ROOT
            / "configs"
            / "curriculum"
            / "opponent_pool_disabled.yaml"
        ).read_text(encoding="utf-8")

        for phase in STEP2_PHASES:
            self.assertIn(f"  {phase}:", curriculum)
        self.assertIn("actor_observation: lidar_only", curriculum)
        self.assertIn("critic_observation: lidar_only", curriculum)
        self.assertIn("replay_scope: ego_only", curriculum)
        self.assertIn("training_supported: false", curriculum)
        self.assertIn("enabled: false", disabled_pool)
        self.assertIn("training_integration: false", disabled_pool)

    def test_curriculum_modules_do_not_import_jax_or_flax(self) -> None:
        source_root = PROJECT_ROOT / "src" / "lidar_racing_rl" / "curriculum"
        for path in source_root.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("import jax", source)
                self.assertNotIn("from flax", source)

    def test_unknown_fields_are_rejected(self) -> None:
        config = copy.deepcopy(_valid_plan())
        config["unexpected"] = True

        with self.assertRaises(CurriculumConfigurationError):
            CurriculumPlan.from_config(config)


if __name__ == "__main__":
    unittest.main()

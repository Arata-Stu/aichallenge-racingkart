"""Dependency-free structural checks for SAC losses and learner state."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = PROJECT_ROOT / "src" / "lidar_racing_rl" / "sac"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"


def _tree(name: str) -> ast.Module:
    return ast.parse((SAC_ROOT / name).read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function: {name}")


def _called_attributes(node: ast.AST) -> set[str]:
    return {
        child.func.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }


def _called_names(node: ast.AST) -> set[str]:
    return {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }


class SACSourceContractTest(unittest.TestCase):
    def test_sac_initializer_keeps_metadata_tools_jax_free(self) -> None:
        tree = _tree("__init__.py")
        imports = [
            node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)
        ]
        self.assertEqual(imports, [])

    def test_bootstrap_mask_depends_only_on_termination(self) -> None:
        function = _function(_tree("losses.py"), "bootstrap_mask")
        positional_names = [argument.arg for argument in function.args.args]
        referenced_names = {
            node.id for node in ast.walk(function) if isinstance(node, ast.Name)
        }

        self.assertEqual(positional_names, ["terminated"])
        self.assertNotIn("truncated", referenced_names)

    def test_soft_target_uses_twin_minimum_entropy_and_stop_gradient(self) -> None:
        function = _function(_tree("losses.py"), "soft_critic_target")
        attributes = _called_attributes(function)

        self.assertIn("minimum", attributes)
        self.assertIn("exp", attributes)
        self.assertIn("stop_gradient", attributes)
        self.assertIn("bootstrap_mask", _called_names(function))

    def test_actor_and_alpha_losses_are_present(self) -> None:
        tree = _tree("losses.py")
        actor = _function(tree, "actor_loss")
        alpha = _function(tree, "alpha_loss")

        self.assertIn("minimum", _called_attributes(actor))
        self.assertIn("exp", _called_attributes(actor))
        self.assertIn("stop_gradient", _called_attributes(alpha))
        alpha_names = {
            node.id for node in ast.walk(alpha) if isinstance(node, ast.Name)
        }
        self.assertIn("log_alpha", alpha_names)

    def test_state_separates_online_target_and_temperature(self) -> None:
        tree = _tree("train_state.py")
        state_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SACTrainState"
        )
        annotated_fields = {
            node.target.id
            for node in state_class.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }

        self.assertTrue(
            {
                "actor_params",
                "critic_params",
                "target_critic_params",
                "log_alpha",
                "actor_opt_state",
                "critic_opt_state",
                "alpha_opt_state",
            }.issubset(annotated_fields)
        )

    def test_polyak_formula_and_compiled_update_guards_are_explicit(self) -> None:
        train_state_tree = _tree("train_state.py")
        learner_tree = _tree("learner.py")
        polyak = _function(train_state_tree, "polyak_update")
        update = _function(learner_tree, "update")
        update_source = ast.unparse(update)

        self.assertIn("target + coefficient * (online - target)", ast.unparse(polyak))
        self.assertGreaterEqual(update_source.count("jax.value_and_grad"), 3)
        self.assertIn("polyak_update", update_source)
        self.assertIn("jax.lax.cond", update_source)
        self.assertIn("_tree_all_finite", update_source)

    def test_update_factory_has_jittable_array_only_call_signature(self) -> None:
        learner_tree = _tree("learner.py")
        update = _function(learner_tree, "update")
        positional_names = [argument.arg for argument in update.args.args]

        self.assertEqual(positional_names, ["state", "batch", "key"])

    def test_checkpoint_schema_tracks_cumulative_environment_progress(self) -> None:
        tree = _tree("checkpoint.py")
        metadata_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CheckpointMetadata"
        )
        annotated_fields = {
            node.target.id
            for node in metadata_class.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        assignments = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance((target := node.targets[0]), ast.Name)
            and isinstance(node.value, ast.Constant)
        }

        self.assertEqual(assignments["CHECKPOINT_SCHEMA_VERSION"], 2)
        self.assertIn("environment_transitions", annotated_fields)

    def test_checkpoint_publication_is_serialized_per_root(self) -> None:
        tree = _tree("checkpoint.py")
        source = (SAC_ROOT / "checkpoint.py").read_text(encoding="utf-8")

        self.assertTrue(
            any(
                isinstance(node, ast.FunctionDef)
                and node.name == "_checkpoint_root_lock"
                for node in ast.walk(tree)
            )
        )
        self.assertIn("fcntl.flock", source)
        self.assertIn("with _checkpoint_root_lock(checkpoint_root):", source)

    def test_trainer_separates_cumulative_progress_from_replay_warmup(self) -> None:
        tree = _tree("trainer.py")
        source = ast.unparse(tree)
        checkpoint_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "save_checkpoint"
        ]

        self.assertIn("cumulative_environment_transitions", source)
        self.assertIn("session_environment_transitions", source)
        self.assertIn("replay_collected_transitions", source)
        self.assertIn(
            "replay_collected_transitions < warmup_transitions",
            source,
        )
        self.assertIn("jax.random.fold_in", source)
        self.assertGreaterEqual(len(checkpoint_calls), 2)
        self.assertTrue(
            all(
                "environment_transitions"
                in {keyword.arg for keyword in call.keywords}
                for call in checkpoint_calls
            )
        )

    def test_default_update_count_matches_vector_collection_size(self) -> None:
        config_root = SAC_ROOT.parents[2] / "configs"
        agent_config = (config_root / "agent" / "sac.yaml").read_text(
            encoding="utf-8"
        )
        environment_config = (config_root / "env" / "base.yaml").read_text(
            encoding="utf-8"
        )
        step1_config = (config_root / "train" / "step1_single_vehicle.yaml").read_text(
            encoding="utf-8"
        )
        gpu_compose = (PROJECT_ROOT / "compose.gpu.yaml").read_text(encoding="utf-8")

        self.assertIn("updates_per_collection: 64", agent_config)
        self.assertIn("actor_learning_rate: 0.0001", agent_config)
        self.assertIn("critic_learning_rate: 0.0001", agent_config)
        self.assertIn("temperature_learning_rate: 0.0001", agent_config)
        self.assertIn("capacity: 250000", agent_config)
        self.assertIn("warmup_transitions: 50000", agent_config)
        self.assertIn("save_interval_updates: 50000", agent_config)
        self.assertIn("keep_last: 20", agent_config)
        self.assertIn("num_envs: 64", environment_config)
        self.assertIn("collision: 50.0", step1_config)
        self.assertIn("off_track: 100.0", step1_config)
        self.assertIn("XLA_PYTHON_CLIENT_PREALLOCATE", gpu_compose)
        self.assertIn("cuda_malloc_async", gpu_compose)

    def test_training_metrics_report_progress_separately_from_reward(self) -> None:
        source = (SAC_ROOT / "trainer.py").read_text(encoding="utf-8")

        for metric in (
            "mean_course_progress_meters_per_transition",
            "mean_course_progress_fraction_per_transition",
            "course_progress_meters_per_simulated_second",
            "collision_rate_per_completed_episode",
            "off_track_rate_per_completed_episode",
            "race_completion_rate",
        ):
            with self.subTest(metric=metric):
                self.assertIn(f'"{metric}"', source)
        self.assertIn("result.diagnostics.progress_delta", source)
        self.assertIn("result.diagnostics.race_complete", source)

    def test_train_and_export_clis_fail_closed_on_v1_model_shape(self) -> None:
        required_paths = (
            "agent.actor.encoder.channels",
            "agent.actor.encoder.kernel_sizes",
            "agent.actor.encoder.strides",
            "agent.actor.encoder.activation",
            "agent.actor.hidden_sizes",
            "agent.actor.action_dim",
            "agent.actor.log_std_min",
            "agent.actor.log_std_max",
            "agent.observation.num_beams",
            "agent.observation.frame_stack",
            "agent.observation.channels_per_frame",
            "agent.critic.count",
            "agent.critic.share_lidar_encoder",
            "agent.critic.hidden_sizes",
        )
        for script_name in ("train.py", "export_policy.py"):
            source = (SCRIPT_ROOT / script_name).read_text(encoding="utf-8")
            for path in required_paths:
                with self.subTest(script=script_name, path=path):
                    self.assertIn(path, source)

    def test_training_and_evaluation_do_not_reintroduce_small_car_raceline(self) -> None:
        source_paths = (
            SAC_ROOT / "trainer.py",
            PROJECT_ROOT
            / "src"
            / "lidar_racing_rl"
            / "evaluation"
            / "evaluator.py",
        )
        for path in source_paths:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("build_reference_waypoints", source)
                self.assertIn("validate_centerline_clearance", source)
                self.assertNotIn("simulator.track.raceline", source)

    def test_unintegrated_step2_features_fail_closed(self) -> None:
        step2_config = (
            PROJECT_ROOT / "configs" / "train" / "step2_four_vehicle.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("- /env: dynamic_lidar", step2_config)
        self.assertNotIn("- /env: domain_randomization", step2_config)
        self.assertIn("enabled: false", step2_config)
        self.assertIn("active_phase: null", step2_config)

        for path in (
            SAC_ROOT / "trainer.py",
            PROJECT_ROOT
            / "src"
            / "lidar_racing_rl"
            / "evaluation"
            / "evaluator.py",
        ):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("curriculum scheduling is not integrated", source)
                self.assertIn("past-policy opponents are not integrated", source)

        make_env_source = (
            PROJECT_ROOT / "src" / "lidar_racing_rl" / "envs" / "make_env.py"
        ).read_text(encoding="utf-8")
        self.assertIn("domain_randomization", make_env_source)
        self.assertIn("vehicle-response domain randomization is not", make_env_source)
        self.assertIn("connected to F1TENTH Gym JAX yet", make_env_source)

    def test_benchmark_num_envs_override_handles_structured_hydra_config(self) -> None:
        source = (SCRIPT_ROOT / "benchmark_env.py").read_text(encoding="utf-8")
        self.assertIn('f"++env.num_envs={args.num_envs}"', source)

    def test_train_primary_configs_compose_into_the_global_package(self) -> None:
        for filename in ("step1_single_vehicle.yaml", "step2_four_vehicle.yaml"):
            source = (
                PROJECT_ROOT / "configs" / "train" / filename
            ).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertTrue(source.startswith("# @package _global_\n"))

    def test_spielberg_training_uses_nominal_f1tenth_geometry(self) -> None:
        profile = (
            PROJECT_ROOT / "configs" / "vehicle" / "f1tenth_nominal.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("profile: f1tenth_nominal", profile)
        self.assertIn("length: 0.58", profile)
        self.assertIn("width: 0.31", profile)
        self.assertIn("wheelbase: 0.3302", profile)
        for filename in ("step1_single_vehicle.yaml", "step2_four_vehicle.yaml"):
            source = (
                PROJECT_ROOT / "configs" / "train" / filename
            ).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertIn("/vehicle: f1tenth_nominal", source)

    def test_awsim_action_limits_are_separate_from_training_geometry(self) -> None:
        deployment = (
            PROJECT_ROOT / "configs" / "deployment" / "awsim.yaml"
        ).read_text(encoding="utf-8")
        exporter = (
            PROJECT_ROOT / "src" / "lidar_racing_rl" / "export" / "export_policy.py"
        ).read_text(encoding="utf-8")
        self.assertIn("steering_max_abs: 0.64", deployment)
        self.assertIn("acceleration_min: -3.2", deployment)
        self.assertIn("acceleration_max: 3.2", deployment)
        self.assertIn('_value(deployment_config, "control")', exporter)

    def test_action_steering_limit_matches_authoritative_vehicle_metadata(self) -> None:
        repository_root = PROJECT_ROOT.parents[2]
        source_paths = (
            PROJECT_ROOT / "configs" / "vehicle" / "aichallenge_kart.yaml",
            PROJECT_ROOT / "scripts" / "install_policy_bundle.py",
            repository_root
            / "aichallenge"
            / "workspace"
            / "src"
            / "aichallenge_submit"
            / "lidar_racing_controller"
            / "config"
            / "lidar_racing_controller.param.yaml",
            repository_root
            / "aichallenge"
            / "workspace"
            / "src"
            / "aichallenge_submit"
            / "lidar_racing_controller"
            / "lidar_racing_controller"
            / "node.py",
        )
        authoritative = (
            repository_root
            / "aichallenge"
            / "workspace"
            / "src"
            / "aichallenge_submit"
            / "racing_kart_description"
            / "config"
            / "vehicle_info.param.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("max_steer_angle: 0.64", authoritative)
        for path in source_paths:
            with self.subTest(path=path.name):
                self.assertIn("0.64", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

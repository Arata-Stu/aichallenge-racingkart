"""Standard-library provenance and deterministic sampling tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lidar_racing_rl.curriculum.opponent_pool import (
    ARCHITECTURE_VERSION,
    SAMPLING_ALGORITHM,
    OpponentPool,
    OpponentPoolConfigurationError,
    OpponentPoolSettings,
    PoolValidationPolicy,
    deterministic_pool_sample,
    load_configured_opponent_pool,
    load_opponent_pool_manifest,
    verify_opponent_checkpoint,
)


CONFIG_SHA256 = "1" * 64
ROOT_COMMIT = "2" * 40
SUBMODULE_COMMIT = "3" * 40


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _policy(
    *,
    config_hash: str = CONFIG_SHA256,
    root_commit: str = ROOT_COMMIT,
    submodule_commit: str = SUBMODULE_COMMIT,
) -> PoolValidationPolicy:
    return PoolValidationPolicy.from_values(
        expected_architecture_version=ARCHITECTURE_VERSION,
        allowed_training_config_sha256=(config_hash,),
        allowed_root_commits=(root_commit,),
        allowed_submodule_commits=(submodule_commit,),
    )


def _write_checkpoint(
    root: Path,
    owner: str,
    *,
    step: int,
    actor_payload: bytes,
) -> tuple[Path, dict[str, object]]:
    directory = root / owner / f"step_{step:012d}"
    directory.mkdir(parents=True)
    learner_payload = f"learner-{owner}-{step}".encode()
    actor_sha256 = _sha256(actor_payload)
    metadata = {
        "schema_version": 2,
        "architecture_version": ARCHITECTURE_VERSION,
        "step": step,
        "environment_transitions": step * 64,
        "config_sha256": CONFIG_SHA256,
        "root_commit": ROOT_COMMIT,
        "submodule_commit": SUBMODULE_COMMIT,
        "payload_sha256": _sha256(learner_payload),
        "actor_payload_sha256": actor_sha256,
        "created_at_utc": "2026-08-28T00:00:00Z",
    }
    (directory / "learner.msgpack").write_bytes(learner_payload)
    (directory / "actor.msgpack").write_bytes(actor_payload)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    entry = {
        "opponent_id": owner,
        "checkpoint_directory": directory.relative_to(root).as_posix(),
        "architecture_version": ARCHITECTURE_VERSION,
        "training_config_sha256": CONFIG_SHA256,
        "root_commit": ROOT_COMMIT,
        "submodule_commit": SUBMODULE_COMMIT,
        "checkpoint_step": step,
        "environment_transitions": step * 64,
        "actor_sha256": actor_sha256,
        "approved_for_opponent_pool": True,
    }
    return directory, entry


def _write_manifest(root: Path, entries: list[dict[str, object]]) -> Path:
    path = root / "opponent_pool.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pool_id": "reviewed-step2-pool",
                "sampling_algorithm": SAMPLING_ALGORITHM,
                "entries": entries,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


class OpponentPoolContractTest(unittest.TestCase):
    def test_enabled_settings_load_a_validated_pool_for_offline_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, entry = _write_checkpoint(root, "alpha", step=7, actor_payload=b"alpha")
            manifest = _write_manifest(root, [entry])
            settings = OpponentPoolSettings.from_config(
                {
                    "schema_version": 1,
                    "enabled": True,
                    "training_integration": False,
                    "manifest": manifest.name,
                    "sampling_algorithm": SAMPLING_ALGORITHM,
                    "seed": 6,
                    "validation": {
                        "expected_architecture_version": ARCHITECTURE_VERSION,
                        "allowed_training_config_sha256": [CONFIG_SHA256],
                        "allowed_root_commits": [ROOT_COMMIT],
                        "allowed_submodule_commits": [SUBMODULE_COMMIT],
                    },
                }
            )

            pool = load_configured_opponent_pool(settings, base_directory=root)

            self.assertEqual(tuple(item.opponent_id for item in pool.entries), ("alpha",))

    def test_manifest_verifies_bytes_provenance_and_sorts_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, alpha = _write_checkpoint(root, "alpha", step=7, actor_payload=b"alpha")
            _, beta = _write_checkpoint(root, "beta", step=11, actor_payload=b"beta")
            manifest = _write_manifest(root, [beta, alpha])

            pool = load_opponent_pool_manifest(manifest, policy=_policy())

            self.assertEqual(
                tuple(entry.opponent_id for entry in pool.entries),
                ("alpha", "beta"),
            )
            self.assertTrue(all(entry.actor_path.is_file() for entry in pool.entries))

    def test_sampling_is_stable_and_manifest_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = [
                _write_checkpoint(root, name, step=index + 1, actor_payload=name.encode())[1]
                for index, name in enumerate(("alpha", "beta", "gamma"))
            ]
            pool = load_opponent_pool_manifest(
                _write_manifest(root, list(reversed(entries))),
                policy=_policy(),
            )
            reversed_pool = OpponentPool(
                pool_id=pool.pool_id,
                sampling_algorithm=pool.sampling_algorithm,
                entries=tuple(reversed(pool.entries)),
                manifest_path=pool.manifest_path,
            )

            first = deterministic_pool_sample(
                pool,
                opponent_count=3,
                seed=6,
                environment_index=2,
                episode_index=9,
            )
            second = deterministic_pool_sample(
                reversed_pool,
                opponent_count=3,
                seed=6,
                environment_index=2,
                episode_index=9,
            )

            self.assertEqual(
                tuple(entry.opponent_id for entry in first),
                tuple(entry.opponent_id for entry in second),
            )
            self.assertEqual(len({entry.opponent_id for entry in first}), 3)

    def test_sampling_requires_explicit_replacement_for_small_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, entry = _write_checkpoint(root, "alpha", step=7, actor_payload=b"alpha")
            pool = load_opponent_pool_manifest(
                _write_manifest(root, [entry]),
                policy=_policy(),
            )

            with self.assertRaises(OpponentPoolConfigurationError):
                deterministic_pool_sample(
                    pool,
                    opponent_count=3,
                    seed=6,
                    environment_index=0,
                    episode_index=0,
                )
            selected = deterministic_pool_sample(
                pool,
                opponent_count=3,
                seed=6,
                environment_index=0,
                episode_index=0,
                replace=True,
            )
            self.assertEqual(tuple(item.opponent_id for item in selected), ("alpha",) * 3)

    def test_manifest_rejects_unapproved_or_mismatched_declarations(self) -> None:
        variants = (
            ("approved_for_opponent_pool", False),
            ("actor_sha256", "0" * 64),
            ("architecture_version", "unknown_architecture"),
        )
        for key, value in variants:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, entry = _write_checkpoint(
                    root,
                    "alpha",
                    step=7,
                    actor_payload=b"alpha",
                )
                entry[key] = value
                manifest = _write_manifest(root, [entry])

                with self.assertRaises(OpponentPoolConfigurationError):
                    load_opponent_pool_manifest(manifest, policy=_policy())

    def test_external_config_and_commit_allowlists_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, entry = _write_checkpoint(root, "alpha", step=7, actor_payload=b"alpha")
            manifest = _write_manifest(root, [entry])
            policies = (
                _policy(config_hash="4" * 64),
                _policy(root_commit="5" * 40),
                _policy(submodule_commit="6" * 40),
            )

            for policy in policies:
                with self.subTest(policy=policy):
                    with self.assertRaises(OpponentPoolConfigurationError):
                        load_opponent_pool_manifest(manifest, policy=policy)

    def test_checkpoint_path_must_stay_below_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, entry = _write_checkpoint(root, "alpha", step=7, actor_payload=b"alpha")
            entry["checkpoint_directory"] = "../outside/step_000000000007"

            with self.assertRaises(OpponentPoolConfigurationError):
                load_opponent_pool_manifest(
                    _write_manifest(root, [entry]),
                    policy=_policy(),
                )

    def test_actual_actor_tampering_is_detected_before_and_after_pool_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, entry = _write_checkpoint(
                root,
                "alpha",
                step=7,
                actor_payload=b"alpha",
            )
            manifest = _write_manifest(root, [entry])
            pool = load_opponent_pool_manifest(manifest, policy=_policy())
            (directory / "actor.msgpack").write_bytes(b"tampered")

            with self.assertRaises(ValueError):
                verify_opponent_checkpoint(pool.entries[0])
            with self.assertRaises(ValueError):
                load_opponent_pool_manifest(manifest, policy=_policy())

    def test_manifest_schema_and_checkpoint_identity_must_be_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, entry = _write_checkpoint(root, "alpha", step=7, actor_payload=b"alpha")
            duplicate = dict(entry)
            duplicate["opponent_id"] = "beta"
            manifest = _write_manifest(root, [entry, duplicate])

            with self.assertRaises(OpponentPoolConfigurationError):
                load_opponent_pool_manifest(manifest, policy=_policy())

    def test_manifest_rejects_unknown_schema_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, entry = _write_checkpoint(root, "alpha", step=7, actor_payload=b"alpha")
            manifest = _write_manifest(root, [entry])
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            raw["unexpected"] = True
            manifest.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaises(OpponentPoolConfigurationError):
                load_opponent_pool_manifest(manifest, policy=_policy())


if __name__ == "__main__":
    unittest.main()

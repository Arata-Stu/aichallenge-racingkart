"""Checkpoint metadata and atomic path contract tests."""

from __future__ import annotations

import json

import jax.numpy as jnp
import pytest
from flax import struct

from lidar_racing_rl.sac.checkpoint import (
    canonical_config_sha256,
    load_checkpoint,
    read_checkpoint_metadata,
    resolve_checkpoint,
    save_checkpoint,
)


@struct.dataclass
class _ToyState:
    step: jnp.ndarray
    parameter: jnp.ndarray


def test_checkpoint_round_trip_and_latest_pointer(tmp_path) -> None:
    state = _ToyState(jnp.asarray(7), jnp.asarray([1.0, 2.0]))
    actor_variables = {"params": {"dense": {"kernel": jnp.ones((2, 2))}}}
    config_hash = canonical_config_sha256({"b": 2, "a": 1})
    saved = save_checkpoint(
        tmp_path,
        state,
        actor_variables,
        step=7,
        environment_transitions=448,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    )
    restored, metadata = load_checkpoint(
        tmp_path,
        _ToyState(jnp.asarray(0), jnp.zeros((2,))),
        expected_architecture_version="lidar-sac-v1",
        expected_config_sha256=config_hash,
    )

    assert resolve_checkpoint(tmp_path) == saved
    assert int(restored.step) == 7
    assert jnp.array_equal(restored.parameter, state.parameter)
    assert metadata.step == 7
    assert metadata.environment_transitions == 448


def test_checkpoint_detects_payload_tampering(tmp_path) -> None:
    config_hash = canonical_config_sha256({"a": 1})
    saved = save_checkpoint(
        tmp_path,
        _ToyState(jnp.asarray(1), jnp.asarray([3.0])),
        {"params": {"dense": {"kernel": jnp.ones((1, 1))}}},
        step=1,
        environment_transitions=64,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    )
    (saved / "learner.msgpack").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(saved, _ToyState(jnp.asarray(0), jnp.zeros((1,))))


def test_metadata_reader_rejects_unknown_fields(tmp_path) -> None:
    config_hash = canonical_config_sha256({"a": 1})
    saved = save_checkpoint(
        tmp_path,
        _ToyState(jnp.asarray(1), jnp.asarray([3.0])),
        {"params": {"dense": {"kernel": jnp.ones((1, 1))}}},
        step=1,
        environment_transitions=64,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    )
    metadata_path = saved / "metadata.json"
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    metadata_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        read_checkpoint_metadata(saved)


def test_checkpoint_retry_repairs_latest_only_for_identical_payload(tmp_path) -> None:
    state = _ToyState(jnp.asarray(7), jnp.asarray([1.0, 2.0]))
    actor_variables = {"params": {"dense": {"kernel": jnp.ones((2, 2))}}}
    config_hash = canonical_config_sha256({"a": 1})
    saved = save_checkpoint(
        tmp_path,
        state,
        actor_variables,
        step=7,
        environment_transitions=448,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    )
    (tmp_path / "LATEST").unlink()

    retried = save_checkpoint(
        tmp_path,
        state,
        actor_variables,
        step=7,
        environment_transitions=448,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    )

    assert retried == saved
    assert resolve_checkpoint(tmp_path) == saved


def test_checkpoint_retry_repairs_malformed_latest_pointer(tmp_path) -> None:
    state = _ToyState(jnp.asarray(7), jnp.asarray([1.0]))
    actor_variables = {"params": {"dense": {"kernel": jnp.ones((1, 1))}}}
    config_hash = canonical_config_sha256({"a": 1})
    saved = save_checkpoint(
        tmp_path,
        state,
        actor_variables,
        step=7,
        environment_transitions=448,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    )
    (tmp_path / "LATEST").write_text("../broken\n", encoding="utf-8")

    save_checkpoint(
        tmp_path,
        state,
        actor_variables,
        step=7,
        environment_transitions=448,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    )

    assert resolve_checkpoint(tmp_path) == saved


def test_checkpoint_retry_rejects_same_step_with_different_progress(tmp_path) -> None:
    state = _ToyState(jnp.asarray(7), jnp.asarray([1.0, 2.0]))
    actor_variables = {"params": {"dense": {"kernel": jnp.ones((2, 2))}}}
    config_hash = canonical_config_sha256({"a": 1})
    save_checkpoint(
        tmp_path,
        state,
        actor_variables,
        step=7,
        environment_transitions=448,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    )

    with pytest.raises(FileExistsError, match="environment_transitions"):
        save_checkpoint(
            tmp_path,
            state,
            actor_variables,
            step=7,
            environment_transitions=512,
            architecture_version="lidar-sac-v1",
            config_sha256=config_hash,
            root_commit="root-sha",
            submodule_commit="submodule-sha",
        )


def test_checkpoint_retry_rejects_corrupted_existing_payload(tmp_path) -> None:
    state = _ToyState(jnp.asarray(7), jnp.asarray([1.0]))
    actor_variables = {"params": {"dense": {"kernel": jnp.ones((1, 1))}}}
    config_hash = canonical_config_sha256({"a": 1})
    saved = save_checkpoint(
        tmp_path,
        state,
        actor_variables,
        step=7,
        environment_transitions=448,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    )
    (saved / "learner.msgpack").write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="checksum"):
        save_checkpoint(
            tmp_path,
            state,
            actor_variables,
            step=7,
            environment_transitions=448,
            architecture_version="lidar-sac-v1",
            config_sha256=config_hash,
            root_commit="root-sha",
            submodule_commit="submodule-sha",
        )


def test_retrying_older_checkpoint_does_not_rewind_latest(tmp_path) -> None:
    config_hash = canonical_config_sha256({"a": 1})
    actor_variables = {"params": {"dense": {"kernel": jnp.ones((1, 1))}}}
    older_state = _ToyState(jnp.asarray(7), jnp.asarray([1.0]))
    newer_state = _ToyState(jnp.asarray(8), jnp.asarray([2.0]))
    older = save_checkpoint(
        tmp_path,
        older_state,
        actor_variables,
        step=7,
        environment_transitions=448,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    )
    newer = save_checkpoint(
        tmp_path,
        newer_state,
        actor_variables,
        step=8,
        environment_transitions=512,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    )

    assert save_checkpoint(
        tmp_path,
        older_state,
        actor_variables,
        step=7,
        environment_transitions=448,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    ) == older
    assert resolve_checkpoint(tmp_path) == newer


@pytest.mark.parametrize("invalid_value", [True, 1.5, -1])
def test_metadata_rejects_invalid_environment_transition_count(
    tmp_path,
    invalid_value,
) -> None:
    config_hash = canonical_config_sha256({"a": 1})
    saved = save_checkpoint(
        tmp_path,
        _ToyState(jnp.asarray(1), jnp.asarray([3.0])),
        {"params": {"dense": {"kernel": jnp.ones((1, 1))}}},
        step=1,
        environment_transitions=64,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    )
    metadata_path = saved / "metadata.json"
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw["environment_transitions"] = invalid_value
    metadata_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="environment_transitions"):
        read_checkpoint_metadata(saved)


def test_metadata_rejects_previous_schema_version(tmp_path) -> None:
    config_hash = canonical_config_sha256({"a": 1})
    saved = save_checkpoint(
        tmp_path,
        _ToyState(jnp.asarray(1), jnp.asarray([3.0])),
        {"params": {"dense": {"kernel": jnp.ones((1, 1))}}},
        step=1,
        environment_transitions=64,
        architecture_version="lidar-sac-v1",
        config_sha256=config_hash,
        root_commit="root-sha",
        submodule_commit="submodule-sha",
    )
    metadata_path = saved / "metadata.json"
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw["schema_version"] = 1
    metadata_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported checkpoint schema"):
        read_checkpoint_metadata(saved)

"""Offline validation and deterministic selection for past-policy opponents.

This module deliberately stops before loading an Actor into the vectorized
environment. A manifest can be reviewed and sampled, but training integration
must be implemented and tested separately before phase 2e can be enabled.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from lidar_racing_rl.export.manifest import ARCHITECTURE_VERSION
from lidar_racing_rl.sac.checkpoint import (
    ACTOR_FILENAME,
    CheckpointMetadata,
    read_checkpoint_metadata,
)


OPPONENT_POOL_SCHEMA_VERSION = 1
OPPONENT_POOL_SETTINGS_SCHEMA_VERSION = 1
SAMPLING_ALGORITHM = "sha256_context_v1"
_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_UINT64_MAX = (1 << 64) - 1


class OpponentPoolConfigurationError(ValueError):
    """Raised when settings or manifests fail the sealed pool contract."""


class OpponentPoolNotIntegratedError(RuntimeError):
    """Raised when phase 2e is requested for live training."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OpponentPoolConfigurationError(f"{label} must be a mapping")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(mapping)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise OpponentPoolConfigurationError(
            f"{label} schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _strict_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OpponentPoolConfigurationError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    if maximum is not None and value > maximum:
        raise OpponentPoolConfigurationError(
            f"{label} must be less than or equal to {maximum}"
        )
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise OpponentPoolConfigurationError(f"{label} must be boolean")
    return value


def _lowercase_hex(value: Any, label: str, lengths: tuple[int, ...]) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        allowed = " or ".join(str(length) for length in lengths)
        raise OpponentPoolConfigurationError(
            f"{label} must be {allowed} lowercase hexadecimal characters"
        )
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise OpponentPoolConfigurationError(
            f"{label} must match {_IDENTIFIER_PATTERN.pattern!r}"
        )
    return value


def _string_collection(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise OpponentPoolConfigurationError(f"{label} must be a list")
    if any(not isinstance(item, str) for item in value):
        raise OpponentPoolConfigurationError(f"{label} must contain strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise OpponentPoolConfigurationError(f"{label} must not contain duplicates")
    return result


@dataclass(frozen=True)
class PoolValidationPolicy:
    """Externally reviewed allowlists; never sourced from the pool manifest."""

    expected_architecture_version: str
    allowed_training_config_sha256: frozenset[str]
    allowed_root_commits: frozenset[str]
    allowed_submodule_commits: frozenset[str]

    @classmethod
    def from_values(
        cls,
        *,
        expected_architecture_version: str,
        allowed_training_config_sha256: Collection[str],
        allowed_root_commits: Collection[str],
        allowed_submodule_commits: Collection[str],
    ) -> PoolValidationPolicy:
        policy = cls(
            expected_architecture_version=expected_architecture_version,
            allowed_training_config_sha256=frozenset(allowed_training_config_sha256),
            allowed_root_commits=frozenset(allowed_root_commits),
            allowed_submodule_commits=frozenset(allowed_submodule_commits),
        )
        policy.validate()
        return policy

    @classmethod
    def from_config(cls, value: Any) -> PoolValidationPolicy:
        label = "opponent_pool.validation"
        config = _mapping(value, label)
        _exact_keys(
            config,
            {
                "expected_architecture_version",
                "allowed_training_config_sha256",
                "allowed_root_commits",
                "allowed_submodule_commits",
            },
            label,
        )
        return cls.from_values(
            expected_architecture_version=config["expected_architecture_version"],
            allowed_training_config_sha256=_string_collection(
                config["allowed_training_config_sha256"],
                f"{label}.allowed_training_config_sha256",
            ),
            allowed_root_commits=_string_collection(
                config["allowed_root_commits"], f"{label}.allowed_root_commits"
            ),
            allowed_submodule_commits=_string_collection(
                config["allowed_submodule_commits"],
                f"{label}.allowed_submodule_commits",
            ),
        )

    def validate(self) -> None:
        if self.expected_architecture_version != ARCHITECTURE_VERSION:
            raise OpponentPoolConfigurationError(
                "opponent pool supports only lidar_actor_conv1d_v1"
            )
        collections = (
            (
                "allowed_training_config_sha256",
                self.allowed_training_config_sha256,
                (64,),
            ),
            ("allowed_root_commits", self.allowed_root_commits, (40, 64)),
            ("allowed_submodule_commits", self.allowed_submodule_commits, (40, 64)),
        )
        for label, values, lengths in collections:
            if not values:
                raise OpponentPoolConfigurationError(f"{label} must not be empty")
            for value in values:
                _lowercase_hex(value, label, lengths)

    def validate_metadata(self, metadata: CheckpointMetadata) -> None:
        self.validate()
        if metadata.architecture_version != self.expected_architecture_version:
            raise OpponentPoolConfigurationError(
                "checkpoint architecture is not approved for the opponent pool"
            )
        if metadata.config_sha256 not in self.allowed_training_config_sha256:
            raise OpponentPoolConfigurationError(
                "checkpoint training config hash is not approved for the opponent pool"
            )
        if metadata.root_commit not in self.allowed_root_commits:
            raise OpponentPoolConfigurationError(
                "checkpoint root commit is not approved for the opponent pool"
            )
        if metadata.submodule_commit not in self.allowed_submodule_commits:
            raise OpponentPoolConfigurationError(
                "checkpoint submodule commit is not approved for the opponent pool"
            )


@dataclass(frozen=True)
class OpponentPoolSettings:
    """Source configuration for offline validation; live integration is gated."""

    schema_version: int
    enabled: bool
    training_integration: bool
    manifest: str | None
    sampling_algorithm: str
    seed: int
    validation_policy: PoolValidationPolicy | None

    @classmethod
    def from_config(cls, value: Any) -> OpponentPoolSettings:
        outer = _mapping(value, "configuration")
        config = _mapping(outer.get("opponent_pool", outer), "opponent_pool")
        _exact_keys(
            config,
            {
                "schema_version",
                "enabled",
                "training_integration",
                "manifest",
                "sampling_algorithm",
                "seed",
                "validation",
            },
            "opponent_pool",
        )
        schema_version = _strict_int(
            config["schema_version"], "opponent_pool.schema_version"
        )
        if schema_version != OPPONENT_POOL_SETTINGS_SCHEMA_VERSION:
            raise OpponentPoolConfigurationError(
                "unsupported opponent pool settings schema version"
            )
        enabled = _strict_bool(config["enabled"], "opponent_pool.enabled")
        training_integration = _strict_bool(
            config["training_integration"], "opponent_pool.training_integration"
        )
        if training_integration:
            raise OpponentPoolNotIntegratedError(
                "checkpoint opponents are not integrated into training"
            )
        sampling_algorithm = config["sampling_algorithm"]
        if sampling_algorithm != SAMPLING_ALGORITHM:
            raise OpponentPoolConfigurationError(
                f"opponent_pool.sampling_algorithm must be {SAMPLING_ALGORITHM}"
            )
        manifest = config["manifest"]
        validation = _mapping(config["validation"], "opponent_pool.validation")
        if enabled:
            if not isinstance(manifest, str) or not manifest:
                raise OpponentPoolConfigurationError(
                    "enabled opponent pool requires a manifest path"
                )
            policy = PoolValidationPolicy.from_config(validation)
        else:
            if manifest is not None:
                raise OpponentPoolConfigurationError(
                    "disabled opponent pool must not name a manifest"
                )
            expected = {
                "expected_architecture_version",
                "allowed_training_config_sha256",
                "allowed_root_commits",
                "allowed_submodule_commits",
            }
            _exact_keys(validation, expected, "opponent_pool.validation")
            if validation["expected_architecture_version"] != ARCHITECTURE_VERSION:
                raise OpponentPoolConfigurationError(
                    "disabled pool must still declare the supported architecture"
                )
            for key in expected - {"expected_architecture_version"}:
                if _string_collection(validation[key], f"opponent_pool.validation.{key}"):
                    raise OpponentPoolConfigurationError(
                        "disabled pool validation allowlists must remain empty"
                    )
            policy = None
        return cls(
            schema_version=schema_version,
            enabled=enabled,
            training_integration=training_integration,
            manifest=manifest,
            sampling_algorithm=sampling_algorithm,
            seed=_strict_int(
                config["seed"],
                "opponent_pool.seed",
                maximum=_UINT64_MAX,
            ),
            validation_policy=policy,
        )

    def require_training_ready(self) -> None:
        """Always reject until an Actor-loading adapter receives integration tests."""

        raise OpponentPoolNotIntegratedError(
            "opponent pool is offline-only; no past policy may enter live training"
        )


@dataclass(frozen=True)
class OpponentCheckpoint:
    """One fully verified Actor checkpoint and immutable provenance snapshot."""

    opponent_id: str
    checkpoint_directory: Path
    actor_path: Path
    architecture_version: str
    training_config_sha256: str
    root_commit: str
    submodule_commit: str
    checkpoint_step: int
    environment_transitions: int
    actor_sha256: str


@dataclass(frozen=True)
class OpponentPool:
    """Validated pool entries sorted by stable opponent identifier."""

    pool_id: str
    sampling_algorithm: str
    entries: tuple[OpponentCheckpoint, ...]
    manifest_path: Path


def _read_json_object(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"opponent pool manifest does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OpponentPoolConfigurationError(
            f"opponent pool manifest is unreadable: {path}"
        ) from error
    return _mapping(raw, "opponent pool manifest")


def _resolve_checkpoint_directory(root: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise OpponentPoolConfigurationError(
            "opponent checkpoint_directory must be a non-empty relative path"
        )
    pure_path = PurePosixPath(raw_path)
    if pure_path.is_absolute() or ".." in pure_path.parts or "\\" in raw_path:
        raise OpponentPoolConfigurationError(
            "opponent checkpoint_directory must stay within the manifest directory"
        )
    try:
        checkpoint_directory = (root / Path(*pure_path.parts)).resolve(strict=True)
        checkpoint_directory.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise OpponentPoolConfigurationError(
            "opponent checkpoint_directory is missing or escapes the manifest directory"
        ) from error
    return checkpoint_directory


def _parse_entry(
    raw_entry: Any,
    *,
    manifest_root: Path,
    policy: PoolValidationPolicy,
) -> OpponentCheckpoint:
    entry = _mapping(raw_entry, "opponent pool entry")
    expected_fields = {
        "opponent_id",
        "checkpoint_directory",
        "architecture_version",
        "training_config_sha256",
        "root_commit",
        "submodule_commit",
        "checkpoint_step",
        "environment_transitions",
        "actor_sha256",
        "approved_for_opponent_pool",
    }
    _exact_keys(entry, expected_fields, "opponent pool entry")
    opponent_id = _identifier(entry["opponent_id"], "opponent_id")
    if entry["approved_for_opponent_pool"] is not True:
        raise OpponentPoolConfigurationError(
            f"opponent {opponent_id} has not been explicitly approved"
        )
    checkpoint_directory = _resolve_checkpoint_directory(
        manifest_root,
        entry["checkpoint_directory"],
    )
    metadata = read_checkpoint_metadata(checkpoint_directory)
    policy.validate_metadata(metadata)
    declared = {
        "architecture_version": entry["architecture_version"],
        "config_sha256": _lowercase_hex(
            entry["training_config_sha256"],
            f"{opponent_id}.training_config_sha256",
            (64,),
        ),
        "root_commit": _lowercase_hex(
            entry["root_commit"], f"{opponent_id}.root_commit", (40, 64)
        ),
        "submodule_commit": _lowercase_hex(
            entry["submodule_commit"],
            f"{opponent_id}.submodule_commit",
            (40, 64),
        ),
        "step": _strict_int(entry["checkpoint_step"], f"{opponent_id}.checkpoint_step"),
        "environment_transitions": _strict_int(
            entry["environment_transitions"],
            f"{opponent_id}.environment_transitions",
        ),
        "actor_payload_sha256": _lowercase_hex(
            entry["actor_sha256"], f"{opponent_id}.actor_sha256", (64,)
        ),
    }
    mismatched = [
        name
        for name, value in declared.items()
        if getattr(metadata, name) != value
    ]
    if mismatched:
        raise OpponentPoolConfigurationError(
            f"opponent {opponent_id} manifest disagrees with checkpoint metadata: "
            f"{', '.join(mismatched)}"
        )
    return OpponentCheckpoint(
        opponent_id=opponent_id,
        checkpoint_directory=checkpoint_directory,
        actor_path=checkpoint_directory / ACTOR_FILENAME,
        architecture_version=metadata.architecture_version,
        training_config_sha256=metadata.config_sha256,
        root_commit=metadata.root_commit,
        submodule_commit=metadata.submodule_commit,
        checkpoint_step=metadata.step,
        environment_transitions=metadata.environment_transitions,
        actor_sha256=metadata.actor_payload_sha256,
    )


def load_opponent_pool_manifest(
    manifest_path: Path,
    *,
    policy: PoolValidationPolicy,
) -> OpponentPool:
    """Verify a pool against external allowlists and the actual checkpoint bytes."""

    policy.validate()
    resolved_manifest = manifest_path.resolve(strict=True)
    manifest = _read_json_object(resolved_manifest)
    _exact_keys(
        manifest,
        {"schema_version", "pool_id", "sampling_algorithm", "entries"},
        "opponent pool manifest",
    )
    schema_version = _strict_int(
        manifest["schema_version"], "opponent pool manifest schema_version"
    )
    if schema_version != OPPONENT_POOL_SCHEMA_VERSION:
        raise OpponentPoolConfigurationError(
            "unsupported opponent pool manifest schema version"
        )
    pool_id = _identifier(manifest["pool_id"], "pool_id")
    sampling_algorithm = manifest["sampling_algorithm"]
    if sampling_algorithm != SAMPLING_ALGORITHM:
        raise OpponentPoolConfigurationError(
            f"pool sampling_algorithm must be {SAMPLING_ALGORITHM}"
        )
    raw_entries = manifest["entries"]
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, str | bytes):
        raise OpponentPoolConfigurationError("opponent pool entries must be a list")
    if not raw_entries:
        raise OpponentPoolConfigurationError("opponent pool must not be empty")
    manifest_root = resolved_manifest.parent.resolve(strict=True)
    entries = tuple(
        sorted(
            (
                _parse_entry(entry, manifest_root=manifest_root, policy=policy)
                for entry in raw_entries
            ),
            key=lambda candidate: candidate.opponent_id,
        )
    )
    identifiers = [entry.opponent_id for entry in entries]
    directories = [entry.checkpoint_directory for entry in entries]
    if len(set(identifiers)) != len(identifiers):
        raise OpponentPoolConfigurationError("opponent pool identifiers must be unique")
    if len(set(directories)) != len(directories):
        raise OpponentPoolConfigurationError(
            "one checkpoint directory must not appear under multiple opponent identifiers"
        )
    return OpponentPool(
        pool_id=pool_id,
        sampling_algorithm=sampling_algorithm,
        entries=entries,
        manifest_path=resolved_manifest,
    )


def load_configured_opponent_pool(
    settings: OpponentPoolSettings,
    *,
    base_directory: Path,
) -> OpponentPool:
    """Resolve and validate an enabled offline pool below a caller-owned root."""

    if not settings.enabled or settings.manifest is None:
        raise OpponentPoolConfigurationError("opponent pool is disabled")
    if settings.validation_policy is None:
        raise OpponentPoolConfigurationError(
            "enabled opponent pool has no validation policy"
        )
    pure_path = PurePosixPath(settings.manifest)
    if pure_path.is_absolute() or ".." in pure_path.parts or "\\" in settings.manifest:
        raise OpponentPoolConfigurationError(
            "configured opponent pool manifest must stay below base_directory"
        )
    root = base_directory.resolve(strict=True)
    try:
        manifest_path = (root / Path(*pure_path.parts)).resolve(strict=True)
        manifest_path.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise OpponentPoolConfigurationError(
            "configured opponent pool manifest is missing or escapes base_directory"
        ) from error
    return load_opponent_pool_manifest(
        manifest_path,
        policy=settings.validation_policy,
    )


def verify_opponent_checkpoint(candidate: OpponentCheckpoint) -> None:
    """Re-verify an entry immediately before a future Actor loader consumes it."""

    metadata = read_checkpoint_metadata(candidate.checkpoint_directory)
    expected = {
        "architecture_version": candidate.architecture_version,
        "config_sha256": candidate.training_config_sha256,
        "root_commit": candidate.root_commit,
        "submodule_commit": candidate.submodule_commit,
        "step": candidate.checkpoint_step,
        "environment_transitions": candidate.environment_transitions,
        "actor_payload_sha256": candidate.actor_sha256,
    }
    mismatched = [
        name for name, value in expected.items() if getattr(metadata, name) != value
    ]
    if mismatched:
        raise OpponentPoolConfigurationError(
            "opponent checkpoint provenance changed after pool validation: "
            f"{', '.join(mismatched)}"
        )


def _uint64(value: Any, label: str) -> int:
    return _strict_int(value, label, maximum=_UINT64_MAX)


def _sampling_context(
    pool: OpponentPool,
    *,
    seed: int,
    environment_index: int,
    episode_index: int,
) -> bytes:
    return b"\x00".join(
        (
            b"lidar-racing-opponent-pool-v1",
            pool.pool_id.encode("ascii"),
            _uint64(seed, "seed").to_bytes(8, "big"),
            _uint64(environment_index, "environment_index").to_bytes(8, "big"),
            _uint64(episode_index, "episode_index").to_bytes(8, "big"),
        )
    )


def deterministic_pool_sample(
    pool: OpponentPool,
    *,
    opponent_count: int,
    seed: int,
    environment_index: int,
    episode_index: int,
    replace: bool = False,
) -> tuple[OpponentCheckpoint, ...]:
    """Select stable opponents without importing JAX or relying on RNG versions."""

    count = _strict_int(opponent_count, "opponent_count", minimum=1)
    if not isinstance(replace, bool):
        raise OpponentPoolConfigurationError("replace must be boolean")
    if pool.sampling_algorithm != SAMPLING_ALGORITHM:
        raise OpponentPoolConfigurationError("opponent pool sampling algorithm changed")
    entries = tuple(sorted(pool.entries, key=lambda entry: entry.opponent_id))
    if not entries:
        raise OpponentPoolConfigurationError("opponent pool must not be empty")
    identifiers = tuple(entry.opponent_id for entry in entries)
    if len(set(identifiers)) != len(identifiers):
        raise OpponentPoolConfigurationError("opponent identifiers must be unique")
    if not replace and count > len(entries):
        raise OpponentPoolConfigurationError(
            "opponent_count exceeds pool size without replacement"
        )
    context = _sampling_context(
        pool,
        seed=seed,
        environment_index=environment_index,
        episode_index=episode_index,
    )
    if not replace:
        ranked = sorted(
            entries,
            key=lambda entry: (
                hashlib.sha256(
                    context + b"\x00candidate\x00" + entry.opponent_id.encode("ascii")
                ).digest(),
                entry.opponent_id,
            ),
        )
        return tuple(ranked[:count])

    selected: list[OpponentCheckpoint] = []
    for slot in range(count):
        digest = hashlib.sha256(
            context + b"\x00slot\x00" + slot.to_bytes(8, "big")
        ).digest()
        selected.append(entries[int.from_bytes(digest, "big") % len(entries)])
    return tuple(selected)


__all__ = [
    "OPPONENT_POOL_SCHEMA_VERSION",
    "OPPONENT_POOL_SETTINGS_SCHEMA_VERSION",
    "SAMPLING_ALGORITHM",
    "OpponentCheckpoint",
    "OpponentPool",
    "OpponentPoolConfigurationError",
    "OpponentPoolNotIntegratedError",
    "OpponentPoolSettings",
    "PoolValidationPolicy",
    "deterministic_pool_sample",
    "load_configured_opponent_pool",
    "load_opponent_pool_manifest",
    "verify_opponent_checkpoint",
]

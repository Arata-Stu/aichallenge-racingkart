"""Portable, atomic learner checkpoints with explicit compatibility metadata."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


CHECKPOINT_SCHEMA_VERSION = 2
LATEST_FILENAME = "LATEST"
PAYLOAD_FILENAME = "learner.msgpack"
ACTOR_FILENAME = "actor.msgpack"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class CheckpointMetadata:
    """Fields required to reject incompatible or corrupted resumes."""

    schema_version: int
    architecture_version: str
    step: int
    environment_transitions: int
    config_sha256: str
    root_commit: str
    submodule_commit: str
    payload_sha256: str
    actor_payload_sha256: str
    created_at_utc: str

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> CheckpointMetadata:
        required = {field.name for field in cls.__dataclass_fields__.values()}
        missing = required - raw.keys()
        extra = raw.keys() - required
        if missing or extra:
            raise ValueError(
                f"checkpoint metadata schema mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        metadata = cls(**raw)
        metadata.validate()
        return metadata

    def validate(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported checkpoint schema version")
        for name in ("step", "environment_transitions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"checkpoint {name} must be a non-negative integer")
        for name in ("config_sha256", "payload_sha256", "actor_payload_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"checkpoint {name} is not a lowercase SHA-256")
        for name in ("architecture_version", "root_commit", "submodule_commit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"checkpoint {name} must not be empty")
        if not isinstance(self.created_at_utc, str) or not self.created_at_utc:
            raise ValueError("checkpoint created_at_utc must not be empty")


def canonical_config_sha256(config: Any) -> str:
    """Hash a resolved JSON-compatible config without ordering ambiguity."""

    serialized = json.dumps(
        config,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def checkpoint_config_sha256(config: Any) -> str:
    """Hash learning semantics while ignoring the operational resume location."""

    normalized = json.loads(
        json.dumps(config, allow_nan=False, ensure_ascii=False, sort_keys=True)
    )
    if isinstance(normalized, dict):
        training = normalized.get("training")
        if isinstance(training, dict):
            training["resume_from"] = None
    return canonical_config_sha256(normalized)


def _write_file_sync(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _checkpoint_root_lock(checkpoint_root: Path) -> Iterator[None]:
    """Serialize numbered publication and ``LATEST`` updates per root.

    The lock is advisory, so every writer must publish through
    :func:`save_checkpoint`. Both supported execution hosts (Linux Docker and
    macOS) provide ``flock`` through the standard-library ``fcntl`` module.
    """

    lock_path = checkpoint_root / ".checkpoint.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    with os.fdopen(descriptor, "a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _checkpoint_name(step: int) -> str:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    return f"step_{step:012d}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_numbered_checkpoint(path: Path) -> CheckpointMetadata:
    """Validate one exact numbered directory without following ``LATEST``."""

    expected_names = {PAYLOAD_FILENAME, ACTOR_FILENAME, METADATA_FILENAME}
    if not path.is_dir():
        raise ValueError(f"checkpoint path is not a directory: {path}")
    actual_names = {entry.name for entry in path.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            "checkpoint directory is incomplete or contains unexpected files: "
            f"{path}"
        )
    try:
        raw = json.loads((path / METADATA_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"checkpoint metadata is unreadable: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    metadata = CheckpointMetadata.from_json(raw)
    if path.name != _checkpoint_name(metadata.step):
        raise ValueError("checkpoint directory name does not match metadata step")
    if _sha256_file(path / PAYLOAD_FILENAME) != metadata.payload_sha256:
        raise ValueError("checkpoint payload checksum mismatch")
    if _sha256_file(path / ACTOR_FILENAME) != metadata.actor_payload_sha256:
        raise ValueError("checkpoint Actor payload checksum mismatch")
    return metadata


def _write_latest(checkpoint_root: Path, checkpoint_name: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{LATEST_FILENAME}.",
        dir=checkpoint_root,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        _write_file_sync(temporary_path, f"{checkpoint_name}\n".encode("utf-8"))
        os.replace(temporary_path, checkpoint_root / LATEST_FILENAME)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _repair_latest_without_rollback(checkpoint_root: Path, final_path: Path) -> None:
    """Publish ``final_path`` unless a newer valid checkpoint is already latest."""

    newer_paths = sorted(
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir()
        and re.fullmatch(r"step_[0-9]{12}", path.name)
        and path.name > final_path.name
    )
    latest_path = checkpoint_root / LATEST_FILENAME
    latest_name: str | None = None
    if latest_path.is_file():
        try:
            candidate = latest_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ValueError("LATEST pointer is unreadable") from error
        if (
            candidate
            and Path(candidate).name == candidate
            and re.fullmatch(r"step_[0-9]{12}", candidate)
        ):
            latest_name = candidate

    if newer_paths:
        if latest_name is None or latest_name <= final_path.name:
            raise ValueError(
                "refusing to repair LATEST to an older checkpoint while newer "
                "numbered checkpoints exist"
            )
        current_latest = checkpoint_root / latest_name
        _read_numbered_checkpoint(current_latest)
        return
    if latest_name == final_path.name:
        return
    if latest_name is not None and latest_name > final_path.name:
        raise ValueError("LATEST points to a missing or invalid newer checkpoint")
    _write_latest(checkpoint_root, final_path.name)


def _recover_matching_checkpoint(
    checkpoint_root: Path,
    final_path: Path,
    desired_metadata: CheckpointMetadata,
) -> Path:
    """Accept an interrupted same-payload publication and repair ``LATEST``."""

    existing = _read_numbered_checkpoint(final_path)
    identity_fields = (
        "schema_version",
        "architecture_version",
        "step",
        "environment_transitions",
        "config_sha256",
        "root_commit",
        "submodule_commit",
        "payload_sha256",
        "actor_payload_sha256",
    )
    mismatched = [
        name
        for name in identity_fields
        if getattr(existing, name) != getattr(desired_metadata, name)
    ]
    if mismatched:
        raise FileExistsError(
            "existing checkpoint does not match the requested retry; "
            f"fields={', '.join(mismatched)}: {final_path}"
        )
    _repair_latest_without_rollback(checkpoint_root, final_path)
    return final_path


def save_checkpoint(
    checkpoint_root: Path,
    learner_state: Any,
    actor_variables: Any,
    *,
    step: int,
    environment_transitions: int,
    architecture_version: str,
    config_sha256: str,
    root_commit: str,
    submodule_commit: str,
) -> Path:
    """Serialize a learner pytree and atomically publish a numbered directory.

    Replay storage and rollout RNG are intentionally not accepted here;
    checkpoints remain small enough for Flax msgpack and contain the model,
    optimizer, target-Critic, learner step, and entropy-temperature state.
    Resume is therefore a documented replay-recollection warm restart rather
    than a bit-exact collector continuation. Existing numbered checkpoints are
    never overwritten.
    """

    from flax import serialization

    checkpoint_root = checkpoint_root.resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    payload = serialization.to_bytes(learner_state)
    actor_payload = serialization.msgpack_serialize(
        serialization.to_state_dict(actor_variables)
    )
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    metadata = CheckpointMetadata(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        architecture_version=architecture_version,
        step=step,
        environment_transitions=environment_transitions,
        config_sha256=config_sha256,
        root_commit=root_commit,
        submodule_commit=submodule_commit,
        payload_sha256=payload_sha256,
        actor_payload_sha256=hashlib.sha256(actor_payload).hexdigest(),
        created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    metadata.validate()
    final_path = checkpoint_root / _checkpoint_name(step)
    metadata_bytes = (
        json.dumps(asdict(metadata), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )

    with _checkpoint_root_lock(checkpoint_root):
        if os.path.lexists(final_path):
            return _recover_matching_checkpoint(checkpoint_root, final_path, metadata)
        temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{final_path.name}.", dir=checkpoint_root)
        )
        try:
            _write_file_sync(temporary_path / PAYLOAD_FILENAME, payload)
            _write_file_sync(temporary_path / ACTOR_FILENAME, actor_payload)
            _write_file_sync(temporary_path / METADATA_FILENAME, metadata_bytes)
            try:
                os.replace(temporary_path, final_path)
            except OSError:
                if os.path.lexists(final_path):
                    if temporary_path.exists():
                        shutil.rmtree(temporary_path)
                    return _recover_matching_checkpoint(
                        checkpoint_root,
                        final_path,
                        metadata,
                    )
                raise
            _repair_latest_without_rollback(checkpoint_root, final_path)
        except BaseException:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)
            raise
    return final_path


def resolve_checkpoint(path: Path) -> Path:
    """Resolve either a numbered checkpoint directory or a root with LATEST."""

    path = path.resolve()
    if (
        (path / PAYLOAD_FILENAME).is_file()
        and (path / ACTOR_FILENAME).is_file()
        and (path / METADATA_FILENAME).is_file()
    ):
        return path
    latest_path = path / LATEST_FILENAME
    if not latest_path.is_file():
        raise FileNotFoundError(f"checkpoint or LATEST pointer not found: {path}")
    name = latest_path.read_text(encoding="utf-8").strip()
    if not name or Path(name).name != name:
        raise ValueError("LATEST must contain one local checkpoint directory name")
    resolved = path / name
    if not resolved.is_dir():
        raise FileNotFoundError(f"LATEST checkpoint directory is missing: {resolved}")
    return resolved


def read_checkpoint_metadata(path: Path) -> CheckpointMetadata:
    """Read and validate metadata without importing JAX or Flax."""

    checkpoint_path = resolve_checkpoint(path)
    return _read_numbered_checkpoint(checkpoint_path)


def load_checkpoint(
    path: Path,
    template_state: Any,
    *,
    expected_architecture_version: str | None = None,
    expected_config_sha256: str | None = None,
) -> tuple[Any, CheckpointMetadata]:
    """Verify integrity and restore bytes into an initialized state template."""

    from flax import serialization

    checkpoint_path = resolve_checkpoint(path)
    metadata = read_checkpoint_metadata(checkpoint_path)
    if (
        expected_architecture_version is not None
        and metadata.architecture_version != expected_architecture_version
    ):
        raise ValueError("checkpoint architecture version does not match the learner")
    if expected_config_sha256 is not None and metadata.config_sha256 != expected_config_sha256:
        raise ValueError("checkpoint config hash does not match the resolved run config")
    payload = (checkpoint_path / PAYLOAD_FILENAME).read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata.payload_sha256:
        raise ValueError("checkpoint payload checksum changed while it was being loaded")
    return serialization.from_bytes(template_state, payload), metadata


def load_actor_variables(path: Path) -> tuple[dict[str, Any], CheckpointMetadata]:
    """Restore standalone Actor variables for evaluation/export without a learner template."""

    from flax import serialization

    checkpoint_path = resolve_checkpoint(path)
    metadata = read_checkpoint_metadata(checkpoint_path)
    payload = (checkpoint_path / ACTOR_FILENAME).read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata.actor_payload_sha256:
        raise ValueError("checkpoint Actor checksum changed while it was being loaded")
    restored = serialization.msgpack_restore(payload)
    if not isinstance(restored, dict) or "params" not in restored:
        raise ValueError("checkpoint Actor payload must contain a params mapping")
    return restored, metadata


def prune_checkpoints(checkpoint_root: Path, keep_last: int) -> tuple[Path, ...]:
    """Remove only old, strictly named numbered checkpoints after a successful save."""

    if isinstance(keep_last, bool) or keep_last < 1:
        raise ValueError("keep_last must be positive")
    root = checkpoint_root.resolve()
    if not root.is_dir():
        return ()
    candidates = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and re.fullmatch(r"step_[0-9]{12}", path.name)
        ),
        key=lambda path: path.name,
    )
    removed = tuple(candidates[:-keep_last])
    for path in removed:
        shutil.rmtree(path)
    return removed


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointMetadata",
    "canonical_config_sha256",
    "checkpoint_config_sha256",
    "load_actor_variables",
    "load_checkpoint",
    "prune_checkpoints",
    "read_checkpoint_metadata",
    "resolve_checkpoint",
    "save_checkpoint",
]

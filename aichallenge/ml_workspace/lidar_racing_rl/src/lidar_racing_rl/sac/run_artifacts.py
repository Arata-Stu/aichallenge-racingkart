"""Reproducibility artifacts for a LiDAR SAC run."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RepositorySnapshot:
    """Read-only Git provenance captured before training starts."""

    root_commit: str
    root_status: str
    root_diff: str
    submodule_commits: dict[str, str]
    submodule_status: dict[str, str]
    submodule_diffs: dict[str, str]


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    return completed.stdout.rstrip()


def _git_diff_untracked(repo: Path) -> str:
    """Represent non-ignored untracked files as ordinary binary-safe patches."""

    names = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    patches: list[str] = []
    for encoded_name in names:
        if not encoded_name:
            continue
        name = encoded_name.decode("utf-8", errors="surrogateescape")
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--binary",
                "--no-index",
                "--",
                "/dev/null",
                name,
            ],
            check=False,
            capture_output=True,
        )
        if completed.returncode not in (0, 1):
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        patches.append(completed.stdout.decode("utf-8", errors="surrogateescape").rstrip())
    return "\n".join(patch for patch in patches if patch)


def _complete_diff(repo: Path) -> str:
    tracked = _git(repo, "diff", "--binary", "HEAD", "--", ".")
    untracked = _git_diff_untracked(repo)
    return "\n".join(part for part in (tracked, untracked) if part)


def capture_repository_snapshot(
    repository_root: Path,
    submodule_paths: Iterable[Path],
) -> RepositorySnapshot:
    """Capture commits and dirty state without changing any repository ref."""

    root = repository_root.resolve()
    commits: dict[str, str] = {}
    statuses: dict[str, str] = {}
    diffs: dict[str, str] = {}
    for raw_path in submodule_paths:
        path = raw_path.resolve()
        try:
            relative = str(path.relative_to(root))
        except ValueError as error:
            raise ValueError("submodule path must stay within repository_root") from error
        commits[relative] = _git(path, "rev-parse", "HEAD")
        statuses[relative] = _git(path, "status", "--short", "--branch")
        diffs[relative] = _complete_diff(path)
    return RepositorySnapshot(
        root_commit=_git(root, "rev-parse", "HEAD"),
        root_status=_git(root, "status", "--short", "--branch"),
        root_diff=_complete_diff(root),
        submodule_commits=commits,
        submodule_status=statuses,
        submodule_diffs=diffs,
    )


def capture_environment() -> dict[str, Any]:
    """Collect non-invasive runtime metadata without importing JAX."""

    packages: dict[str, str] = {}
    for distribution in (
        "distrax",
        "flax",
        "hydra-core",
        "jax",
        "jaxlib",
        "numpy",
        "omegaconf",
        "optax",
        "torch",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def _atomic_write(path: Path, content: str | bytes) -> None:
    payload = content.encode("utf-8") if isinstance(content, str) else content
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _status_is_dirty(status: str) -> bool:
    return any(line and not line.startswith("## ") for line in status.splitlines())


def write_run_artifacts(
    run_directory: Path,
    *,
    resolved_config_yaml: str,
    repository: RepositorySnapshot,
    environment: dict[str, Any] | None = None,
) -> None:
    """Write the blueprint provenance files atomically before collection."""

    run_directory = run_directory.resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    files = {
        "resolved_config.yaml": resolved_config_yaml.rstrip() + "\n",
        "git_status.txt": repository.root_status.rstrip() + "\n",
        "git_diff.patch": repository.root_diff.rstrip() + "\n",
        "root_commit.txt": repository.root_commit.rstrip() + "\n",
        "submodule_commits.txt": (
            json.dumps(repository.submodule_commits, indent=2, sort_keys=True) + "\n"
        ),
        "submodule_status.txt": (
            json.dumps(repository.submodule_status, indent=2, sort_keys=True) + "\n"
        ),
        "submodule_diffs.patch": (
            json.dumps(repository.submodule_diffs, indent=2, sort_keys=True) + "\n"
        ),
        "environment.txt": (
            json.dumps(environment or capture_environment(), indent=2, sort_keys=True)
            + "\n"
        ),
        "run_manifest.json": (
            json.dumps(
                {
                    "schema_version": 1,
                    "root_commit": repository.root_commit,
                    "root_dirty": _status_is_dirty(repository.root_status),
                    "submodule_commits": repository.submodule_commits,
                    "submodule_dirty": {
                        path: _status_is_dirty(status)
                        for path, status in repository.submodule_status.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ),
    }
    existing = [name for name in files if (run_directory / name).exists()]
    if existing:
        raise FileExistsError(f"run provenance already exists: {', '.join(sorted(existing))}")
    for name, content in files.items():
        _atomic_write(run_directory / name, content)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one finite JSON metrics record and flush it to durable storage."""

    serialized = json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


__all__ = [
    "RepositorySnapshot",
    "append_jsonl",
    "capture_environment",
    "capture_repository_snapshot",
    "write_run_artifacts",
]

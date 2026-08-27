#!/usr/bin/env python3
"""Report the JAX runtime and repository provenance used for training."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[2]
F1TENTH_SUBMODULE = PROJECT_ROOT / "repos" / "f1tenth_gym_jax"


def _distribution_version(name: str) -> str:
    """Return an installed distribution version without importing the package."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _git_commit(path: Path) -> str:
    """Return the current commit for a Git worktree or a readable status."""
    if not path.exists():
        return "not initialized"

    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"

    return result.stdout.strip()


def _jax_environment() -> dict[str, str]:
    """Collect explicitly configured JAX/XLA environment variables."""
    prefixes = ("JAX_", "XLA_")
    return {key: value for key, value in sorted(os.environ.items()) if key.startswith(prefixes)}


def main() -> int:
    """Print backend diagnostics and return non-zero when JAX cannot initialize."""
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"JAX distribution: {_distribution_version('jax')}")
    print(f"jaxlib distribution: {_distribution_version('jaxlib')}")
    print(f"Root repository commit: {_git_commit(REPOSITORY_ROOT)}")
    print(f"F1TENTH Gym JAX commit: {_git_commit(F1TENTH_SUBMODULE)}")

    environment = _jax_environment()
    if environment:
        print("JAX/XLA settings:")
        for key, value in environment.items():
            print(f"  {key}={value}")
    else:
        print("JAX/XLA settings: none")

    try:
        import jax

        backend = jax.default_backend()
        devices = jax.devices()
    except Exception as error:  # Backend/plugin failures should remain visible to the caller.
        print(f"JAX initialization: FAILED ({type(error).__name__}: {error})", file=sys.stderr)
        return 1

    print(f"JAX import version: {jax.__version__}")
    print(f"JAX backend: {backend}")
    print(f"JAX device count: {len(devices)}")
    for index, device in enumerate(devices):
        device_kind = getattr(device, "device_kind", "unknown")
        print(f"  [{index}] platform={device.platform} kind={device_kind} id={device.id}")

    gpu_names = sorted(
        {
            str(getattr(device, "device_kind", "unknown"))
            for device in devices
            if device.platform == "gpu"
        }
    )
    print(f"GPU: {', '.join(gpu_names) if gpu_names else 'not available'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

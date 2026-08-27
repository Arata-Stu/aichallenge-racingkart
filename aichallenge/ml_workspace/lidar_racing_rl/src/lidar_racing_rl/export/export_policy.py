"""Complete, fail-closed Flax Actor to PyTorch deployment export."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lidar_racing_rl.export.manifest import (
    ARCHITECTURE_VERSION,
    build_policy_manifest,
    sha256_file,
    write_policy_manifest,
)
from lidar_racing_rl.models.parameter_conversion import (
    build_torch_actor_from_flax,
    deterministic_parity_error,
    save_torch_state_dict,
)
from lidar_racing_rl.sac.checkpoint import (
    ACTOR_FILENAME,
    checkpoint_config_sha256,
    load_actor_variables,
    resolve_checkpoint,
)


@dataclass(frozen=True)
class ExportResult:
    """Published artifact paths and measured conversion error."""

    output_directory: Path
    flax_policy: Path
    torch_policy: Path
    manifest: Path
    config_snapshot: Path
    maximum_absolute_error: float


def _value(mapping: Mapping[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"missing export configuration: {'.'.join(path)}")
        current = current[key]
    return current


def export_policy_bundle(
    *,
    checkpoint: Path,
    output_directory: Path,
    deployment_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    config_snapshot_yaml: str,
    parity_seed: int = 0,
    parity_batch_size: int = 16,
) -> ExportResult:
    """Convert, verify, and atomically publish all blueprint artifacts."""

    from lidar_racing_rl.models.actor_flax import TanhGaussianActor

    if isinstance(parity_batch_size, bool) or parity_batch_size < 1:
        raise ValueError("parity_batch_size must be positive")
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"export output already exists: {output_directory}")

    resolved_checkpoint = resolve_checkpoint(checkpoint)
    actor_variables, checkpoint_metadata = load_actor_variables(resolved_checkpoint)
    if checkpoint_metadata.architecture_version != ARCHITECTURE_VERSION:
        raise ValueError("checkpoint architecture is not supported by this exporter")
    if checkpoint_config_sha256(training_config) != checkpoint_metadata.config_sha256:
        raise ValueError("training config snapshot does not match checkpoint config hash")
    actor = TanhGaussianActor()
    torch_actor = build_torch_actor_from_flax(actor_variables)

    generator = np.random.default_rng(parity_seed)
    parity_input = generator.uniform(
        0.0,
        1.0,
        size=(parity_batch_size, 8, 360),
    ).astype(np.float32)
    maximum_absolute_error = deterministic_parity_error(
        actor,
        actor_variables,
        torch_actor,
        parity_input,
    )
    tolerance = float(_value(deployment_config, "model", "flax_torch_max_absolute_error"))
    if not np.isfinite(maximum_absolute_error) or maximum_absolute_error > tolerance:
        raise ValueError(
            "Flax/PyTorch parity failed: "
            f"max_absolute_error={maximum_absolute_error}, tolerance={tolerance}"
        )

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
    )
    try:
        flax_path = temporary_directory / "policy_flax.msgpack"
        torch_path = temporary_directory / "policy_torch.pt"
        manifest_path = temporary_directory / "policy_manifest.json"
        config_path = temporary_directory / "config_snapshot.yaml"

        shutil.copyfile(resolved_checkpoint / ACTOR_FILENAME, flax_path)
        save_torch_state_dict(torch_path, torch_actor.state_dict())

        preprocessing = _value(deployment_config, "preprocessing")
        control = _value(deployment_config, "control")
        manifest = build_policy_manifest(
            model_checksum_sha256=sha256_file(torch_path),
            training_config_hash=checkpoint_metadata.config_sha256,
            root_repository_commit=checkpoint_metadata.root_commit,
            f1tenth_gym_jax_commit=checkpoint_metadata.submodule_commit,
            beam_count=int(_value(preprocessing, "canonical_beams")),
            frame_stack=int(_value(preprocessing, "frame_stack")),
            scan_channels=len(tuple(_value(preprocessing, "channels"))),
            field_of_view=float(_value(preprocessing, "field_of_view")),
            range_max=float(_value(preprocessing, "expected_range_max")),
            steering_max_abs=float(_value(control, "steering_max_abs")),
            acceleration_min=float(_value(control, "acceleration_min")),
            acceleration_max=float(_value(control, "acceleration_max")),
        )
        manifest["conversion"] = {
            "parity_batch_size": parity_batch_size,
            "parity_seed": parity_seed,
            "maximum_absolute_error": maximum_absolute_error,
            "tolerance": tolerance,
        }
        write_policy_manifest(manifest_path, manifest)
        config_path.write_text(config_snapshot_yaml.rstrip() + "\n", encoding="utf-8")

        os.replace(temporary_directory, output_directory)
    except BaseException:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        raise

    return ExportResult(
        output_directory=output_directory,
        flax_policy=output_directory / "policy_flax.msgpack",
        torch_policy=output_directory / "policy_torch.pt",
        manifest=output_directory / "policy_manifest.json",
        config_snapshot=output_directory / "config_snapshot.yaml",
        maximum_absolute_error=maximum_absolute_error,
    )


def result_as_json(result: ExportResult) -> str:
    """Serialize a CLI-safe export summary."""

    return json.dumps(
        {
            "output_directory": str(result.output_directory),
            "policy_flax": str(result.flax_policy),
            "policy_torch": str(result.torch_policy),
            "policy_manifest": str(result.manifest),
            "config_snapshot": str(result.config_snapshot),
            "maximum_absolute_error": result.maximum_absolute_error,
        },
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )


__all__ = ["ExportResult", "export_policy_bundle", "result_as_json"]

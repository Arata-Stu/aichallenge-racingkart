"""PyTorch Actor architecture and fail-closed policy artifact loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn


SUPPORTED_ARCHITECTURE_VERSION = "lidar_actor_conv1d_v1"


class PolicyLoadError(RuntimeError):
    """Raised when a model artifact cannot be trusted or loaded."""


def _nonempty_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PolicyLoadError(f"manifest field {key!r} must be a non-empty string")
    return value.strip()


def _lowercase_hex(
    data: Mapping[str, object],
    key: str,
    *,
    lengths: tuple[int, ...],
) -> str:
    value = data.get(key)
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or re.fullmatch(r"[0-9a-f]+", value) is None
    ):
        expected = " or ".join(str(length) for length in lengths)
        raise PolicyLoadError(
            f"manifest field {key!r} must be {expected} lowercase hex characters"
        )
    return value


def _positive_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PolicyLoadError(f"manifest field {key!r} must be a positive integer")
    return value


def _mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise PolicyLoadError(f"manifest field {key!r} must be an object")
    return value


def _finite_float(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyLoadError(f"manifest field {key!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PolicyLoadError(f"manifest field {key!r} must be finite")
    return result


def _integer_tuple(
    data: Mapping[str, object],
    key: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    raw = data.get(key, default)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise PolicyLoadError(f"architecture field {key!r} must be an integer array")
    values = tuple(raw)
    if not values or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise PolicyLoadError(f"architecture field {key!r} must contain integers")
    if any(value <= 0 for value in values):
        raise PolicyLoadError(f"architecture field {key!r} must contain positive values")
    return values


@dataclass(frozen=True)
class ActorArchitecture:
    """Shape-defining parameters shared by the Flax and PyTorch actors."""

    beam_count: int = 360
    frame_stack: int = 4
    scan_channels: int = 2
    conv_channels: tuple[int, ...] = (32, 64, 64)
    kernel_sizes: tuple[int, ...] = (8, 4, 3)
    strides: tuple[int, ...] = (4, 2, 1)
    hidden_dim: int = 256
    action_dim: int = 2
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @property
    def input_channels(self) -> int:
        return self.frame_stack * self.scan_channels

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, object],
        architecture_data: Mapping[str, object],
    ) -> ActorArchitecture:
        architecture = cls(
            beam_count=_positive_int(manifest, "beam_count"),
            frame_stack=_positive_int(manifest, "frame_stack"),
            scan_channels=_positive_int(manifest, "scan_channels"),
            conv_channels=_integer_tuple(
                architecture_data,
                "conv_channels",
                (),
            ),
            kernel_sizes=_integer_tuple(
                architecture_data,
                "kernel_sizes",
                (),
            ),
            strides=_integer_tuple(architecture_data, "strides", ()),
            hidden_dim=_positive_int(architecture_data, "hidden_dim"),
            action_dim=_positive_int(architecture_data, "action_dim"),
            log_std_min=_finite_float(architecture_data, "log_std_min"),
            log_std_max=_finite_float(architecture_data, "log_std_max"),
        )
        architecture.validate()
        return architecture

    def validate(self) -> None:
        exact_contract = (
            self.beam_count == 360
            and self.frame_stack == 4
            and self.scan_channels == 2
            and self.conv_channels == (32, 64, 64)
            and self.kernel_sizes == (8, 4, 3)
            and self.strides == (4, 2, 1)
            and self.hidden_dim == 256
            and self.action_dim == 2
            and math.isclose(self.log_std_min, -5.0, rel_tol=0.0, abs_tol=1.0e-12)
            and math.isclose(self.log_std_max, 2.0, rel_tol=0.0, abs_tol=1.0e-12)
        )
        if not exact_contract:
            raise PolicyLoadError(
                "manifest architecture does not match lidar_actor_conv1d_v1"
            )


class LidarActor(nn.Module):
    """Three-layer Conv1D Gaussian Actor with deterministic tanh inference."""

    def __init__(self, architecture: ActorArchitecture | None = None):
        super().__init__()
        self.architecture = architecture or ActorArchitecture()
        self.architecture.validate()

        layers: list[nn.Module] = []
        input_channels = self.architecture.input_channels
        output_length = self.architecture.beam_count
        for output_channels, kernel_size, stride in zip(
            self.architecture.conv_channels,
            self.architecture.kernel_sizes,
            self.architecture.strides,
            strict=True,
        ):
            layers.append(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            layers.append(nn.ReLU())
            output_length = (output_length - kernel_size) // stride + 1
            if output_length <= 0:
                raise PolicyLoadError("Conv1D architecture collapses the LiDAR beam dimension")
            input_channels = output_channels

        self.encoder = nn.Sequential(*layers)
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * output_length, self.architecture.hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(self.architecture.hidden_dim, self.architecture.action_dim)
        self.log_std_head = nn.Linear(self.architecture.hidden_dim, self.architecture.action_dim)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected_shape = (
            self.architecture.input_channels,
            self.architecture.beam_count,
        )
        if observation.ndim != 3 or tuple(observation.shape[1:]) != expected_shape:
            raise ValueError(
                f"expected Actor input [batch, {expected_shape[0]}, {expected_shape[1]}], "
                f"got {tuple(observation.shape)}"
            )
        features = self.trunk(self.encoder(observation))
        mean = self.mean_head(features)
        log_std = torch.clamp(
            self.log_std_head(features),
            self.architecture.log_std_min,
            self.architecture.log_std_max,
        )
        return mean, log_std

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self(observation)
        return torch.tanh(mean)


@dataclass(frozen=True)
class ActionScaling:
    steering_max_abs: float
    acceleration_min: float
    acceleration_max: float


@dataclass(frozen=True)
class PolicyManifest:
    """Validated policy metadata required to reproduce deployment semantics."""

    architecture_version: str
    architecture: ActorArchitecture
    input_range_max: float
    field_of_view: float
    action_scaling: ActionScaling
    model_checksum_sha256: str
    training_config_hash: str
    root_repository_commit: str
    f1tenth_gym_jax_commit: str
    export_timestamp: str

    @classmethod
    def load(cls, path: str | Path) -> PolicyManifest:
        manifest_path = Path(path)
        if not manifest_path.is_file():
            raise PolicyLoadError(f"policy manifest not found: {manifest_path}")
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PolicyLoadError(f"failed to read policy manifest: {error}") from error
        if not isinstance(raw, Mapping):
            raise PolicyLoadError("policy manifest root must be a JSON object")

        architecture_version = _nonempty_string(raw, "architecture_version")
        if architecture_version != SUPPORTED_ARCHITECTURE_VERSION:
            raise PolicyLoadError(
                f"unsupported architecture_version {architecture_version!r}; "
                f"expected {SUPPORTED_ARCHITECTURE_VERSION!r}"
            )
        architecture_data = raw.get("architecture", {})
        if not isinstance(architecture_data, Mapping):
            raise PolicyLoadError("manifest architecture must be an object")
        architecture = ActorArchitecture.from_manifest(raw, architecture_data)

        normalization = _mapping(raw, "range_normalization")
        if _nonempty_string(normalization, "type") != "divide_by_range_max":
            raise PolicyLoadError("unsupported range normalization")
        if not math.isclose(_finite_float(normalization, "output_min"), 0.0):
            raise PolicyLoadError("normalized range output_min must be 0.0")
        if not math.isclose(_finite_float(normalization, "output_max"), 1.0):
            raise PolicyLoadError("normalized range output_max must be 1.0")
        input_range_max = _finite_float(normalization, "range_max")
        if input_range_max <= 0.0:
            raise PolicyLoadError("range_normalization.range_max must be positive")

        field_of_view = _finite_float(raw, "field_of_view")
        if field_of_view <= 0.0:
            raise PolicyLoadError("field_of_view must be positive")

        validity = _mapping(raw, "validity")
        if not math.isclose(_finite_float(validity, "valid"), 1.0):
            raise PolicyLoadError("validity.valid must be 1.0")
        if not math.isclose(_finite_float(validity, "invalid"), 0.0):
            raise PolicyLoadError("validity.invalid must be 0.0")

        scaling_data = _mapping(raw, "action_scaling")
        scaling = ActionScaling(
            steering_max_abs=_finite_float(scaling_data, "steering_max_abs"),
            acceleration_min=_finite_float(scaling_data, "acceleration_min"),
            acceleration_max=_finite_float(scaling_data, "acceleration_max"),
        )
        if scaling.steering_max_abs <= 0.0:
            raise PolicyLoadError("steering_max_abs must be positive")
        if scaling.acceleration_max <= scaling.acceleration_min:
            raise PolicyLoadError("acceleration bounds must be increasing")

        checksum_data = raw.get("model_checksum")
        if isinstance(checksum_data, Mapping):
            if _nonempty_string(checksum_data, "algorithm").lower() != "sha256":
                raise PolicyLoadError("only SHA-256 model checksums are supported")
            checksum = _nonempty_string(checksum_data, "value").lower()
        elif isinstance(checksum_data, str):
            checksum = checksum_data.lower().removeprefix("sha256:")
        else:
            raise PolicyLoadError("model_checksum must be a string or object")
        if re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
            raise PolicyLoadError("model_checksum must contain a 64-character SHA-256 digest")

        return cls(
            architecture_version=architecture_version,
            architecture=architecture,
            input_range_max=input_range_max,
            field_of_view=field_of_view,
            action_scaling=scaling,
            model_checksum_sha256=checksum,
            training_config_hash=_lowercase_hex(
                raw,
                "training_config_hash",
                lengths=(64,),
            ),
            root_repository_commit=_lowercase_hex(
                raw,
                "root_repository_commit",
                lengths=(40, 64),
            ),
            f1tenth_gym_jax_commit=_lowercase_hex(
                raw,
                "f1tenth_gym_jax_commit",
                lengths=(40, 64),
            ),
            export_timestamp=_nonempty_string(raw, "export_timestamp"),
        )

    def validate_runtime(
        self,
        *,
        beam_count: int,
        frame_stack: int,
        scan_channels: int,
        input_range_max: float,
        angle_min: float,
        angle_max: float,
        steering_max_abs: float,
        acceleration_min: float,
        acceleration_max: float,
    ) -> None:
        expected_shape = (beam_count, frame_stack, scan_channels)
        manifest_shape = (
            self.architecture.beam_count,
            self.architecture.frame_stack,
            self.architecture.scan_channels,
        )
        if manifest_shape != expected_shape:
            raise PolicyLoadError(
                f"manifest observation shape {manifest_shape} does not match runtime "
                f"shape {expected_shape}"
            )
        runtime_field_of_view = angle_max - angle_min
        input_comparisons = (
            ("range_max", self.input_range_max, input_range_max),
            ("field_of_view", self.field_of_view, runtime_field_of_view),
        )
        for name, manifest_value, runtime_value in input_comparisons:
            if not math.isclose(manifest_value, runtime_value, rel_tol=0.0, abs_tol=1e-6):
                raise PolicyLoadError(
                    f"manifest {name}={manifest_value} does not match runtime {runtime_value}"
                )
        comparisons = (
            ("steering_max_abs", self.action_scaling.steering_max_abs, steering_max_abs),
            ("acceleration_min", self.action_scaling.acceleration_min, acceleration_min),
            ("acceleration_max", self.action_scaling.acceleration_max, acceleration_max),
        )
        for name, manifest_value, runtime_value in comparisons:
            if not math.isclose(manifest_value, runtime_value, rel_tol=0.0, abs_tol=1e-6):
                raise PolicyLoadError(
                    f"manifest {name}={manifest_value} does not match runtime {runtime_value}"
                )

    def verify_model_checksum(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise PolicyLoadError(f"policy state_dict not found: {path}")
        digest = hashlib.sha256()
        try:
            with path.open("rb") as model_file:
                for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            raise PolicyLoadError(f"failed to checksum policy state_dict: {error}") from error
        if digest.hexdigest() != self.model_checksum_sha256:
            raise PolicyLoadError("policy state_dict checksum does not match manifest")


class PolicyRuntime:
    """Loaded deterministic Actor bound to a verified manifest."""

    def __init__(self, *, actor: LidarActor, manifest: PolicyManifest, device: torch.device):
        self.actor = actor
        self.manifest = manifest
        self.device = device

    @classmethod
    def load(
        cls,
        *,
        model_path: str | Path,
        manifest_path: str | Path,
        device: str,
        expected_beam_count: int,
        expected_frame_stack: int,
        expected_scan_channels: int,
        expected_range_max: float,
        expected_angle_min: float,
        expected_angle_max: float,
        expected_steering_max_abs: float,
        expected_acceleration_min: float,
        expected_acceleration_max: float,
    ) -> PolicyRuntime:
        manifest = PolicyManifest.load(manifest_path)
        manifest.validate_runtime(
            beam_count=expected_beam_count,
            frame_stack=expected_frame_stack,
            scan_channels=expected_scan_channels,
            input_range_max=expected_range_max,
            angle_min=expected_angle_min,
            angle_max=expected_angle_max,
            steering_max_abs=expected_steering_max_abs,
            acceleration_min=expected_acceleration_min,
            acceleration_max=expected_acceleration_max,
        )
        manifest.verify_model_checksum(model_path)

        try:
            torch_device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise PolicyLoadError(f"invalid PyTorch device {device!r}: {error}") from error
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise PolicyLoadError("CUDA policy device requested but CUDA is unavailable")

        actor = LidarActor(manifest.architecture).to(torch_device)
        try:
            loaded = torch.load(model_path, map_location=torch_device, weights_only=True)
        except Exception as error:
            raise PolicyLoadError(f"failed to load policy state_dict: {error}") from error
        if isinstance(loaded, Mapping) and isinstance(loaded.get("state_dict"), Mapping):
            loaded = loaded["state_dict"]
        if not isinstance(loaded, Mapping) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in loaded.items()
        ):
            raise PolicyLoadError("model file must contain a PyTorch tensor state_dict")
        try:
            actor.load_state_dict(dict(loaded), strict=True)
        except (RuntimeError, ValueError) as error:
            raise PolicyLoadError(
                f"state_dict does not match manifest architecture: {error}"
            ) from error
        actor.eval()
        return cls(actor=actor, manifest=manifest, device=torch_device)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Return the deterministic normalized action for one canonical observation."""
        expected_shape = (
            self.manifest.architecture.input_channels,
            self.manifest.architecture.beam_count,
        )
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != expected_shape:
            raise ValueError(f"expected observation shape {expected_shape}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("Actor observation contains NaN or Inf")

        tensor = torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor.deterministic_action(tensor)
        if action.shape != (1, 2) or not bool(torch.all(torch.isfinite(action)).item()):
            raise RuntimeError("Actor returned an invalid action")
        return action[0].detach().cpu().numpy().astype(np.float32, copy=False)

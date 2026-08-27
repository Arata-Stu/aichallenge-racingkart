"""Strict Flax-to-PyTorch Actor parameter conversion and parity checking."""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _unwrap_params(variables: Mapping[str, Any]) -> Mapping[str, Any]:
    params = variables.get("params", variables)
    if not isinstance(params, Mapping):
        raise TypeError("Flax Actor parameters must be a mapping")
    return params


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise KeyError(f"missing Flax Actor parameter: {path}/{key}")
    return mapping[key]


def _layer(params: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    current: Any = params
    traversed: list[str] = []
    for name in path:
        traversed.append(name)
        if not isinstance(current, Mapping):
            raise TypeError(f"Flax Actor path is not a mapping: {'/'.join(traversed[:-1])}")
        current = _required(current, name, "/".join(traversed[:-1]))
    if not isinstance(current, Mapping):
        raise TypeError(f"Flax Actor layer is not a mapping: {'/'.join(path)}")
    return current


def flax_actor_to_torch_state_dict(variables: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Convert exact Flax kernels to the state_dict expected by ROS 2.

    Flax Conv kernels are ``[kernel, in, out]`` while PyTorch uses
    ``[out, in, kernel]``.  Flax Dense kernels are ``[in, out]`` while
    PyTorch uses ``[out, in]``.
    """

    import torch

    params = _unwrap_params(variables)
    encoder = _layer(params, "encoder")
    converted: OrderedDict[str, Any] = OrderedDict()
    expected_conv_shapes = (
        ((8, 8, 32), (32,)),
        ((4, 32, 64), (64,)),
        ((3, 64, 64), (64,)),
    )
    for index, (kernel_shape, bias_shape) in enumerate(expected_conv_shapes):
        layer = _layer(encoder, f"conv_{index}")
        kernel = np.asarray(_required(layer, "kernel", f"encoder/conv_{index}"))
        bias = np.asarray(_required(layer, "bias", f"encoder/conv_{index}"))
        if kernel.shape != kernel_shape or bias.shape != bias_shape:
            raise ValueError(f"unexpected Flax shape at encoder/conv_{index}")
        torch_index = 2 * index
        converted[f"encoder.{torch_index}.weight"] = torch.from_numpy(
            np.ascontiguousarray(kernel.transpose(2, 1, 0))
        )
        converted[f"encoder.{torch_index}.bias"] = torch.from_numpy(
            np.ascontiguousarray(bias)
        )

    dense = _layer(encoder, "dense")
    dense_kernel = np.asarray(_required(dense, "kernel", "encoder/dense"))
    dense_bias = np.asarray(_required(dense, "bias", "encoder/dense"))
    if dense_kernel.shape != (2624, 256) or dense_bias.shape != (256,):
        raise ValueError("unexpected Flax shape at encoder/dense")
    converted["trunk.1.weight"] = torch.from_numpy(
        np.ascontiguousarray(dense_kernel.transpose(1, 0))
    )
    converted["trunk.1.bias"] = torch.from_numpy(np.ascontiguousarray(dense_bias))

    for flax_name, torch_name in (
        ("mean_head", "mean_head"),
        ("log_std_head", "log_std_head"),
    ):
        layer = _layer(params, flax_name)
        kernel = np.asarray(_required(layer, "kernel", flax_name))
        bias = np.asarray(_required(layer, "bias", flax_name))
        if kernel.shape != (256, 2) or bias.shape != (2,):
            raise ValueError(f"unexpected Flax shape at {flax_name}")
        converted[f"{torch_name}.weight"] = torch.from_numpy(
            np.ascontiguousarray(kernel.transpose(1, 0))
        )
        converted[f"{torch_name}.bias"] = torch.from_numpy(np.ascontiguousarray(bias))
    return converted


def build_torch_actor_from_flax(variables: Mapping[str, Any]) -> Any:
    """Create a CPU eval Actor and strictly load every converted tensor."""

    from lidar_racing_rl.models.actor_torch import TorchLidarActor

    actor = TorchLidarActor()
    actor.load_state_dict(flax_actor_to_torch_state_dict(variables), strict=True)
    actor.eval()
    return actor


def deterministic_parity_error(
    flax_actor: Any,
    flax_variables: Mapping[str, Any],
    torch_actor: Any,
    observation: Any,
) -> float:
    """Return maximum absolute deterministic-action error on one input batch."""

    import jax
    import torch

    array = np.asarray(observation, dtype=np.float32)
    flax_output = flax_actor.apply(
        flax_variables,
        array,
        method=flax_actor.deterministic_action,
    )
    with torch.inference_mode():
        torch_output = torch_actor.deterministic_action(torch.from_numpy(array)).cpu().numpy()
    flax_array = np.asarray(jax.device_get(flax_output), dtype=np.float32)
    if flax_array.shape != torch_output.shape:
        raise ValueError("Flax and PyTorch deterministic outputs have different shapes")
    if not np.all(np.isfinite(flax_array)) or not np.all(np.isfinite(torch_output)):
        raise ValueError("non-finite deterministic output during parity check")
    return float(np.max(np.abs(flax_array - torch_output)))


def save_torch_state_dict(path: Path, state_dict: Mapping[str, Any]) -> None:
    """Atomically save weights only; never serialize a Python module object."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(OrderedDict(state_dict), temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "build_torch_actor_from_flax",
    "deterministic_parity_error",
    "flax_actor_to_torch_state_dict",
    "save_torch_state_dict",
]

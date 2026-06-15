#!/usr/bin/env python3
"""Run model inference for every sensor frame and save a reviewable result."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import torch


APP_DIR = Path(__file__).resolve().parent
ML_ROOT = APP_DIR.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transform_images(
    images: np.ndarray,
    image_height: int,
    image_width: int,
    color_space: str,
) -> np.ndarray:
    transformed = []
    for image in images:
        frame = np.asarray(image)
        if frame.shape[:2] != (image_height, image_width):
            frame = cv2.resize(
                frame,
                (image_width, image_height),
                interpolation=cv2.INTER_LINEAR,
            )
        if color_space == "yuv":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
        transformed.append(frame.astype(np.float32) / 255.0)
    return np.stack(transformed, axis=0).transpose(0, 3, 1, 2)


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.model_type == "pilot_net":
        module = load_module(
            ML_ROOT / "pilot_net" / "lib" / "model.py",
            "learning_studio_pilot_model",
        )
        model = module.PilotNet(
            image_height=args.image_height,
            image_width=args.image_width,
            output_dim=args.output_dim,
        )
    else:
        module = load_module(
            ML_ROOT / "tiny_lidar_net" / "lib" / "model.py",
            "learning_studio_tiny_lidar_model",
        )
        model_class = getattr(module, args.architecture)
        model = model_class(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
        )
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_targets(
    sequence_dir: Path,
    output_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steers = np.load(sequence_dir / "steers.npy", mmap_mode="r")
    accelerations = np.load(sequence_dir / "accelerations.npy", mmap_mode="r")
    if output_dim == 1:
        targets = np.asarray(steers, dtype=np.float32).reshape(-1, 1)
    else:
        targets = np.stack(
            (
                np.asarray(accelerations, dtype=np.float32),
                np.asarray(steers, dtype=np.float32),
            ),
            axis=1,
        )
    return steers, accelerations, targets


def evaluate_sequence(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    sequence_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    steers, accelerations, targets = load_targets(
        sequence_dir,
        args.output_dim,
    )
    if args.model_type == "pilot_net":
        samples = np.load(sequence_dir / "images.npy", mmap_mode="r")
    else:
        samples = np.load(sequence_dir / "scans.npy", mmap_mode="r")
        if samples.ndim != 2 or samples.shape[1] != args.input_dim:
            raise ValueError(
                f"LiDAR input_dim mismatch in {sequence_dir}: "
                f"dataset={samples.shape}, configured={args.input_dim}"
            )
    if not (len(samples) == len(steers) == len(accelerations)):
        raise ValueError(f"Data length mismatch: {sequence_dir}")

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            end = min(start + args.batch_size, len(samples))
            if args.model_type == "pilot_net":
                batch = transform_images(
                    samples[start:end],
                    args.image_height,
                    args.image_width,
                    args.color_space,
                )
            else:
                batch = np.asarray(samples[start:end], dtype=np.float32)
                batch = np.clip(
                    np.nan_to_num(batch),
                    0.0,
                    args.max_range,
                ) / args.max_range
                batch = batch[:, None, :]
            tensor = torch.from_numpy(batch).to(device)
            predictions.append(model(tensor).cpu().numpy())
    prediction_array = (
        np.concatenate(predictions, axis=0)
        if predictions
        else np.empty((0, args.output_dim), dtype=np.float32)
    )
    return prediction_array, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E2E model frame by frame.")
    parser.add_argument(
        "--model-type",
        choices=("pilot_net", "tiny_lidar_net"),
        required=True,
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-height", type=int, default=66)
    parser.add_argument("--image-width", type=int, default=200)
    parser.add_argument("--color-space", choices=("rgb", "yuv"), default="yuv")
    parser.add_argument(
        "--architecture",
        choices=("TinyLidarNet", "TinyLidarNetSmall"),
        default="TinyLidarNet",
    )
    parser.add_argument("--input-dim", type=int, default=750)
    parser.add_argument("--max-range", type=float, default=30.0)
    parser.add_argument("--output-dim", type=int, choices=(1, 2), default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    required_sample = "images.npy" if args.model_type == "pilot_net" else "scans.npy"
    sequence_dirs = sorted(
        path
        for path in args.dataset_dir.iterdir()
        if path.is_dir()
        and all(
            (path / filename).exists()
            for filename in (required_sample, "steers.npy", "accelerations.npy")
        )
    )
    if not sequence_dirs:
        raise RuntimeError(f"No valid sequences found in {args.dataset_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Model: {args.model_type}")
    print(f"[INFO] Device: {device}")
    model = build_model(args, device)

    all_predictions = []
    all_targets = []
    sequences = []
    frame_start = 0
    for number, sequence_dir in enumerate(sequence_dirs, start=1):
        print(f"[{number}/{len(sequence_dirs)}] Evaluating {sequence_dir.name}")
        predictions, targets = evaluate_sequence(
            args,
            model,
            device,
            sequence_dir,
        )
        all_predictions.append(predictions)
        all_targets.append(targets)
        sequences.append(
            {
                "name": sequence_dir.name,
                "path": str(sequence_dir.resolve()),
                "start": frame_start,
                "count": len(targets),
            }
        )
        frame_start += len(targets)

    if frame_start == 0:
        raise RuntimeError(f"No frames found in {args.dataset_dir}")

    predictions = np.concatenate(all_predictions, axis=0).astype(np.float32)
    targets = np.concatenate(all_targets, axis=0).astype(np.float32)
    absolute_error = np.abs(predictions - targets)
    mae = absolute_error.mean(axis=1)
    if args.output_dim == 1:
        steer_error = absolute_error[:, 0]
        accel_error = np.zeros_like(steer_error)
    else:
        accel_error = absolute_error[:, 0]
        steer_error = absolute_error[:, 1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "results.npz",
        predictions=predictions,
        targets=targets,
        mae=mae.astype(np.float32),
        steer_error=steer_error.astype(np.float32),
        accel_error=accel_error.astype(np.float32),
    )
    summary = {
        "frame_count": int(len(targets)),
        "mean_mae": float(mae.mean()),
        "p95_mae": float(np.percentile(mae, 95)),
        "max_mae": float(mae.max()),
        "mean_steer_error": float(steer_error.mean()),
        "mean_accel_error": float(accel_error.mean()),
    }
    model_config = {
        "output_dim": args.output_dim,
    }
    if args.model_type == "pilot_net":
        model_config.update(
            {
                "image_height": args.image_height,
                "image_width": args.image_width,
                "color_space": args.color_space,
            }
        )
    else:
        model_config.update(
            {
                "architecture": args.architecture,
                "input_dim": args.input_dim,
                "max_range": args.max_range,
            }
        )
    manifest = {
        "created_at": utc_now(),
        "model_type": args.model_type,
        "model_label": (
            "PilotNet" if args.model_type == "pilot_net" else "TinyLiDARNet"
        ),
        "dataset_name": args.dataset_name,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "model": model_config,
        "summary": summary,
        "sequences": sequences,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

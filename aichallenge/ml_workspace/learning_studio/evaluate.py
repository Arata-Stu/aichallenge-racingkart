#!/usr/bin/env python3
"""Run PilotNet inference for every frame and save a reviewable result."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch


APP_DIR = Path(__file__).resolve().parent
PILOT_ROOT = APP_DIR.parent / "pilot_net"
sys.path.insert(0, str(PILOT_ROOT))

from lib.model import PilotNet  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    batch = np.stack(transformed, axis=0)
    return batch.transpose(0, 3, 1, 2)


def evaluate_sequence(
    model: torch.nn.Module,
    device: torch.device,
    sequence_dir: Path,
    image_height: int,
    image_width: int,
    color_space: str,
    output_dim: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    images = np.load(sequence_dir / "images.npy", mmap_mode="r")
    steers = np.load(sequence_dir / "steers.npy", mmap_mode="r")
    accelerations = np.load(sequence_dir / "accelerations.npy", mmap_mode="r")
    if not (len(images) == len(steers) == len(accelerations)):
        raise ValueError(f"Data length mismatch: {sequence_dir}")

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            batch = transform_images(
                images[start:end],
                image_height,
                image_width,
                color_space,
            )
            tensor = torch.from_numpy(batch).to(device)
            predictions.append(model(tensor).cpu().numpy())
    prediction_array = (
        np.concatenate(predictions, axis=0)
        if predictions
        else np.empty((0, output_dim), dtype=np.float32)
    )
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
    return prediction_array, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PilotNet frame by frame.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-height", type=int, default=66)
    parser.add_argument("--image-width", type=int, default=200)
    parser.add_argument("--output-dim", type=int, choices=(1, 2), default=2)
    parser.add_argument("--color-space", choices=("rgb", "yuv"), default="yuv")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    sequence_dirs = sorted(
        path
        for path in args.dataset_dir.iterdir()
        if path.is_dir()
        and all(
            (path / filename).exists()
            for filename in ("images.npy", "steers.npy", "accelerations.npy")
        )
    )
    if not sequence_dirs:
        raise RuntimeError(f"No valid sequences found in {args.dataset_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    model = PilotNet(
        image_height=args.image_height,
        image_width=args.image_width,
        output_dim=args.output_dim,
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    all_predictions = []
    all_targets = []
    sequences = []
    frame_start = 0
    for number, sequence_dir in enumerate(sequence_dirs, start=1):
        print(f"[{number}/{len(sequence_dirs)}] Evaluating {sequence_dir.name}")
        predictions, targets = evaluate_sequence(
            model,
            device,
            sequence_dir,
            args.image_height,
            args.image_width,
            args.color_space,
            args.output_dim,
            args.batch_size,
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
    manifest = {
        "created_at": utc_now(),
        "dataset_name": args.dataset_name,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "model": {
            "image_height": args.image_height,
            "image_width": args.image_width,
            "output_dim": args.output_dim,
            "color_space": args.color_space,
        },
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

#!/usr/bin/env python3
"""Offline evaluation and UI-ready prediction export for trajectory RSU fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.data import BevTrajectorySequenceDataset, RsuTrajectorySequenceDataset
from lib.model import BevBezierTrajectoryNet, RsuBezierTrajectoryFusionNet, RsuTrajectoryFusionNet


TrajectoryModel = RsuTrajectoryFusionNet | RsuBezierTrajectoryFusionNet | BevBezierTrajectoryNet
TrajectoryDataset = RsuTrajectorySequenceDataset | BevTrajectorySequenceDataset


def load_checkpoint(path: Path, device: torch.device) -> tuple[TrajectoryModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no model_state_dict")
    cfg = checkpoint.get("config", {})
    model_cfg, data_cfg = cfg.get("model", {}), cfg.get("data", {})
    architecture = model_cfg.get("architecture")
    if architecture not in {
        "trajectory_multimodal", "trajectory_bezier_v2", "bev_trajectory_bezier_v1"
    }:
        raise ValueError("Offline trajectory evaluation requires a trajectory checkpoint")
    if architecture == "bev_trajectory_bezier_v1":
        model = BevBezierTrajectoryNet(
            bev_channels=len(data_cfg.get("bev_channels", [0, 1, 2, 3, 4, 5])),
            vehicle_state_dim=int(model_cfg.get("vehicle_state_dim", 1)),
            output_dim=int(model_cfg.get("output_dim", 2)),
            frame_feature_dim=int(model_cfg.get("frame_feature_dim", 128)),
            temporal_hidden_dim=int(model_cfg.get("temporal_hidden_dim", 128)),
            fusion_hidden_dim=int(model_cfg.get("fusion_hidden_dim", 128)),
            trajectory_modes=int(model_cfg.get("trajectory_modes", 4)),
            trajectory_steps=int(data_cfg.get("trajectory_steps", 20)),
            trajectory_dim=int(model_cfg.get("trajectory_dim", 3)),
            trajectory_anchor_count=int(model_cfg.get("trajectory_anchor_count", 4)),
            max_anchor_step_normalized=float(model_cfg.get("max_anchor_step_normalized", 0.24)),
            max_anchor_heading_delta=float(model_cfg.get("max_anchor_heading_delta", 1.2)),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        return model, cfg
    model_class = (
        RsuBezierTrajectoryFusionNet
        if architecture == "trajectory_bezier_v2"
        else RsuTrajectoryFusionNet
    )
    model_args = dict(
        scan_dim=int(model_cfg.get("scan_dim", 1080)),
        rsu_count=int(model_cfg.get("rsu_count", 6)),
        rsu_meta_dim=int(model_cfg.get("rsu_meta_dim", 5)),
        vehicle_state_dim=int(model_cfg.get("vehicle_state_dim", 1)),
        output_dim=int(model_cfg.get("output_dim", 2)),
        frame_feature_dim=int(model_cfg.get("frame_feature_dim", 128)),
        temporal_hidden_dim=int(model_cfg.get("temporal_hidden_dim", 128)),
        fusion_hidden_dim=int(model_cfg.get("fusion_hidden_dim", 128)),
        distance_decay_m=float(model_cfg.get("distance_decay_m", 35.0)),
        top_k_rsus=int(model_cfg.get("top_k_rsus", 2)),
        trajectory_modes=int(model_cfg.get("trajectory_modes", 4)),
        trajectory_steps=int(data_cfg.get("trajectory_steps", 20)),
        trajectory_dim=int(model_cfg.get("trajectory_dim", 3)),
    )
    if model_class is RsuBezierTrajectoryFusionNet:
        model_args.update(
            trajectory_anchor_count=int(model_cfg.get("trajectory_anchor_count", 4)),
            max_anchor_step_normalized=float(model_cfg.get("max_anchor_step_normalized", 0.24)),
            max_anchor_heading_delta=float(model_cfg.get("max_anchor_heading_delta", 1.2)),
        )
    model = model_class(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


def dataset_from_config(path: Path, cfg: dict[str, Any]) -> TrajectoryDataset:
    data = cfg.get("data", {})
    if cfg.get("model", {}).get("architecture") == "bev_trajectory_bezier_v1":
        return BevTrajectorySequenceDataset(
            path,
            history_len=int(data.get("history_len", 5)),
            acceleration_scale=float(data.get("acceleration_scale", 2.0)),
            trajectory_steps=int(data.get("trajectory_steps", 20)),
            trajectory_dt=float(data.get("trajectory_dt", 0.25)),
            trajectory_distance_scale=float(data.get("trajectory_distance_scale", 50.0)),
            max_speed=float(data.get("max_speed", 15.0)),
            bev_channels=tuple(data.get("bev_channels", [0, 1, 2, 3, 4, 5])),
            bev_height=int(data.get("bev_height", 160)),
            bev_width=int(data.get("bev_width", 140)),
        )
    return RsuTrajectorySequenceDataset(
        path,
        history_len=int(data.get("history_len", 5)),
        max_range=float(data.get("max_range", 45.0)),
        acceleration_scale=float(data.get("acceleration_scale", 2.0)),
        trajectory_steps=int(data.get("trajectory_steps", 20)),
        trajectory_dt=float(data.get("trajectory_dt", 0.25)),
        trajectory_distance_scale=float(data.get("trajectory_distance_scale", 50.0)),
        max_speed=float(data.get("max_speed", 15.0)),
    )


def evaluate_sequence(
    model: TrajectoryModel, dataset: TrajectoryDataset,
    device: torch.device, batch_size: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "sample_indices", "controls", "target_controls", "trajectories",
            "target_trajectories", "mode_probabilities", "gates",
        )
    }
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key not in {"target", "trajectory", "sample_index"}
            }
            if "bev" in inputs:
                output = model(inputs["bev"], vehicle_state=inputs.get("vehicle_state"))
            else:
                output = model(
                    inputs["ego_scans"], inputs["rsu_scans"], inputs["rsu_meta"],
                    vehicle_state=inputs.get("vehicle_state"), rsu_mask=inputs.get("rsu_mask"),
                )
            arrays["sample_indices"].append(batch["sample_index"].numpy())
            arrays["controls"].append(output.control.cpu().numpy())
            arrays["target_controls"].append(batch["target"].numpy())
            arrays["trajectories"].append(output.trajectories.cpu().numpy())
            arrays["target_trajectories"].append(batch["trajectory"].numpy())
            arrays["mode_probabilities"].append(torch.softmax(output.mode_logits, dim=-1).cpu().numpy())
            arrays["gates"].append(output.gates.cpu().numpy())
    result = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    distance_scale = dataset.trajectory_distance_scale
    speed_scale = dataset.max_speed
    acceleration_scale = dataset.acceleration_scale
    result["trajectories"][..., :2] *= distance_scale
    result["trajectories"][..., 2] *= speed_scale
    result["target_trajectories"][..., :2] *= distance_scale
    result["target_trajectories"][..., 2] *= speed_scale
    result["controls"][:, 0] *= acceleration_scale
    result["target_controls"][:, 0] *= acceleration_scale
    selected_modes = result["mode_probabilities"].argmax(axis=1)
    selected = result["trajectories"][np.arange(len(selected_modes)), selected_modes]
    target = result["target_trajectories"]
    displacement = np.linalg.norm(selected[..., :2] - target[..., :2], axis=-1)
    control_error = result["controls"] - result["target_controls"]
    metrics = {
        "samples": float(len(selected)),
        "ade_m": float(displacement.mean()),
        "fde_m": float(displacement[:, -1].mean()),
        "speed_mae_mps": float(np.abs(selected[..., 2] - target[..., 2]).mean()),
        "acceleration_mae": float(np.abs(control_error[:, 0]).mean()),
        "steering_mae_rad": float(np.abs(control_error[:, 1]).mean()),
    }
    result["selected_modes"] = selected_modes.astype(np.int64)
    return metrics, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--version", default="default")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model, cfg = load_checkpoint(args.checkpoint.resolve(), device)
    sequence_dirs = sorted(
        path for path in args.dataset_dir.resolve().iterdir()
        if path.is_dir() and (
            (path / "ego_scans.npy").is_file() or (path / "bev_frames.npy").is_file()
        )
    )
    if not sequence_dirs:
        raise SystemExit(f"No sequences found in {args.dataset_dir}")
    args.output.mkdir(parents=True, exist_ok=True)
    sequence_output = args.output / "sequences"
    sequence_output.mkdir(exist_ok=True)
    manifest: list[dict[str, Any]] = []
    weighted: dict[str, float] = {}
    total_samples = 0
    for number, sequence_dir in enumerate(sequence_dirs):
        dataset = dataset_from_config(sequence_dir, cfg)
        if not len(dataset):
            print(f"[SKIP] {sequence_dir.name}: shorter than the configured history/horizon")
            continue
        metrics, arrays = evaluate_sequence(model, dataset, device, args.batch_size)
        file_name = f"sequence_{number:04d}.npz"
        np.savez_compressed(sequence_output / file_name, **arrays)
        samples = int(metrics["samples"])
        total_samples += samples
        for key, value in metrics.items():
            if key != "samples":
                weighted[key] = weighted.get(key, 0.0) + value * samples
        manifest.append({"id": sequence_dir.name, "file": file_name, "metrics": metrics})
        print(
            f"[{number + 1}/{len(sequence_dirs)}] {sequence_dir.name}: "
            f"ADE={metrics['ade_m']:.3f}m FDE={metrics['fde_m']:.3f}m"
        )
    if total_samples == 0:
        raise SystemExit("No evaluable samples remain after history/horizon filtering")
    summary = {key: value / total_samples for key, value in weighted.items()}
    summary["samples"] = total_samples
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "version": args.version,
        "split": args.split, "device": str(device), "metrics": summary,
        "sequences": manifest,
        "trajectory": {
            "architecture": str(cfg["model"].get("architecture", "trajectory_multimodal")),
            "steps": int(cfg["data"]["trajectory_steps"]),
            "dt": float(cfg["data"]["trajectory_dt"]),
            "modes": int(cfg["model"]["trajectory_modes"]),
            "anchor_count": int(cfg["model"].get("trajectory_anchor_count", 0)),
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Offline evaluation saved to {args.output}")


if __name__ == "__main__":
    main()

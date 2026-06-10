from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset

from lib.data import LidarTrajectorySequenceDataset, MultiSeqLidarTrajectoryDataset
from lib.runtime import build_model, load_checkpoint, resolve_device


def make_sequence_dataset(seq_dir: Path, cfg: DictConfig) -> LidarTrajectorySequenceDataset:
    return LidarTrajectorySequenceDataset(
        seq_dir=seq_dir,
        history_length=cfg.data.history_length,
        history_stride=cfg.data.history_stride,
        future_num_points=cfg.data.future_num_points,
        future_stride=cfg.data.future_stride,
        max_range=cfg.data.max_range,
        target_max_x=cfg.data.target_max_x,
        target_max_y=cfg.data.target_max_y,
        min_future_forward=cfg.data.min_future_forward,
    )


def load_dataset(dataset_dir: Path, cfg: DictConfig) -> ConcatDataset:
    if (dataset_dir / "scan_inputs.npy").exists():
        return ConcatDataset([make_sequence_dataset(dataset_dir, cfg)])

    return MultiSeqLidarTrajectoryDataset(
        dataset_root=dataset_dir,
        history_length=cfg.data.history_length,
        history_stride=cfg.data.history_stride,
        future_num_points=cfg.data.future_num_points,
        future_stride=cfg.data.future_stride,
        max_range=cfg.data.max_range,
        target_max_x=cfg.data.target_max_x,
        target_max_y=cfg.data.target_max_y,
        min_future_forward=cfg.data.min_future_forward,
    )


def locate_sample(
    dataset: ConcatDataset,
    global_index: int,
) -> Tuple[LidarTrajectorySequenceDataset, int, int]:
    dataset_index = bisect.bisect_right(dataset.cumulative_sizes, global_index)
    previous_size = 0 if dataset_index == 0 else dataset.cumulative_sizes[dataset_index - 1]
    local_index = global_index - previous_size
    sequence = dataset.datasets[dataset_index]
    raw_index = sequence.indices[local_index]
    return sequence, local_index, raw_index


def select_indices(total: int, start: int, count: int, stride: int) -> List[int]:
    if total < 1:
        return []
    if start < 0 or start >= total:
        raise ValueError(f"start-index must be in [0, {total - 1}], got {start}.")

    count = min(max(1, count), total - start)
    if stride > 0:
        return list(range(start, min(total, start + count * stride), stride))

    return np.unique(np.linspace(start, total - 1, count, dtype=np.int64)).tolist()


def load_scan_angles(
    sequence: LidarTrajectorySequenceDataset,
    num_rays: int,
    fallback_angle_min_deg: float,
    fallback_angle_max_deg: float,
) -> np.ndarray:
    angle_path = sequence.seq_dir / "scan_angles.npy"
    if angle_path.exists():
        angles = np.load(angle_path).astype(np.float32)
        if len(angles) == num_rays:
            return angles

    metadata_path = sequence.seq_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open() as f:
            metadata = json.load(f)
        geometry = metadata.get("scan_geometry", {})
        if (
            geometry.get("num_rays") == num_rays
            and "angle_min" in geometry
            and "angle_max" in geometry
        ):
            return np.linspace(
                float(geometry["angle_min"]),
                float(geometry["angle_max"]),
                num_rays,
                dtype=np.float32,
            )

    return np.linspace(
        np.deg2rad(fallback_angle_min_deg),
        np.deg2rad(fallback_angle_max_deg),
        num_rays,
        dtype=np.float32,
    )


def load_timestamp(sequence: LidarTrajectorySequenceDataset, raw_index: int) -> Optional[float]:
    timestamp_path = sequence.seq_dir / "timestamps.npy"
    if not timestamp_path.exists():
        return None
    timestamps = np.load(timestamp_path, mmap_mode="r")
    return float(timestamps[raw_index]) / 1e9


def predict(
    model: torch.nn.Module,
    scan_history: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(scan_history).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        path, control_points = model(tensor, return_control_points=True)
    return path[0].cpu().numpy(), control_points[0].cpu().numpy()


def trajectory_metrics(prediction: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    distances = np.linalg.norm(prediction - target, axis=1)
    return float(distances.mean()), float(distances[-1])


def plot_sample(
    output_path: Path,
    sequence: LidarTrajectorySequenceDataset,
    global_index: int,
    raw_index: int,
    scan_history: np.ndarray,
    target: np.ndarray,
    angles: np.ndarray,
    obstacle_threshold_m: float,
    prediction: Optional[np.ndarray] = None,
    control_points: Optional[np.ndarray] = None,
    timestamp_sec: Optional[float] = None,
) -> Dict[str, float | int | str | None]:
    scans_m = scan_history * sequence.max_range
    latest = scans_m[-1]
    free_scan = latest[0]
    obstacle_scan = latest[1]
    diff = latest[2]
    obstacle_mask = diff > obstacle_threshold_m
    angle_deg = np.rad2deg(angles)

    free_x = free_scan * np.cos(angles)
    free_y = free_scan * np.sin(angles)
    obstacle_x = obstacle_scan * np.cos(angles)
    obstacle_y = obstacle_scan * np.sin(angles)
    plot_step = max(1, len(angles) // 720)

    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    ax_scan, ax_profile, ax_history, ax_path = axes.flat

    ax_scan.plot(
        free_x[::plot_step],
        free_y[::plot_step],
        ".",
        color="#377eb8",
        markersize=2,
        alpha=0.65,
        label="virtual scan",
    )
    ax_scan.plot(
        obstacle_x[::plot_step],
        obstacle_y[::plot_step],
        ".",
        color="#4daf4a",
        markersize=2,
        alpha=0.65,
        label="with obstacles",
    )
    if obstacle_mask.any():
        ax_scan.scatter(
            obstacle_x[obstacle_mask],
            obstacle_y[obstacle_mask],
            s=15,
            color="#e41a1c",
            label="obstacle difference",
            zorder=3,
        )
    ax_scan.arrow(0.0, 0.0, 2.0, 0.0, width=0.08, color="black", length_includes_head=True)
    ax_scan.set_title("Latest scan in ego frame")
    ax_scan.set_xlabel("forward x [m]")
    ax_scan.set_ylabel("left y [m]")
    ax_scan.set_aspect("equal", adjustable="box")
    ax_scan.grid(True, alpha=0.25)
    ax_scan.legend(loc="best")

    ax_profile.plot(angle_deg, free_scan, color="#377eb8", linewidth=1.1, label="virtual scan")
    ax_profile.plot(
        angle_deg,
        obstacle_scan,
        color="#4daf4a",
        linewidth=1.1,
        label="with obstacles",
    )
    ax_profile.fill_between(
        angle_deg,
        0.0,
        diff,
        where=obstacle_mask,
        color="#e41a1c",
        alpha=0.45,
        label="diff",
    )
    ax_profile.set_title("Latest range profile")
    ax_profile.set_xlabel("ray angle [deg]")
    ax_profile.set_ylabel("range / difference [m]")
    ax_profile.set_xlim(float(angle_deg[0]), float(angle_deg[-1]))
    ax_profile.set_ylim(0.0, sequence.max_range * 1.03)
    ax_profile.grid(True, alpha=0.25)
    ax_profile.legend(loc="best")

    image = ax_history.imshow(
        scans_m[:, 2, :],
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[float(angle_deg[0]), float(angle_deg[-1]), 0, len(scan_history) - 1],
        cmap="magma",
        vmin=0.0,
        vmax=max(obstacle_threshold_m, float(scans_m[:, 2, :].max())),
    )
    ax_history.set_title("Obstacle difference history")
    ax_history.set_xlabel("ray angle [deg]")
    ax_history.set_ylabel("history frame (old to new)")
    figure.colorbar(image, ax=ax_history, label="difference [m]")

    origin = np.zeros((1, 2), dtype=np.float32)
    target_with_origin = np.concatenate([origin, target], axis=0)
    ax_path.plot(
        target_with_origin[:, 0],
        target_with_origin[:, 1],
        "-o",
        color="#377eb8",
        markersize=3,
        linewidth=2,
        label="ground truth",
    )
    ade = None
    fde = None
    if prediction is not None:
        prediction_with_origin = np.concatenate([origin, prediction], axis=0)
        ax_path.plot(
            prediction_with_origin[:, 0],
            prediction_with_origin[:, 1],
            "-o",
            color="#e41a1c",
            markersize=3,
            linewidth=2,
            label="prediction",
        )
        ade, fde = trajectory_metrics(prediction, target)
    if control_points is not None:
        control_with_origin = np.concatenate([origin, control_points], axis=0)
        ax_path.plot(
            control_with_origin[:, 0],
            control_with_origin[:, 1],
            "--s",
            color="#984ea3",
            markersize=4,
            linewidth=1,
            alpha=0.8,
            label="Bezier control points",
        )
    ax_path.arrow(0.0, 0.0, 1.0, 0.0, width=0.04, color="black", length_includes_head=True)
    ax_path.set_title("Future trajectory" if ade is None else f"Future trajectory | ADE={ade:.3f} m FDE={fde:.3f} m")
    ax_path.set_xlabel("forward x [m]")
    ax_path.set_ylabel("left y [m]")
    ax_path.set_aspect("equal", adjustable="datalim")
    ax_path.grid(True, alpha=0.25)
    ax_path.legend(loc="best")

    active_fraction = float(obstacle_mask.mean())
    timestamp_text = "unknown" if timestamp_sec is None else f"{timestamp_sec:.3f} s"
    figure.suptitle(
        f"sequence={sequence.seq_dir.name} | dataset_index={global_index} | "
        f"raw_index={raw_index} | stamp={timestamp_text} | "
        f"obstacle_rays={active_fraction * 100.0:.2f}%"
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    return {
        "dataset_index": global_index,
        "sequence": sequence.seq_dir.name,
        "raw_index": raw_index,
        "timestamp_sec": timestamp_sec,
        "obstacle_ray_fraction": active_fraction,
        "ade_m": ade,
        "fde_m": fde,
    }


def collect_dataset_statistics(
    dataset: ConcatDataset,
    indices: Sequence[int],
    obstacle_threshold_m: float,
) -> Dict[str, np.ndarray | float]:
    endpoints = []
    final_distances = []
    obstacle_fractions = []

    for index in indices:
        sequence, _, _ = locate_sample(dataset, index)
        scan_history, target = dataset[index]
        latest_diff_m = scan_history[-1, 2] * sequence.max_range
        endpoints.append(target[-1])
        final_distances.append(np.linalg.norm(target[-1]))
        obstacle_fractions.append(np.mean(latest_diff_m > obstacle_threshold_m))

    endpoint_array = np.asarray(endpoints, dtype=np.float32)
    obstacle_array = np.asarray(obstacle_fractions, dtype=np.float32)
    return {
        "endpoints": endpoint_array,
        "final_distances": np.asarray(final_distances, dtype=np.float32),
        "obstacle_fractions": obstacle_array,
        "obstacle_active_ratio": float(np.mean(obstacle_array > 0.0)),
    }


def collect_sequence_integrity(dataset: ConcatDataset) -> List[Dict[str, float | int | str | None]]:
    results = []
    for sequence in dataset.datasets:
        timestamp_path = sequence.seq_dir / "timestamps.npy"
        median_dt_sec = None
        max_dt_sec = None
        non_monotonic_timestamps = None

        if timestamp_path.exists():
            timestamps = np.load(timestamp_path, mmap_mode="r")
            timestamp_deltas = np.diff(timestamps.astype(np.float64)) / 1e9
            if len(timestamp_deltas) > 0:
                positive_deltas = timestamp_deltas[timestamp_deltas > 0.0]
                median_dt_sec = (
                    None if len(positive_deltas) == 0 else float(np.median(positive_deltas))
                )
                max_dt_sec = float(np.max(timestamp_deltas))
                non_monotonic_timestamps = int(np.sum(timestamp_deltas <= 0.0))
            else:
                non_monotonic_timestamps = 0

        results.append(
            {
                "sequence": sequence.seq_dir.name,
                "raw_samples": int(len(sequence.scans)),
                "valid_samples": int(len(sequence)),
                "channels": int(sequence.scans.shape[1]),
                "num_rays": int(sequence.scans.shape[2]),
                "median_dt_sec": median_dt_sec,
                "max_dt_sec": max_dt_sec,
                "non_monotonic_timestamps": non_monotonic_timestamps,
            }
        )
    return results


def plot_summary(
    output_path: Path,
    dataset: ConcatDataset,
    statistics: Dict[str, np.ndarray | float],
    sequence_integrity: Sequence[Dict[str, float | int | str | None]],
    sample_metrics: Sequence[Dict[str, float | int | str | None]],
) -> None:
    endpoints = statistics["endpoints"]
    final_distances = statistics["final_distances"]
    obstacle_fractions = statistics["obstacle_fractions"]
    sequence_names = [sequence.seq_dir.name for sequence in dataset.datasets]
    sequence_sizes = [len(sequence) for sequence in dataset.datasets]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax_endpoint, ax_obstacles, ax_distance, ax_fourth = axes.flat

    ax_endpoint.scatter(endpoints[:, 0], endpoints[:, 1], s=10, alpha=0.5, color="#377eb8")
    ax_endpoint.set_title("Ground-truth final points")
    ax_endpoint.set_xlabel("forward x [m]")
    ax_endpoint.set_ylabel("left y [m]")
    ax_endpoint.grid(True, alpha=0.25)

    ax_obstacles.hist(obstacle_fractions * 100.0, bins=30, color="#e41a1c", alpha=0.8)
    ax_obstacles.set_title(
        f"Obstacle-ray fraction | active samples={statistics['obstacle_active_ratio'] * 100.0:.1f}%"
    )
    ax_obstacles.set_xlabel("rays with diff above threshold [%]")
    ax_obstacles.set_ylabel("samples")
    ax_obstacles.grid(True, alpha=0.25)

    timestamp_maxima = [
        float(item["max_dt_sec"])
        for item in sequence_integrity
        if item["max_dt_sec"] is not None
    ]
    if timestamp_maxima:
        ax_distance.bar(
            np.arange(len(timestamp_maxima)),
            timestamp_maxima,
            color="#4daf4a",
            alpha=0.8,
        )
        ax_distance.set_title("Maximum timestamp gap per sequence")
        ax_distance.set_xlabel("sequence")
        ax_distance.set_ylabel("maximum gap [s]")
    else:
        ax_distance.hist(final_distances, bins=30, color="#4daf4a", alpha=0.8)
        ax_distance.set_title("Ground-truth final distance")
        ax_distance.set_xlabel("distance [m]")
        ax_distance.set_ylabel("samples")
    ax_distance.grid(True, alpha=0.25)

    prediction_metrics = [item for item in sample_metrics if item["ade_m"] is not None]
    if prediction_metrics:
        ade = [float(item["ade_m"]) for item in prediction_metrics]
        fde = [float(item["fde_m"]) for item in prediction_metrics]
        positions = np.arange(len(prediction_metrics))
        ax_fourth.plot(positions, ade, "-o", markersize=3, label="ADE")
        ax_fourth.plot(positions, fde, "-o", markersize=3, label="FDE")
        ax_fourth.set_title("Prediction errors for rendered samples")
        ax_fourth.set_xlabel("rendered sample")
        ax_fourth.set_ylabel("error [m]")
        ax_fourth.legend(loc="best")
    else:
        positions = np.arange(len(sequence_names))
        ax_fourth.bar(positions, sequence_sizes, color="#984ea3", alpha=0.8)
        ax_fourth.set_title("Valid samples per sequence")
        ax_fourth.set_xlabel("sequence")
        ax_fourth.set_ylabel("valid samples")
        if len(sequence_names) <= 20:
            ax_fourth.set_xticks(positions, sequence_names, rotation=45, ha="right")
    ax_fourth.grid(True, alpha=0.25)

    figure.suptitle(
        f"LiDAR trajectory dataset summary | sequences={len(dataset.datasets)} | "
        f"valid_samples={len(dataset)} | analyzed_samples={len(endpoints)}"
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Render offline LiDAR trajectory dataset and inference comparisons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=script_dir / "config" / "train.yaml")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "visualizations")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Dataset-index stride. 0 selects samples evenly across the dataset.",
    )
    parser.add_argument("--summary-samples", type=int, default=2000)
    parser.add_argument("--obstacle-threshold-m", type=float, default=0.05)
    parser.add_argument("--fallback-angle-min-deg", type=float, default=-135.0)
    parser.add_argument("--fallback-angle-max-deg", type=float, default=135.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    cfg = OmegaConf.load(args.config.expanduser().resolve())
    state_dict = None
    checkpoint_metadata: Dict[str, object] = {}

    if args.checkpoint:
        state_dict, checkpoint_cfg, checkpoint_metadata = load_checkpoint(
            args.checkpoint.expanduser().resolve(),
            map_location=device,
        )
        if checkpoint_cfg is not None:
            cfg = checkpoint_cfg

    dataset_dir = args.dataset_dir.expanduser().resolve()
    dataset = load_dataset(dataset_dir, cfg)
    selected_indices = select_indices(
        total=len(dataset),
        start=args.start_index,
        count=args.num_samples,
        stride=args.stride,
    )
    summary_indices = select_indices(
        total=len(dataset),
        start=0,
        count=min(args.summary_samples, len(dataset)),
        stride=0,
    )

    model = None
    if state_dict is not None:
        model = build_model(cfg).to(device)
        model.load_state_dict(state_dict)
        model.eval()

    output_dir = args.output_dir.expanduser().resolve()
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_metrics = []
    for global_index in selected_indices:
        sequence, _, raw_index = locate_sample(dataset, global_index)
        scan_history, target = dataset[global_index]
        angles = load_scan_angles(
            sequence,
            scan_history.shape[-1],
            args.fallback_angle_min_deg,
            args.fallback_angle_max_deg,
        )
        prediction = None
        control_points = None
        if model is not None:
            prediction, control_points = predict(model, scan_history, device)

        output_path = sample_dir / f"sample_{global_index:06d}_{sequence.seq_dir.name}.png"
        sample_metrics.append(
            plot_sample(
                output_path=output_path,
                sequence=sequence,
                global_index=global_index,
                raw_index=raw_index,
                scan_history=scan_history,
                target=target,
                angles=angles,
                obstacle_threshold_m=args.obstacle_threshold_m,
                prediction=prediction,
                control_points=control_points,
                timestamp_sec=load_timestamp(sequence, raw_index),
            )
        )

    statistics = collect_dataset_statistics(
        dataset=dataset,
        indices=summary_indices,
        obstacle_threshold_m=args.obstacle_threshold_m,
    )
    sequence_integrity = collect_sequence_integrity(dataset)
    plot_summary(
        output_path=output_dir / "dataset_summary.png",
        dataset=dataset,
        statistics=statistics,
        sequence_integrity=sequence_integrity,
        sample_metrics=sample_metrics,
    )

    prediction_metrics = [item for item in sample_metrics if item["ade_m"] is not None]
    report = {
        "dataset_dir": str(dataset_dir),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint.expanduser().resolve()),
        "checkpoint_metadata": checkpoint_metadata,
        "num_sequences": len(dataset.datasets),
        "num_valid_samples": len(dataset),
        "num_analyzed_samples": len(summary_indices),
        "obstacle_active_ratio": statistics["obstacle_active_ratio"],
        "mean_obstacle_ray_fraction": float(np.mean(statistics["obstacle_fractions"])),
        "mean_ade_m": (
            None
            if not prediction_metrics
            else float(np.mean([float(item["ade_m"]) for item in prediction_metrics]))
        ),
        "mean_fde_m": (
            None
            if not prediction_metrics
            else float(np.mean([float(item["fde_m"]) for item in prediction_metrics]))
        ),
        "sequences": sequence_integrity,
        "samples": sample_metrics,
    }
    with (output_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved summary: {output_dir / 'dataset_summary.png'}")
    print(f"Saved sample comparisons: {sample_dir}")
    print(f"Saved report: {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()

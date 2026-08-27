#!/usr/bin/env python3
"""Inspect and visualize a preprocessed RSU fusion dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RSU_COLORS = (
    "#ff5050",
    "#ffa500",
    "#ffe600",
    "#3cdc64",
    "#00c8ff",
    "#be64ff",
)


@dataclass(frozen=True)
class FusionDataset:
    root: Path
    ego_scans: np.ndarray
    rsu_scans: np.ndarray
    rsu_meta: np.ndarray
    targets: np.ndarray
    rsu_mask: np.ndarray
    vehicle_state: np.ndarray | None

    @property
    def sample_count(self) -> int:
        return int(self.ego_scans.shape[0])

    @property
    def rsu_count(self) -> int:
        return int(self.rsu_scans.shape[1])


def _load_array(root: Path, name: str, required: bool = True) -> np.ndarray | None:
    path = root / name
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required dataset file: {path}")
        return None
    return np.load(path, mmap_mode="r", allow_pickle=False)


def load_dataset(root: Path) -> FusionDataset:
    """Load arrays using memory maps and reject incompatible shapes early."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")

    ego = _load_array(root, "ego_scans.npy")
    rsu = _load_array(root, "rsu_scans.npy")
    meta = _load_array(root, "rsu_meta.npy")
    targets = _load_array(root, "targets.npy")
    mask = _load_array(root, "rsu_mask.npy", required=False)
    vehicle_state = _load_array(root, "vehicle_state.npy", required=False)
    assert ego is not None and rsu is not None and meta is not None and targets is not None

    errors: list[str] = []
    if ego.ndim != 2:
        errors.append(f"ego_scans.npy must be [N, R], got {ego.shape}")
    if rsu.ndim != 3:
        errors.append(f"rsu_scans.npy must be [N, S, R], got {rsu.shape}")
    if meta.ndim != 3:
        errors.append(f"rsu_meta.npy must be [N, S, M], got {meta.shape}")
    if targets.ndim != 2:
        errors.append(f"targets.npy must be [N, C], got {targets.shape}")
    if errors:
        raise ValueError("\n".join(errors))

    n = int(ego.shape[0])
    s = int(rsu.shape[1])
    if n == 0:
        errors.append("Dataset contains no synchronized samples")
    if ego.shape[1] == 0:
        errors.append("ego_scans contains no rays")
    if s == 0 or rsu.shape[2] == 0:
        errors.append("rsu_scans must contain at least one RSU and one ray")
    if rsu.shape[0] != n:
        errors.append(f"rsu_scans sample count {rsu.shape[0]} != ego sample count {n}")
    if meta.shape[:2] != (n, s):
        errors.append(f"rsu_meta leading shape {meta.shape[:2]} != {(n, s)}")
    if targets.shape[0] != n:
        errors.append(f"targets sample count {targets.shape[0]} != ego sample count {n}")
    if targets.shape[1] < 2:
        errors.append(f"targets must contain at least 2 controls, got {targets.shape[1]}")
    if ego.shape[1] != rsu.shape[2]:
        errors.append(f"ego rays {ego.shape[1]} != RSU rays {rsu.shape[2]}")
    if mask is not None and mask.shape != (n, s):
        errors.append(f"rsu_mask shape {mask.shape} != {(n, s)}")
    if vehicle_state is not None and (vehicle_state.ndim != 2 or vehicle_state.shape[0] != n):
        errors.append(f"vehicle_state must be [N, V] with N={n}, got {vehicle_state.shape}")
    if errors:
        raise ValueError("\n".join(errors))

    if mask is None:
        mask = np.ones((n, s), dtype=np.bool_)

    return FusionDataset(root, ego, rsu, meta, targets, mask, vehicle_state)


def _sample_rows(array: np.ndarray, limit: int) -> np.ndarray:
    if len(array) <= limit:
        return np.asarray(array)
    indices = np.linspace(0, len(array) - 1, num=limit, dtype=np.int64)
    return np.asarray(array[indices])


def _scan_stats(array: np.ndarray, max_range: float, limit: int) -> dict[str, float]:
    values = _sample_rows(array, limit)
    total = max(1, values.size)
    finite = np.isfinite(values)
    finite_values = values[finite]
    return {
        "finite_pct": 100.0 * float(finite.sum()) / total,
        "nan_pct": 100.0 * float(np.isnan(values).sum()) / total,
        "posinf_pct": 100.0 * float(np.isposinf(values).sum()) / total,
        "negative_pct": 100.0 * float((finite_values < 0.0).sum()) / max(1, finite_values.size),
        "over_range_pct": 100.0 * float((finite_values > max_range).sum()) / max(1, finite_values.size),
        "min": float(finite_values.min()) if finite_values.size else float("nan"),
        "max": float(finite_values.max()) if finite_values.size else float("nan"),
    }


def build_report(dataset: FusionDataset, max_range: float, stats_samples: int) -> tuple[str, list[str]]:
    """Return a human-readable integrity report and actionable warnings."""
    ego_stats = _scan_stats(dataset.ego_scans, max_range, stats_samples)
    rsu_flat = dataset.rsu_scans.reshape(dataset.sample_count, -1)
    rsu_stats = _scan_stats(rsu_flat, max_range, stats_samples)
    mask = np.asarray(dataset.rsu_mask, dtype=np.bool_)
    target_sample = _sample_rows(dataset.targets, stats_samples)
    warnings: list[str] = []

    if not np.isfinite(target_sample).all():
        warnings.append("targets contain NaN or Inf")
    if ego_stats["negative_pct"] > 0.0 or rsu_stats["negative_pct"] > 0.0:
        warnings.append("scan arrays contain negative finite ranges")
    if ego_stats["over_range_pct"] > 0.0 or rsu_stats["over_range_pct"] > 0.0:
        warnings.append(f"scan arrays contain finite ranges above --max-range={max_range:g}")
    if not mask.any():
        warnings.append("all RSU samples are masked as unavailable")
    availability = 100.0 * mask.mean(axis=0)
    low_availability = np.flatnonzero(availability < 95.0)
    if low_availability.size:
        labels = ", ".join(f"RSU {index + 1:02d}={availability[index]:.1f}%" for index in low_availability)
        warnings.append(f"RSU synchronization availability is below 95%: {labels}")
    if dataset.rsu_meta.shape[2] < 5:
        warnings.append("rsu_meta has no age_s column (expected index 4)")
    elif np.allclose(np.asarray(dataset.rsu_meta[..., 0]), 0.0):
        warnings.append("RSU distance metadata is all zero; distance gating cannot use RSU geometry")
    for column in range(min(2, target_sample.shape[1])):
        finite_target = target_sample[:, column][np.isfinite(target_sample[:, column])]
        if finite_target.size and np.ptp(finite_target) < 1e-6:
            warnings.append(f"target[{column}] is constant; check Joycon teacher commands")

    lines = [
        f"Dataset: {dataset.root}",
        f"Samples: {dataset.sample_count}",
        f"ego_scans: {dataset.ego_scans.shape} {dataset.ego_scans.dtype}",
        f"rsu_scans: {dataset.rsu_scans.shape} {dataset.rsu_scans.dtype}",
        f"rsu_meta: {dataset.rsu_meta.shape} {dataset.rsu_meta.dtype}",
        f"targets: {dataset.targets.shape} {dataset.targets.dtype}",
        f"rsu_mask: {dataset.rsu_mask.shape} availability={np.round(availability, 1).tolist()}%",
        (
            "ego ranges(sampled): "
            f"finite={ego_stats['finite_pct']:.1f}% +inf={ego_stats['posinf_pct']:.1f}% "
            f"nan={ego_stats['nan_pct']:.3f}% min={ego_stats['min']:.3f} max={ego_stats['max']:.3f}"
        ),
        (
            "RSU ranges(sampled): "
            f"finite={rsu_stats['finite_pct']:.1f}% +inf={rsu_stats['posinf_pct']:.1f}% "
            f"nan={rsu_stats['nan_pct']:.3f}% min={rsu_stats['min']:.3f} max={rsu_stats['max']:.3f}"
        ),
    ]
    if dataset.rsu_meta.shape[2] >= 5:
        age = np.asarray(dataset.rsu_meta[..., 4])
        valid_age = age[mask & np.isfinite(age)]
        lines.append(
            "RSU age_s: "
            + (
                f"mean={valid_age.mean():.4f} max={valid_age.max():.4f}"
                if valid_age.size
                else "no finite valid values"
            )
        )
    for column in range(min(2, dataset.targets.shape[1])):
        finite_target = target_sample[:, column][np.isfinite(target_sample[:, column])]
        if finite_target.size:
            lines.append(
                f"target[{column}]: min={finite_target.min():.4f} "
                f"max={finite_target.max():.4f} mean={finite_target.mean():.4f}"
            )
    lines.append("Integrity: " + ("OK" if not warnings else f"WARN ({len(warnings)})"))
    lines.extend(f"  - {warning}" for warning in warnings)
    return "\n".join(lines), warnings


def _display_scan(scan: np.ndarray, max_range: float) -> np.ndarray:
    return np.clip(np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=0.0), 0.0, max_range)


def create_viewer(
    dataset: FusionDataset,
    initial_index: int,
    max_range: float,
    target_names: tuple[str, str],
) -> tuple[Any, Any]:
    """Create an interactive Matplotlib viewer and return (figure, update)."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    plt.style.use("dark_background")
    rsu_columns = min(3, max(1, dataset.rsu_count))
    rsu_rows = int(np.ceil(dataset.rsu_count / rsu_columns))
    figure = plt.figure(figsize=(15, 3.0 + 2.5 * rsu_rows), constrained_layout=False)
    grid = figure.add_gridspec(
        rsu_rows + 2,
        rsu_columns,
        height_ratios=[1.25] + [1.0] * rsu_rows + [1.1],
        hspace=0.58,
        wspace=0.25,
    )
    figure.subplots_adjust(bottom=0.1, top=0.93)

    ego_axis = figure.add_subplot(grid[0, :])
    ego_x = np.linspace(-1.0, 1.0, dataset.ego_scans.shape[1])
    (ego_line,) = ego_axis.plot(ego_x, np.zeros_like(ego_x), color="white", linewidth=1.0)
    ego_axis.set_xlabel("normalized ray position")
    ego_axis.set_ylabel("range [m]")
    ego_axis.set_ylim(0.0, max_range * 1.03)
    ego_axis.grid(alpha=0.2)

    rsu_axes = []
    rsu_lines = []
    rsu_x = np.linspace(-1.0, 1.0, dataset.rsu_scans.shape[2])
    for sensor in range(dataset.rsu_count):
        axis = figure.add_subplot(grid[1 + sensor // rsu_columns, sensor % rsu_columns])
        color = RSU_COLORS[sensor % len(RSU_COLORS)]
        (line,) = axis.plot(rsu_x, np.zeros_like(rsu_x), color=color, linewidth=0.9)
        axis.set_ylim(0.0, max_range * 1.03)
        axis.set_ylabel("range [m]")
        axis.grid(alpha=0.2)
        rsu_axes.append(axis)
        rsu_lines.append(line)

    target_axis = figure.add_subplot(grid[-1, :])
    plot_indices = np.unique(
        np.linspace(0, dataset.sample_count - 1, num=min(dataset.sample_count, 5000), dtype=np.int64)
    )
    target_values = np.asarray(dataset.targets[plot_indices, :2])
    target_axis.plot(plot_indices, target_values[:, 0], label=target_names[0], color="#00c8ff")
    target_axis.plot(plot_indices, target_values[:, 1], label=target_names[1], color="#ff8c50")
    cursor = target_axis.axvline(initial_index, color="white", linewidth=1.0, alpha=0.8)
    target_axis.set_xlabel("synchronized sample index")
    target_axis.set_ylabel("control target")
    target_axis.grid(alpha=0.2)
    target_axis.legend(loc="upper right")

    slider_axis = figure.add_axes((0.15, 0.025, 0.7, 0.025))
    slider = Slider(
        slider_axis,
        "sample",
        0,
        dataset.sample_count - 1,
        valinit=initial_index,
        valstep=1,
    )

    def update(index_value: int | float) -> None:
        index = int(np.clip(round(float(index_value)), 0, dataset.sample_count - 1))
        ego_scan = np.asarray(dataset.ego_scans[index])
        ego_line.set_ydata(_display_scan(ego_scan, max_range))
        ego_finite = 100.0 * np.isfinite(ego_scan).mean()
        ego_axis.set_title(f"Ego Virtual Scan — sample {index}/{dataset.sample_count - 1}, finite {ego_finite:.1f}%")

        for sensor, (axis, line) in enumerate(zip(rsu_axes, rsu_lines)):
            scan = np.asarray(dataset.rsu_scans[index, sensor])
            valid = bool(dataset.rsu_mask[index, sensor])
            line.set_ydata(_display_scan(scan, max_range))
            line.set_alpha(1.0 if valid else 0.25)
            meta = np.asarray(dataset.rsu_meta[index, sensor])
            details = []
            if meta.size >= 1 and np.isfinite(meta[0]):
                details.append(f"d={meta[0]:.1f}m")
            if meta.size >= 5 and np.isfinite(meta[4]):
                details.append(f"age={meta[4]:.3f}s")
            state = "valid" if valid else "MISSING"
            suffix = ", ".join([state, *details])
            axis.set_title(f"RSU {sensor + 1:02d} — {suffix}")

        cursor.set_xdata([index, index])
        target = np.asarray(dataset.targets[index, :2])
        figure.suptitle(
            f"{dataset.root.name} | {target_names[0]}={target[0]:+.4f}, "
            f"{target_names[1]}={target[1]:+.4f} | arrows/PageUp/PageDown to navigate"
        )
        figure.canvas.draw_idle()

    def on_slider(value: float) -> None:
        update(value)

    def on_key(event: Any) -> None:
        current = int(round(slider.val))
        steps = {"left": -1, "right": 1, "pageup": 10, "pagedown": -10}
        if event.key in steps:
            slider.set_val(np.clip(current + steps[event.key], 0, dataset.sample_count - 1))
        elif event.key == "home":
            slider.set_val(0)
        elif event.key == "end":
            slider.set_val(dataset.sample_count - 1)

    slider.on_changed(on_slider)
    figure.canvas.mpl_connect("key_press_event", on_key)
    update(initial_index)
    return figure, update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and visualize NumPy arrays produced by preprocess_bag_to_npy.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Sequence directory containing *.npy files")
    parser.add_argument("--index", type=int, default=0, help="Initial synchronized sample index")
    parser.add_argument("--max-range", type=float, default=45.0, help="Display clipping range in meters")
    parser.add_argument("--stats-samples", type=int, default=2000, help="Maximum rows used for summary statistics")
    parser.add_argument(
        "--target-names",
        nargs=2,
        metavar=("LONGITUDINAL", "LATERAL"),
        default=("acceleration", "steering"),
        help="Labels for targets[:, 0:2]",
    )
    parser.add_argument("--save", type=Path, help="Save the selected sample as a PNG/PDF/SVG")
    parser.add_argument("--no-show", action="store_true", help="Do not open the interactive Matplotlib window")
    parser.add_argument("--report-only", action="store_true", help="Only validate arrays and print the report")
    parser.add_argument("--fail-on-warning", action="store_true", help="Return a non-zero status for integrity warnings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_range <= 0.0:
        raise SystemExit("--max-range must be positive")
    if args.stats_samples < 1:
        raise SystemExit("--stats-samples must be at least 1")

    try:
        dataset = load_dataset(args.dataset)
        report, warnings = build_report(dataset, args.max_range, args.stats_samples)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        raise SystemExit(f"Dataset validation failed:\n{exc}") from exc

    print(report)
    if warnings and args.fail_on_warning:
        raise SystemExit(1)
    if args.report_only:
        return
    if not 0 <= args.index < dataset.sample_count:
        raise SystemExit(f"--index must be between 0 and {dataset.sample_count - 1}")

    figure, _ = create_viewer(dataset, args.index, args.max_range, tuple(args.target_names))
    if args.save:
        output = args.save.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, bbox_inches="tight")
        print(f"Saved visualization: {output}")
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    elif warnings:
        print("Visualization saved with integrity warnings; review the report above.")


if __name__ == "__main__":
    main()

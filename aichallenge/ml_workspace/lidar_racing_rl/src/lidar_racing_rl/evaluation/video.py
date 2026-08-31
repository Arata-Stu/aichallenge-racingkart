"""Render one deterministic evaluation episode as a track animation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationTrace:
    """Host-side samples for one evaluated Ego episode."""

    center_x: tuple[float, ...]
    center_y: tuple[float, ...]
    center_yaw: tuple[float, ...]
    left_widths: tuple[float, ...]
    right_widths: tuple[float, ...]
    poses: tuple[tuple[float, float, float], ...]
    speeds: tuple[float, ...]
    actions: tuple[tuple[float, float], ...]
    cumulative_progress: tuple[float, ...]
    race_complete: tuple[bool, ...]
    collision: tuple[bool, ...]
    off_track: tuple[bool, ...]
    truncated: tuple[bool, ...]
    control_dt: float
    track_length: float
    vehicle_length: float
    vehicle_width: float

    def validate(self) -> None:
        """Reject malformed or non-finite trace data before rendering."""

        track_count = len(self.center_x)
        if track_count < 3 or any(
            len(values) != track_count
            for values in (
                self.center_y,
                self.center_yaw,
                self.left_widths,
                self.right_widths,
            )
        ):
            raise ValueError("evaluation video requires aligned track arrays")
        sample_count = len(self.poses)
        if sample_count < 2 or any(
            len(values) != sample_count
            for values in (
                self.speeds,
                self.actions,
                self.cumulative_progress,
                self.race_complete,
                self.collision,
                self.off_track,
                self.truncated,
            )
        ):
            raise ValueError("evaluation video requires aligned rollout arrays")
        scalars = (
            *self.center_x,
            *self.center_y,
            *self.center_yaw,
            *self.left_widths,
            *self.right_widths,
            *(value for pose in self.poses for value in pose),
            *self.speeds,
            *(value for action in self.actions for value in action),
            *self.cumulative_progress,
            self.control_dt,
            self.track_length,
            self.vehicle_length,
            self.vehicle_width,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("evaluation video trace contains non-finite values")
        if (
            self.control_dt <= 0.0
            or self.track_length <= 0.0
            or self.vehicle_length <= 0.0
            or self.vehicle_width <= 0.0
        ):
            raise ValueError("evaluation video dimensions and timing must be positive")


def frame_stride(*, control_dt: float, fps: int, playback_speed: float) -> int:
    """Return the integer simulator-sample stride nearest the requested speed."""

    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("control_dt must be finite and positive")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        raise ValueError("fps must be a positive integer")
    if not math.isfinite(playback_speed) or playback_speed <= 0.0:
        raise ValueError("playback_speed must be finite and positive")
    return max(1, round(playback_speed * fps * control_dt))


def _frame_indices(
    sample_count: int,
    *,
    control_dt: float,
    fps: int,
    playback_speed: float,
) -> tuple[int, ...]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    stride = frame_stride(
        control_dt=control_dt,
        fps=fps,
        playback_speed=playback_speed,
    )
    indices = list(range(0, sample_count, stride))
    if indices[-1] != sample_count - 1:
        indices.append(sample_count - 1)
    return tuple(indices)


def render_evaluation_video(
    trace: EvaluationTrace,
    output: Path,
    *,
    fps: int = 20,
    playback_speed: float = 4.0,
) -> None:
    """Render ``trace`` to GIF, or MP4 when an ffmpeg executable is available."""

    trace.validate()
    suffix = output.suffix.lower()
    if suffix not in {".gif", ".mp4"}:
        raise ValueError("evaluation video output must end in .gif or .mp4")
    indices = _frame_indices(
        len(trace.poses),
        control_dt=trace.control_dt,
        fps=fps,
        playback_speed=playback_speed,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
    from matplotlib.patches import Polygon

    center_x = np.asarray(trace.center_x)
    center_y = np.asarray(trace.center_y)
    center_yaw = np.asarray(trace.center_yaw)
    left_widths = np.asarray(trace.left_widths)
    right_widths = np.asarray(trace.right_widths)
    normal_x = -np.sin(center_yaw)
    normal_y = np.cos(center_yaw)
    left_x = center_x + left_widths * normal_x
    left_y = center_y + left_widths * normal_y
    right_x = center_x - right_widths * normal_x
    right_y = center_y - right_widths * normal_y
    poses = np.asarray(trace.poses)
    actions = np.asarray(trace.actions)
    speeds = np.asarray(trace.speeds)
    progress = np.asarray(trace.cumulative_progress)

    all_x = np.concatenate((left_x, right_x))
    all_y = np.concatenate((left_y, right_y))
    padding = max(float(np.ptp(all_x)), float(np.ptp(all_y))) * 0.04
    figure, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    figure.patch.set_facecolor("#f8fafc")
    axis.set_facecolor("#f8fafc")
    axis.plot(left_x, left_y, color="#111827", linewidth=1.3)
    axis.plot(right_x, right_y, color="#111827", linewidth=1.3)
    axis.plot(
        center_x,
        center_y,
        color="#a855f7",
        linewidth=0.9,
        linestyle="--",
        alpha=0.75,
    )
    axis.set_xlim(float(np.min(all_x) - padding), float(np.max(all_x) + padding))
    axis.set_ylim(float(np.min(all_y) - padding), float(np.max(all_y) + padding))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Deterministic LiDAR policy rollout")
    (trail,) = axis.plot([], [], color="#0284c7", linewidth=2.0, alpha=0.9)
    (current,) = axis.plot([], [], "o", color="#dc2626", markersize=4)
    car = Polygon(np.zeros((4, 2)), closed=True, facecolor="#ef4444", alpha=0.45)
    axis.add_patch(car)
    status = axis.text(
        0.01,
        0.99,
        "",
        transform=axis.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        color="#111827",
        bbox={"facecolor": "#ffffff", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )

    local_corners = np.asarray(
        [
            [0.5 * trace.vehicle_length, 0.5 * trace.vehicle_width],
            [0.5 * trace.vehicle_length, -0.5 * trace.vehicle_width],
            [-0.5 * trace.vehicle_length, -0.5 * trace.vehicle_width],
            [-0.5 * trace.vehicle_length, 0.5 * trace.vehicle_width],
        ]
    )

    def update(frame_number: int) -> tuple[object, ...]:
        index = indices[frame_number]
        x, y, yaw = poses[index]
        rotation = np.asarray([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        car.set_xy(local_corners @ rotation.T + np.asarray([x, y]))
        trail.set_data(poses[: index + 1, 0], poses[: index + 1, 1])
        current.set_data([x], [y])
        completion = 100.0 * progress[index] / trace.track_length
        terminal = (
            "race_complete"
            if trace.race_complete[index]
            else "collision"
            if trace.collision[index]
            else "off_track"
            if trace.off_track[index]
            else "truncated"
            if trace.truncated[index]
            else "running"
        )
        status.set_text(
            f"step {index:4d}  t={index * trace.control_dt:6.2f} s\n"
            f"speed={speeds[index]:6.2f} m/s\n"
            f"steer={actions[index, 0]:+6.3f}  accel={actions[index, 1]:+6.3f}\n"
            f"Frenet progress={progress[index]:7.2f}/{trace.track_length:.2f} m "
            f"({completion:5.1f}%)\n"
            f"state={terminal}"
        )
        return trail, current, car, status

    animation = FuncAnimation(
        figure,
        update,
        frames=len(indices),
        interval=1000.0 / fps,
        blit=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        if not writers.is_available("ffmpeg"):
            raise RuntimeError(
                "MP4 output requires ffmpeg in the container; use a .gif output instead"
            )
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2400)
    try:
        animation.save(output, writer=writer, dpi=110)
    finally:
        plt.close(figure)


__all__ = ["EvaluationTrace", "frame_stride", "render_evaluation_video"]

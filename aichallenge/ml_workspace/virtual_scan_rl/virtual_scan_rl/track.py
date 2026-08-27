"""Race-line projection constrained locally to avoid hairpin cross-lane jumps."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


def completed_lap_count(
    accumulated_progress_m: float,
    track_length_m: float,
    awsim_lap_transitions: int,
) -> int:
    """Fuse geometric progress with AWSIM's start-line counter semantics."""
    track_laps = max(0, int(accumulated_progress_m / max(track_length_m, 1e-6)))
    counter_laps = max(0, int(awsim_lap_transitions) - 1)
    return max(track_laps, counter_laps)


class TrackProgress:
    def __init__(
        self,
        raceline_csv: str,
        *,
        search_back: int = 15,
        search_forward: int = 50,
        max_step_m: float = 4.0,
    ) -> None:
        self.points = self._load_points(raceline_csv)
        if len(self.points) < 3:
            raise ValueError("Raceline must contain at least three points")
        self.segment_start = self.points
        self.segment_end = np.roll(self.points, -1, axis=0)
        self.segment_vec = self.segment_end - self.segment_start
        self.segment_len = np.linalg.norm(self.segment_vec, axis=1)
        if np.any(self.segment_len < 1e-6):
            raise ValueError("Raceline contains duplicate adjacent points")
        self.cumulative = np.concatenate(
            (np.array([0.0]), np.cumsum(self.segment_len, dtype=np.float64))
        )
        self.total_length = float(self.cumulative[-1])
        self.search_back = int(search_back)
        self.search_forward = int(search_forward)
        self.max_step_m = float(max_step_m)
        self.segment_index: int | None = None
        self.s: float | None = None
        self.accumulated_progress = 0.0

    @staticmethod
    def _load_points(path: str) -> np.ndarray:
        csv_path = Path(path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Raceline CSV not found: {csv_path}")
        points: list[tuple[float, float]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "x" not in reader.fieldnames or "y" not in reader.fieldnames:
                raise ValueError(f"Raceline CSV requires x,y columns: {csv_path}")
            for row in reader:
                points.append((float(row["x"]), float(row["y"])))
        raw_points = np.asarray(points, dtype=np.float64)
        if raw_points.ndim != 2 or raw_points.shape[1] != 2:
            raise ValueError(f"Raceline CSV contains no usable x,y points: {csv_path}")
        if not np.all(np.isfinite(raw_points)):
            raise ValueError(f"Raceline CSV contains non-finite x,y values: {csv_path}")

        # Some trajectory CSVs explicitly close the loop by repeating the first
        # point as the final row. TrackProgress already creates that closing
        # segment with np.roll(), so retaining the repeated row would introduce
        # a zero-length segment. Also tolerate repeated samples inside the file.
        if len(raw_points) > 1:
            keep = np.concatenate(
                (
                    np.array([True]),
                    np.linalg.norm(np.diff(raw_points, axis=0), axis=1) >= 1e-6,
                )
            )
            raw_points = raw_points[keep]
        if len(raw_points) > 1 and np.linalg.norm(raw_points[-1] - raw_points[0]) < 1e-6:
            raw_points = raw_points[:-1]
        return raw_points

    def reset(self, x: float, y: float, yaw: float | None = None) -> tuple[float, float]:
        index, s, distance = self._project(x, y, None, yaw)
        self.segment_index = index
        self.s = s
        self.accumulated_progress = 0.0
        return s, distance

    def update(self, x: float, y: float) -> tuple[float, float, float]:
        if self.segment_index is None or self.s is None:
            s, distance = self.reset(x, y)
            return 0.0, s, distance
        index, new_s, distance = self._project(x, y, self.segment_index, None)
        delta = new_s - self.s
        if delta < -0.5 * self.total_length:
            delta += self.total_length
        elif delta > 0.5 * self.total_length:
            delta -= self.total_length
        if abs(delta) > self.max_step_m:
            delta = 0.0
        self.segment_index = index
        self.s = new_s
        self.accumulated_progress += delta
        return float(delta), float(new_s), float(distance)

    def _project(
        self,
        x: float,
        y: float,
        center_index: int | None,
        yaw: float | None,
    ) -> tuple[int, float, float]:
        count = len(self.points)
        if center_index is None:
            indices = np.arange(count, dtype=np.int64)
        else:
            offsets = np.arange(-self.search_back, self.search_forward + 1)
            indices = (center_index + offsets) % count
        starts = self.segment_start[indices]
        vectors = self.segment_vec[indices]
        lengths_sq = np.sum(vectors * vectors, axis=1)
        query = np.array([x, y], dtype=np.float64)
        ratios = np.clip(np.sum((query - starts) * vectors, axis=1) / lengths_sq, 0.0, 1.0)
        projected = starts + ratios[:, None] * vectors
        distances_sq = np.sum((projected - query) ** 2, axis=1)
        scores = distances_sq.copy()
        if center_index is None and yaw is not None:
            heading = np.array([math.cos(yaw), math.sin(yaw)])
            alignment = np.sum(vectors / self.segment_len[indices, None] * heading, axis=1)
            scores += np.where(alignment >= 0.0, 0.0, 25.0)
        local = int(np.argmin(scores))
        index = int(indices[local])
        s = float(self.cumulative[index] + ratios[local] * self.segment_len[index])
        return index, s % self.total_length, float(math.sqrt(distances_sq[local]))

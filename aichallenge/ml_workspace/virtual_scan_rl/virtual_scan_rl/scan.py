"""Virtual LaserScan preprocessing and temporal stacking."""

from __future__ import annotations

from collections import deque

import numpy as np


def sanitize_scan(ranges, *, num_rays: int, max_range_m: float) -> np.ndarray:
    """Return a fixed-size float32 scan normalized to [0, 1]."""
    values = np.asarray(ranges, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("LaserScan contains no ranges")
    values = np.nan_to_num(values, nan=max_range_m, posinf=max_range_m, neginf=0.0)
    values = np.clip(values, 0.0, max_range_m)
    if values.size != num_rays:
        old_x = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
        new_x = np.linspace(0.0, 1.0, num_rays, dtype=np.float32)
        values = np.interp(new_x, old_x, values).astype(np.float32)
    return values / float(max_range_m)


class ScanHistory:
    def __init__(self, frames: int, num_rays: int) -> None:
        if frames < 1 or num_rays < 1:
            raise ValueError("frames and num_rays must be positive")
        self.frames = int(frames)
        self.num_rays = int(num_rays)
        self._values: deque[np.ndarray] = deque(maxlen=self.frames)

    def reset(self, scan: np.ndarray) -> np.ndarray:
        value = self._validate(scan)
        self._values.clear()
        for _ in range(self.frames):
            self._values.append(value.copy())
        return self.value()

    def append(self, scan: np.ndarray) -> np.ndarray:
        value = self._validate(scan)
        if not self._values:
            return self.reset(value)
        self._values.append(value.copy())
        return self.value()

    def value(self) -> np.ndarray:
        if not self._values:
            return np.ones((self.frames, self.num_rays), dtype=np.float32)
        return np.stack(tuple(self._values), axis=0).astype(np.float32, copy=False)

    def _validate(self, scan: np.ndarray) -> np.ndarray:
        value = np.asarray(scan, dtype=np.float32)
        if value.shape != (self.num_rays,):
            raise ValueError(f"Expected scan shape {(self.num_rays,)}, got {value.shape}")
        return value


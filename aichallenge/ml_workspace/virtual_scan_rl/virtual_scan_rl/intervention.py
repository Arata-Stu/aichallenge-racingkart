"""Human-intervention helpers and intervention demonstration recording."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np


def trigger_press(axis_value: float) -> float:
    """Convert DualShock released=+1/pressed=-1 axis to [0, 1]."""
    return float(np.clip((1.0 - float(axis_value)) * 0.5, 0.0, 1.0))


def joy_action(
    axes,
    *,
    steer_axis: int,
    positive_axis: int,
    negative_axis: int,
    deadzone: float,
) -> np.ndarray:
    def axis(index: int, default: float) -> float:
        return float(axes[index]) if 0 <= index < len(axes) else default

    steering = float(np.clip(axis(steer_axis, 0.0), -1.0, 1.0))
    throttle = trigger_press(axis(positive_axis, 1.0))
    brake = trigger_press(axis(negative_axis, 1.0))
    longitudinal = throttle - brake
    if abs(longitudinal) < deadzone:
        longitudinal = 0.0
    return np.array([steering, longitudinal], dtype=np.float32)


class InterventionRecorder:
    def __init__(self, directory: str, flush_samples: int = 500) -> None:
        self.directory = Path(directory)
        self.flush_samples = max(1, int(flush_samples))
        self._records: list[dict[str, np.ndarray | float | int]] = []
        self._part = 0
        self._session = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"

    def add(
        self,
        *,
        scan: np.ndarray,
        state: np.ndarray,
        proposed_action: np.ndarray,
        executed_action: np.ndarray,
        reward: float,
    ) -> None:
        self._records.append(
            {
                "scan": np.asarray(scan, dtype=np.float16).copy(),
                "state": np.asarray(state, dtype=np.float32).copy(),
                "proposed_action": np.asarray(proposed_action, dtype=np.float32).copy(),
                "executed_action": np.asarray(executed_action, dtype=np.float32).copy(),
                "reward": float(reward),
            }
        )
        if len(self._records) >= self.flush_samples:
            self.flush()

    def flush(self) -> Path | None:
        if not self._records:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        output = self.directory / f"interventions_{self._session}_{self._part:04d}.npz"
        np.savez_compressed(
            output,
            scan=np.stack([r["scan"] for r in self._records]),
            state=np.stack([r["state"] for r in self._records]),
            proposed_action=np.stack([r["proposed_action"] for r in self._records]),
            executed_action=np.stack([r["executed_action"] for r in self._records]),
            reward=np.asarray([r["reward"] for r in self._records], dtype=np.float32),
        )
        self._records.clear()
        self._part += 1
        return output


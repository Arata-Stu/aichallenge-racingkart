"""Rule-based longitudinal control used with steering-only TinyLiDARNet models."""

from __future__ import annotations


class RuleBasedAccelerationController:
    """Use full acceleration for launch, then a limited cruise command.

    Hysteresis prevents the acceleration command from rapidly switching when the
    measured speed is close to the configured threshold.
    """

    def __init__(
        self,
        *,
        startup_acceleration: float = 1.0,
        cruise_acceleration: float = 0.7,
        speed_threshold_kmh: float = 15.0,
        speed_hysteresis_kmh: float = 1.0,
    ) -> None:
        if speed_threshold_kmh <= 0.0:
            raise ValueError("speed_threshold_kmh must be positive")
        if speed_hysteresis_kmh < 0.0:
            raise ValueError("speed_hysteresis_kmh must be non-negative")
        if speed_hysteresis_kmh >= speed_threshold_kmh:
            raise ValueError("speed_hysteresis_kmh must be smaller than speed_threshold_kmh")

        self.startup_acceleration = float(startup_acceleration)
        self.cruise_acceleration = float(cruise_acceleration)
        self.speed_threshold_kmh = float(speed_threshold_kmh)
        self.speed_hysteresis_kmh = float(speed_hysteresis_kmh)
        self._startup_mode = True

    @property
    def startup_mode(self) -> bool:
        return self._startup_mode

    def command(self, speed_mps: float) -> float:
        speed_kmh = abs(float(speed_mps)) * 3.6
        upper_threshold = self.speed_threshold_kmh + self.speed_hysteresis_kmh
        lower_threshold = self.speed_threshold_kmh - self.speed_hysteresis_kmh

        if self._startup_mode and speed_kmh >= upper_threshold:
            self._startup_mode = False
        elif not self._startup_mode and speed_kmh <= lower_threshold:
            self._startup_mode = True

        if self._startup_mode:
            return self.startup_acceleration
        return self.cruise_acceleration

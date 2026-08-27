"""TensorBoard metrics specific to driving and human intervention."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class DrivingMetricsCallback(BaseCallback):
    def __init__(self, log_every_steps: int = 500) -> None:
        super().__init__(verbose=0)
        self.log_every_steps = max(1, int(log_every_steps))
        self.window_steps = 0
        self.window_interventions = 0
        self.last_info: dict = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            self.window_steps += 1
            self.window_interventions += int(bool(info.get("human_intervention", False)))
            self.last_info = info
        if self.n_calls % self.log_every_steps == 0 and self.window_steps:
            self.logger.record(
                "driving/human_intervention_rate",
                self.window_interventions / self.window_steps,
            )
            for source, target in (
                ("speed_mps", "driving/speed_mps"),
                ("track_distance_m", "driving/track_distance_m"),
                ("total_progress_m", "driving/episode_progress_m"),
                ("lap_time_s", "driving/lap_time_s"),
            ):
                if source in self.last_info:
                    self.logger.record(target, float(self.last_info[source]))
            self.window_steps = 0
            self.window_interventions = 0
        return True


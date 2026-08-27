"""SAC variant that stores the action actually executed after Joy takeover."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import SAC


class InterventionSAC(SAC):
    def _store_transition(
        self, replay_buffer, buffer_action, new_obs, reward, dones, infos
    ) -> None:
        actual_buffer_action = np.array(buffer_action, dtype=np.float32, copy=True)
        for index, info in enumerate(infos):
            executed = info.get("executed_action")
            if executed is None:
                continue
            env_action = np.asarray(executed, dtype=np.float32).reshape(1, -1)
            actual_buffer_action[index] = self.policy.scale_action(env_action)[0]
        super()._store_transition(
            replay_buffer, actual_buffer_action, new_obs, reward, dones, infos
        )


import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from stable_baselines3.common.vec_env import DummyVecEnv

from virtual_scan_rl.sac import InterventionSAC


class DummyEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), 0.0, False, False, {}


def test_intervention_action_is_stored_in_replay_buffer():
    env = DummyVecEnv([DummyEnv])
    model = InterventionSAC("MlpPolicy", env, buffer_size=10, learning_starts=1)
    model._last_obs = env.reset()
    model._store_transition(
        model.replay_buffer,
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        np.array([False]),
        [{"executed_action": np.array([0.75, -0.5], dtype=np.float32)}],
    )
    np.testing.assert_allclose(model.replay_buffer.actions[0, 0], [0.75, -0.5])


import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from virtual_scan_rl.model import TinyLidarFeatureExtractor


def test_tiny_lidar_feature_shape():
    observation_space = gym.spaces.Dict(
        {
            "scan": gym.spaces.Box(0.0, 1.0, shape=(4, 1080), dtype=np.float32),
            "state": gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
        }
    )
    extractor = TinyLidarFeatureExtractor(observation_space, features_dim=128)
    result = extractor(
        {
            "scan": torch.zeros((2, 4, 1080)),
            "state": torch.zeros((2, 5)),
        }
    )
    assert tuple(result.shape) == (2, 128)


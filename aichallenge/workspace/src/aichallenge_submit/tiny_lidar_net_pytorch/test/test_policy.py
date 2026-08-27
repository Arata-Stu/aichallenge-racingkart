import pytest

torch = pytest.importorskip("torch")

from tiny_lidar_net_pytorch.model import build_model
from tiny_lidar_net_pytorch.policy import TinyLidarTorchPolicy


def test_checkpoint_round_trip_and_scan_resize(tmp_path):
    model = build_model("small", input_dim=1080, output_dim=2)
    checkpoint_path = tmp_path / "model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "small",
            "input_dim": 1080,
            "output_dim": 2,
            "max_range": 30.0,
        },
        checkpoint_path,
    )

    policy = TinyLidarTorchPolicy(checkpoint_path, device="cpu")
    processed = policy.preprocess([0.0, float("nan"), float("inf"), 3.0])
    assert processed.shape == (1, 1, 1080)
    assert torch.isfinite(processed).all()
    acceleration, steering = policy.predict([1.0] * 750)
    assert -1.0 <= acceleration <= 1.0
    assert -1.0 <= steering <= 1.0


def test_preprocess_does_not_modify_tensor_input(tmp_path):
    model = build_model("small", input_dim=1080, output_dim=2)
    checkpoint_path = tmp_path / "model.pth"
    torch.save({"model_state_dict": model.state_dict(), "architecture": "small"}, checkpoint_path)
    policy = TinyLidarTorchPolicy(checkpoint_path, device="cpu")
    source = torch.tensor([-1.0, 40.0])

    policy.preprocess(source)

    assert torch.equal(source, torch.tensor([-1.0, 40.0]))

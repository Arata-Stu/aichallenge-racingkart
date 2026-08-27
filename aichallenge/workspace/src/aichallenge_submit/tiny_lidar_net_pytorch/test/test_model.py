import pytest

torch = pytest.importorskip("torch")

from tiny_lidar_net_pytorch.model import build_model


@pytest.mark.parametrize("architecture", ["normal", "small"])
def test_model_output_shape(architecture):
    model = build_model(architecture, input_dim=1080, output_dim=2)
    output = model(torch.zeros(3, 1, 1080))
    assert output.shape == (3, 2)
    assert torch.all(output <= 1.0)
    assert torch.all(output >= -1.0)


def test_rejects_unknown_architecture():
    with pytest.raises(ValueError):
        build_model("unknown")

import torch

from climate_forecasting.model import ClimateLSTM, ModelConfig


def test_model_forward_shapes():
    cfg = ModelConfig(input_size=6, hidden_size=16, num_layers=1, bidirectional=False)
    model = ClimateLSTM(cfg)

    b, t, f = 2, 7, cfg.input_size
    x = torch.zeros((b, t, f), dtype=torch.float32)
    y = model(x)

    assert y.shape == (b, t, 1)

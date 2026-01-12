import torch

from climate_forecasting.model import ClimateLSTM, ModelConfig


def test_model_forward_shapes():
    cfg = ModelConfig(input_size=6, hidden_size=16, num_layers=1, bidirectional=False)
    model = ClimateLSTM(cfg)

    batch_size, lookback, num_features = 2, 7, cfg.input_size
    features_tensor = torch.zeros(
        (batch_size, lookback, num_features), dtype=torch.float32
    )
    predictions = model(features_tensor)

    assert predictions.shape == (batch_size, lookback, 1)

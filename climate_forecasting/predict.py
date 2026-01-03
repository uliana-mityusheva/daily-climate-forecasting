from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from climate_forecasting.data import (
    DataConfig,
    load_scaler,
    inverse_transform_meantemp,
    create_windows_seq2seq,
    _read_raw_df,
    _feature_engineering,
    ALL_COLS,
)
from climate_forecasting.model import ClimateLSTM, ModelConfig
import json


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: Path, device: torch.device) -> ClimateLSTM:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    state = ckpt["state_dict"]

    raw_cfg = ckpt.get("model_cfg", {})
    cfg = ModelConfig(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg

    model = ClimateLSTM(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_on_test(
    model: ClimateLSTM,
    cfg: DataConfig,
) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True, parents=True)

    # -------- data --------
    df = _read_raw_df(cfg.raw_path)
    df = _feature_engineering(df)
    df = df.loc[:, ("date", *ALL_COLS)].copy()

    series = df.loc[:, ALL_COLS].to_numpy(dtype=np.float32)

    pack = load_scaler(cfg.processed_dir, cfg.scaler_name)
    scaler = pack["scaler"]

    series_scaled = scaler.transform(series).astype(np.float32)

    # time split (same logic as training)
    n = len(series_scaled)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))
    test_scaled = series_scaled[val_end:]

    X_test, y_test = create_windows_seq2seq(test_scaled, cfg.lookback)

    # -------- inference --------
    device = next(model.parameters()).device
    X = torch.from_numpy(X_test).float().to(device)

    y_true_scaled = y_test[:, -1, 0]          # last timestep (numpy)
    y_hat = model(X)                          # [B,T,1]
    y_pred_scaled = y_hat[:, -1, 0].cpu().numpy()

    # -------- inverse transform --------
    # reference rows: last row of each window
    ref_rows = test_scaled[cfg.lookback - 1 : cfg.lookback - 1 + len(y_pred_scaled)]

    y_true = inverse_transform_meantemp(y_true_scaled, ref_rows, scaler)
    y_pred = inverse_transform_meantemp(y_pred_scaled, ref_rows, scaler)

    # -------- metrics --------
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # compatible with older sklearn

    print(f"TEST MAE  (°C): {mae:.3f}")
    print(f"TEST RMSE (°C): {rmse:.3f}")

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
    }

    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # -------- save CSV --------
    out_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    out_df.to_csv(reports_dir / "test_predictions.csv", index=False)

    # -------- plot --------
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.title("Test predictions (meantemp)")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "test_predictions.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/model_best.pt"),
    )
    args = parser.parse_args()

    cfg = DataConfig()
    device = _get_device()
    print(f"Device: {device}")

    model = load_model(args.model_path, device)
    predict_on_test(model, cfg)


if __name__ == "__main__":
    main()
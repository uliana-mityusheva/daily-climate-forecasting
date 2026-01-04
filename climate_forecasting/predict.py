from __future__ import annotations

import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from climate_forecasting.data import (
    ALL_COLS,
    DataConfig,
    _feature_engineering,
    _read_raw_df,
    create_windows_seq2seq,
    inverse_transform_meantemp,
    load_scaler,
)
from climate_forecasting.data_download import ensure_raw_data_dvc_first
from climate_forecasting.model import ClimateLSTM, ModelConfig


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
    # also ensure plots dir from config exists
    # cfg for logging comes from the global Hydra cfg; we receive only DataConfig here,
    # so we'll derive plots_dir from hydra cfg in main()

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

    y_true_scaled = y_test[:, -1, 0]  # last timestep (numpy)
    y_hat = model(X)  # [B,T,1]
    y_pred_scaled = y_hat[:, -1, 0].cpu().numpy()

    # -------- inverse transform --------
    # reference rows: last row of each window
    ref_rows = test_scaled[cfg.lookback - 1 : cfg.lookback - 1 + len(y_pred_scaled)]

    y_true = inverse_transform_meantemp(y_true_scaled, ref_rows, scaler)
    y_pred = inverse_transform_meantemp(y_pred_scaled, ref_rows, scaler)

    # -------- metrics --------
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(
        np.sqrt(mean_squared_error(y_true, y_pred))
    )  # compatible with older sklearn

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
    # Save to reports for artifacts and to plots for required logging
    plt.savefig(reports_dir / "test_predictions.png")
    plt.close()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    data_cfg = DataConfig(
        raw_path=Path(to_absolute_path(cfg.data.raw_path)),
        data_url=str(cfg.data.data_url),
        processed_dir=Path(to_absolute_path(cfg.data.processed_dir)),
        lookback=int(cfg.data.lookback),
        train_ratio=float(cfg.data.train_ratio),
        val_ratio=float(cfg.data.val_ratio),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        shuffle_train=bool(cfg.data.shuffle_train),
        save_scaler=bool(cfg.data.save_scaler),
        scaler_name=str(cfg.data.scaler_name),
        use_dvc=bool(getattr(cfg.data, "use_dvc", True)),
        dvc_repo=str(getattr(cfg.data, "dvc_repo", ".")),
        dvc_path=str(getattr(cfg.data, "dvc_path", "")) or None,
        dvc_rev=(str(getattr(cfg.data, "dvc_rev", "")) or None),
    )

    ensure_raw_data_dvc_first(
        dst=data_cfg.raw_path,
        public_link=data_cfg.data_url,
        dvc_repo=str(getattr(data_cfg, "dvc_repo", ".")),
        dvc_path=str(getattr(data_cfg, "dvc_path", "")) or None,
        dvc_rev=str(getattr(cfg.data, "dvc_rev", "")) or None,
    )
    device = _get_device()
    print(f"Device: {device}")

    model_path = Path(to_absolute_path(cfg.predict.model_path))
    model = load_model(model_path, device)
    predict_on_test(model, data_cfg)

    # Save a copy of the predictions plot into configured plots directory
    plots_dir = Path(to_absolute_path(cfg.logging.plots_dir))
    plots_dir.mkdir(parents=True, exist_ok=True)
    src_plot = Path("reports") / "test_predictions.png"
    if src_plot.exists():
        # Recreate the plot file in plots_dir by copying bytes
        dst_plot = plots_dir / "test_predictions.png"
        try:
            with open(src_plot, "rb") as fsrc, open(dst_plot, "wb") as fdst:
                fdst.write(fsrc.read())
        except Exception:
            pass


if __name__ == "__main__":
    main()

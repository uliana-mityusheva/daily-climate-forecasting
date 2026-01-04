# climate_forecasting/train.py
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn
from torch.optim import Adam

from climate_forecasting.data import DataConfig, create_dataloaders
from climate_forecasting.data_download import ensure_raw_data
from climate_forecasting.model import ClimateLSTM, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0

    save_dir: Path = Path("artifacts")
    ckpt_name: str = "model_best.pt"

    # set 0 to disable early stopping
    early_stopping_patience: int = 10


def _to_jsonable(d: dict) -> dict:
    """Convert Path values to str
    so checkpoint is safe to unpickle across torch versions."""
    out = {}
    for k, v in d.items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
) -> None:
    """
    Save a safe checkpoint:
      - weights (state_dict)
      - JSON-serializable configs (Path -> str)
    Also writes a sibling .json with configs for convenience.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "model_cfg": _to_jsonable(model_cfg.__dict__),
        "data_cfg": _to_jsonable(data_cfg.__dict__),
    }
    torch.save(payload, path)

    json_path = path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model_cfg": payload["model_cfg"], "data_cfg": payload["data_cfg"]},
            f,
            indent=2,
            ensure_ascii=False,
        )


class LSTMModule(pl.LightningModule):
    def __init__(self, model_cfg: ModelConfig, lr: float, weight_decay: float):
        super().__init__()
        self.model = ClimateLSTM(model_cfg)
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay

        # For plotting after training
        self.train_loss_hist = []
        self.val_loss_hist = []
        self.train_last_mse_hist = []
        self.val_last_mse_hist = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        features, targets = batch
        preds = self(features)
        loss = self.criterion(preds, targets)
        last_mse = torch.mean((preds[:, -1, 0] - targets[:, -1, 0]) ** 2)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train/last_mse", last_mse, prog_bar=False, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx: int):
        features, targets = batch
        preds = self(features)
        loss = self.criterion(preds, targets)
        last_mse = torch.mean((preds[:, -1, 0] - targets[:, -1, 0]) ** 2)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/last_mse", last_mse, prog_bar=False, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_last_mse": last_mse}

    def test_step(self, batch, batch_idx: int):
        features, targets = batch
        preds = self(features)
        loss = self.criterion(preds, targets)
        last_mse = torch.mean((preds[:, -1, 0] - targets[:, -1, 0]) ** 2)
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test/last_mse", last_mse, prog_bar=False, on_step=False, on_epoch=True
        )
        return {"test_loss": loss, "test_last_mse": last_mse}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def on_train_epoch_end(self):
        # Collect last logged metrics for plotting
        metrics = self.trainer.callback_metrics
        if "train/loss" in metrics:
            self.train_loss_hist.append(float(metrics["train/loss"].detach().cpu()))
        if "train/last_mse" in metrics:
            self.train_last_mse_hist.append(
                float(metrics["train/last_mse"].detach().cpu())
            )

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "val/loss" in metrics:
            self.val_loss_hist.append(float(metrics["val/loss"].detach().cpu()))
        if "val/last_mse" in metrics:
            self.val_last_mse_hist.append(float(metrics["val/last_mse"].detach().cpu()))


def _build_data_config(cfg: DictConfig) -> DataConfig:
    """Create DataConfig from Hydra config with proper Path types and absolute paths."""
    return DataConfig(
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
    )


def _build_model_config(cfg: DictConfig) -> ModelConfig:
    return ModelConfig(
        input_size=int(cfg.model.input_size),
        hidden_size=int(cfg.model.hidden_size),
        num_layers=int(cfg.model.num_layers),
        dropout=float(cfg.model.dropout),
        bidirectional=bool(cfg.model.bidirectional),
        out_size=int(cfg.model.out_size),
    )


def _build_train_config(cfg: DictConfig) -> TrainConfig:
    return TrainConfig(
        epochs=int(cfg.train.epochs),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        save_dir=Path(to_absolute_path(cfg.train.save_dir)),
        ckpt_name=str(cfg.train.ckpt_name),
        early_stopping_patience=int(cfg.train.early_stopping_patience),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print the active config for traceability
    print(OmegaConf.to_yaml(cfg))

    # Configs
    data_cfg = _build_data_config(cfg)
    ensure_raw_data(public_link=data_cfg.data_url, dst=data_cfg.raw_path)
    model_cfg = _build_model_config(cfg)
    train_cfg = _build_train_config(cfg)

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(data_cfg)

    # Lightning module
    lit_module = LSTMModule(
        model_cfg, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(train_cfg.save_dir),
        filename=Path(train_cfg.ckpt_name).stem,
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        save_last=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=train_cfg.early_stopping_patience,
        mode="min",
    )

    # MLflow logger
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
        run_name=cfg.logging.run_name,
    )
    # Log hyperparameters
    mlf_logger.log_hyperparams(
        {
            **{f"data/{k}": v for k, v in data_cfg.__dict__.items()},
            **{f"model/{k}": v for k, v in model_cfg.__dict__.items()},
            **{f"train/{k}": v for k, v in train_cfg.__dict__.items()},
        }
    )

    # Log git commit SHA as code version
    try:
        commit_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
        mlf_logger.experiment.log_param(mlf_logger.run_id, "git_commit", commit_sha)
    except Exception:
        pass

    trainer = pl.Trainer(
        max_epochs=train_cfg.epochs,
        logger=mlf_logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
    )

    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # After fit, load best checkpoint,
    # save a plain PyTorch checkpoint for predict.py compatibility, then test
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        print(f"Best Lightning checkpoint: {best_ckpt}")
        lit_best = LSTMModule.load_from_checkpoint(
            best_ckpt,
            model_cfg=model_cfg,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        model = lit_best.model
        # Save in previous plain format for predict.py
        ckpt_path = train_cfg.save_dir / "model_best.pt"
        save_checkpoint(ckpt_path, model, model_cfg, data_cfg)
        print(f"Saved plain checkpoint for inference: {ckpt_path}")

    # Save training plots to configured plots directory
    plots_dir = Path(to_absolute_path(cfg.logging.plots_dir))
    plots_dir.mkdir(parents=True, exist_ok=True)

    if getattr(lit_module, "train_loss_hist", None):
        plt.figure()
        plt.plot(lit_module.train_loss_hist, label="train/loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "train_loss.png")
        plt.close()

    if getattr(lit_module, "val_loss_hist", None):
        plt.figure()
        plt.plot(lit_module.val_loss_hist, label="val/loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "val_loss.png")
        plt.close()

    if getattr(lit_module, "val_last_mse_hist", None):
        plt.figure()
        plt.plot(lit_module.val_last_mse_hist, label="val/last_mse")
        plt.xlabel("Epoch")
        plt.ylabel("Last-step MSE")
        plt.title("Validation Last-step MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "val_last_mse.png")
        plt.close()

    # Test with the best model (Lightning will load from checkpoint if provided)
    trainer.test(lit_module if not best_ckpt else lit_best, dataloaders=test_loader)


if __name__ == "__main__":
    main()

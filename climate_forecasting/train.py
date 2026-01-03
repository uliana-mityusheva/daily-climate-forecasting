# climate_forecasting/train.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import json

import torch
from torch import nn
from torch.optim import Adam

from climate_forecasting.data import DataConfig, create_dataloaders
from climate_forecasting.model import ModelConfig, ClimateLSTM


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0

    save_dir: Path = Path("artifacts")
    ckpt_name: str = "model_best.pt"

    # set 0 to disable early stopping
    early_stopping_patience: int = 10


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_jsonable(d: dict) -> dict:
    """Convert Path values to str so checkpoint is safe to unpickle across torch versions."""
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_last_mse = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = criterion(y_hat, y)
        total_loss += float(loss.item())

        last_mse = torch.mean((y_hat[:, -1, 0] - y[:, -1, 0]) ** 2)
        total_last_mse += float(last_mse.item())

        n_batches += 1

    if n_batches == 0:
        return {"loss": float("nan"), "last_mse": float("nan")}

    return {"loss": total_loss / n_batches, "last_mse": total_last_mse / n_batches}


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_last_mse = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(x)

        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

        last_mse = torch.mean((y_hat[:, -1, 0] - y[:, -1, 0]) ** 2)
        total_last_mse += float(last_mse.item())

        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1), "last_mse": total_last_mse / max(n_batches, 1)}


def main() -> None:
    # Configs
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    device = get_device()
    print("Device:", device)

    train_loader, val_loader, test_loader = create_dataloaders(data_cfg)

    model = ClimateLSTM(model_cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    patience_left = train_cfg.early_stopping_patience

    ckpt_path = train_cfg.save_dir / train_cfg.ckpt_name

    for epoch in range(1, train_cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d}/{train_cfg.epochs} | "
            f"train_loss={tr['loss']:.6f} train_last_mse={tr['last_mse']:.6f} | "
            f"val_loss={va['loss']:.6f} val_last_mse={va['last_mse']:.6f}"
        )

        improved = va["loss"] < best_val - 1e-12
        if improved:
            best_val = va["loss"]
            best_epoch = epoch
            save_checkpoint(ckpt_path, model, model_cfg, data_cfg)
            print(f"  ✅ Saved best checkpoint (val_loss={best_val:.6f})")
            patience_left = train_cfg.early_stopping_patience
        else:
            if train_cfg.early_stopping_patience > 0:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"  🛑 Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                    break

    # Load best checkpoint and evaluate on test
    if ckpt_path.exists():
        # PyTorch 2.6+ default weights_only=True may reject non-weights objects.
        # We saved JSON-serializable configs, but still explicitly allow full load for our trusted checkpoint.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded best checkpoint from {ckpt_path}")

    te = evaluate(model, test_loader, criterion, device)
    print(f"TEST | loss={te['loss']:.6f} last_mse={te['last_mse']:.6f}")


if __name__ == "__main__":
    main()
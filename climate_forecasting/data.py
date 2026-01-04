from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

DATE_COL = "date"

# Kaggle-style columns / features
COL_MONTH = "month"
COL_DAY = "day"
COL_HUM = "humidity"
COL_WIND = "wind_speed"
COL_PRESS = "meanpressure"
COL_RATIO = "humidity_pressure_ratio"
COL_TARGET = "meantemp"

ALL_COLS = (COL_MONTH, COL_DAY, COL_HUM, COL_WIND, COL_PRESS, COL_RATIO, COL_TARGET)
FEATURE_COLS = (COL_MONTH, COL_DAY, COL_HUM, COL_WIND, COL_PRESS, COL_RATIO)  # 6
TARGET_COL = COL_TARGET  # 1

TARGET_INDEX = ALL_COLS.index(TARGET_COL)  # 6
N_FEATURES = len(FEATURE_COLS)  # 6


@dataclass(frozen=True)
class DataConfig:
    """Configuration container for the data pipeline."""

    raw_path: Path = Path("data/raw/daily_climate_data.csv")
    data_url: str = "https://disk.360.yandex.ru/d/8K8JWqynAo7XvA"
    # DVC integration
    use_dvc: bool = True
    dvc_repo: str | None = "."  # current repo
    dvc_path: str | None = "data/raw/daily_climate_data.csv"
    dvc_rev: str | None = None
    processed_dir: Path = Path("data/processed")

    lookback: int = 7
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False

    # For time-series, do not shuffle by default.
    shuffle_train: bool = False

    save_scaler: bool = True
    scaler_name: str = "minmax_scaler.joblib"


class ClimateSeq2SeqDataset(Dataset):
    """
    features: [lookback, 6]
    targets: [lookback, 1]  (sequence of meantemp)
    """

    def __init__(self, features_seq: np.ndarray, targets_seq: np.ndarray):
        assert (
            features_seq.ndim == 3 and features_seq.shape[2] == N_FEATURES
        ), f"features must be [N,T,6], got {features_seq.shape}"
        assert (
            targets_seq.ndim == 3 and targets_seq.shape[2] == 1
        ), f"targets must be [N,T,1], got {targets_seq.shape}"
        assert (
            features_seq.shape[0] == targets_seq.shape[0]
            and features_seq.shape[1] == targets_seq.shape[1]
        ), "features and targets must align"

        self.features = torch.from_numpy(features_seq.astype(np.float32))
        self.targets = torch.from_numpy(targets_seq.astype(np.float32))

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]


def _read_raw_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if DATE_COL not in df.columns:
        raise ValueError(f"Expected '{DATE_COL}' column, got: {df.columns.tolist()}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="raise")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if df[DATE_COL].isna().any():
        raise ValueError("NaT in date column after parsing.")

    # If duplicate dates exist, keep the last value
    df = df.drop_duplicates(subset=[DATE_COL], keep="last").reset_index(drop=True)
    return df


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Extract month/day from date
    df[COL_MONTH] = df[DATE_COL].dt.month.astype(np.int64)
    df[COL_DAY] = df[DATE_COL].dt.day.astype(np.int64)

    # Humidity/pressure ratio with safe handling of inf/nan
    if COL_PRESS not in df.columns or COL_HUM not in df.columns:
        raise ValueError(
            f"Need columns '{COL_HUM}' and '{COL_PRESS}' for ratio feature."
        )

    df[COL_RATIO] = df[COL_HUM] / df[COL_PRESS].replace(0, np.nan)
    df[COL_RATIO] = df[COL_RATIO].replace([np.inf, -np.inf], np.nan)

    # Drop rows where ratio is undefined
    df = df.dropna(subset=[COL_RATIO]).reset_index(drop=True)
    return df


def _time_splits(
    n: int, train_ratio: float, val_ratio: float
) -> Tuple[slice, slice, slice]:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be in [0,1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def _ensure_enough_rows(name: str, n: int, lookback: int) -> None:
    if n < lookback:
        raise ValueError(
            f"{name} split too small for lookback={lookback}: n={n}. "
            f"Increase split size or reduce lookback."
        )


def _fit_scaler(train_values: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train_values)
    return scaler


def create_windows_seq2seq(
    series_scaled: np.ndarray, lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    series_scaled: [N, 7] aligned with ALL_COLS, already scaled by MinMaxScaler
    Returns:
      features: [M, lookback, 6]
      targets: [M, lookback, 1]
    """
    if series_scaled.ndim != 2 or series_scaled.shape[1] != len(ALL_COLS):
        raise ValueError(
            f"Expected [N,7] array aligned with ALL_COLS, got {series_scaled.shape}"
        )

    n = series_scaled.shape[0]
    m = n - lookback + 1
    if m <= 0:
        raise ValueError(f"Not enough rows: N={n}, lookback={lookback}")

    features = np.zeros((m, lookback, N_FEATURES), dtype=np.float32)
    targets = np.zeros((m, lookback, 1), dtype=np.float32)

    feat_mat = series_scaled[:, :N_FEATURES]
    targ_vec = series_scaled[:, TARGET_INDEX]

    for i in range(m):
        features[i] = feat_mat[i : i + lookback]
        targets[i, :, 0] = targ_vec[i : i + lookback]

    return features, targets


def prepare_datasets(
    cfg: DataConfig,
) -> Tuple[ClimateSeq2SeqDataset, ClimateSeq2SeqDataset, ClimateSeq2SeqDataset]:
    df = _read_raw_df(cfg.raw_path)
    df = _feature_engineering(df)

    required = (COL_HUM, COL_WIND, COL_PRESS, COL_TARGET)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available: {df.columns.tolist()}"
        )

    # IMPORTANT: enforce column order to match ALL_COLS
    df = df.loc[:, (DATE_COL, *ALL_COLS)].copy()

    series = df.loc[:, ALL_COLS].to_numpy(dtype=np.float32)  # [N,7]
    train_sl, val_sl, test_sl = _time_splits(
        len(series), cfg.train_ratio, cfg.val_ratio
    )

    _ensure_enough_rows("train", train_sl.stop - train_sl.start, cfg.lookback)
    _ensure_enough_rows("val", val_sl.stop - val_sl.start, cfg.lookback)
    _ensure_enough_rows("test", test_sl.stop - test_sl.start, cfg.lookback)

    scaler = _fit_scaler(series[train_sl])
    series_scaled = scaler.transform(series).astype(np.float32)

    if cfg.save_scaler:
        cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": scaler,
                "all_cols": ALL_COLS,
                "feature_cols": FEATURE_COLS,
                "target_col": TARGET_COL,
            },
            cfg.processed_dir / cfg.scaler_name,
        )

    features_train, targets_train = create_windows_seq2seq(
        series_scaled[train_sl], cfg.lookback
    )
    features_val, targets_val = create_windows_seq2seq(
        series_scaled[val_sl], cfg.lookback
    )
    features_test, targets_test = create_windows_seq2seq(
        series_scaled[test_sl], cfg.lookback
    )

    return (
        ClimateSeq2SeqDataset(features_train, targets_train),
        ClimateSeq2SeqDataset(features_val, targets_val),
        ClimateSeq2SeqDataset(features_test, targets_test),
    )


def create_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = prepare_datasets(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def load_scaler(
    processed_dir: Path = Path("data/processed"),
    scaler_name: str = "minmax_scaler.joblib",
) -> Dict[str, Any]:
    path = processed_dir / scaler_name
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}")
    pack = joblib.load(path)
    if not isinstance(pack, dict) or "scaler" not in pack:
        raise ValueError(
            f"Unexpected scaler pack format in {path}. Expected dict with key 'scaler'."
        )
    return pack


def inverse_transform_meantemp(
    meantemp_scaled: np.ndarray,
    reference_scaled_rows: np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """
    Convert meantemp from scaled [0,1] back to original units (°C),
    when scaler was fit on ALL_COLS (7-dim).
    """
    mt = meantemp_scaled.reshape(-1)
    ref = np.asarray(reference_scaled_rows, dtype=np.float32).copy()

    if ref.ndim != 2 or ref.shape[1] != len(ALL_COLS):
        raise ValueError(f"reference_scaled_rows must be [K,7], got {ref.shape}")
    if ref.shape[0] != mt.shape[0]:
        raise ValueError("reference rows count must match meantemp values count")

    ref[:, TARGET_INDEX] = mt
    inv = scaler.inverse_transform(ref)
    return inv[:, TARGET_INDEX]

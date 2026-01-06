from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from climate_forecasting.data import DataConfig
from climate_forecasting.data_download import ensure_raw_data_dvc_first


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
        dvc_rev=str(getattr(cfg.data, "dvc_rev", "")) or None,
    )

    ensure_raw_data_dvc_first(
        dst=data_cfg.raw_path,
        public_link=data_cfg.data_url,
        dvc_repo=data_cfg.dvc_repo,
        dvc_path=data_cfg.dvc_path,
        dvc_rev=data_cfg.dvc_rev,
    )


if __name__ == "__main__":
    main()

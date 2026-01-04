from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import mlflow.onnx
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    tracking_uri = cfg.logging.tracking_uri
    experiment_name = cfg.logging.experiment_name
    onnx_path = Path(to_absolute_path(cfg.export.onnx_path))

    model_name = str(getattr(cfg.serve, "model_name", "daily_climate"))
    register_model = bool(getattr(cfg.serve, "register", True))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="register_model") as run:
        mlflow.onnx.log_model(onnx_model=str(onnx_path), artifact_path="model")
        run_id = run.info.run_id
        print(f"Logged ONNX model to MLflow run_id={run_id}")

        if register_model:
            model_uri = f"runs:/{run_id}/model"
            result = mlflow.register_model(model_uri=model_uri, name=model_name)
            print(
                f"""Registered model name='{model_name}',
                version={result.version} from {model_uri}"""
            )
        else:
            print("Registration disabled; model available under the run artifacts.")


if __name__ == "__main__":
    main()

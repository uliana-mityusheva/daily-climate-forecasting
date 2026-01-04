from __future__ import annotations

import sys

import fire

from climate_forecasting.convert_trt import main as convert_trt_main
from climate_forecasting.export import main as export_main

# We import the Hydra entrypoints and call them directly.
# Each is decorated with @hydra.main and can be invoked with no arguments
# to read configs from the default configs/ directory.
from climate_forecasting.get_data import main as get_data_main
from climate_forecasting.predict import main as predict_main
from climate_forecasting.register_model import main as register_model_main
from climate_forecasting.train import main as train_main


def _run_hydra(entrypoint, overrides: tuple[str, ...]):
    prev_argv = sys.argv
    try:
        sys.argv = [prev_argv[0], *overrides]
        return entrypoint()
    finally:
        sys.argv = prev_argv


class Commands:
    def get_data(self, *overrides: str):
        """Download or fetch data via DVC according to configs.

        Example: dcf get_data data.raw_path=data/raw/file.csv
        """
        return _run_hydra(get_data_main, overrides)

    def train(self, *overrides: str):
        """Run model training with Hydra configs.

        Example: dcf train model.hidden_size=128 train.epochs=5
        """
        return _run_hydra(train_main, overrides)

    def predict(self, *overrides: str):
        """Run inference on the test split and produce reports/plots.

        Example: dcf predict data.lookback=10
        """
        return _run_hydra(predict_main, overrides)

    def export(self, *overrides: str):
        """Export trained model to ONNX.

        Example: dcf export export.onnx_path=artifacts/model.onnx
        """
        return _run_hydra(export_main, overrides)

    def convert_trt(self, *overrides: str):
        """Convert ONNX model to TensorRT engine using trtexec.

        Example: dcf convert_trt export.onnx_path=artifacts/model.onnx
        """
        return _run_hydra(convert_trt_main, overrides)

    def register_model(self, *overrides: str):
        """Log and (optionally) register the ONNX model in MLflow.

        Example: dcf register_model serve.register=false
        """
        return _run_hydra(register_model_main, overrides)


def main():
    fire.Fire(Commands)


if __name__ == "__main__":
    sys.exit(main())

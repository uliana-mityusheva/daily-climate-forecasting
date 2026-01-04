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


class Commands:
    def get_data(self):
        """Download or fetch data via DVC according to configs."""
        return get_data_main()

    def train(self):
        """Run model training with Hydra configs."""
        return train_main()

    def predict(self):
        """Run inference on the test split and produce reports/plots."""
        return predict_main()

    def export(self):
        """Export trained model to ONNX."""
        return export_main()

    def convert_trt(self):
        """Convert ONNX model to TensorRT engine using trtexec."""
        return convert_trt_main()

    def register_model(self):
        """Log and (optionally) register the ONNX model in MLflow."""
        return register_model_main()


def main():
    fire.Fire(Commands)


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from climate_forecasting.model import ClimateLSTM, ModelConfig


def _load_model_from_ckpt(
    ckpt_path: Path, device: torch.device
) -> tuple[ClimateLSTM, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    raw_cfg = ckpt.get("model_cfg", {})
    model_cfg = ModelConfig(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    model = ClimateLSTM(model_cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def _export_onnx(
    model: ClimateLSTM,
    example_input: torch.Tensor,
    out_path: Path,
    opset: int,
    dynamic: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = (
        {"features": {0: "batch", 1: "time"}, "y_hat": {0: "batch", 1: "time"}}
        if dynamic
        else None
    )
    torch.onnx.export(
        model,
        example_input,
        out_f=str(out_path),
        input_names=["features"],
        output_names=["y_hat"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )


def _validate_with_onnxruntime(onnx_path: Path, example_input: torch.Tensor) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception:
        print("onnxruntime not available; skipping runtime validation.")
        return

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    x = example_input.detach().cpu().numpy().astype(np.float32)
    y = sess.run([out_name], {inp_name: x})[0]
    print(f"ONNX runtime validation ok. Output shape: {y.shape}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cpu")
    ckpt_path = Path(to_absolute_path(cfg.export.model_path))
    onnx_path = Path(to_absolute_path(cfg.export.onnx_path))
    opset = int(cfg.export.opset_version)
    dynamic = bool(cfg.export.dynamic_shapes)

    model, ckpt = _load_model_from_ckpt(ckpt_path, device)

    # Determine example input shape from configs
    raw_data_cfg = ckpt.get("data_cfg", {})
    lookback = int(raw_data_cfg.get("lookback", 7))
    input_size = int(getattr(model.cfg, "input_size", 6))
    example = torch.zeros((1, lookback, input_size), dtype=torch.float32, device=device)

    _export_onnx(model, example, onnx_path, opset=opset, dynamic=dynamic)
    print(f"Exported ONNX model to: {onnx_path}")

    _validate_with_onnxruntime(onnx_path, example)


if __name__ == "__main__":
    main()

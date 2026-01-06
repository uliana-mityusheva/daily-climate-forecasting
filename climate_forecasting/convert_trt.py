from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


def _default_trt_path(onnx_path: Path) -> Path:
    if onnx_path.suffix.lower() == ".onnx":
        return onnx_path.with_suffix(".trt")
    return Path("artifacts") / "model.trt"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    onnx_path = Path(to_absolute_path(cfg.export.onnx_path))
    trt_path = _default_trt_path(onnx_path)
    trt_path.parent.mkdir(parents=True, exist_ok=True)

    trtexec = shutil.which("trtexec")
    if not trtexec:
        raise SystemExit(
            """TensorRT 'trtexec' binary not found in PATH.
            Install NVIDIA TensorRT and ensure 'trtexec' is available."""
        )

    cmd = [
        trtexec,
        f"--onnx={str(onnx_path)}",
        f"--saveEngine={str(trt_path)}",
        "--explicitBatch",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved TensorRT engine to: {trt_path}")


if __name__ == "__main__":
    main()

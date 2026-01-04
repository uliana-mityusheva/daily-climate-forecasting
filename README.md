# Daily Climate Forecasting

## Description

This project implements a complete machine learning pipeline for **daily temperature forecasting** based on historical climate data.

The main goal of the project is to demonstrate a full MLOps workflow, including:

- data loading and preprocessing,
- feature engineering,
- model training,
- checkpointing,
- inference on test data,
- logging of evaluation metrics and visualizations.

The project is based on the Kaggle dataset _Daily Climate Time Series Data_ and follows the modeling approach presented in a public Kaggle notebook using an LSTM neural network.

---

## Setup

This project uses `uv` for environment management and `DVC` for data. Follow these steps on a clean machine:

1. Install uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create venv and install dependencies

```
uv venv .venv
uv sync
```

3. Install pre-commit and run checks (should be green)

```
uv run pre-commit install
uv run pre-commit run -a
```

4. Start MLflow tracking server (local)

```
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 8080
```

5. Get data (DVC local storage or fallback)

- Preferred: use DVC to pull data into the workspace

```
mkdir -p ../dvc_store
uv run dvc pull && uv run dvc checkout
```

- If the DVC remote is empty/unavailable, train/infer will automatically download the CSV from the public Yandex Disk link on first run.

Unified CLI

- You can use a single entrypoint for all commands:

```
uv run dcf <command>
```

Where `<command>` is one of: `get_data`, `train`, `predict`, `export`, `convert_trt`, `register_model`.

## Data

The raw dataset is stored in: `data/raw/daily_climate_data.csv`

The dataset contains daily climate observations, including:

- mean temperature,
- humidity,
- wind speed,
- atmospheric pressure,
- date.

### Preprocessing

The preprocessing pipeline includes:

- parsing and sorting data chronologically,
- removing duplicate dates,
- feature engineering:
  - month and day extracted from the date,
  - humidity-to-pressure ratio,
- splitting data into **train / validation / test** sets using a **time-based split**,
- scaling numerical features using `MinMaxScaler`, fitted **only on the training split**.

Processed artifacts such as scalers are stored locally and are not committed to git.

---

## Model

The model is an **LSTM-based sequence-to-sequence neural network** implemented in PyTorch.

Model characteristics:

- Input: sliding window of past climate observations,
- Output: predicted temperature sequence,
- During inference, the **last timestep** of the predicted sequence is interpreted as a one-day-ahead temperature forecast.

The model architecture and its hyperparameters are saved together with the trained weights in a checkpoint file.

---

## Training

Model training is implemented using PyTorch.

Training details:

- Loss function: Mean Squared Error (MSE),
- Optimization is performed on the training split,
- Validation split is used for **early stopping**,
- The best model checkpoint (based on validation loss) is saved to: `artifacts/model_best.pt`

Early stopping is applied using a chronological validation set, ensuring that no future information is used during training.

---

## Train

Run end-to-end training (loads data, preprocesses, trains, logs to MLflow, saves checkpoints and plots):

```
uv run dcf train
```

Hydra overrides example (change hyperparameters from CLI):

```
uv run dcf train \
  model.hidden_size=128 train.epochs=5 data.lookback=10
```

Artifacts produced:

```
artifacts/model_best.pt
plots/train_loss.png
plots/val_loss.png
plots/val_last_mse.png
```

## Inference

Inference is implemented in a separate script and runs **only on the test split**.

During inference:

- predictions are generated for each sliding window in the test data,
- the last timestep of each predicted sequence is treated as the next-day forecast,
- predictions are inverse-transformed back to degrees Celsius.

Inference produces the following artifacts:

```text
reports/
├── metrics.json
├── test_predictions.csv
├── test_predictions.png
```

Prediction tables and plots are generated automatically and are not stored in git.

---

## Infer

Run inference on the test split and generate evaluation artifacts:

```
uv run dcf predict
```

Outputs:

```
reports/metrics.json
reports/test_predictions.csv
reports/test_predictions.png
plots/test_predictions.png
```

Input data format (CSV columns expected):

- `date` (YYYY-MM-DD)
- `meantemp` (target)
- `humidity`
- `wind_speed`
- `meanpressure`

The pipeline performs feature engineering (month, day, ratio) internally.

Example: the raw CSV is stored at `data/raw/daily_climate_data.csv` (via DVC or downloaded from the public link on first run).

---

## Production Preparation

Export trained model to ONNX:

```
uv run dcf export
```

Output:

```
artifacts/model.onnx
```

Convert ONNX to TensorRT (requires NVIDIA TensorRT installed; `trtexec` must be in PATH):

```
uv run dcf convert_trt
```

Output:

```
artifacts/model.trt
```

Delivery artifacts (for deployment):

- `artifacts/model_best.pt` (weights + configs for offline usage)
- `artifacts/model.onnx` (portable inference format)
- `artifacts/model.trt` (TensorRT engine; optional, GPU optimized)
- `data/processed/minmax_scaler.joblib` (scaler and metadata)
- Configs under `configs/` (Hydra YAMLs)

---

## Serving (MLflow Serving)

1. Start MLflow tracking server (if not running):

```
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 8080
```

2. Ensure ONNX exists, then log the model to MLflow (no registry):

```
uv run dcf export
uv run dcf register_model serve.register=false
```

3. Serve the logged model (replace <run_id> from previous step):

```
mlflow models serve -m "runs:/<run_id>/model" -p 5005 --no-conda
```

4. Invoke with HTTP (example payload; shape [1, lookback=7, features=6]):

```
curl -X POST http://127.0.0.1:5005/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"features": [[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]]}}'
```

Note: to get meaningful predictions, pass scaled features consistent with the training scaler.

## Metrics

The following evaluation metrics are computed on the test split:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Metrics are saved in machine-readable format: `reports/metrics.json`

---

## How to Run

### Train the model

To train the model and save the best checkpoint, run:

```
uv run dcf train
```

The training process includes data preprocessing, model training, validation, and checkpointing.
The best model is saved to the `artifacts/` directory.

---

### Run inference on test data

To run inference on the test split and generate evaluation artifacts, run:

```
uv run dcf predict
```

After running inference, the following files will be generated automatically:

```text
reports/
├── metrics.json
├── test_predictions.csv
├── test_predictions.png
```

The `metrics.json` file contains evaluation results.
Prediction tables and plots are generated as inference artifacts and are not stored in git.

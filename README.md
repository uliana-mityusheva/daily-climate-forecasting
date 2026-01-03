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

The project is based on the Kaggle dataset *Daily Climate Time Series Data* and follows the modeling approach presented in a public Kaggle notebook using an LSTM neural network.

---

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
uv run python -m climate_forecasting.train
```

The training process includes data preprocessing, model training, validation, and checkpointing.
The best model is saved to the `artifacts/` directory.

---

### Run inference on test data
To run inference on the test split and generate evaluation artifacts, run:

```
uv run python -m climate_forecasting.predict
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
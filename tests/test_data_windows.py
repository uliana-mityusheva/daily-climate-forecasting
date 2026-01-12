import numpy as np

from climate_forecasting.data import (
    ALL_COLS,
    N_FEATURES,
    TARGET_INDEX,
    create_windows_seq2seq,
)


def test_create_windows_seq2seq_shapes_and_alignment():
    num_rows = 10
    lookback = 5

    # Construct a scaled series [N,7] aligned with ALL_COLS
    series = np.zeros((num_rows, len(ALL_COLS)), dtype=np.float32)

    # Fill features with increasing values per feature for determinism
    for feature_index in range(N_FEATURES):
        series[:, feature_index] = np.arange(num_rows, dtype=np.float32) + feature_index

    # Target is separate pattern
    target = np.arange(num_rows, dtype=np.float32) * 10.0
    series[:, TARGET_INDEX] = target

    features, targets = create_windows_seq2seq(series, lookback)

    # number of windows
    num_windows = num_rows - lookback + 1
    assert features.shape == (num_windows, lookback, N_FEATURES)
    assert targets.shape == (num_windows, lookback, 1)

    # Check target alignment: each window's target equals slice of the target vector
    for start_index in range(num_windows):
        expected = target[start_index : start_index + lookback]
        np.testing.assert_allclose(targets[start_index, :, 0], expected)

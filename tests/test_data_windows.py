import numpy as np

from climate_forecasting.data import (
    ALL_COLS,
    N_FEATURES,
    TARGET_INDEX,
    create_windows_seq2seq,
)


def test_create_windows_seq2seq_shapes_and_alignment():
    n = 10
    lookback = 5

    # Construct a scaled series [N,7] aligned with ALL_COLS
    series = np.zeros((n, len(ALL_COLS)), dtype=np.float32)

    # Fill features with increasing values per feature for determinism
    for j in range(N_FEATURES):
        series[:, j] = np.arange(n, dtype=np.float32) + j

    # Target is separate pattern
    target = np.arange(n, dtype=np.float32) * 10.0
    series[:, TARGET_INDEX] = target

    features, targets = create_windows_seq2seq(series, lookback)

    # m = n - lookback + 1
    m = n - lookback + 1
    assert features.shape == (m, lookback, N_FEATURES)
    assert targets.shape == (m, lookback, 1)

    # Check target alignment: each window's target equals slice of the target vector
    for i in range(m):
        expected = target[i : i + lookback]
        np.testing.assert_allclose(targets[i, :, 0], expected)

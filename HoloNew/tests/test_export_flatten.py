import numpy as np
import pytest
from HoloNew.evaluation.export.flatten import to_columns


def test_orders_columns_with_time_first():
    time = np.array([0.0, 0.1, 0.2])
    channels = {"dynamics/com/x": np.array([1.0, 2.0, 3.0]),
                "diag/foot_slip": np.array([0.0, 0.5, 0.0])}
    header, table = to_columns(time, channels)
    assert header == ["time", "dynamics/com/x", "diag/foot_slip"]
    assert table.shape == (3, 3)
    np.testing.assert_allclose(table[:, 0], time)
    np.testing.assert_allclose(table[:, 1], [1.0, 2.0, 3.0])


def test_empty_channels_gives_time_only():
    time = np.array([0.0, 0.1])
    header, table = to_columns(time, {})
    assert header == ["time"]
    assert table.shape == (2, 1)


def test_rejects_wrong_length_or_2d():
    time = np.array([0.0, 0.1, 0.2])
    with pytest.raises(ValueError):
        to_columns(time, {"bad": np.array([1.0, 2.0])})
    with pytest.raises(ValueError):
        to_columns(time, {"bad2d": np.zeros((3, 2))})

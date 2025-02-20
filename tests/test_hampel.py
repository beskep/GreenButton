"""
Hampel Filter test.

from https://github.com/MichaelisTrofficus/hampel_filter/blob/master/tests/test_functionality.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import hypothesis
import hypothesis.strategies as st
import numpy as np
import polars as pl

from greenbutton.anomaly.hampel import HampelFilter

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def _hampel(value: ArrayLike, window_size: int = 4, min_samples: int | None = None):
    return (
        HampelFilter(window_size=window_size, min_samples=min_samples)(value)
        .select('filtered')
        .to_numpy()
        .ravel()
    )


def test_hampel_filter_no_outliers():
    # Test when there are no outliers, data should remain unchanged
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered = _hampel(data)
    assert np.allclose(data, filtered)


def test_hampel_filter_with_outliers():
    # Test with outliers, they should be replaced by medians within the window
    data = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0])
    filtered = _hampel(data, window_size=5)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0])
    assert np.allclose(expected, filtered)


def test_hampel_filter_large_window():
    # Test with a large window, should have no effect as the window is too large
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered = _hampel(data, window_size=10)
    assert np.allclose(data, filtered)


def test_hampel_filter_custom_threshold():
    # Test with a custom threshold, should replace the outlier
    data = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0])
    filtered = _hampel(data, window_size=5)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0])
    assert np.allclose(expected, filtered)


def test_hampel_filter_empty_data():
    # Test with an empty data array, should return an empty array
    data = np.array([])
    filtered = _hampel(data)
    assert len(filtered) == 0


def test_hampel_filter_one_point():
    # Test with one data point, should return the same point
    data = np.array([42.0])
    filtered = _hampel(data)
    assert np.allclose(data, filtered)


def test_hampel_filter_three_points_with_outliers():
    # Test with three data points and outliers, should replace the outliers
    data = np.array([1.0, 100.0, 3.0])
    filtered = _hampel(data, window_size=3, min_samples=0)
    expected = np.array([1.0, 3.0, 3.0])
    assert np.allclose(expected, filtered)


@hypothesis.given(st.lists(st.floats(), min_size=4))
def test_hampel_filter_with_dataframe(values: list[float]):
    hf = HampelFilter(window_size=4)

    v1 = hf(values)
    v2 = hf(values, value='value')
    v3 = hf(pl.DataFrame({'value': values}), value=pl.col('value'))

    assert np.allclose(
        v1['filtered'].to_numpy(),
        v2['filtered'].to_numpy(),
        equal_nan=True,
    )
    assert np.allclose(
        v1['filtered'].to_numpy(),
        v3['filtered'].to_numpy(),
        equal_nan=True,
    )

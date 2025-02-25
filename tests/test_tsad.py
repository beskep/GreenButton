from __future__ import annotations

import hypothesis
import hypothesis.strategies as st
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from greenbutton.anomaly import tsad


@hypothesis.given(st.integers(100, 1000))
@hypothesis.settings(deadline=None, max_examples=5)
def test_tsad(size: int):
    rng = np.random.default_rng(42)

    index = rng.integers(10, size - 10)
    values = rng.normal(0, 1, size=size)
    values[index] = 1000

    times = (np.datetime64('2000-01-01') + np.arange(size)).astype('datetime64[ms]')

    data = pl.DataFrame({'time': times, 'value': values})
    detector = tsad.Detector()
    detected = detector(data)

    assert detected.height == size
    assert detected['outlier'].to_numpy()[index]

    grid = detector.plot(detected)
    assert isinstance(grid, sns.FacetGrid)
    plt.close(grid.figure)

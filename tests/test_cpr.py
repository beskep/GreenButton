from __future__ import annotations

import dataclasses as dc
from typing import Literal

import hypothesis
import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from matplotlib.axes import Axes

from greenbutton import cpr


@dc.dataclass
class Dataset:
    base: float  # 기저부하
    t_h: float  # 난방 시작 온도
    t_c: float  # 냉방 시작 온도
    beta_h: float  # 난방 민감도
    beta_c: float  # 냉방 민감도

    hc: Literal['h', 'c', 'hc']
    n: int

    temperature: np.ndarray = dc.field(init=False)
    energy: np.ndarray = dc.field(init=False)

    def __post_init__(self):
        self.base = np.round(self.base, 2)

        self.temperature = np.linspace(
            2 * self.t_h, 2 * self.t_c, num=self.n, endpoint=True
        )

        # 정수 균형점 온도
        self.t_h = np.round(self.t_h) if 'h' in self.hc else -np.inf
        self.t_c = np.round(self.t_c) if 'c' in self.hc else np.inf

        zeros = np.zeros_like(self.temperature)
        self.energy = (
            self.base
            + np.maximum(zeros, self.t_h - self.temperature) * self.beta_h
            + np.maximum(zeros, self.temperature - self.t_c) * self.beta_c
        )

    def dataframe(self):
        return pl.DataFrame({'temperature': self.temperature, 'energy': self.energy})


def test_cpr():
    dataset = Dataset(base=1, t_h=-5, t_c=5, beta_h=1, beta_c=1, hc='hc', n=100)
    sr = cpr.RelativeSearchRange(1 / 4, 3 / 4, delta=1)

    estimator = cpr.CprEstimator.create(x=dataset.temperature, y=dataset.energy)
    model = estimator.fit(heating=sr, cooling=sr, method='brute', operation='best')

    assert model.is_valid

    sample = model.data.dataframe.head()
    cols = ['temperature', 'energy', 'HDD', 'CDD', 'Epb', 'Eph', 'Epc', 'Ep']
    assert model.predict(sample).columns == cols
    assert model.disaggregate(sample).columns == [*cols, 'Edb', 'Edh', 'Edc']

    assert isinstance(model.plot(), Axes)


@hypothesis.given(
    st.builds(
        Dataset,
        base=st.floats(1, 42),
        t_h=st.floats(-2, -1),
        t_c=st.floats(1, 2),
        beta_h=st.floats(1, 42),
        beta_c=st.floats(1, 42),
        hc=st.sampled_from(['h', 'c', 'hc']),
        n=st.integers(100, 1000),
    ),
    st.sampled_from(['dataframe', 'array']),
    st.sampled_from(['brute', 'numerical']),
)
@hypothesis.settings(deadline=None, max_examples=20)
def test_cpr_hypothesis(dataset: Dataset, inputs, method):
    sr = cpr.AbsoluteSearchRange(-2, 2, delta=1)
    estimator = cpr.CprEstimator()

    if inputs == 'array':
        estimator.set_data(x=dataset.temperature, y=dataset.energy)
    else:
        estimator.set_data(dataset.dataframe())

    model = estimator.fit(heating=sr, cooling=sr, method=method)
    assert model.is_valid

    if method == 'numerical':
        return

    # 기저부하
    assert model.coef['Intercept'] == pytest.approx(dataset.base)

    # 난방 민감도
    if 'h' in dataset.hc:
        assert model.change_points[0] == pytest.approx(dataset.t_h)
        assert model.coef['HDD'] == pytest.approx(dataset.beta_h)
    else:
        assert np.isnan(model.change_points[0])
        assert 'HDD' not in model.coef

    # 냉방 민감도
    if 'c' in dataset.hc:
        assert model.change_points[1] == pytest.approx(dataset.t_c)
        assert model.coef['CDD'] == pytest.approx(dataset.beta_c)
    else:
        assert np.isnan(model.change_points[1])
        assert 'CDD' not in model.coef

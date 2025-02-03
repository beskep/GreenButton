from __future__ import annotations

import dataclasses as dc
from typing import Literal

import hypothesis
import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest

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
    seed: int

    temperature: np.ndarray = dc.field(init=False)
    energy: np.ndarray = dc.field(init=False)

    def __post_init__(self):
        tr = (2 * self.t_h, 2 * self.t_c)
        self.base = np.round(self.base, 2)
        self.t_h = np.round(self.t_h) if 'h' in self.hc else -np.inf
        self.t_c = np.round(self.t_c) if 'c' in self.hc else np.inf

        rng = np.random.default_rng(self.seed)
        self.temperature = rng.uniform(*tr, size=self.n)
        zeros = np.zeros_like(self.temperature)
        noise = rng.normal(loc=0, scale=0.005, size=self.n)
        self.energy = (
            self.base
            + np.maximum(zeros, self.t_h - self.temperature) * self.beta_h
            + np.maximum(zeros, self.temperature - self.t_c) * self.beta_c
            + noise
        )

    def dataframe(self):
        return pl.DataFrame({'temperature': self.temperature, 'energy': self.energy})


@hypothesis.given(
    st.builds(
        Dataset,
        base=st.floats(1, 42),
        t_h=st.floats(-5, -1),
        t_c=st.floats(1, 5),
        beta_h=st.floats(1, 42),
        beta_c=st.floats(1, 42),
        hc=st.sampled_from(['h', 'c', 'hc']),
        n=st.integers(100, 1000),
        seed=st.integers(42),
    )
)
def test_cpr(dataset: Dataset):
    search_range = cpr.SearchRange(delta=1)
    regression = cpr.ChangePointRegression(dataset.dataframe())

    model = regression.optimize(
        heating=search_range if 'h' in dataset.hc else None,
        cooling=search_range if 'c' in dataset.hc else None,
        optimizer='brute',
    )
    coef = model.coef()

    rel = 0.02
    assert coef['Intercept'] == pytest.approx(dataset.base, rel=rel)

    if 'h' in dataset.hc:
        assert model.change_point[0] == pytest.approx(dataset.t_h, rel=rel)
        assert coef['HDD'] == pytest.approx(dataset.beta_h, rel=rel)

    if 'c' in dataset.hc:
        assert model.change_point[1] == pytest.approx(dataset.t_c, rel=rel)
        assert coef['CDD'] == pytest.approx(dataset.beta_c, rel=rel)

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
import pytest

from greenbutton import cpr


@dataclass
class Dataset:
    base: float  # 기저부하
    t_h: float  # 난방 시작 온도
    t_c: float  # 냉방 시작 온도
    beta_h: float  # 난방 민감도
    beta_c: float  # 냉방 민감도

    temperature: np.ndarray
    energy: np.ndarray

    @classmethod
    def random(cls, n: int = 1000, *, heating: bool = True, cooling: bool = True):
        if not (heating or cooling):
            raise ValueError

        rng = np.random.default_rng()

        base = rng.uniform(0, 1)
        t_h = rng.uniform(-2, -1) if heating else -np.inf
        t_c = rng.uniform(1, 2) if cooling else np.inf
        beta_h = rng.uniform(0, 1)
        beta_c = rng.uniform(0, 1)

        temperature = rng.uniform(-4, 4, size=n)
        zeros = np.zeros_like(temperature)
        noise = rng.normal(loc=0, scale=0.01, size=n)
        energy = (
            base
            + np.maximum(zeros, t_h - temperature) * beta_h
            + np.maximum(zeros, temperature - t_c) * beta_c
            + noise
        )

        return cls(
            base=base,
            t_h=t_h,
            t_c=t_c,
            beta_h=beta_h,
            beta_c=beta_c,
            temperature=temperature,
            energy=energy,
        )

    def dataframe(self):
        return pl.DataFrame({'temperature': self.temperature, 'energy': self.energy})


@pytest.mark.parametrize(
    ('heating', 'cooling'),
    [(True, True), (True, False), (False, True)],
)
@pytest.mark.parametrize('optimizer', [None, 'brute'])
def test_cpr(*, heating: bool, cooling: bool, optimizer: Literal['brute'] | None):
    dataset = Dataset.random(heating=heating, cooling=cooling)

    search_range = cpr.SearchRange(delta=0.1)
    regression = cpr.ChangePointRegression(dataset.dataframe())
    model = regression.optimize(
        heating=search_range if heating else None,
        cooling=search_range if cooling else None,
        optimizer=optimizer,
    )
    coef = model.coef()

    rel = 0.1 if optimizer is None else 0.05
    assert coef['Intercept'] == pytest.approx(dataset.base, rel=rel)

    if heating:
        assert model.change_point[0] == pytest.approx(dataset.t_h, rel=rel)
        assert coef['HDD'] == pytest.approx(dataset.beta_h, rel=rel)

    if cooling:
        assert model.change_point[1] == pytest.approx(dataset.t_c, rel=rel)
        assert coef['CDD'] == pytest.approx(dataset.beta_c, rel=rel)

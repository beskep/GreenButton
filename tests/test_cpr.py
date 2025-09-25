import dataclasses as dc
import math
from typing import Literal

import hypothesis
import hypothesis.strategies as st
import msgspec
import numpy as np
import polars as pl
import polars.testing
import pytest
from matplotlib.axes import Axes

from greenbutton import cpr
from scripts import cpr as script


def test_search_range_error():
    with pytest.raises(cpr.SearchRangeError, match='nan in'):
        cpr.AbsoluteSearchRange(math.nan, 0.8)
    with pytest.raises(cpr.SearchRangeError, match=r'self.vmin=0.8 >= self.vmax=0.2'):
        cpr.AbsoluteSearchRange(0.8, 0.2)
    with pytest.raises(cpr.SearchRangeError, match=r'self.delta=-1 <= 0'):
        cpr.AbsoluteSearchRange(0.2, 0.8, -1)

    with pytest.raises(cpr.SearchRangeError, match='nan in'):
        cpr.RelativeSearchRange(np.nan, 0.8)
    with pytest.raises(cpr.SearchRangeError, match=r'self.vmin=0.8 >= self.vmax=0.2'):
        cpr.RelativeSearchRange(0.8, 0.2)
    with pytest.raises(cpr.SearchRangeError, match=r'self.delta=-1 <= 0'):
        cpr.RelativeSearchRange(0.2, 0.8, -1)
    with pytest.raises(cpr.SearchRangeError, match=r'self.vmin=0 <= 0'):
        cpr.RelativeSearchRange(0, 0.8)
    with pytest.raises(cpr.SearchRangeError, match=r'self.vmax=1 >= 1'):
        cpr.RelativeSearchRange(0.2, 1)


@dc.dataclass
class Dataset:
    base: float  # 기저부하
    t_h: float  # 난방 시작 온도
    t_c: float  # 냉방 시작 온도
    beta_h: float  # 난방 민감도
    beta_c: float  # 냉방 민감도

    hc: Literal['h', 'c', 'hc']
    n: int
    noise: float = 0

    temperature: np.ndarray = dc.field(init=False)
    heating: np.ndarray = dc.field(init=False)
    cooling: np.ndarray = dc.field(init=False)
    energy: np.ndarray = dc.field(init=False)
    datetime: np.ndarray = dc.field(init=False)

    def __post_init__(self):
        self.base = np.round(self.base, 2)
        self.temperature = np.linspace(
            2 * self.t_h, 2 * self.t_c, num=self.n, endpoint=True
        )

        # 정수 균형점 온도
        self.t_h = np.round(self.t_h) if 'h' in self.hc else -np.inf
        self.t_c = np.round(self.t_c) if 'c' in self.hc else np.inf

        self.heating = np.maximum(0, self.t_h - self.temperature) * self.beta_h
        self.cooling = np.maximum(0, self.temperature - self.t_c) * self.beta_c
        self.energy = self.base + self.heating + self.cooling

        rng = np.random.default_rng()

        if self.noise:
            self.energy += rng.normal(0, scale=self.noise, size=self.energy.size)

        # 임의의 날짜
        dt = np.datetime64('2000-01-01') + rng.integers(0, 1000, size=self.energy.size)
        self.datetime = np.datetime_as_string(dt)

    def dataframe(self):
        return pl.DataFrame({
            'temperature': self.temperature,
            'energy': self.energy,
            'datetime': self.datetime,
        })


def test_cpr():
    dataset = Dataset(base=1, t_h=-5, t_c=5, beta_h=1, beta_c=1, hc='hc', n=100)
    sr = cpr.RelativeSearchRange(1 / 4, 3 / 4, delta=1)

    estimator = cpr.CprEstimator(x=dataset.temperature, y=dataset.energy)
    analysis = estimator.fit(heating=sr, cooling=sr, method='brute', operation='best')

    assert analysis.is_valid
    sample = estimator.data.dataframe.head()
    cols = ['temperature', 'energy', 'HDD', 'CDD', 'Epb', 'Eph', 'Epc', 'Ep']
    assert analysis.predict(sample).columns == cols
    assert analysis.disaggregate(sample).columns == [*cols, 'Edb', 'Edh', 'Edc']

    assert isinstance(analysis.plot(sample), Axes)

    model = analysis.from_dataframe(analysis.model_frame)
    assert analysis.change_points == model.change_points
    assert analysis.coef == model.coef
    assert analysis.validity == model.validity


def test_cpr_not_enough_data():
    dataset = Dataset(base=1, t_h=-5, t_c=5, beta_h=1, beta_c=1, hc='hc', n=3)

    with pytest.raises(
        cpr.NotEnoughDataError, match='At least 4 valid samples are required'
    ):
        cpr.CprEstimator(x=dataset.temperature, y=dataset.energy)


@hypothesis.given(
    st.builds(
        Dataset,
        base=st.floats(1, 42),
        t_h=st.integers(-2, -1),
        t_c=st.integers(1, 2),
        beta_h=st.floats(1, 42),
        beta_c=st.floats(1, 42),
        hc=st.sampled_from(['h', 'c', 'hc']),
        n=st.integers(100, 1000),
    ),
    st.sampled_from(['dataframe', 'array']),
    st.sampled_from(['brute', 'numerical']),
)
@hypothesis.settings(deadline=None, max_examples=20)
def test_cpr_hypothesis(data: Dataset, inputs, method):
    sr = cpr.AbsoluteSearchRange(-2, 2, delta=1)

    if inputs == 'array':
        estimator = cpr.CprEstimator(
            x=data.temperature, y=data.energy, datetime=data.datetime
        )
    else:
        estimator = cpr.CprEstimator(data.dataframe())

    model = estimator.fit(heating=sr, cooling=sr, method=method)
    assert model.is_valid

    if method == 'numerical':
        return

    energy = model.disaggregate(estimator.data.dataframe)

    # 기저부하
    assert model.coef['Intercept'] == pytest.approx(data.base)

    # 난방
    if 'h' in data.hc:
        assert model.change_points[0] == pytest.approx(data.t_h)
        assert model.coef['HDD'] == pytest.approx(data.beta_h)
        np.testing.assert_allclose(data.heating, energy['Eph'].to_numpy())
    else:
        assert np.isnan(model.change_points[0])
        assert 'HDD' not in model.coef
        assert energy.select(pl.col('Eph').eq(0).all()).item()

    # 냉방
    if 'c' in data.hc:
        assert model.change_points[1] == pytest.approx(data.t_c)
        assert model.coef['CDD'] == pytest.approx(data.beta_c)
        np.testing.assert_allclose(data.cooling, energy['Epc'].to_numpy())
    else:
        assert np.isnan(model.change_points[1])
        assert 'CDD' not in model.coef
        assert energy.select(pl.col('Epc').eq(0).all()).item()

    # 분할
    polars.testing.assert_series_equal(
        energy['energy'], pl.Series('energy', data.energy)
    )
    polars.testing.assert_series_equal(
        energy['energy'],
        energy.select(
            pl.col('Epb').fill_null(0)
            + pl.col('Eph').fill_null(0)
            + pl.col('Epc').fill_null(0)
        ).to_series(),
        check_names=False,
    )


@hypothesis.given(
    st.builds(
        Dataset,
        base=st.floats(1, 42),
        t_h=st.integers(-2, -1),
        t_c=st.integers(1, 2),
        beta_h=st.floats(1, 42),
        beta_c=st.floats(1, 42),
        hc=st.sampled_from(['h', 'c', 'hc']),
        n=st.integers(10, 200),
        noise=st.floats(0, 1),
    ),
)
@hypothesis.settings(deadline=None, max_examples=20)
def test_cpr_script(dataset: Dataset):
    obs = {
        'temperature': dataset.temperature.tolist(),
        'energy': dataset.energy.tolist(),
        'datetime': dataset.datetime.tolist(),
    }

    try:
        analyzed = script.analyze(
            msgspec.json.encode({
                'observations': obs,
                'search_range': {'delta': 1.0},
            }).decode(),
            plot='html',
            mode='return',
        )
    except cpr.OptimizationError:
        return

    assert analyzed is not None
    assert analyzed.plot is not None

    predicted = script.predict(
        model=msgspec.json.encode(analyzed.model).decode(),
        observations=msgspec.json.encode(obs).decode(),
        plot='json',
        mode='return',
    )
    assert predicted is not None
    assert predicted.plot is not None

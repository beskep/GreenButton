"""
냉난방 민감도 분석을 위한 Change Point Regression.

References
----------
[1] https://eemeter.readthedocs.io/
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypedDict, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy import optimize as opt


class OptimizationError(ValueError):
    pass


class OptimizeBoundError(ValueError):
    def __init__(
        self,
        heating,
        cooling,
        required: Literal[1, 2, 'ge1', None] = None,
    ) -> None:
        self.heating = heating
        self.cooling = cooling

        match required:
            case 1:
                msg = '냉난방 탐색범위 중 하나만 지정해야 합니다.'
            case 2:
                msg = '냉난방 탐색 범위를 모두 지정해야 합니다.'
            case 'ge1':
                msg = '냉난방 탐색범위 중 최소 하나를 지정해야 합니다.'
            case _:
                msg = ''

        bound = f'({heating=!r}, {cooling=!r})'

        super().__init__(f'{msg} {bound}' if msg else bound)


class Model(TypedDict):
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    T: np.ndarray
    pval: np.ndarray
    r2: float
    adj_r2: float
    # CI 생략
    df_model: int
    df_resid: int
    residuals: np.ndarray
    X: np.ndarray
    y: np.ndarray
    pred: np.ndarray


@dataclass
class SearchRange:
    ratio: bool = True
    vmin: float = 0.05
    vmax: float = 0.95
    delta: float = 1.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.ratio and self.vmin > 1:
            msg = f'{self.vmin=} > 1'
            raise ValueError(msg)

        if self.ratio and self.vmax > 1:
            msg = f'{self.vmax=} > 1'
            raise ValueError(msg)

        if self.vmin >= self.vmax:
            msg = f'{self.vmin=} >= {self.vmax=}'
            raise ValueError(msg)

        if self.delta <= 0:
            msg = f'{self.delta=} <= 0'
            raise ValueError(msg)

        return self

    @property
    def bounds(self):
        return (self.vmin, self.vmax)

    def update_bounds(self, tmin: float, tmax: float) -> None:
        n = -int(np.floor(np.log10(self.delta)))

        if self.ratio:
            r = tmax - tmin
            self.vmin = np.round(tmin + r * self.vmin, n)
            self.vmax = np.round(tmin + r * self.vmax, n)
        else:
            self.vmin = np.round(np.max([tmin, self.vmin]), n)
            self.vmax = np.round(np.min([tmax, self.vmax]), n)

        self.ratio = False

    def slice(self):
        return slice(self.vmin, self.vmax, self.delta)


class SearchRanges(NamedTuple):
    trange: np.ndarray
    heating: SearchRange | None
    cooling: SearchRange | None


DEFAULT_RANGE = SearchRange()


@dataclass
class Optimized:
    param: np.ndarray
    optimizer: str
    optimize_result: opt.OptimizeResult | None
    model: Model
    dataframe: pl.DataFrame


class PlotStyle(TypedDict, total=False):
    scatter: dict
    line: dict
    axvline: dict


@dataclass
class ChangePointRegression:
    data: pl.DataFrame

    x: str = 'temperature'
    y: str = 'energy'
    hdd: str = 'HDD'
    cdd: str = 'CDD'

    target: Literal['r2', 'adj_r2'] = 'adj_r2'
    x_coef: bool = False

    def x_range(self):
        return (
            self.data.select(vmin=pl.min(self.x), vmax=pl.max(self.x))
            .to_numpy()
            .ravel()
        )

    def degree_day(
        self,
        th: float = np.nan,
        tc: float = np.nan,
        data: pl.DataFrame | None = None,
    ):
        if data is None:
            data = self.data

        x = pl.col(self.x)
        return data.with_columns(
            # HDD
            pl.when(not np.isnan(th))
            .then(pl.max_horizontal(pl.lit(0), th - x))
            .otherwise(pl.lit(None))
            .alias(self.hdd),
            # CDD
            pl.when(not np.isnan(tc))
            .then(pl.max_horizontal(pl.lit(0), x - tc))
            .otherwise(pl.lit(None))
            .alias(self.cdd),
        )

    @overload
    def fit(self, th, tc, *, as_dataframe: Literal[True]) -> pd.DataFrame | None: ...

    @overload
    def fit(self, th, tc, *, as_dataframe: Literal[False] = ...) -> Model: ...

    def fit(self, th: float = np.nan, tc: float = np.nan, *, as_dataframe=False):
        if th >= tc:
            return None if as_dataframe else {'adj_r2': -np.inf, 'r2': -np.inf}

        df = self.degree_day(th=th, tc=tc)

        # x, heating, cooling 순서
        variables = tuple(
            v
            for b, v in zip(
                [self.x_coef, not np.isnan(th), not np.isnan(tc)],
                [self.x, self.hdd, self.cdd],
                strict=True,
            )
            if b
        )

        model = pg.linear_regression(
            X=df.select(variables).to_pandas(),
            y=df.select(self.y).to_series(),
            add_intercept=True,
            as_dataframe=as_dataframe,
        )

        if not as_dataframe:
            coef = dict(zip(model['names'], model['coef'], strict=True))
            if coef.get('HDD', 0) > 0 or coef.get('CDD', 0) < 0:
                # FIXME
                return {'adj_r2': -np.inf, 'r2': -np.inf} | model

        return model

    def _fit(self, th: float = np.nan, tc: float = np.nan):
        return -self.fit(th=th, tc=tc, as_dataframe=False)[self.target]

    def _fit_heating(self, th: float):
        return -self.fit(th=th, tc=np.nan, as_dataframe=False)[self.target]

    def _fit_cooling(self, tc: float):
        return -self.fit(th=np.nan, tc=tc, as_dataframe=False)[self.target]

    def _fit_array(self, array: np.ndarray):
        return -self.fit(*array, as_dataframe=False)[self.target]

    def _search_ranges(
        self,
        heating: SearchRange | None = DEFAULT_RANGE,
        cooling: SearchRange | None = DEFAULT_RANGE,
    ):
        if heating is None and cooling is None:
            raise OptimizeBoundError(heating, cooling, required='ge1')

        r = self.x_range()
        if heating is not None:
            heating.update_bounds(*r)
        if cooling is not None:
            cooling.update_bounds(*r)

        return SearchRanges(trange=r, heating=heating, cooling=cooling)

    def optimize_brute(
        self,
        heating: SearchRange | None = DEFAULT_RANGE,
        cooling: SearchRange | None = DEFAULT_RANGE,
        **kwargs,
    ) -> np.ndarray:
        match self._search_ranges(heating, cooling):
            case _, None, None:
                raise AssertionError
            case _, h, None:
                assert h is not None
                fn = self._fit_heating
                ranges = [h.slice()]
            case _, None, c:
                assert c is not None
                fn = self._fit_cooling
                ranges = [c.slice()]
            case _, h, c:
                fn = self._fit_array
                ranges = [h.slice(), c.slice()]
            case _:
                raise AssertionError

        with warnings.catch_warnings(action='ignore'):
            xmin = opt.brute(fn, ranges=ranges, **kwargs)

        assert not isinstance(xmin, tuple)
        return xmin

    def optimize_scalar(
        self,
        heating: SearchRange | None = DEFAULT_RANGE,
        cooling: SearchRange | None = DEFAULT_RANGE,
        **kwargs,
    ) -> opt.OptimizeResult:
        match self._search_ranges(heating, cooling):
            case _, None, None:
                raise AssertionError
            case _, h, None:
                assert h is not None
                return opt.minimize_scalar(self._fit_heating, bounds=h.bounds, **kwargs)
            case _, None, c:
                assert c is not None
                return opt.minimize_scalar(self._fit_cooling, bounds=c.bounds, **kwargs)
            case _:
                raise OptimizeBoundError(heating, cooling, required=1)

    def optimize_multivariable(
        self,
        heating: SearchRange = DEFAULT_RANGE,
        cooling: SearchRange = DEFAULT_RANGE,
        x0: np.ndarray | None = None,
        **kwargs,
    ) -> opt.OptimizeResult:
        r = self._search_ranges(heating, cooling)
        assert r.heating is not None
        assert r.cooling is not None

        if x0 is None:
            x0 = r.trange[0] + (r.trange[1] - r.trange[0]) * np.array([0.2, 0.8])

        return opt.minimize(
            self._fit_array,
            x0=x0,
            bounds=[r.heating.bounds, r.cooling.bounds],
            **kwargs,
        )

    def optimize(
        self,
        heating: SearchRange | None = DEFAULT_RANGE,
        cooling: SearchRange | None = DEFAULT_RANGE,
        optimizer: Literal['multivariable', 'scalar', 'brute', None] = None,
        **kwargs,
    ):
        if heating is None and cooling is None:
            raise OptimizeBoundError(heating, cooling, required='ge1')

        if optimizer is None:
            optimizer = (
                'multivariable'
                if (heating is not None and cooling is not None)
                else 'scalar'
            )

        res: opt.OptimizeResult | None

        match optimizer:
            case 'multivariable':
                if heating is None or cooling is None:
                    raise OptimizeBoundError(heating, cooling, required=2)

                res = self.optimize_multivariable(heating, cooling, **kwargs)
                p = res['x']
            case 'scalar':
                res = self.optimize_scalar(heating, cooling, **kwargs)
                p = [res['x']]
            case 'brute':
                p = self.optimize_brute(heating, cooling, **kwargs)
                res = None
            case e:
                msg = f'잘못된 최적화 방법: {e}'
                raise ValueError(msg)

        if np.size(p) == 1:
            param = np.array([np.nan, p[0]] if heating is None else [p[0], np.nan])
        else:
            param = np.array(p)

        if (df := self.fit(*param, as_dataframe=True)) is None:
            raise OptimizationError

        return Optimized(
            param=param,
            optimizer=optimizer,
            optimize_result=res,
            model=self.fit(*param, as_dataframe=False),
            dataframe=pl.from_pandas(df),
        )

    def segments(
        self,
        data: Sequence[float] | NDArray | Optimized,
        /,
        xmin: float | None = None,
        xmax: float | None = None,
    ):
        if isinstance(data, Optimized):
            param = tuple(data.param)
            model = data.model
        else:
            param = tuple(data)  # th, tc
            model = self.fit(*data, as_dataframe=False)

        coef = dict(zip(model['names'], model['coef'], strict=True))

        r = self.x_range()
        x = pl.DataFrame({
            self.x: [float(xmin or r[0]), *sorted(param), float(xmax or r[1])]
        }).with_row_index()

        pred = (
            self.degree_day(param[0], param[1], data=x)
            .with_columns(pl.lit(1.0).alias('Intercept'))
            .unpivot(index='index')
            .with_columns(
                coef=pl.col('variable').replace_strict(
                    coef, default=0.0, return_dtype=pl.Float64
                )
            )
            .with_columns((pl.col('value') * pl.col('coef')).alias(self.y))
            .group_by('index')
            .agg(pl.sum(self.y))
        )

        return x.join(pred, on='index', how='inner').sort('index')

    def plot(
        self,
        data: Sequence[float] | NDArray | Optimized | pl.DataFrame,
        /,
        *,
        ax: Axes | None = None,
        axvline: bool = True,
        style: PlotStyle | None = None,
    ):
        segments = data if isinstance(data, pl.DataFrame) else self.segments(data)

        if ax is None:
            ax = plt.gca()

        style = style or {}
        kws: dict[str, Any] = (
            {'c': 'gray', 'alpha': 0.5, 'zorder': 2.1}  ##
            | style.get('scatter', {})
        )
        kwl: dict[str, Any] = {'zorder': 2.2} | style.get('line', {})
        kwa: dict[str, Any] = {'ls': '--', 'c': 'gray'} | style.get('axvline', {})

        sns.scatterplot(self.data, x=self.x, y=self.y, ax=ax, **kws)
        sns.lineplot(segments, x=self.x, y=self.y, ax=ax, **kwl)

        if axvline:
            points = (
                segments.filter(pl.col('index').is_in([0, segments.height - 1]).not_())
                .select(self.x)
                .to_numpy()
                .ravel()
            )
            for x in points:
                ax.axvline(x, **kwa)

        return ax


if __name__ == '__main__':
    from itertools import product

    from loguru import logger

    from greenbutton.utils import cnsl, set_logger

    set_logger()
    pl.Config.set_tbl_cols(10)
    pl.Config.set_tbl_rows(50)

    class TestDataset:
        X = tuple(range(8))
        Y = (2, 1, 0, 0, 0, 0, 2, 4)

        def __init__(
            self,
            th: float = 2,
            tc: float = 5,
            seed: int | None = None,
            scale=1,
        ) -> None:
            self.th = th
            self.tc = tc
            self.seed = seed
            self.scale = scale

        def df(self, *, dd=False):
            y = np.array(self.Y).astype(float)
            if self.seed is not None:
                rng = np.random.default_rng(self.seed)
                y += rng.normal(0, scale=self.scale, size=len(y))

            df = pl.DataFrame({'x': self.X, 'y': y})

            if dd and not np.isnan(self.th):
                df = df.with_columns(
                    hdd=pl.min_horizontal(pl.lit(0), pl.col('x') - self.th)
                )
            if dd and not np.isnan(self.tc):
                df = df.with_columns(
                    cdd=pl.max_horizontal(pl.lit(0), pl.col('x') - self.tc)
                )

            return df

    data = TestDataset()
    df = data.df()
    cnsl.print(df)

    cpr = ChangePointRegression(data=df, x='x', y='y', target='adj_r2', x_coef=False)
    cnsl.print(cpr.degree_day(th=2, tc=5))

    for h, c, m in product(
        [DEFAULT_RANGE, None],
        [DEFAULT_RANGE, None],
        [None, 'brute'],
    ):
        logger.info('')
        logger.info('h={!r} c={!r} m={}', h, c, m)

        try:
            optim = cpr.optimize(h, c, m)  # type: ignore[arg-type]
        except ValueError as _e:
            logger.error(_e)
        else:
            cnsl.print(optim)

    optim = cpr.optimize()
    cpr.plot(optim)
    plt.show()

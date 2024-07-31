"""
냉난방 민감도 분석을 위한 Change Point Regression.

References
----------
[1] https://eemeter.readthedocs.io/
"""
# pylint: disable=unsubscriptable-object

import warnings
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import BaseModel, NonNegativeFloat, PositiveFloat, model_validator
from scipy import optimize as opt
from typing_extensions import TypedDict


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


class Variable(TypedDict, total=False):
    x: str
    y: str
    hdd: str
    cdd: str


class VariableModel(BaseModel):
    x: str = 'temperature'
    y: str = 'energy'
    hdd: str = 'HDD'
    cdd: str = 'CDD'

    def independent(
        self,
        *,
        heating: bool = True,
        cooling: bool = True,
        x: bool = False,
    ):
        return tuple(
            v
            for b, v in zip(
                [x, heating, cooling],
                [self.x, self.hdd, self.cdd],
                strict=True,
            )
            if b
        )


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


class SearchRange(BaseModel):
    ratio: bool = True
    vmin: NonNegativeFloat = 0.05
    vmax: NonNegativeFloat = 0.95
    delta: PositiveFloat = 1.0

    @model_validator(mode='after')
    def _validate_range(self):
        vmin = self.vmin
        vmax = self.vmax

        if self.ratio and vmin > 1:
            msg = f'{vmin=} > 1'
            raise ValueError(msg)

        if self.ratio and vmax > 1:
            msg = f'{vmax=} > 1'
            raise ValueError(msg)

        if vmin >= vmax:
            msg = f'{vmin=} >= {vmax=}'
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


class Optimized(BaseModel):
    model_config = {'arbitrary_types_allowed': True}

    param: np.ndarray
    optimizer: str
    optimize_result: opt.OptimizeResult | None
    model: Model
    dataframe: pl.DataFrame


class PlotStyle(TypedDict, total=False):
    scatter: dict
    line: dict
    axvline: dict


class ChangePointRegression:
    # XXX np.nan 대신 math.nan?
    # TODO MSE 최소화 기준

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        variable: Variable | VariableModel | None = None,
        target: Literal['r2', 'adj_r2'] = 'adj_r2',
        x_coef: bool = False,
    ) -> None:
        self._data = data
        self._v = (
            VariableModel.model_validate(variable) if variable else VariableModel()
        )
        self._t = target
        self._xc = x_coef

    def x_range(self):
        return (
            self._data.select(vmin=pl.min(self._v.x), vmax=pl.max(self._v.x))
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
            data = self._data

        x = pl.col(self._v.x)
        return data.with_columns(
            # HDD
            pl.when(not np.isnan(th))
            .then(pl.max_horizontal(pl.lit(0), th - x))
            .otherwise(pl.lit(None))
            .alias(self._v.hdd),
            # CDD
            pl.when(not np.isnan(tc))
            .then(pl.max_horizontal(pl.lit(0), x - tc))
            .otherwise(pl.lit(None))
            .alias(self._v.cdd),
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
        variables = self._v.independent(
            heating=not np.isnan(th), cooling=not np.isnan(tc), x=self._xc
        )

        model = pg.linear_regression(
            X=df.select(variables).to_pandas(),
            y=df.select(self._v.y).to_series(),
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
        return -self.fit(th=th, tc=tc, as_dataframe=False)[self._t]

    def _fit_heating(self, th: float):
        return -self.fit(th=th, tc=np.nan, as_dataframe=False)[self._t]

    def _fit_cooling(self, tc: float):
        return -self.fit(th=np.nan, tc=tc, as_dataframe=False)[self._t]

    def _fit_array(self, array: np.ndarray):
        return -self.fit(*array, as_dataframe=False)[self._t]

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

        r = self.x_range()
        if xmin is None:
            xmin = r[0]
        if xmax is None:
            xmax = r[1]

        coef = dict(zip(model['names'], model['coef'], strict=True))
        x = pl.DataFrame({self._v.x: [xmin, *sorted(param), xmax]}).with_row_index()
        pred = (
            self.degree_day(param[0], param[1], data=x)
            .with_columns(pl.lit(1).alias('Intercept'))
            .melt(id_vars='index')
            .with_columns(
                coef=pl.col('variable').replace(coef, return_dtype=pl.Float64)
            )
            .with_columns((pl.col('value') * pl.col('coef')).alias(self._v.y))
            .group_by('index')
            .agg(pl.sum(self._v.y))
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
        kws: dict[str, Any] = {'c': 'gray', 'alpha': 0.5, 'zorder': 2.1} | style.get(
            'scatter', {}
        )
        kwl: dict[str, Any] = {'zorder': 2.2} | style.get('line', {})
        kwa: dict[str, Any] = {'ls': '--', 'c': 'gray'} | style.get('axvline', {})

        sns.scatterplot(self._data, x=self._v.x, y=self._v.y, ax=ax, **kws)
        sns.lineplot(segments, x=self._v.x, y=self._v.y, ax=ax, **kwl)

        if axvline:
            points = (
                segments.filter(pl.col('index').is_in([0, segments.height - 1]).not_())
                .select(self._v.x)
                .to_numpy()
                .ravel()
            )
            for x in points:
                ax.axvline(x, **kwa)

        return ax


class _TestDataset:
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


if __name__ == '__main__':
    from itertools import product

    from loguru import logger

    from greenbutton.utils import cnsl, set_logger

    set_logger()
    pl.Config.set_tbl_cols(10)
    pl.Config.set_tbl_rows(50)

    _data = _TestDataset()
    _df = _data.df()
    cnsl.print(_df)

    cpr = ChangePointRegression(
        data=_df,
        variable=VariableModel(x='x', y='y'),
        target='adj_r2',
        x_coef=False,
    )

    cnsl.print(cpr.degree_day(th=2, tc=5))

    for _h, _c, m in product(
        [DEFAULT_RANGE, None],
        [DEFAULT_RANGE, None],
        [None, 'brute'],
    ):
        logger.info('{!r} {!r} {}', _h, _c, m)

        try:
            optim = cpr.optimize(_h, _c, m)  # type: ignore[arg-type]
        except ValueError as _e:
            logger.error(_e)
        else:
            cnsl.print(optim)

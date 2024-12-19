"""
냉난방 민감도 분석을 위한 Change Point Regression.

References
----------
[1] https://eemeter.readthedocs.io/
"""

from __future__ import annotations

import dataclasses as dc
import warnings
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, TypedDict, overload

import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import polars as pl
import seaborn as sns
from scipy import optimize as opt

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd
    from matplotlib.axes import Axes

Optimizer = Literal['multivariable', 'scalar', 'brute'] | None


class CprError(ValueError):
    pass


class SearchRangeError(CprError):
    pass


class NotEnoughDataError(CprError):
    def __init__(self, required: int, given: int) -> None:
        self.required = required
        self.given = given

        super().__init__(
            f'At least {required!r} valid samples are required. {given!r} given.'
        )


class OptimizationError(CprError):
    pass


class OptimizeBoundError(CprError):
    MSG: ClassVar[dict] = {
        1: '냉난방 탐색범위 중 하나만 지정해야 합니다.',
        2: '냉난방 탐색 범위를 모두 지정해야 합니다.',
        'ge1': '냉난방 탐색범위 중 최소 하나를 지정해야 합니다.',
    }

    def __init__(
        self,
        heating: SearchRange | None,
        cooling: SearchRange | None,
        required: Literal[1, 2, 'ge1'] | None = None,
    ) -> None:
        self.heating = heating
        self.cooling = cooling

        bound = f'({heating=!r}, {cooling=!r})'
        if msg := self.MSG.get(required):
            bound = f'{msg} {bound}'

        super().__init__(bound)


class LinearModel(TypedDict):
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


@dc.dataclass
class SearchRange:
    ratio: bool = True
    vmin: float = 0.05
    vmax: float = 0.95
    delta: float = 1.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        if any(np.isnan(x) for x in [self.vmin, self.vmax, self.delta]):
            msg = f'nan in {self}'
            raise SearchRangeError(msg)

        if self.ratio and self.vmin > 1:
            msg = f'{self.vmin=} > 1'
            raise SearchRangeError(msg)

        if self.ratio and self.vmax > 1:
            msg = f'{self.vmax=} > 1'
            raise SearchRangeError(msg)

        if self.vmin >= self.vmax:
            msg = f'{self.vmin=} >= {self.vmax=}'
            raise SearchRangeError(msg)

        if self.delta <= 0:
            msg = f'{self.delta=} <= 0'
            raise SearchRangeError(msg)

        return self

    @property
    def bounds(self):
        return (self.vmin, self.vmax)

    def update_bounds(self, tmin: float, tmax: float):
        n = -int(np.floor(np.log10(self.delta)))

        if self.ratio:
            r = tmax - tmin
            self.vmin = np.round(tmin + r * self.vmin, n)
            self.vmax = np.round(tmin + r * self.vmax, n)
        else:
            self.vmin = np.round(np.min([tmin, self.vmin]), n)
            self.vmax = np.round(np.max([tmax, self.vmax]), n)

        self.ratio = False
        self.validate()
        return self

    def slice(self):
        return slice(self.vmin, self.vmax, self.delta)


class _SearchRanges(NamedTuple):
    trange: np.ndarray
    heating: SearchRange | None
    cooling: SearchRange | None


DEFAULT_RANGE = SearchRange()


class PlotStyle(TypedDict, total=False):
    line: dict
    scatter: dict
    axvline: dict | None

    shuffle: bool
    xmin: float | None
    xmax: float | None


@dc.dataclass(frozen=True)
class ChangePointModel:
    change_point: np.ndarray

    optimizer: Optimizer
    optimize_result: opt.OptimizeResult | None

    model_dict: LinearModel
    model_frame: pl.DataFrame

    cpr: ChangePointRegression

    DEFAULT_STYLE: ClassVar[PlotStyle] = {
        'scatter': {'zorder': 2.1, 'color': 'gray', 'alpha': 0.75},
        'line': {'zorder': 2.2, 'color': 'gray', 'alpha': 0.75},
        'axvline': {'ls': '--', 'color': 'gray', 'alpha': 0.5},
        'shuffle': True,
    }

    @property
    def is_valid(self):
        return self.model_dict['r2'] != 0

    def coef(self) -> dict[str, float]:
        return dict(zip(self.model_dict['names'], self.model_dict['coef'], strict=True))

    def predict(
        self, data: pl.DataFrame | Sequence | np.ndarray | None = None
    ) -> pl.DataFrame:
        """
        기온으로부터 기저, 냉방, 난방 에너지 사용량 예측.

        Parameters
        ----------
        data : pl.DataFrame | Sequence | np.ndarray | None, optional
            입력 데이터. `pl.DataFrame`인 경우, CPR 분석 시 설정한
            `temperature` 열이 있어야 함.

        Returns
        -------
        pl.DataFrame
            열 목록:
            - `temperature`: 입력 변수(기온)
            - 'HDD': 난방도일
            - 'CDD': 냉방도일
            - 'Epb': 예상 기저 사용량
            - 'Eph': 예상 난방 사용량
            - 'Epc': 예상 냉방 사용량
            - 'Ep': 예상 총 사용량
        """
        if data is None:
            df = self.cpr.data
        elif isinstance(data, pl.DataFrame):
            df = data
        else:
            df = pl.DataFrame({self.cpr.temperature: data})

        coef = dict.fromkeys(['Intercept', 'HDD', 'CDD'], 1.0) | self.coef()
        return (
            self.cpr.degree_day(
                th=self.change_point[0], tc=self.change_point[1], data=df
            )
            .with_columns(
                pl.lit(coef['Intercept']).alias('Epb'),
                pl.col('HDD').mul(coef['HDD']).alias('Eph'),
                pl.col('CDD').mul(coef['CDD']).alias('Epc'),
            )
            .with_columns(
                (
                    pl.col('Epb').fill_null(0)
                    + pl.col('Eph').fill_null(0)
                    + pl.col('Epc').fill_null(0)
                ).alias('Ep')
            )
        )

    def disaggregate(self, data: pl.DataFrame | None = None) -> pl.DataFrame:
        """
        기온과 총 에너지 사용량으로부터 기저, 난방, 냉방 사용량 분리.

        Parameters
        ----------
        data : pl.DataFrame | None, optional
            입력 데이터. CPR 분석 시 설정한 `temperature`와 `energy` 열이 있어야 함.

        Returns
        -------
        pl.DataFrame
            `predict()`의 결과에 다음 열 추가:
            - 'Edb': 분리된 기저 사용량.
            - 'Edh': 분리된 난방 사용량.
            - 'Edc': 분리된 냉방 사용량.
        """
        ratio = pl.col(self.cpr.energy) / pl.col('Ep')
        return self.predict(data).with_columns(
            pl.col('Epb').fill_null(0).mul(ratio).alias('Edb'),
            pl.col('Eph').fill_null(0).mul(ratio).alias('Edh'),
            pl.col('Epc').fill_null(0).mul(ratio).alias('Edc'),
        )

    def _segments(self, xmin: float | None = None, xmax: float | None = None):
        r = self.cpr.x_range()
        points = [
            r[0] if xmin is None else xmin,
            *sorted(self.change_point),
            r[1] if xmax is None else xmax,
        ]
        data = pl.DataFrame(
            pl.Series(name=self.cpr.temperature, values=points, dtype=pl.Float64)
        )
        return self.predict(data)

    def plot(
        self,
        *,
        ax: Axes | None = None,
        segments: bool = True,
        scatter: bool | pl.DataFrame = True,
        style: PlotStyle | None = None,
    ):
        if ax is None:
            ax = plt.gca()

        style = self.DEFAULT_STYLE | (style or {})

        if scatter is not False:
            data = self.cpr.data if scatter is True else scatter
            if style.get('shuffle', True):
                data = data.sample(fraction=1, shuffle=True)

            sns.scatterplot(
                data.to_pandas(),
                x=self.cpr.temperature,
                y=self.cpr.energy,
                ax=ax,
                **style.get('scatter', {}),
            )

        if segments:
            sns.lineplot(
                self._segments(xmin=style.get('xmin'), xmax=style.get('xmax')),
                x=self.cpr.temperature,
                y='Ep',
                ax=ax,
                **style.get('line', {}),
            )

            if s := style.get('axvline'):
                for x in self.change_point:
                    ax.axvline(x, **s)

        return ax


@dc.dataclass
class ChangePointRegression:
    data: pl.DataFrame
    temperature: str = 'T'
    energy: str = 'E'

    target: Literal['r2', 'adj_r2'] = 'r2'
    x_coef: bool = False
    min_samples: int = 4

    allow_single_heating_cooling: bool = False
    _cp: tuple[float, float] = dc.field(init=False)  # 허용 가능한 균형점 온도 범위

    def __post_init__(self):
        if self.data.height < self.min_samples:
            raise NotEnoughDataError(required=self.min_samples, given=self.data.height)

        if self.allow_single_heating_cooling:
            self._cp = (-np.inf, np.inf)
        else:
            t = (
                self.data.select(pl.col(self.temperature).unique().sort())
                .to_numpy()
                .ravel()
            )
            self._cp = (float(t[1]), float(t[-2]))
            assert self._cp[0] < self._cp[1]

    def x_range(self) -> tuple[float, float]:
        return (
            self.data.select(pl.min(self.temperature)).item(),
            self.data.select(pl.max(self.temperature)).item(),
        )

    def degree_day(
        self,
        th: float = np.nan,
        tc: float = np.nan,
        data: pl.DataFrame | None = None,
    ):
        if data is None:
            data = self.data

        t = pl.col(self.temperature)
        return data.with_columns(
            # HDD
            pl.when(not np.isnan(th))
            .then(pl.max_horizontal(pl.lit(0), th - t))
            .otherwise(pl.lit(None))
            .alias('HDD'),
            # CDD
            pl.when(not np.isnan(tc))
            .then(pl.max_horizontal(pl.lit(0), t - tc))
            .otherwise(pl.lit(None))
            .alias('CDD'),
        )

    def _is_valid_change_point(self, th: float, tc: float) -> bool:
        # th, tc가 nan인 경우를 고려해 유효하지 않은 케이스 판단
        # e.g. th = nan, tc = 42일 때, th >= tc == False

        if np.isnan(th) and np.isnan(tc):
            return False

        if th >= tc:
            return False

        return (
            self.allow_single_heating_cooling  ##
            or not (th < self._cp[0] or tc > self._cp[1])
        )

    @overload
    def fit(self, th, tc, *, as_dataframe: Literal[True]) -> pd.DataFrame | None: ...

    @overload
    def fit(self, th, tc, *, as_dataframe: Literal[False] = ...) -> LinearModel: ...

    def fit(self, th: float = np.nan, tc: float = np.nan, *, as_dataframe=False):
        if self.data.height < self.min_samples:
            raise NotEnoughDataError(required=self.min_samples, given=self.data.height)

        if not self._is_valid_change_point(th=th, tc=tc):
            return None if as_dataframe else {'adj_r2': 0, 'r2': 0}

        df = self.degree_day(th=th, tc=tc)

        # 기울기를 계산할 독립변수 목록
        # x, heating, cooling 순서
        is_indep = [self.x_coef, not np.isnan(th), not np.isnan(tc)]
        names = [self.temperature, 'HDD', 'CDD']
        indep_vars = tuple(v for i, v in zip(is_indep, names, strict=True) if i)

        model = pg.linear_regression(
            X=df.select(indep_vars).to_pandas(),
            y=df.select(self.energy).to_series(),
            add_intercept=True,
            as_dataframe=as_dataframe,
        )

        if not as_dataframe:
            coef = dict(zip(model['names'], model['coef'], strict=True))
            if coef.get('HDD', 0) < 0 or coef.get('CDD', 0) < 0:
                # 유효하지 않은 모델 기각
                return model | {'adj_r2': 0, 'r2': 0}

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
    ) -> _SearchRanges:
        if heating is None and cooling is None:
            raise OptimizeBoundError(heating, cooling, required='ge1')

        r = self.x_range()
        if heating is not None:
            heating.update_bounds(*r)
        if cooling is not None:
            cooling.update_bounds(*r)

        return _SearchRanges(trange=np.array(r), heating=heating, cooling=cooling)

    def _optimize_brute(
        self,
        heating: SearchRange | None = DEFAULT_RANGE,
        cooling: SearchRange | None = DEFAULT_RANGE,
        **kwargs,
    ) -> np.ndarray:
        fn: Callable[..., float]
        match self._search_ranges(heating, cooling):
            case _, None, None:
                raise AssertionError
            case (_, h, None) if h is not None:
                fn = self._fit_heating
                ranges = [h.slice()]
            case (_, None, c) if c is not None:
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

    def _optimize_scalar(
        self,
        heating: SearchRange | None = DEFAULT_RANGE,
        cooling: SearchRange | None = DEFAULT_RANGE,
        **kwargs,
    ) -> opt.OptimizeResult:
        match self._search_ranges(heating, cooling):
            case _, None, None:
                raise AssertionError
            case (_, h, None) if h is not None:
                r = opt.minimize_scalar(self._fit_heating, bounds=h.bounds, **kwargs)
            case (_, None, c) if c is not None:
                r = opt.minimize_scalar(self._fit_cooling, bounds=c.bounds, **kwargs)
            case _:
                raise OptimizeBoundError(heating, cooling, required=1)

        assert isinstance(r, opt.OptimizeResult)
        return r

    def _optimize_multivariable(
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

    def _optimize(
        self,
        heating: SearchRange | None,
        cooling: SearchRange | None,
        optimizer: Optimizer,
        **kwargs,
    ) -> tuple[np.ndarray, opt.OptimizeResult | None]:
        match optimizer:
            case 'multivariable':
                if heating is None or cooling is None:
                    raise OptimizeBoundError(heating, cooling, required=2)

                res = self._optimize_multivariable(heating, cooling, **kwargs)
                param = res['x']
            case 'scalar':
                res = self._optimize_scalar(heating, cooling, **kwargs)
                param = np.array([res['x']])
            case 'brute':
                param = self._optimize_brute(heating, cooling, **kwargs)
                res = None
            case e:
                msg = f'잘못된 최적화 방법: {e}'
                raise ValueError(msg)

        return param, res

    def optimize(
        self,
        heating: SearchRange | None = DEFAULT_RANGE,
        cooling: SearchRange | None = DEFAULT_RANGE,
        optimizer: Optimizer = 'brute',
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

        param, res = self._optimize(
            heating=heating, cooling=cooling, optimizer=optimizer, **kwargs
        )

        # change point
        if np.size(param) == 1:
            cp = np.array([np.nan, param[0]] if heating is None else [param[0], np.nan])
        else:
            cp = np.array(param)

        # dataframe
        if (pddf := self.fit(*cp, as_dataframe=True)) is None:
            msg = 'No valid model'
            raise OptimizationError(msg)

        cpdf = pl.DataFrame({'names': ['HDD', 'CDD'], 'change_point': cp})
        model_frame = (
            pl.from_pandas(pddf)
            .join(cpdf, on='names', how='left')
            .select('names', 'change_point', pl.all().exclude('names', 'change_point'))
        )

        return ChangePointModel(
            change_point=cp,
            optimizer=optimizer,
            optimize_result=res,
            model_dict=self.fit(*cp, as_dataframe=False),
            model_frame=model_frame,
            cpr=self,
        )

    def optimize_multi_models(
        self,
        heating: SearchRange = DEFAULT_RANGE,
        cooling: SearchRange = DEFAULT_RANGE,
        optimizer: Optimizer = 'brute',
        **kwargs,
    ):
        models: list[ChangePointModel] = []

        for h, c in [(heating, cooling), (heating, None), (None, cooling)]:
            try:
                models.append(
                    self.optimize(heating=h, cooling=c, optimizer=optimizer, **kwargs)
                )
            except OptimizationError:
                pass

        if not models:
            raise OptimizationError

        return max(models, key=lambda x: x.model_dict[self.target])


if __name__ == '__main__':
    from itertools import product

    import rich
    from loguru import logger

    from greenbutton.utils import LogHandler

    console = rich.get_console()

    LogHandler.set()
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(50)

    @dc.dataclass
    class _TestDataset:
        temperature: Sequence[float] = tuple(range(8))
        energy: Sequence[float] = (2, 1, 0, 0, 0, 0, 2, 4)
        th: float = 2
        tc: float = 5
        seed: int | None = None
        scale: float = 1

        def df(self, *, dd=False):
            y = np.array(self.energy).astype(float)
            if self.seed is not None:
                rng = np.random.default_rng(self.seed)
                y += rng.normal(0, scale=self.scale, size=len(y))

            df = pl.DataFrame({'T': self.temperature, 'E': y})

            if dd and not np.isnan(self.th):
                df = df.with_columns(
                    hdd=pl.min_horizontal(pl.lit(0), pl.col('T') - self.th)
                )
            if dd and not np.isnan(self.tc):
                df = df.with_columns(
                    cdd=pl.max_horizontal(pl.lit(0), pl.col('E') - self.tc)
                )

            return df

    data = _TestDataset().df()
    console.print(data)

    cpr = ChangePointRegression(data=data)
    console.print(cpr.degree_day(th=2, tc=5))

    for h, c, m in product(
        [DEFAULT_RANGE, None],
        [DEFAULT_RANGE, None],
        [None, 'brute'],
    ):
        if h is None and c is None:
            continue

        logger.info('')
        logger.info('h={!r} c={!r} m={}', h, c, m)

        model = cpr.optimize(h, c, m)  # type: ignore[arg-type]
        console.print(model.disaggregate())

    model = cpr.optimize()
    model.plot()
    plt.show()

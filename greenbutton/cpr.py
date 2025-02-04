"""
냉난방 민감도 분석을 위한 Change Point Regression.

References
----------
[1] https://eemeter.readthedocs.io/
"""

from __future__ import annotations

import abc
import dataclasses as dc
import datetime as dt
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, overload

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import polars as pl
import seaborn as sns
from matplotlib import cm
from numpy.typing import ArrayLike, NDArray
from scipy import optimize as opt

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from matplotlib.axes import Axes

Optimizer = Literal['multivariable', 'scalar', 'brute']
_FloatArray = Sequence[float] | NDArray[np.float64]
_DateArray = Sequence[dt.datetime | str] | NDArray[np.datetime64]
_X = Literal['hc', 'h', 'c']


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


class LinearModelDict(TypedDict, total=False):
    """pingouin으로 분석한 선형회귀분석 결과."""

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
class SearchRange(abc.ABC):
    vmin: float  # 탐색 최소 온도
    vmax: float  # 탐색 최대 온도
    delta: float  # 탐색 간격 [°C] (brute force 탐색 시 이용)

    def __post_init__(self):
        self.validate()

    def validate(self):
        if any(np.isnan(x) for x in [self.vmin, self.vmax, self.delta]):
            msg = f'nan in {self}'
            raise SearchRangeError(msg)

        if self.vmin >= self.vmax:
            msg = f'{self.vmin=} >= {self.vmax=}'
            raise SearchRangeError(msg)

        if self.delta <= 0:
            msg = f'{self.delta=} <= 0'
            raise SearchRangeError(msg)

        return self

    def slice(self):
        return slice(self.vmin, self.vmax, self.delta)

    @abc.abstractmethod
    def update(self, vmin: float, vmax: float) -> AbsoluteSearchRange:
        """
        분석에 이용할 외기온 범위에 맞춰 탐색 범위 업데이트.

        Parameters
        ----------
        vmin : float
            외기온도 최소값 [°C].
        vmax : float
            외기온도 최대값 [°C].

        Returns
        -------
        AbsoluteSearchRange
        """


@dc.dataclass
class AbsoluteSearchRange(SearchRange):
    """CPR 모델 균형점 온도 탐색 범위 (vmin, vmax가 °C 단위)."""

    def update(self, vmin: float, vmax: float) -> AbsoluteSearchRange:
        """
        분석에 이용할 외기온 범위에 맞춰 탐색 범위 업데이트.

        Parameters
        ----------
        vmin : float
            외기온도 최소값 [°C].
        vmax : float
            외기온도 최대값 [°C].

        Returns
        -------
        AbsoluteSearchRange

        Examples
        --------
        >>> AbsoluteSearchRange(vmin=-1, vmax=1, delta=0.1).update(vmin=-42, vmax=0)
        AbsoluteSearchRange(vmin=-42, vmax=1, delta=0.1)
        """
        n = -int(np.floor(np.log10(self.delta)))

        return AbsoluteSearchRange(
            vmin=np.round(np.min([vmin, self.vmin]), n),
            vmax=np.round(np.max([vmax, self.vmax]), n),
            delta=self.delta,
        )

    @property
    def bounds(self):
        return (self.vmin, self.vmax)


@dc.dataclass
class RelativeSearchRange(SearchRange):
    """외기온의 최대, 최소 온도에 상대적인 탐색 범위."""

    def validate(self):
        if self.vmin <= 0:
            msg = f'{self.vmin=} <= 0'
            raise SearchRangeError(msg)

        if self.vmax >= 1:
            msg = f'{self.vmax=} >= 1'
            raise SearchRangeError(msg)

        return super().validate()

    def update(self, vmin: float, vmax: float) -> AbsoluteSearchRange:
        """
        분석에 이용할 외기온 범위에 맞춰 탐색 범위 업데이트.

        Parameters
        ----------
        vmin : float
            외기온도 최소값 [°C].
        vmax : float
            외기온도 최대값 [°C].

        Returns
        -------
        AbsoluteSearchRange

        Examples
        --------
        >>> RelativeSearchRange(vmin=0.2, vmax=0.8, delta=0.1).update(vmin=10, vmax=20)
        AbsoluteSearchRange(vmin=12.0, vmax=18.0, delta=0.1)
        """
        n = -int(np.floor(np.log10(self.delta)))
        r = vmax - vmin

        return AbsoluteSearchRange(
            vmin=np.round(vmin + r * self.vmin, n),
            vmax=np.round(vmin + r * self.vmax, n),
            delta=self.delta,
        )


DEFAULT_RANGE = RelativeSearchRange(vmin=0.05, vmax=0.95, delta=0.1)


@dc.dataclass(frozen=True)
class CprConfig:
    target: Literal['r2', 'adj_r2'] = 'r2'
    """최적화 대상. r² 또는 adj-r²가 최대가 되는 모델 탐색."""

    const_baseline: bool = True
    """True면 temperature의 계수를 계산하지 않고 일정한 baseline을 계산."""

    min_samples: int = 4
    """최소 샘플 개수."""

    allow_single_hvac_point: bool = False
    """냉·난방 구간에 데이터가 하나만 존재하는 모델의 허용 여부."""

    pvalue_threshold: float = 0.05
    """냉난방 민감도의 유효성 기준."""


class PlotStyle(TypedDict, total=False):
    line: dict
    scatter: dict
    axvline: dict | None

    shuffle: bool
    datetime_hue: bool
    xmin: float | None
    xmax: float | None


@dc.dataclass
class CprData:
    dataframe: pl.DataFrame
    conf: CprConfig

    # 정렬 후 (t[0], t[1], t[-2], t[-1])
    temp_range: tuple[float, float, float, float] = dc.field(init=False)

    def __post_init__(self):
        self.dataframe = self.dataframe.drop_nulls(['temperature', 'energy'])

        if (c := self.dataframe.height) < self.conf.min_samples:
            raise NotEnoughDataError(required=self.conf.min_samples, given=c)

        t = self.dataframe.select(pl.col('temperature').sort()).to_numpy().ravel()
        self.temp_range = (t[0], t[1], t[-2], t[-1])

    @classmethod
    def create(
        cls,
        x: _FloatArray | pl.DataFrame,
        y: _FloatArray | None = None,
        datetime: _DateArray | None = None,
        conf: CprConfig | None = None,
    ):
        conf = conf or CprConfig()
        if isinstance(x, pl.DataFrame):
            return cls(x, conf=conf)

        if y is None:
            msg = 'y must be provided if x is not a DataFrame.'
            raise ValueError(msg)

        data = pl.DataFrame({'temperature': x, 'energy': y})

        if datetime is not None:
            dt = pl.Series('datetime', datetime)
            if dt.dtype is pl.String:
                dt = dt.str.to_datetime()

            data = data.with_columns(dt)

        return cls(data, conf=conf)

    def degree_day(
        self,
        th: float = np.nan,
        tc: float = np.nan,
        data: pl.DataFrame | None = None,
    ):
        if data is None:
            data = self.dataframe

        t = pl.col('temperature')
        return data.with_columns(
            pl.when(np.isnan(th))
            .then(pl.lit(None))
            .otherwise(pl.max_horizontal(pl.lit(0), th - t))
            .alias('HDD'),
            pl.when(np.isnan(tc))
            .then(pl.lit(None))
            .otherwise(pl.max_horizontal(pl.lit(0), t - tc))
            .alias('CDD'),
        )

    def _is_valid_change_points(self, th: float, tc: float):
        if np.isnan(th) and np.isnan(tc):
            return False

        if th >= tc:
            return False

        return (
            self.conf.allow_single_hvac_point  ##
            or not (th < self.temp_range[1] or tc > self.temp_range[2])
        )

    @overload
    def fit(
        self, th: float, tc: float, *, as_dataframe: Literal[True]
    ) -> pd.DataFrame | None: ...

    @overload
    def fit(
        self, th: float, tc: float, *, as_dataframe: Literal[False] = ...
    ) -> LinearModelDict | None: ...

    def fit(
        self,
        th: float = np.nan,
        tc: float = np.nan,
        *,
        as_dataframe: bool = False,
    ):
        if not self._is_valid_change_points(th=th, tc=tc):
            return None

        data = self.degree_day(th=th, tc=tc)

        # 독립변수 목록
        variables = ['temperature', 'HDD', 'CDD']
        is_indep = [not self.conf.const_baseline, not np.isnan(th), not np.isnan(tc)]
        independants = [v for v, i in zip(variables, is_indep, strict=True) if i]

        return pg.linear_regression(
            X=data.select(independants).to_pandas(),
            y=data.select('energy').to_series(),
            add_intercept=True,
            as_dataframe=as_dataframe,
        )

    def _fit(self, cp: tuple[float, float] | np.ndarray = (np.nan, np.nan)) -> float:
        """
        최적화 목적함수 (낮을수록 우수).

        Parameters
        ----------
        cp : tuple[float, float] | np.ndarray, optional
            (Th, Tc)

        Returns
        -------
        float
            -r² 또는 -(adj-r²)
        """
        if (model := self.fit(th=cp[0], tc=cp[1], as_dataframe=False)) is None:
            return np.inf

        coef = dict(zip(model['names'], model['coef'], strict=True))
        if any(coef.get(x, 0) < 0 for x in ['HDD', 'CDD']):
            # 유효하지 않은 모델 -> inf
            return np.inf

        pvalue = dict(zip(model['names'], model['pval'], strict=True))
        if any(pvalue.get(x, 0) > self.conf.pvalue_threshold for x in ['HDD', 'CDD']):
            # 냉난방 민감도 p-value가 유요하지 않은 모델 -> 0
            return 0

        return -model[self.conf.target]

    def object_function(self, x: _X) -> Callable[..., float]:
        """
        최적화 대상(냉난방)별 목적함수 반환.

        Parameters
        ----------
        x : _X
            최적화 대상

        Returns
        -------
        Callable[..., float]
        """
        match x:
            case 'h':
                return lambda x: self._fit((x, np.nan))
            case 'c':
                return lambda x: self._fit((np.nan, x))
            case 'hc':
                return self._fit


@dc.dataclass(frozen=True)
class CprModel:
    change_point: np.ndarray

    model_dict: LinearModelDict
    model_frame: pl.DataFrame

    optimizer: Optimizer
    optimize_result: opt.OptimizeResult | None

    data: CprData

    DEFAULT_STYLE: ClassVar[PlotStyle] = {
        'scatter': {'zorder': 2.1, 'color': 'gray', 'alpha': 0.5, 'palette': 'flare'},
        'line': {'zorder': 2.2, 'color': 'gray', 'alpha': 0.75},
        'axvline': {'ls': '--', 'color': 'gray', 'alpha': 0.5},
        'shuffle': True,
        'datetime_hue': True,
    }

    @property
    def is_valid(self) -> bool:
        return self.model_dict['r2'] > 0

    def coef(self) -> dict[str, float]:
        return dict(zip(self.model_dict['names'], self.model_dict['coef'], strict=True))

    def predict(self, data: pl.DataFrame | ArrayLike | None = None) -> pl.DataFrame:
        """
        기온으로부터 기저, 냉방, 난방 에너지 사용량 예측.

        Parameters
        ----------
        data : pl.DataFrame | ArrayLike | None, optional
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
            df = self.data.dataframe
        elif isinstance(data, pl.DataFrame):
            df = data
        else:
            df = pl.DataFrame({'temperature': data})

        coef = dict.fromkeys(['Intercept', 'HDD', 'CDD'], 1.0) | self.coef()
        return (
            self.data.degree_day(
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
        ratio = pl.col('energy') / pl.col('Ep')
        return self.predict(data).with_columns(
            pl.col('Epb').fill_null(0).mul(ratio).alias('Edb'),
            pl.col('Eph').fill_null(0).mul(ratio).alias('Edh'),
            pl.col('Epc').fill_null(0).mul(ratio).alias('Edc'),
        )

    def _segments(self, xmin: float | None = None, xmax: float | None = None):
        points = [
            self.data.temp_range[0] if xmin is None else xmin,
            *sorted(self.change_point),
            self.data.temp_range[-1] if xmax is None else xmax,
        ]
        data = pl.DataFrame(pl.Series('temperature', values=points, dtype=pl.Float64))
        return self.predict(data)

    @staticmethod
    def _plot_scatter(data: pl.DataFrame, style: PlotStyle, ax: Axes):
        dt = 'datetime'
        kwargs = style.get('scatter', {})
        palette = kwargs.pop('palette', 'flare')

        if not (style.get('datetime_hue') and dt in data.columns):
            sm = None
        else:
            data = data.with_columns(pl.col(dt).dt.epoch('d').alias('epoch'))
            sm = cm.ScalarMappable(cmap=palette, norm=mpl.colors.Normalize())
            kwargs |= {
                'hue': 'epoch',
                'palette': palette,
                'hue_norm': sm.norm,
                'legend': False,
            }

        if style.get('shuffle', True):
            data = data.sample(fraction=1, shuffle=True, seed=42)

        sns.scatterplot(data.to_pandas(), x='temperature', y='energy', ax=ax, **kwargs)
        if sm is not None:
            cb = plt.colorbar(sm, ax=ax)
            loc = mdates.AutoDateLocator()
            cb.ax.yaxis.set_major_locator(loc)
            cb.ax.yaxis.set_major_formatter(mdates.AutoDateFormatter(loc))

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
            data = self.data.dataframe if scatter is True else scatter
            self._plot_scatter(data=data, style=style, ax=ax)

        if segments:
            sns.lineplot(
                self._segments(xmin=style.get('xmin'), xmax=style.get('xmax')),
                x='temperature',
                y='Ep',
                ax=ax,
                **style.get('line', {}),
            )

            if s := style.get('axvline'):
                for x in self.change_point:
                    ax.axvline(x, **s)

        return ax


@dc.dataclass
class CprEstimator:
    data: CprData | None = None
    conf: CprConfig = dc.field(default_factory=CprConfig)

    @staticmethod
    def _data(
        conf: CprConfig,
        x: _FloatArray | pl.DataFrame | CprData,
        y: _FloatArray | None = None,
        datetime: _DateArray | None = None,
    ):
        if isinstance(x, CprData):
            data = x
            data.conf = conf
        else:
            data = CprData.create(x=x, y=y, datetime=datetime, conf=conf)

        return data

    @classmethod
    def create(
        cls,
        x: _FloatArray | pl.DataFrame | CprData,
        y: _FloatArray | None = None,
        datetime: _DateArray | None = None,
        conf: CprConfig | None = None,
    ):
        if conf is None:
            conf = CprConfig()

        data = cls._data(conf=conf, x=x, y=y, datetime=datetime)
        return cls(data=data, conf=conf)

    def set_data(
        self,
        x: _FloatArray | pl.DataFrame | CprData,
        y: _FloatArray | None = None,
        datetime: _DateArray | None = None,
    ):
        self.data = self._data(conf=self.conf, x=x, y=y, datetime=datetime)
        return self

    def _update_search_range(self, r: SearchRange):
        if self.data is None:
            msg = 'Data is not set'
            raise CprError(msg)

        indices = (0, -1) if self.conf.allow_single_hvac_point else (1, 2)
        return r.update(
            vmin=self.data.temp_range[indices[0]],
            vmax=self.data.temp_range[indices[1]],
        )

    def _fit_brute(
        self,
        heating: AbsoluteSearchRange | None,
        cooling: AbsoluteSearchRange | None,
        **kwargs,
    ) -> np.ndarray:
        assert self.data is not None

        x: _X
        match heating, cooling:
            case (h, None) if h is not None:
                x = 'h'
                ranges = [h.slice()]
            case (None, c) if c is not None:
                x = 'c'
                ranges = [c.slice()]
            case h, c:
                assert h is not None
                assert c is not None
                x = 'hc'
                ranges = [h.slice(), c.slice()]

        fn = self.data.object_function(x)
        xmin = opt.brute(fn, ranges=ranges, **kwargs)
        assert not isinstance(xmin, tuple)
        return xmin

    def _fit_scalar(
        self,
        heating: AbsoluteSearchRange | None,
        cooling: AbsoluteSearchRange | None,
        **kwargs,
    ) -> opt.OptimizeResult:
        assert self.data is not None

        match heating, cooling:
            case (h, None) if h is not None:
                r = opt.minimize_scalar(
                    self.data.object_function('h'), bounds=h.bounds, **kwargs
                )
            case (None, c) if c is not None:
                r = opt.minimize_scalar(
                    self.data.object_function('c'), bounds=c.bounds, **kwargs
                )
            case _:
                raise OptimizeBoundError(heating, heating, required=1)

        assert isinstance(r, opt.OptimizeResult)
        return r

    def _fit_multivariable(
        self,
        heating: AbsoluteSearchRange,
        cooling: AbsoluteSearchRange,
        x0: np.ndarray | None = None,
        **kwargs,
    ):
        assert self.data is not None

        if x0 is None:
            # 초기 추정치를 온도 범위의 0.2, 0.8 지점으로 설정
            tr = self.data.temp_range
            x0 = tr[0] + (tr[-1] - tr[0]) * np.array([0.2, 0.8])

        return opt.minimize(
            self.data.object_function('hc'),
            x0=x0,
            bounds=[heating.bounds, cooling.bounds],
            **kwargs,
        )

    def _fit_with(
        self,
        heating: AbsoluteSearchRange | None,
        cooling: AbsoluteSearchRange | None,
        optimizer: Optimizer,
        **kwargs,
    ) -> tuple[np.ndarray, opt.OptimizeResult | None]:
        match optimizer:
            case 'brute':
                param = self._fit_brute(heating=heating, cooling=cooling, **kwargs)
                res = None
            case 'multivariable':
                if heating is None or cooling is None:
                    raise OptimizeBoundError(heating, cooling, required=2)

                res = self._fit_multivariable(
                    heating=heating, cooling=cooling, **kwargs
                )
                param = res['x']
            case 'scalar':
                res = self._fit_scalar(heating=heating, cooling=cooling)
                param = np.array([res['x']])  # TODO Th, Tc 분리
            case _:
                raise AssertionError(optimizer)

        return param, res

    def _fit(
        self,
        heating: SearchRange | None = DEFAULT_RANGE,
        cooling: SearchRange | None = DEFAULT_RANGE,
        optimizer: Optimizer | None = 'brute',
        **kwargs,
    ) -> CprModel:
        if self.data is None:
            msg = 'Data is not set'
            raise ValueError(msg)

        if heating is None and cooling is None:
            raise OptimizeBoundError(heating, cooling, required='ge1')

        h = None if heating is None else self._update_search_range(heating)
        c = None if cooling is None else self._update_search_range(cooling)

        if optimizer is None:
            optimizer = (
                'scalar' if (heating is None or cooling is None) else 'multivariable'
            )

        param, res = self._fit_with(h, c, optimizer, **kwargs)

        # change point (heating, cooling)
        if np.size(param) == 1:
            cp = np.array([np.nan, param[0]] if heating is None else [param[0], np.nan])
        else:
            cp = np.array(param)

        if (model_dict := self.data.fit(*cp, as_dataframe=False)) is None:
            msg = f'No valid model (cp={cp})'
            raise OptimizationError(msg)

        # dataframe
        pddf = self.data.fit(*cp, as_dataframe=True)
        assert pddf is not None

        cpdf = pl.DataFrame({'names': ['HDD', 'CDD'], 'change_point': cp})
        model_frame = (
            pl.from_pandas(pddf)
            .join(cpdf, on='names', how='left')
            .select('names', 'change_point', pl.all().exclude('names', 'change_point'))
        )

        return CprModel(
            change_point=cp,
            optimizer=optimizer,
            optimize_result=res,
            model_dict=model_dict,
            model_frame=model_frame,
            data=self.data,
        )

    def fit(
        self,
        heating: SearchRange = DEFAULT_RANGE,
        cooling: SearchRange = DEFAULT_RANGE,
        optimizer: Optimizer | None = 'brute',
        model: Literal['h', 'c', 'hc', 'best'] = 'best',
        **kwargs,
    ):
        if model == 'best' and optimizer not in {'brute', None}:
            msg = 'optimizer must be set to "brute" or None to search the best model.'
            raise ValueError(msg)

        kwargs['optimizer'] = optimizer

        match model:
            case 'h':
                return self._fit(heating=heating, cooling=None, **kwargs)
            case 'c':
                return self._fit(heating=None, cooling=cooling, **kwargs)
            case 'hc':
                return self._fit(heating=heating, cooling=cooling, **kwargs)

        heating = self._update_search_range(heating)
        cooling = self._update_search_range(cooling)
        models: list[CprModel] = []

        for h, c in [(heating, cooling), (heating, None), (None, cooling)]:
            try:
                models.append(self._fit(heating=h, cooling=c, **kwargs))
            except OptimizationError:
                pass

        if not models:
            msg = 'No valid model'
            raise OptimizationError(msg)

        return max(models, key=lambda x: x.model_dict[self.conf.target])


if __name__ == '__main__':
    import rich

    console = rich.get_console()

    data = pl.DataFrame({
        'temperature': list(range(8)),
        'energy': (
            np.array([2, 1, 0, 0, 0, 0, 2, 4])
            + np.random.default_rng(42).normal(0, 0.1, 8)
        ),
        'datetime': pl.date_range(
            pl.date(2000, 1, 1), pl.date(2000, 1, 8), interval='1d', eager=True
        ),
    })

    estimator = CprEstimator().set_data(data)
    sr = RelativeSearchRange(0.05, 0.95, delta=1)
    model = estimator.fit(heating=sr, cooling=sr, optimizer='brute')

    console.print(data)
    console.rule()
    console.print(model)

    model.plot()
    plt.show()

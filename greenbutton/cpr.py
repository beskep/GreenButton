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
import enum
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, overload

import matplotlib.colors as mcolors
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
    import pandas as pd
    from matplotlib.axes import Axes
    from polars._typing import FrameType

type Operation = Literal['hc', 'h', 'c']
type Method = Literal['brute', 'numerical']

type _FloatArray = NDArray[np.float64]
type _FloatSequence = Sequence[float] | NDArray[np.float64]
type _DateSequence = Sequence[dt.datetime | str] | NDArray[np.datetime64]


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


class LinearModelDict(TypedDict):
    """pingouin으로 분석한 선형회귀분석 결과."""

    names: list[str]
    coef: _FloatArray
    se: _FloatArray
    T: _FloatArray
    pval: _FloatArray
    r2: float
    adj_r2: float
    # CI 생략
    df_model: int
    df_resid: int
    residuals: _FloatArray
    X: _FloatArray
    y: _FloatArray
    pred: _FloatArray


@dc.dataclass
class SearchRange(abc.ABC):
    vmin: float  # 탐색 최소 온도
    vmax: float  # 탐색 최대 온도

    delta: float  # 탐색 간격/해상도 [°C]
    # (brute force 탐색, 최적화 후 균형점 온도 반올림에 이용)

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

    def decimals(self, amin: int = 0):
        return min(amin, -int(np.floor(np.log10(self.delta))))

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
        return AbsoluteSearchRange(
            vmin=np.min([vmin, self.vmin]),
            vmax=np.max([vmax, self.vmax]),
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
        r = vmax - vmin

        return AbsoluteSearchRange(
            vmin=vmin + r * self.vmin,
            vmax=vmin + r * self.vmax,
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


def degree_day(data: FrameType, th: float = np.nan, tc: float = np.nan):
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


class Validity(enum.IntEnum):
    VALID = 1
    INSIGNIFICANT = 0
    INVALID = -1


def check_model_validity(
    model: LinearModelDict | None, conf: CprConfig
) -> tuple[Validity, float]:
    if model is None:
        return Validity.INVALID, np.nan

    coef = dict(zip(model['names'], model['coef'], strict=True))
    pvalue = dict(zip(model['names'], model['pval'], strict=True))
    r2 = model[conf.target]  # r2 or adj-r2

    if any(coef.get(x, 0) < 0 for x in ['HDD', 'CDD']) or model['r2'] <= 0:
        # 냉난방 민감도가 음수 또는 r2가 0
        return Validity.INVALID, r2

    if (
        pvalue.get('HDD', 0) > conf.pvalue_threshold
        or pvalue.get('CDD', 0) > conf.pvalue_threshold
    ):
        # 냉난방 민감도 p-value가 유효하지 않음
        return Validity.INSIGNIFICANT, r2

    return Validity.VALID, r2


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
        x: _FloatSequence | pl.DataFrame,
        y: _FloatSequence | None = None,
        datetime: _DateSequence | None = None,
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

        data = degree_day(self.dataframe, th=th, tc=tc)

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

    def _fit(self, cp: tuple[float, float] | _FloatArray = (np.nan, np.nan)) -> float:
        """
        최적화 목적함수 (낮을수록 우수).

        Parameters
        ----------
        cp : tuple[float, float] | _FloatArray, optional
            (Th, Tc)

        Returns
        -------
        float
            -r² 또는 -(adj-r²)
        """
        model = self.fit(th=cp[0], tc=cp[1], as_dataframe=False)
        validity, r2 = check_model_validity(model, self.conf)

        match validity:
            case Validity.VALID:
                return -r2
            case Validity.INSIGNIFICANT:
                return 0
            case Validity.INVALID:
                return np.inf

    def object_function(self, operation: Operation) -> Callable[..., float]:
        """
        최적화 대상(냉난방)별 목적함수 반환.

        Parameters
        ----------
        operation: Operation
            최적화 대상

        Returns
        -------
        Callable[..., float]
        """
        match operation:
            case 'h':
                return lambda x: self._fit((x, np.nan))
            case 'c':
                return lambda x: self._fit((np.nan, x))
            case 'hc':
                return self._fit


@dc.dataclass
class Optimizer:
    data: CprData
    heating: AbsoluteSearchRange
    cooling: AbsoluteSearchRange
    kwargs: dict

    def brute(self, operation: Operation) -> _FloatArray:
        match operation:
            case 'h':
                ranges = [self.heating.slice()]
            case 'c':
                ranges = [self.cooling.slice()]
            case 'hc':
                ranges = [self.heating.slice(), self.cooling.slice()]

        fn = self.data.object_function(operation)
        xmin = opt.brute(fn, ranges=ranges, **self.kwargs)
        assert not isinstance(xmin, tuple)
        return xmin

    def scalar(self, operation: Operation) -> opt.OptimizeResult:
        match operation:
            case 'hc':
                raise OptimizeBoundError(self.heating, self.cooling, required=1)
            case 'h':
                bounds = self.heating.bounds
            case 'c':
                bounds = self.cooling.bounds

        return opt.minimize_scalar(  # pyright: ignore[reportReturnType]
            self.data.object_function(operation),
            bounds=bounds,
            **self.kwargs,
        )

    def multivariable(self, x0: _FloatArray | None = None) -> opt.OptimizeResult:
        if x0 is None:
            # 초기 추정치를 온도 범위의 0.2, 0.8 지점으로 설정
            a, *_, b = self.data.temp_range
            x0 = a + (b - a) * np.array([0.2, 0.8])

        return opt.minimize(
            self.data.object_function('hc'),
            x0=x0,
            bounds=[self.heating.bounds, self.cooling.bounds],
            **self.kwargs,
        )

    def _optimize(self, operation: Operation, method: Method):
        match operation, method:
            case _, 'brute':
                param = self.brute(operation=operation)
                res = None
            case 'h' | 'c', 'numerical':
                res = self.scalar(operation=operation)
                param = np.array([res['x']])
            case 'hc', 'numerical':
                res = self.multivariable()
                param = res['x']

        match operation:
            case 'h':
                cp = (round(float(param[0]), self.heating.decimals()), np.nan)
            case 'c':
                cp = (np.nan, round(float(param[0]), self.cooling.decimals()))
            case 'hc':
                cp = (
                    round(float(param[0]), self.heating.decimals()),
                    round(float(param[1]), self.cooling.decimals()),
                )

        if (model_dict := self.data.fit(*cp, as_dataframe=False)) is None:
            msg = f'No valid model (cp={cp})'
            raise OptimizationError(msg)

        return CprModel(
            change_points=cp,
            model_dict=model_dict,
            data=self.data,
            validity=check_model_validity(model_dict, conf=self.data.conf)[0],
            optimize_method=method,
            optimize_result=res,
        )

    def __call__(
        self,
        operation: Operation | Literal['best'],
        method: Method = 'brute',
    ):
        if operation != 'best':
            return self._optimize(operation=operation, method=method)

        models: list[CprModel] = []
        for op in ['h', 'c', 'hc']:
            try:
                models.append(self._optimize(operation=op, method=method))  # type: ignore[arg-type]
            except OptimizationError:
                pass

        if not models or all(x.validity <= 0 for x in models):
            msg = 'No valid model'
            raise OptimizationError(msg)

        def key(model: CprModel):
            return (model.validity, model.model_dict[self.data.conf.target])

        return max(models, key=key)


@dc.dataclass(frozen=True)
class CprModel:
    change_points: tuple[float, float]
    model_dict: LinearModelDict
    data: CprData

    validity: Validity
    optimize_method: Method
    optimize_result: opt.OptimizeResult | None

    DEFAULT_STYLE: ClassVar[PlotStyle] = {
        'scatter': {'zorder': 2.1, 'color': 'gray', 'alpha': 0.5, 'palette': 'flare'},
        'line': {'zorder': 2.2, 'color': 'gray', 'alpha': 0.75},
        'axvline': {'ls': '--', 'color': 'gray', 'alpha': 0.5},
        'shuffle': True,
        'datetime_hue': True,
    }
    OBSERVATIONS: ClassVar[set[str]] = {'X', 'y', 'pred', 'residuals'}

    @cached_property
    def model_frame(self):
        cp = 'change_points'
        cpdf = pl.DataFrame({'names': ['HDD', 'CDD'], cp: self.change_points})
        data = {k: v for k, v in self.model_dict.items() if k not in self.OBSERVATIONS}
        return (
            pl.DataFrame(data)
            .join(cpdf, on='names', how='left')
            .select('names', cp, pl.all().exclude('names', cp))
        )

    @property
    def is_valid(self) -> bool:
        return self.validity > 0

    @cached_property
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
            다음 열 포함:
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

        coef = dict.fromkeys(['Intercept', 'HDD', 'CDD'], 1.0) | self.coef
        return (
            degree_day(df, *self.change_points)
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
            *sorted(self.change_points),
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
            sm = cm.ScalarMappable(cmap=palette, norm=mcolors.Normalize())
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
                for x in self.change_points:
                    ax.axvline(x, **s)

        return ax


@dc.dataclass
class CprEstimator:
    data: CprData | None = None
    conf: CprConfig = dc.field(default_factory=CprConfig)

    @staticmethod
    def _data(
        conf: CprConfig,
        x: _FloatSequence | pl.DataFrame | CprData,
        y: _FloatSequence | None = None,
        datetime: _DateSequence | None = None,
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
        x: _FloatSequence | pl.DataFrame | CprData,
        y: _FloatSequence | None = None,
        datetime: _DateSequence | None = None,
        conf: CprConfig | None = None,
    ):
        if conf is None:
            conf = CprConfig()

        data = cls._data(conf=conf, x=x, y=y, datetime=datetime)
        return cls(data=data, conf=conf)

    def set_data(
        self,
        x: _FloatSequence | pl.DataFrame | CprData,
        y: _FloatSequence | None = None,
        datetime: _DateSequence | None = None,
    ):
        self.data = self._data(conf=self.conf, x=x, y=y, datetime=datetime)
        return self

    def _update_search_range(self, r: SearchRange):
        assert self.data is not None
        tr = self.data.temp_range
        allow = self.conf.allow_single_hvac_point
        return r.update(vmin=tr[0] if allow else tr[1], vmax=tr[-1] if allow else tr[2])

    def fit(
        self,
        heating: SearchRange = DEFAULT_RANGE,
        cooling: SearchRange = DEFAULT_RANGE,
        method: Method = 'brute',
        operation: Operation | Literal['best'] = 'best',
        **kwargs,
    ):
        if self.data is None:
            msg = 'Data is not set'
            raise ValueError(msg)

        optimize = Optimizer(
            data=self.data,
            heating=self._update_search_range(heating),
            cooling=self._update_search_range(cooling),
            kwargs=kwargs,
        )
        return optimize(operation=operation, method=method)


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
    model = estimator.fit(heating=sr, cooling=sr)

    console.print(data)

    console.rule()
    console.print(model)

    console.rule()
    with pl.Config() as conf:
        conf.set_tbl_cols(20)
        console.print(model.disaggregate())

    model.plot()
    plt.show()

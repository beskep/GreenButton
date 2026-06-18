"""경희대학교 extended change point model 아이디어 테스트."""

import dataclasses as dc
import enum
import functools
import itertools
import warnings
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.optimize as opt
import seaborn as sns
import statsmodels.api as sm
import structlog
from matplotlib.figure import Figure

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils.cli import App

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from statsmodels.iolib.summary import Summary

type Building = Literal['KEPCO', 'KEA']
type Delta = Literal['day', 'hour']
type Period = Literal['summer', 'winter', 'spring-autumn'] | None
type TempVar = Literal['Te', 'dT', 'Tiw']

BUILDINGS: tuple[Building, ...] = ('KEPCO', 'KEA')
DELTAS: tuple[Delta, ...] = ('day', 'hour')
PERIODS: tuple[Period, ...] = ('summer', 'winter', 'spring-autumn', None)
TEMP_VARS: tuple[TempVar, ...] = ('Te', 'dT', 'Tiw')

logger = structlog.stdlib.get_logger()
app = App(
    config=[
        cyclopts.config.Toml(f'config/{x}.toml', use_commands_as_keys=False)
        for x in ['.experiment', 'experiment']
    ],
)


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'ecpm'

    EUI_THRESHOLD: float = 0.1

    # gross floor area [m²]
    GFA: ClassVar[dict[Building, float]] = {'KEPCO': 5208.81, 'KEA': 24348.0}

    def read_building_data(
        self,
        bldg: Building,
        delta: Delta = 'day',
        *,
        holiday: bool | None = False,
    ):
        index = '01' if bldg == 'KEPCO' else '02'
        src = self.dirs.database / f'{index}.{bldg}-BEMS-{delta}.parquet'
        lf = (
            pl
            .scan_parquet(src)
            .drop_nulls()
            .rename({
                'electricity_kWh': 'energy',
                'temp_external': 'te',
                'temp_internal': 'ti',
            })
            .with_columns(pl.col('energy').truediv(self.GFA[bldg]).alias('EUI'))
        )

        if delta == 'day':
            lf = lf.filter(pl.col('EUI') >= self.EUI_THRESHOLD)

        if isinstance(holiday, bool):
            lf = lf.filter(pl.col('holiday') == holiday)

        return lf.collect()

    def read_weather(
        self,
        bldg: Building | None = None,
        *,
        holiday: bool | None = False,
    ):
        lf = (
            pl
            .scan_parquet(self.dirs.database / 'extended.parquet')
            .filter(pl.col('EUI') >= self.EUI_THRESHOLD)
            .with_columns()
        )

        if bldg is not None:
            lf = lf.filter(pl.col('building') == bldg)

        if isinstance(holiday, bool):
            lf = lf.filter(pl.col('holiday') == holiday)

        return lf.collect()


@app.command
@dc.dataclass
class BldgGridPlot:
    conf: Config

    def plot(self, bldg: Building, delta: Delta):
        data = (
            self.conf
            .read_building_data(bldg, delta)
            .with_columns((pl.col('ti') - pl.col('te')).alias(r'$\Delta T$'))
            .rename({'te': '$T_{ext}$', 'ti': '$T_{int}$'})
        )

        if delta == 'day':
            hue = None
        else:
            hue = 'office hour'
            data = data.with_columns(
                (
                    (pl.col('datetime').dt.time() >= pl.time(9, 0))
                    & (pl.col('datetime').dt.time() <= pl.time(18, 0))
                ).alias(hue)
            )

        grid = (
            sns
            .PairGrid(data.drop('datetime', 'energy', 'holiday').to_pandas(), hue=hue)
            .map_lower(sns.scatterplot, alpha=0.25)
            .map_diag(sns.histplot)
        )

        if delta == 'hour':
            grid.add_legend()

        grid.savefig(self.conf.dirs.analysis / f'01.BldgPairGrid-{bldg}-{delta}.png')
        plt.close(grid.figure)

    def __call__(self):
        utils.mpl.MplTheme().grid().apply()

        for bldg, delta in itertools.product(BUILDINGS, DELTAS):
            logger.info('%s %s', bldg, delta)
            self.plot(bldg, delta)


def _months(period: Period):
    match period:
        case None:
            return ()
        case 'summer':
            return (6, 7, 8)
        case 'winter':
            return (12, 1, 2)
        case 'spring-autumn':
            return (3, 4, 5, 9, 10, 11)


@app.command
@dc.dataclass
class WeatherGrid:
    conf: Config
    alpha: float = 0.25

    def plot(self, bldg: Building, period: Period):
        data = (
            self.conf
            .read_weather(bldg)
            .with_columns((pl.col('Te') - pl.col('Ti')).alias('Te-Ti'))
            .drop('Ti', 'RH')
        )

        if period is not None:
            months = _months(period)
            data = data.filter(pl.col('date').dt.month().is_in(months))

        grid = (
            sns
            .PairGrid(data.drop('date', 'holiday', 'energy').to_pandas(), height=2)
            .map_lower(sns.scatterplot, alpha=self.alpha)
            .map_upper(sns.scatterplot, alpha=self.alpha)
            .map_diag(sns.histplot)
        )

        p = '' if period is None else f'-{period}'
        grid.savefig(self.conf.dirs.analysis / f'02.WeatherPairGrid-{bldg}{p}.png')
        grid.savefig(self.conf.dirs.analysis / f'02.WeatherPairGrid-{bldg}{p}.svg')
        plt.close(grid.figure)

    def __call__(self):
        utils.mpl.MplTheme().grid().apply()

        for bldg, period in itertools.product(BUILDINGS, PERIODS):
            self.plot(bldg, period)


class _Model(enum.StrEnum):
    CPM = 'CPM'
    """E = Eb + β_h max(0, Th-Te) + β_c max(0, Te-Tc)"""

    ADDITIVE = 'ADD'
    """
    E = Eb + β_h  max(0, Th -Te) + β_c  max(0, Te-Tc)
           + β_h' max(0, Th'-Ti) + β_c' max(0, Ti-Tc')
    """

    MULTIPLICATIVE = 'MULT'
    """
    ΔT = Ti - Te  # (대부분) 양수
    E  = Eb + β_h (ΔT + τ_h) max(0, Th - Te) + β_c (ΔT + τ_c) max(0, Te - Tc)
    """


@dc.dataclass
class _Dataset:
    data: pl.DataFrame
    building: Building
    tvar: TempVar
    xvar: tuple[str, ...] = ('Tiw', 'Te', 'dT', 'Pv', 'I')
    yvar: str = 'EUI'

    def __post_init__(self):
        self.data = (
            self.data
            # NOTE: Te - Ti 대신 Ti - Te (대부분 양수) 적용
            .with_columns((pl.col('Tiw') - pl.col('Te')).alias('dT'))
            .filter(pl.col('building') == self.building)
            .select([*self.xvar, self.yvar])
            .drop_nulls()
        )

    @functools.cached_property
    def t(self):
        return self.data[self.tvar].to_numpy()

    @functools.cached_property
    def y(self):
        return self.data['EUI'].to_numpy()


@dc.dataclass
class _Optimizer:
    model: _Model
    dataset: _Dataset
    extend_weather: bool = False  # TODO I, Pv 하나씩만 추가한 모델도 분석

    PENALTY: ClassVar[float] = 1e8
    XLABEL: ClassVar[dict[TempVar, str]] = {
        'Te': 'External Temperature $T_{ext}$',
        'Tiw': 'Internal Temperature $T_{int}$',
        'dT': r'Temperature Difference ($\Delta T = T_{ext} - T_{int})$',
    }

    def fit_linear_model(self, cp: np.ndarray):
        t = pl.col(self.dataset.tvar)
        data = self.dataset.data.with_columns(
            pl.lit(1.0).alias('ones'),
            pl.max_horizontal(pl.lit(0), cp[0] - t).alias('hdd'),
            pl.max_horizontal(pl.lit(0), t - cp[1]).alias('cdd'),
        )
        x_vars = ['ones', 'hdd', 'cdd']

        # NOTE: X 1, 2, 3번째 변수는 ones, HDD, CDD -> baseline, beta_h, beta_c로 유지
        match self.model:
            case _Model.CPM:
                pass
            case _Model.ADDITIVE:
                # cp=[Th, Tc, Th', Tc']
                ti = pl.col('Tiw')
                data = data.with_columns(
                    pl.max_horizontal(pl.lit(0), cp[2] - ti).alias('hddi'),
                    pl.max_horizontal(pl.lit(0), ti - cp[3]).alias('cddi'),
                )
                x_vars.extend(['hddi', 'cddi'])
            case _Model.MULTIPLICATIVE:
                # cp=[Th, Tc, tau_h, tau_c]  # noqa: ERA001
                dt = pl.col('dT')
                data = data.with_columns(
                    'ones',
                    ((dt + cp[2]) * pl.col('hdd')).alias('hdd'),
                    ((dt + cp[3]) * pl.col('cdd')).alias('cdd'),
                )
            case _:
                raise ValueError(self.model)

        if self.extend_weather:
            x_vars.extend(['I', 'Pv'])

        return sm.OLS(endog=self.dataset.y, exog=data.select(x_vars).to_numpy()).fit()

    def bound(self):
        tiw = (10.0, 40.0)
        match self.dataset.tvar:
            case 'Te':
                t = (0.0, 25.0)
            case 'dT':
                t = (-15.0, 10.0)
            case 'Tiw':
                t = tiw

        match self.model:
            case _Model.CPM:
                return (t, t)
            case _Model.ADDITIVE:
                return (t, t, tiw, tiw)
            case _Model.MULTIPLICATIVE:
                shift = (-5.0, 50.0)
                return (t, t, shift, shift)
            case _:
                raise ValueError(self.model)

    def object(self, cp: np.ndarray) -> np.ndarray:
        model = self.fit_linear_model(cp)

        # t_h > t_c일 경우 패널티
        p1 = self.PENALTY * max(0, cp[0] - cp[1]) ** 2

        beta: NDArray[np.floating]
        match self.model:
            case _Model.CPM | _Model.MULTIPLICATIVE:
                beta = model.params[1:3]
            case _Model.ADDITIVE:
                beta = model.params[1:5]
                p1 += self.PENALTY * max(0, cp[2] - cp[3]) ** 2
            case _:
                raise ValueError(self.model)

        # 음수 beta 패널티
        p2 = self.PENALTY * np.sum(np.square(np.minimum(0, beta)))

        resid = np.append(model.resid, [p1, p2])
        return np.sum(np.square(resid))

    def optimize(self) -> opt.OptimizeResult:
        r = opt.differential_evolution(self.object, bounds=self.bound(), rng=42)
        if not r.success:
            msg = 'Failed to optimize'
            raise ValueError(msg)
        return r

    def plot(self, r: opt.OptimizeResult, /):
        tvar = self.dataset.tvar
        linear_model = self.fit_linear_model(r.x)

        # summary
        summary: Summary = linear_model.summary()
        summary.add_extra_txt(['change_points:', *(str(x) for x in r.x)])

        # CPM
        fig = Figure()
        ax = fig.subplots()
        scatter = (
            self.dataset.data
            .select(
                tvar,
                pl.col(self.dataset.yvar).alias('Measured'),
                pl.Series('Predicted', linear_model.fittedvalues),
            )
            .unpivot(index=tvar)
            .with_columns()
        )
        ax.axhline(linear_model.params[0], ls=':', c='gray', alpha=0.5)
        ax.axvline(r.x[0], ls=':', c='gray', alpha=0.5)
        ax.axvline(r.x[1], ls=':', c='gray', alpha=0.5)
        sns.scatterplot(
            scatter,
            x=tvar,
            y='value',
            hue='variable',
            style='variable',
            alpha=0.5,
            s=10,
            ax=ax,
        )
        ax.text(
            0.02,
            0.05,
            f'$r^2={linear_model.rsquared:.4f}$',
            transform=ax.transAxes,
        )
        ax.set_ylim(0)
        ax.set_xlabel(f'{self.XLABEL[tvar]} [°C]')
        ax.set_ylabel('EUI [kWh/m²]')
        ax.legend(title='', markerscale=2)

        # residual
        data = (
            self.dataset.data
            .with_columns(
                period=pl
                .when(pl.col(tvar) < r.x[0])
                .then(pl.lit('H'))
                .when(pl.col(tvar) > r.x[1])
                .then(pl.lit('C'))
                .otherwise(pl.lit('B'))
            )
            .with_columns(residual=linear_model.resid)
            .unpivot(index=['period', 'residual'])
        )
        grid = (
            sns
            .FacetGrid(
                data,
                hue='period',
                hue_order=['H', 'B', 'C'],
                palette=['orangered', 'dimgray', 'steelblue'],
                col='variable',
                col_wrap=3,
                sharex=False,
                despine=False,
            )
            .map_dataframe(sns.scatterplot, x='value', y='residual', alpha=0.25, s=10)
            .set_axis_labels('', 'residual')
            .set_titles('{col_name} vs residual')
        )

        return summary, fig, grid

    def __call__(self):
        r = self.optimize()
        plots = self.plot(r)

        return r, *plots


@app.command
@dc.dataclass
class Ecpm:
    conf: Config

    def __call__(self):
        (
            utils.mpl
            .MplTheme()
            .grid(show=False)
            .tick(which='both', direction='in', color='.5')
            .apply()
        )

        data = self.conf.read_weather()
        output = self.conf.dirs.analysis / 'ECPM'
        output.mkdir(exist_ok=True)

        for bldg, tvar, model, extend_weather in itertools.product(
            BUILDINGS, TEMP_VARS, _Model, (False, True)
        ):
            if model != _Model.CPM and tvar != 'Te':
                continue

            e = ' ExtW' if extend_weather else ''
            case = f'{bldg} T={tvar} model={model}{e}'
            logger.info(case)

            optimizer = _Optimizer(
                model=model,
                dataset=_Dataset(data, building=bldg, tvar=tvar),
                extend_weather=extend_weather,
            )
            _, summary, fig, grid = optimizer()

            output.joinpath(f'{case}.csv').write_text(summary.as_csv())
            output.joinpath(f'{case}.txt').write_text(summary.as_text())
            fig.savefig(output / f'{case} scatter.png')
            fig.savefig(output / f'{case} scatter.svg')
            grid.savefig(output / f'{case} residual.png')
            grid.savefig(output / f'{case} residual.svg')
            plt.close('all')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='The figure layout has changed to tight')
    app()

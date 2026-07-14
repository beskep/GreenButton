"""Extended change point model 분석."""

import dataclasses as dc
import enum
import functools
import itertools
import warnings
from typing import TYPE_CHECKING, ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.optimize as opt
import seaborn as sns
import statsmodels.api as sm
import structlog
from matplotlib.figure import Figure

from greenbutton import utils
from greenbutton.utils import tqdm
from scripts.exp.ecpm.common import Config, app

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from matplotlib.axes import Axes
    from numpy.typing import NDArray
    from statsmodels.regression.linear_model import RegressionResults

type TempVar = Literal['Te', 'Te-Ti', 'Tiw']
type ExtraWeather = Literal['I', 'Pv', 'I+Pv'] | None

TEMP_VARS: tuple[TempVar, ...] = ('Te', 'Te-Ti', 'Tiw')
EXTRA_WEATHER: tuple[ExtraWeather, ...] = (None, 'I', 'Pv', 'I+Pv')

logger = structlog.stdlib.get_logger()


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
    building: str
    tvar: TempVar
    xvar: tuple[str, ...] = ('Tiw', 'Te', 'Ti-Te', 'Te-Ti', 'Pv', 'I')
    yvar: str = 'EUI'

    def __post_init__(self):
        self.data = (
            self.data
            .with_columns((pl.col('Tiw') - pl.col('Te')).alias('Ti-Te'))
            .with_columns((pl.col('Te') - pl.col('Tiw')).alias('Te-Ti'))
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


def _r2_score(ytrue, ypred):
    return 1 - (
        (pl.col(ytrue) - pl.col(ypred)).pow(2).sum()
        / (pl.col(ytrue) - pl.mean(ytrue)).pow(2).sum()
    )


@dc.dataclass
class _Optimizer:
    model: _Model
    dataset: _Dataset
    extra_weather: ExtraWeather = None

    _PENALTY: ClassVar[float] = 1e4
    XLABEL: ClassVar[dict[TempVar, str]] = {
        'Te': 'External Temperature $T_{ext}$',
        'Tiw': 'Internal Temperature $T_{int}$',
        'Te-Ti': r'Temperature Difference ($\Delta T = T_{ext} - T_{int})$',
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
                # NOTE: Te - Ti 대신 대부분의 경우 양수인 Ti - Te 적용
                dt = pl.col('Ti-Te')
                data = data.with_columns(
                    'ones',
                    ((dt + cp[2]) * pl.col('hdd')).alias('hdd'),
                    ((dt + cp[3]) * pl.col('cdd')).alias('cdd'),
                )
            case _:
                raise ValueError(self.model)

        if self.extra_weather:
            x_vars.extend(self.extra_weather.split('+'))

        return sm.OLS(endog=self.dataset.y, exog=data.select(x_vars).to_numpy()).fit()

    def bound(self):
        tiw = (10.0, 40.0)
        match self.dataset.tvar:
            case 'Te':
                t = (0.0, 25.0)
            case 'Te-Ti':
                t = (-20.0, 20.0)
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

    @functools.cached_property
    def penalty(self):
        y = self.dataset.y
        sse = np.sum(np.square(y - y.mean()))
        return sse * self._PENALTY

    def object(self, cp: np.ndarray) -> np.ndarray:
        model = self.fit_linear_model(cp)

        # t_h > t_c일 경우 패널티
        p1 = self.penalty * max(0, cp[0] - cp[1]) ** 2

        beta: NDArray[np.floating]
        match self.model:
            case _Model.CPM | _Model.MULTIPLICATIVE:
                beta = model.params[1:3]
            case _Model.ADDITIVE:
                beta = model.params[1:5]
                p1 += self.penalty * max(0, cp[2] - cp[3]) ** 2
            case _:
                raise ValueError(self.model)

        # 음수 beta 패널티
        p2 = self.penalty * np.sum(np.square(np.minimum(0, beta)))

        resid = np.append(model.resid, [p1, p2])
        return np.sum(np.square(resid))

    def _optimize(self) -> opt.OptimizeResult:
        r = opt.differential_evolution(
            self.object, bounds=self.bound(), popsize=30, rng=42
        )
        if not r.success:
            msg = 'Failed to optimize'
            raise ValueError(msg)
        return r

    @functools.cached_property
    def optimize_result(self):
        r = self._optimize()
        if not r.success:
            logger.warning('Optimization failed', case=self)
        return r

    @functools.cached_property
    def linear_model(self) -> RegressionResults:
        return self.fit_linear_model(self.optimize_result.x)

    @functools.cached_property
    def summary(self):
        summary = self.linear_model.summary2()
        summary.add_text(f'change_points={self.optimize_result.x.round(2).tolist()}')
        return summary

    @functools.cached_property
    def period_data(self):
        cp = self.optimize_result.x
        return self.dataset.data.with_columns(
            period=pl
            .when(pl.col(self.dataset.tvar) < cp[0])
            .then(pl.lit('H'))
            .when(pl.col(self.dataset.tvar) > cp[1])
            .then(pl.lit('C'))
            .otherwise(pl.lit('B')),
            ypred=self.linear_model.fittedvalues,
            residual=self.linear_model.resid,
        )

    def plot(self):
        cp = self.optimize_result.x
        linear_model = self.linear_model
        tvar = self.dataset.tvar

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
        ax.axvline(cp[0], ls=':', c='gray', alpha=0.5)
        ax.axvline(cp[1], ls=':', c='gray', alpha=0.5)
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
        grid = (
            sns
            .FacetGrid(
                self.period_data
                .drop('ypred', 'Ti-Te')
                .unpivot(index=['period', 'residual'])
                .with_columns(),
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

        return fig, grid

    def stats(self):
        stats = (
            self.period_data
            .group_by('period')
            .agg(
                _r2_score('EUI', 'ypred').alias('r2'),
                (pl.col('ypred') - pl.col('EUI')).pow(2).mean().sqrt().alias('RMSE'),
            )
            .unpivot(index='period')
            .with_columns(
                pl.format('season.{}.{}', 'period', 'variable').alias('variable'),
            )
        )
        return dict(zip(stats['variable'], stats['value'], strict=True))


@dc.dataclass(frozen=True)
class _Case:
    building: str
    tvar: TempVar
    model: _Model
    ext: ExtraWeather

    @functools.cached_property
    def name(self):
        e = f'_Ext{self.ext}' if self.ext else ''
        return f'{self.building} T={self.tvar} model={self.model}{e}'

    def __str__(self):
        return self.name

    @classmethod
    def iter(cls, buildings: Collection[str]):
        for bldg, tvar, model, ext in itertools.product(
            buildings, TEMP_VARS, _Model, EXTRA_WEATHER
        ):
            if model != _Model.CPM and tvar != 'Te':
                continue

            yield cls(building=bldg, tvar=tvar, model=model, ext=ext)


@app.command
@dc.dataclass
class Ecpm:
    conf: Config
    _: dc.KW_ONLY
    building: tuple[str, ...] | None = None
    plot: bool = True

    @functools.cached_property
    def output(self):
        output = self.conf.dirs.analysis / 'ECPM'
        output.mkdir(exist_ok=True)
        return output

    def fit(self, data: pl.DataFrame, case: _Case):
        logger.info(case.name)

        optimizer = _Optimizer(
            model=case.model,
            dataset=_Dataset(data, building=case.building, tvar=case.tvar),
            extra_weather=case.ext,
        )

        self.output.joinpath(f'{case} OLS.txt').write_text(optimizer.summary.as_text())

        if self.plot:
            scatter, residual = optimizer.plot()
            scatter.savefig(self.output / f'{case} scatter.png')
            scatter.savefig(self.output / f'{case} scatter.svg')
            residual.savefig(self.output / f'{case} residual.png')
            plt.close('all')

        lm = optimizer.linear_model
        rmse = np.sqrt(np.mean(np.square(lm.resid)))
        cvrmse = rmse / np.mean(lm.model.endog)
        s = {
            'r.squared': lm.rsquared,
            'r.squared.adj': lm.rsquared_adj,
            'p-value': lm.f_pvalue,
            'AIC': lm.aic,
            'BIC': lm.bic,
            'RMSE': rmse,
            'CV(RMSE)': cvrmse,
            **optimizer.stats(),
        }

        key = dc.asdict(case)
        return [{**key, 'variable': k, 'value': v} for k, v in s.items()]

    def __call__(self):
        (
            utils.mpl
            .MplTheme()
            .grid(show=False)
            .tick(which='both', direction='in', color='.5')
            .apply()
        )

        data = (
            pl
            .scan_parquet(list(self.conf.dirs.database.glob('DATA-*.parquet')))
            .filter(pl.col('holiday').not_())
            .collect()
        )
        match self.building:
            case None:
                buildings = data['building'].unique().sort().to_list()
            case _:
                buildings = list(self.building)

        cases = list(_Case.iter(buildings))
        stats_dicts = [self.fit(data, x) for x in tqdm(cases)]

        stats = pl.from_dicts(itertools.chain.from_iterable(stats_dicts))
        stats.write_csv(self.conf.dirs.analysis / '02.ecpm.stats.csv')
        stats.write_parquet(self.conf.dirs.analysis / '02.ecpm.stats.parquet')


def _order(it: Iterable):
    return {v: idx for idx, v in enumerate(it)}


@app.command
@dc.dataclass
class SummaryPlot:
    conf: Config

    @staticmethod
    def plot(data: pl.DataFrame):
        fig = Figure()
        axes = fig.subplots(1, 2, sharey=True)

        def plot(ax: Axes, data: pl.DataFrame):
            sns.scatterplot(data, x='value', y='case', hue='hue', ax=ax, alpha=0.8)

        plot(
            axes[0],
            data
            .filter(pl.col('variable').str.contains('r-squared'))
            .rename({'r2': 'hue'})
            .with_columns(),
        )
        plot(
            axes[1],
            data
            .filter(pl.col('stat') == 'r2', pl.col('period') != 'Base')
            .rename({'period': 'hue'})
            .with_columns(),
        )

        ax: Axes
        for ax, title in zip(axes, ['모델 $r^2$', '냉난방 기간별 $r^2$'], strict=True):
            ax.legend(title='')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(title, loc='left', weight=500)

            if title.startswith('모델'):
                ax.set_xlim(0.2, 1.0)
            else:
                ax.set_xlim(0, 1)

            for tl in ax.get_yticklabels():
                tl.set_horizontalalignment('left')

            ax.tick_params('y', pad=70)

        return fig

    def __call__(self):
        (
            utils.mpl
            .MplTheme(font={'math': 'cm'})
            .grid()
            .apply({'legend.fontsize': 'x-small', 'legend.borderpad': 0.2})
        )

        data = (
            pl
            .scan_parquet(self.conf.dirs.analysis / '02.ecpm.stats.parquet')
            .filter(pl.col('tvar') != 'Tiw')
            .with_columns(
                pl.col('model').replace({'ADD': 'Add', 'MULT': 'Mult'}),
                pl.col('tvar').replace_strict({
                    'Te': '',
                    'Te-Ti': r'$(\Delta T)$',
                }),
                pl.col('ext').str.replace('Pv', 'P_v'),
                pl
                .col('variable')
                .str.extract_groups(r'^season-(?P<period>[HCB])-(?P<stat>.*)$')
                .alias('m'),
            )
            .unnest('m')
            .with_columns(
                pl.concat_str(
                    'model',
                    'tvar',
                    pl.format(' ${}$', 'ext').fill_null(''),
                ).alias('case'),
                pl
                .col('variable')
                .replace_strict(
                    {'r-squared': '$r^2$', 'r-squared-adj': '$r^2_{adj}$'}, default=None
                )
                .alias('r2'),
                pl
                .col('period')
                .replace({'H': 'Heating', 'C': 'Cooling', 'B': 'Base'})
                .alias('period'),
                pl
                .col('period')
                .replace_strict({'H': 1, 'C': 0, 'B': 2}, return_dtype=pl.Int8)
                .alias('period-index'),
            )
            .sort(
                pl.col('model').replace_strict(_order(['CPM', 'Add', 'Mult'])),
                pl.col('tvar'),
                pl.col('ext').replace_strict(_order([None, 'I', 'P_v', 'I+P_v'])),
                'r2',
                'period-index',
            )
            .collect()
        )

        output = self.conf.dirs.analysis / 'r2'
        output.mkdir(exist_ok=True)

        for (bldg,), df in data.group_by('building', maintain_order=True):
            fig = self.plot(df)
            fig.savefig(output / f'r2.{bldg}.png')
            fig.savefig(output / f'r2.{bldg}.svg')


if __name__ == '__main__':
    warnings.filterwarnings(
        'ignore',
        message='divide by zero encountered in scalar divide',
    )
    warnings.filterwarnings(
        'ignore',
        message='The figure layout has changed to tight',
    )

    app()

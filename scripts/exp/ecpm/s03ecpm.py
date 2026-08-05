"""2026-08-04 ECPM 추가 분석 (ADD, MULT 제외)."""

import dataclasses as dc
import functools
import itertools
import typing
from typing import TYPE_CHECKING, ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.optimize as opt
import seaborn as sns
import statsmodels.api as sm
import structlog
from matplotlib.figure import Figure
from tqdm.rich import tqdm

from greenbutton import utils
from scripts.exp.ecpm.common import Config, app

if TYPE_CHECKING:
    from collections.abc import Sequence

    from statsmodels.regression.linear_model import RegressionResults

type Temperature = Literal['Te', 'dT']
type Energy = Literal['consumption', 'consumption+generation']
type ModelType = Literal[
    'CPM',
    'CPMw1',  # I+Pv
    'CPMw2',  # Ih+Ic+Pvh+Pvc
    'CPMw3',  # Ih+Pvc
    'dCPM',
    'dCPMw1',
    'dCPMw2',
    'dCPMw3',
]

ENERGY: tuple[Energy, ...] = typing.get_args(Energy.__value__)
MODEL_TYPE: tuple[ModelType, ...] = typing.get_args(ModelType.__value__)

logger = structlog.stdlib.get_logger()


@dc.dataclass
class Dataset:
    data: pl.DataFrame
    building: str
    xvar: tuple[str, ...] = ('Te', 'dT', 'Pv', 'I')
    yvar: Energy = 'consumption'

    def __post_init__(self):
        self.data = (
            self.data
            .filter(pl.col('building') == self.building)
            .with_columns((pl.col('Te') - pl.col('Tiw')).alias('dT'))
            .select([*self.xvar, self.yvar])
            .drop_nulls()
        )

    @functools.cached_property
    def y(self):
        return self.data[self.yvar].to_numpy()


def _r2_score(ytrue, ypred):
    return 1 - (
        (pl.col(ytrue) - pl.col(ypred)).pow(2).sum()
        / (pl.col(ytrue) - pl.mean(ytrue)).pow(2).sum()
    )


@dc.dataclass(frozen=True)
class _Case:
    building: str
    model: ModelType

    energy: Energy = 'consumption'

    @functools.cached_property
    def name(self):
        e = '' if self.energy == 'consumption' else '.c+g'
        return f'{self.building}.{self.model}{e}'

    def __str__(self):
        return self.name

    @functools.cached_property
    def t(self):
        return 'dT' if self.model.startswith('d') else 'Te'

    @functools.cached_property
    def exog(self):
        match self.model[-2:]:
            case 'w1':
                weather = ['I', 'Pv']
            case 'w2':
                weather = ['Ih', 'Ic', 'Pvh', 'Pvc']
            case 'w3':
                weather = ['Ih', 'Pvc']
            case _:
                weather = []

        exog = ['HDD', 'CDD', *weather]

        if self.model.startswith('d'):
            exog = [x if x in {'I', 'Pv'} else f'd{x}' for x in exog]

        return ('Const', *exog)

    @classmethod
    def iter(cls, buildings: Sequence[str]):
        for bldg, model in itertools.product(buildings, MODEL_TYPE):
            yield cls(bldg, model)


@dc.dataclass
class Model:
    dataset: Dataset

    t: Temperature
    exog: Sequence[str] = ('Const', 'HDD', 'CDD')

    XLABEL: ClassVar[dict[str, str]] = {
        'Te': 'External Temperature $T_{ext}$',
        'dT': r'Temperature Difference ($\Delta T = T_{ext} - T_{int})$',
    }
    _BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        'Te': (0.0, 25.0),
        'dT': (-20.0, 20.0),
    }

    def _fit_linear_model(self, cp: np.ndarray):
        te = pl.col('Te')
        dt = pl.col('dT')
        data = self.dataset.data.with_columns(
            pl.lit(1.0).alias('Const'),
            # Te
            pl.max_horizontal(pl.lit(0), cp[0] - te).alias('HDD'),
            pl.max_horizontal(pl.lit(0), te - cp[1]).alias('CDD'),
            ((te < cp[0]).cast(pl.Float64) * pl.col('Pv')).alias('Pvh'),
            ((te > cp[1]).cast(pl.Float64) * pl.col('Pv')).alias('Pvc'),
            ((te < cp[0]).cast(pl.Float64) * pl.col('I')).alias('Ih'),
            ((te > cp[1]).cast(pl.Float64) * pl.col('I')).alias('Ic'),
            # dT
            pl.max_horizontal(pl.lit(0), cp[0] - dt).alias('dHDD'),
            pl.max_horizontal(pl.lit(0), dt - cp[1]).alias('dCDD'),
            ((dt < cp[0]).cast(pl.Float64) * pl.col('Pv')).alias('dPvh'),
            ((dt > cp[1]).cast(pl.Float64) * pl.col('Pv')).alias('dPvc'),
            ((dt < cp[0]).cast(pl.Float64) * pl.col('I')).alias('dIh'),
            ((dt > cp[1]).cast(pl.Float64) * pl.col('I')).alias('dIc'),
        )
        exog = data.select(self.exog).to_numpy()
        return sm.OLS(endog=self.dataset.y, exog=exog).fit()

    def bound(self):
        match self.t:
            case 'Te':
                r = (0.0, 25.0)
            case 'dT':
                r = (-20.0, 20.0)

        return (r, r)

    @functools.cached_property
    def _penalty(self):
        y = self.dataset.y
        return np.sum(np.square(y - y.mean()))

    def _object(self, cp: np.ndarray) -> np.ndarray:
        model = self._fit_linear_model(cp)

        # t_h > t_c일 경우 패널티
        # TODO 빼고 테스트
        p = self._penalty * max(0, cp[0] - cp[1]) ** 2

        return np.sum(np.square(model.resid)) + np.square(p)

    def _optimize(self) -> opt.OptimizeResult:
        r = opt.differential_evolution(
            self._object,
            bounds=self.bound(),
            popsize=30,
            rng=42,
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
        return self._fit_linear_model(self.optimize_result.x)

    @functools.cached_property
    def summary(self):
        summary = self.linear_model.summary2(xname=self.exog)
        summary.add_text(f'change_points={self.optimize_result.x.round(2).tolist()}')

        return summary

    @functools.cached_property
    def period_data(self):
        t = pl.col(self.t)
        cp = self.optimize_result.x
        return self.dataset.data.with_columns(
            period=pl
            .when(t < cp[0])
            .then(pl.lit('heating'))
            .when(t > cp[1])
            .then(pl.lit('cooling'))
            .otherwise(pl.lit('baseload')),
            ypred=self.linear_model.fittedvalues,
            residual=self.linear_model.resid,
        )

    def plot(self, *, residual: bool = False):
        cp = self.optimize_result.x
        linear_model = self.linear_model

        # CPM
        fig = Figure()
        ax = fig.subplots()
        scatter = (
            self.dataset.data
            .select(
                self.t,
                pl.col(self.dataset.yvar).alias('Measured'),
                pl.Series('Predicted', linear_model.fittedvalues),
            )
            .unpivot(index=self.t)
            .with_columns()
        )
        ax.axhline(linear_model.params[0], ls=':', c='gray', alpha=0.5)
        ax.axvline(cp[0], ls=':', c='gray', alpha=0.5)
        ax.axvline(cp[1], ls=':', c='gray', alpha=0.5)
        sns.scatterplot(
            scatter,
            x=self.t,
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
        ax.set_xlabel(f'{self.XLABEL[self.t]} [°C]')
        ax.set_ylabel('EUI [kWh/m²]')
        ax.legend(title='', markerscale=2)

        # residual
        if not residual:
            grid = None
        else:
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
                .map_dataframe(
                    sns.scatterplot, x='value', y='residual', alpha=0.25, s=10
                )
                .set_axis_labels('', 'residual')
                .set_titles('{col_name} vs residual')
            )

        return fig, grid

    def stats(self):
        y = self.dataset.yvar
        stats = (
            self.period_data
            .group_by('period')
            .agg(
                _r2_score(y, 'ypred').alias('r2'),
                (pl.col('ypred') - pl.col(y)).pow(2).mean().sqrt().alias('RMSE'),
            )
            .unpivot(index='period')
            .with_columns(
                pl.format('season.{}.{}', 'period', 'variable').alias('variable'),
            )
        )
        return dict(zip(stats['variable'], stats['value'], strict=True))


@app.command
@dc.dataclass
class Fit:
    conf: Config
    _: dc.KW_ONLY
    building: tuple[str, ...] | None = ('KEPCO', 'KEA', 'EnergyX')
    plot: Literal['scatter', 'all'] | None = None

    @functools.cached_property
    def output(self):
        output = self.conf.dirs.analysis / 'ECPM.season'
        output.mkdir(exist_ok=True)
        return output

    @functools.cached_property
    def data(self):
        return (
            pl
            .scan_parquet(
                list(self.conf.dirs.database.glob('DATA-*.parquet')),
                missing_columns='insert',
            )
            .filter(pl.col('holiday').not_())
            .with_columns(
                (pl.col('consumption') + pl.col('generation')).alias(
                    'consumption+generation'
                )
            )
            .collect()
        )

    def _fit(self, case: _Case):
        dataset = Dataset(self.data, building=case.building, yvar=case.energy)

        model = Model(dataset=dataset, t=case.t, exog=case.exog)
        self.output.joinpath(f'{case}.OLS.txt').write_text(model.summary.as_text())

        if self.plot:
            scatter, residual = model.plot(residual=self.plot == 'all')
            scatter.savefig(self.output / f'{case}.scatter.png')
            if residual is not None:
                residual.savefig(self.output / f'{case}.residual.png')
                plt.close('all')

        lm = model.linear_model
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
            **model.stats(),
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

        match self.building:
            case None:
                buildings = self.data['building'].unique().sort().to_list()
            case _:
                buildings = list(self.building)

        cases = list(_Case.iter(buildings))
        stats_dicts = [self._fit(x) for x in tqdm(cases)]
        stats_dicts = [x for x in stats_dicts if x is not None]

        stats = pl.from_dicts(itertools.chain.from_iterable(stats_dicts))
        stats.write_parquet(self.conf.dirs.analysis / '02.ecpm.season.stats.parquet')
        stats.write_csv(self.conf.dirs.analysis / '02.ecpm.season.stats.csv')
        (
            stats
            .with_columns(pl.col('value').round_sig_figs(4))
            .pivot('variable', index=['building', 'model', 'energy'], values='value')
            .write_csv(self.conf.dirs.analysis / '02.ecpm.season.stats.wide.csv')
        )


@app.command
@dc.dataclass
class Evaluate:
    conf: Config

    @functools.cached_property
    def data(self):
        return (
            pl
            .scan_parquet(self.conf.dirs.analysis / '02.ecpm.season.stats.parquet')
            .with_columns(
                pl.format(
                    '${}$',
                    pl
                    .col('model')
                    .str.replace(r'^d', r'\Delta T \ ')
                    .str.replace_many({
                        'w1': '+I+P_v',
                        'w2': '+I_h+I_c+P_{vh}+P_{vc}',
                        'w3': '+I_h+P_{vc}',
                    }),
                ).alias('model'),
                pl
                .col('variable')
                .replace_strict(
                    {
                        'r.squared': 'r2',
                        'r.squared.adj': 'r2',
                        'season.heating.r2': 'r2',
                        'season.cooling.r2': 'r2',
                        'RMSE': 'RMSE',
                        'season.heating.RMSE': 'RMSE',
                        'season.cooling.RMSE': 'RMSE',
                        'AIC': 'InformationCriterion',
                        'BIC': 'InformationCriterion',
                    },
                    default=None,
                )
                .alias('group'),
                pl.col('variable').replace({
                    'r.squared': '$r^2$',
                    'r.squared.adj': '$r^2_{adj}$',
                    'season.heating.r2': 'Heating $r^2$',
                    'season.cooling.r2': 'Cooling $r^2$',
                    'season.heating.RMSE': 'Heating RMSE',
                    'season.cooling.RMSE': 'Cooling RMSE',
                }),
            )
            .collect()
        )

    @functools.cached_property
    def output(self):
        d = self.conf.dirs.analysis / '04.evaluation'
        d.mkdir(exist_ok=True)
        return d

    @staticmethod
    def plot(data: pl.DataFrame, hue: str | None = None, xlabel: str | None = None):
        fig = Figure()
        ax = fig.subplots()

        sns.scatterplot(data, x='value', y='model', hue=hue, ax=ax)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        ax.set_ylabel('')
        if (legend := ax.get_legend()) is not None:
            legend.set_title('')

        return fig

    def __call__(self):
        utils.mpl.MplTheme().grid().apply()

        data = self.data.drop_nulls('group')
        for (bldg, group), df in data.group_by(['building', 'group']):
            logger.info('bldg=%s, group=%s', bldg, group)

            fig = self.plot(df, hue='variable', xlabel=group)
            fig.savefig(self.output / f'{bldg}.{group}.png')


if __name__ == '__main__':
    app()

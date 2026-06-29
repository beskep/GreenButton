import dataclasses as dc
import functools
import typing
import warnings
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import polars as pl
import polars.selectors as cs
import scipy.optimize as opt
import seaborn as sns
import statsmodels.api as sm
import structlog
from matplotlib.figure import Figure
from rich.progress import track

from greenbutton import utils
from greenbutton.utils.cli import App

if TYPE_CHECKING:
    from statsmodels.iolib.summary import Summary
    from statsmodels.regression.linear_model import RegressionResults

type Xvar = Literal['Te', 'Te+Pv', 'Te+I', 'Te+Pv+I']

XVARS: tuple[Xvar, ...] = typing.get_args(Xvar.__value__)

logger = structlog.stdlib.get_logger()
app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml', root_keys='hybrid', use_commands_as_keys=False
    )
)


@dc.dataclass(frozen=True)
class _Cpm:
    data: pl.DataFrame
    xvar: Xvar = 'Te'

    _PENALTY: ClassVar[float] = 1e4

    @functools.cached_property
    def y(self):
        return self.data['EUI'].to_numpy()

    @functools.cached_property
    def exog(self):
        return ('ones', 'hdd', 'cdd', *(x for x in self.xvar.split('+') if x != 'Te'))

    def fit_linear_model(self, cp: np.ndarray):
        t = pl.col('Te')
        data = self.data.with_columns(
            pl.lit(1.0).alias('ones'),
            pl.max_horizontal(pl.lit(0), cp[0] - t).alias('hdd'),
            pl.max_horizontal(pl.lit(0), t - cp[1]).alias('cdd'),
        )
        x = data.select(self.exog).to_numpy()
        return sm.OLS(endog=self.y, exog=x).fit()

    @functools.cached_property
    def penalty(self):
        sse = np.sum(np.square(self.y - self.y.mean()))
        return sse * self._PENALTY

    def object(self, cp: np.ndarray) -> np.ndarray:
        model = self.fit_linear_model(cp)

        # t_h > t_c일 경우 패널티
        p1 = self.penalty * max(0, cp[0] - cp[1]) ** 2

        # 음수 beta 패널티
        beta: np.ndarray = model.params[1:3]
        p2 = self.penalty * np.sum(np.square(np.minimum(0, beta)))

        resid = np.append(model.resid, [p1, p2])
        return np.sum(np.square(resid))

    def _optimize(self) -> opt.OptimizeResult:
        r = opt.differential_evolution(self.object, bounds=((5, 20), (5, 20)), rng=42)
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

    def summary(self):
        xvars = ['const', '(Th-Te)^+', '(Te-Tc)^+', *self.xvar.split('+')]
        summary: Summary = self.linear_model.summary()
        summary.add_extra_txt([
            f'exog={xvars}',
            f'change_points={self.optimize_result.x.round(2).tolist()}',
        ])
        return summary

    def stats(self):
        lm = self.linear_model
        params = {f'param:{k}': v for k, v in zip(self.exog, lm.params, strict=True)}
        return {
            'observations_count': lm.nobs,
            'r2': lm.rsquared,
            'r2adj': lm.rsquared_adj,
            'aic': lm.aic,
            'bic': lm.bic,
            'p-value': lm.f_pvalue,
            **params,
        }

    def plot(self):
        cp = self.optimize_result.x
        lm = self.linear_model

        # CPM
        fig = Figure()
        ax = fig.subplots()
        scatter = (
            self.data
            .select(
                'Te',
                pl.col('EUI').alias('Measured'),
                pl.Series('Predicted', lm.fittedvalues),
            )
            .unpivot(index='Te')
            .with_columns()
        )
        ax.axhline(lm.params[0], ls=':', c='gray', alpha=0.5)
        ax.axvline(cp[0], ls=':', c='gray', alpha=0.5)
        ax.axvline(cp[1], ls=':', c='gray', alpha=0.5)
        sns.scatterplot(
            scatter,
            x='Te',
            y='value',
            hue='variable',
            style='variable',
            alpha=0.5,
            s=10,
            ax=ax,
        )
        ax.text(0.02, 0.05, f'$r^2={lm.rsquared:.4f}$', transform=ax.transAxes)
        ax.set_ylim(0)
        ax.set_xlabel('External Temperature [°C]')
        ax.set_ylabel('EUI [kWh/m²]')
        ax.legend(title='', markerscale=2)

        return fig


@app.command
@dc.dataclass
class Cpm:
    root: Path
    outlier_detection: Literal['IQR', 'LOF'] = 'IQR'
    min_n: int = 10

    @functools.cached_property
    def output(self):
        output = self.root / '03.CPM'
        output.mkdir(exist_ok=True)

        for v in XVARS:
            output.joinpath(v).mkdir(exist_ok=True)

        return output

    @functools.cached_property
    def raw(self):
        outlier = f'outlier_{self.outlier_detection.lower()}'
        index = ('type', 'building', 'date', 'Te', 'Pv', 'I')
        data = (
            pl
            .scan_parquet(self.root / '01.consumption.day.LOF.parquet')
            .filter(
                pl.col('holiday').not_(),
                pl.col(outlier).not_(),
            )
            .collect()
            .pivot('energy', index=index, values='EUI')
            .with_columns((pl.col('전력') + pl.col('열')).alias('합계'))
            .unpivot(index=index, variable_name='energy', value_name='EUI')
        )

        bldg_index = (
            data
            .select('type', 'building')
            .unique()
            .sort(pl.all())
            .with_row_index(offset=1)
        )

        return data.join(bldg_index, on=['type', 'building'], how='left')

    def _cpm(self, data: pl.DataFrame, xvar: Xvar):
        data = (
            (data)
            .filter(pl.col('energy') == '합계')
            .drop_nulls(['EUI', *xvar.split('+')])
        )

        if data.height < self.min_n:
            return None

        cpm = _Cpm(data, xvar=xvar)

        index = data['index'][0]
        type_ = data['type'][0]
        building = data['building'][0]
        name = f'{index:03d}.{type_}.{building}.CPM.{xvar}'

        output = self.output / xvar
        output.joinpath(f'{name}.txt').write_text(cpm.summary().as_text())
        cpm.plot().savefig(output / f'{name}.png')

        return {
            'index': index,
            'type': type_,
            'building': building,
            'xvar': xvar,
            'name': name,
            **cpm.stats(),
        }

    def __call__(self):
        index = ('type', 'building')
        bldg_count = self.raw.select(index).n_unique()

        def it():
            for _, data in self.raw.group_by(index):
                for xvar in XVARS:
                    if r := self._cpm(data, xvar=xvar):
                        yield r

        dicts = list(track(it(), total=bldg_count * len(XVARS)))
        models = pl.from_dicts(dicts)
        models.write_parquet(self.output / 'models.parquet')
        models.write_excel(self.output / 'models.xlsx')


@app.command
@dc.dataclass
class IEuiCorr(Cpm):
    """일사량-사용량 상관분석."""

    root: Path

    @staticmethod
    def corr(data: pl.DataFrame):
        index = data['index'][0]
        type_ = data['type'][0]
        building = data['building'][0]
        r = pg.corr(data['EUI'].to_numpy(), data['I'])
        return pl.from_pandas(r).select(
            pl.lit(index).alias('index'),
            pl.lit(type_).alias('type'),
            pl.lit(building).alias('building'),
            pl.all(),
        )

    def __call__(self):
        index = ('type', 'building')
        bldg_count = self.raw.select(index).n_unique()

        def it():
            for _, data in self.raw.group_by(index):
                yield self.corr(data)

        dicts = list(track(it(), total=bldg_count))
        r = pl.concat(dicts).sort(pl.all())

        r.write_excel(self.root / '04.stats/EUIvsI.xlsx')
        r.describe().write_excel(self.root / '04.stats/EUIvsI.desc.xlsx')

        return r


@app.command
@dc.dataclass
class Inspect:
    """CPM 모델 파라미터 분포."""

    root: Path

    @functools.cached_property
    def output(self):
        d = self.root / '04.stats'
        d.mkdir(exist_ok=True)
        return d

    @functools.cached_property
    def models(self):
        return (
            pl
            .scan_parquet(self.root / '03.CPM/models.parquet')
            .with_columns(
                pl.format(
                    '${}$',
                    pl.col('xvar').str.replace_many(['Te', 'Pv'], ['T_e', 'P_v']),
                ).alias('model')
            )
            .collect()
        )

    def stats(self):
        stats = (
            self.models
            .unpivot(cs.float(), index='model')
            .group_by('model', 'variable')
            .agg(pl.mean('value').alias('mean'), pl.std('value').alias('std'))
            .with_columns(
                pl
                .col('variable')
                .str.strip_prefix('param:')
                .replace({
                    'r2': '$r^2$',
                    'r2adj': '$r^2_adj$',
                    'aic': 'AIC',
                    'bic': 'BIC',
                    'ones': '$E_b$',
                    'hdd': r'$beta_H$',
                    'cdd': r'$beta_C$',
                    'Pv': r'$beta_{P_v}$',
                    'I': r'$beta_I$',
                })
                .alias('variable2')
            )
            .sort(pl.all())
        )
        stats.write_csv(self.output / 'stats.csv')
        table = (
            stats
            .with_columns(cs.float().round_sig_figs(3))
            .with_columns(pl.format('{}±{}', 'mean', 'std').alias('value'))
            .pivot('variable2', index='model', values='value', sort_columns=True)
            .rename(lambda x: f'[{x}]')
            .with_columns(pl.format('[{}]', pl.all().fill_null('-')))
        )

        table.select('[model]', cs.contains('r^2', 'AIC', 'BIC')).write_csv(
            self.output / 'stats.table.r2.csv', include_bom=True
        )
        table.select(~cs.contains('r^2', 'AIC', 'BIC')).write_csv(
            self.output / 'stats.table.params.csv', include_bom=True
        )

    def stats_grid(self, *, by_type: bool):
        data = (
            self.models
            .unpivot(['r2', 'r2adj', 'aic', 'bic'], index=('type', 'model'))
            .with_columns(
                pl.col('variable').replace({
                    'r2': '$r^2$',
                    'r2adj': '$r^2_{adj}$',
                    'aic': 'AIC',
                    'bic': 'BIC',
                })
            )
            .with_columns()
        )

        grid = sns.FacetGrid(
            data,
            row='model',
            col='variable',
            sharex='col',
            sharey=False,
            hue='type' if by_type else None,
            hue_order=('공공', '다소비') if by_type else None,
            margin_titles=True,
            aspect=16 / 9,
            height=2.5,
            despine=False,
        ).map_dataframe(sns.histplot, x='value', kde=True)

        grid.savefig(self.output / f'stats{".by-type" if by_type else ""}.png')
        plt.close(grid.figure)

    def beta_grid(self, *, by_type: bool):
        beta = (
            self.models
            .select('type', 'model', cs.starts_with('param:'))
            .unpivot(index=['type', 'model'])
            .with_columns(
                pl
                .col('variable')
                .str.strip_prefix('param:')
                .replace_strict({
                    'ones': '$E_b$',
                    'hdd': r'$\beta_H$',
                    'cdd': r'$\beta_C$',
                    'Pv': r'$\beta_{P_v}$',
                    'I': r'$\beta_I$',
                })
                .alias('variable')
            )
        )

        grid = sns.FacetGrid(
            beta,
            row='model',
            col='variable',
            sharex='col',
            sharey=False,
            hue='type' if by_type else None,
            hue_order=('공공', '다소비') if by_type else None,
            margin_titles=True,
            aspect=4 / 3,
            height=2.5,
            despine=False,
        ).map_dataframe(sns.histplot, x='value')

        grid.axes_dict['$T_e$', r'$\beta_{P_v}$'].set_xlim(-0.05, 0.05)
        grid.axes_dict['$T_e$', r'$\beta_I$'].set_xlim(-0.05, 0.05)

        grid.savefig(self.output / f'beta{".by-type" if by_type else ""}.png')
        plt.close(grid.figure)

    def __call__(self):
        self.stats()
        self.stats_grid(by_type=False)
        self.stats_grid(by_type=True)
        self.beta_grid(by_type=False)
        self.beta_grid(by_type=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='The figure layout has changed to tight')
    utils.mpl.MplTheme(font={'math': 'stix'}).grid().apply()
    app()

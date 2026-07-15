"""2026-07-23 중간보고 자료용 그래프 등."""

import dataclasses as dc
import functools
import itertools
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Literal

import cyclopts
import numpy as np
import polars as pl
import seaborn as sns
import structlog
from matplotlib.figure import Figure
from scipy import stats
from skmisc import loess
from tqdm.rich import tqdm

from greenbutton import utils
from greenbutton.utils.cli import App
from scripts.ami.hybrid.s03cpm import BatchCpm, Cpm, Energy

if TYPE_CHECKING:
    from matplotlib.axes import Axes

DIR = '99.presentation'

logger = structlog.stdlib.get_logger()
app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml', root_keys='hybrid', use_commands_as_keys=False
    )
)


@app.command
@dc.dataclass
class Anomaly:
    """이상치 처리 과정 설명."""

    root: Path

    def calculate(self, span: float = 0.5):
        # 0002.공공.경기도 고양시.전력.False
        # 0002.공공.경기도 고양시.열.False
        data = (
            pl
            .scan_parquet(self.root / '01.raw.daily.parquet')
            .filter(
                pl.col('bldg.case') == '0002.공공.경기도 고양시',
                pl.col('energy') == '열',
                pl.col('holiday').not_(),
                pl.col('eui') < 10,  # noqa: PLR2004
            )
            .sort('Te')
            .collect()
        )

        x = data.select('Te').to_numpy()
        y = data['eui'].to_numpy()

        # mean model
        mm = loess.loess(x, y)
        mm.model.span = span
        ypred = mm.predict(x).values

        # sigma model
        resid = np.abs(y - ypred)
        sm = loess.loess(x, resid)
        sm.model.span = span
        resid_pred = sm.predict(x).values

        ci = (
            stats.t.ppf(1 - (1 - 0.99) / 2, data.height - 2)
            * np.maximum(resid_pred, np.quantile(resid, 0.1))
            / np.sqrt(2 / np.pi)
        )

        return (
            data
            .with_columns(
                pl.Series('fit', ypred),
                pl.Series('resid', resid),
                pl.Series('resid_pred', resid_pred),
                pl.Series('lower', ypred - ci),
                pl.Series('upper', ypred + ci),
            )
            .with_columns(
                anomaly=pl
                .col('eui')
                .is_between('lower', 'upper')
                .replace_strict({True: 'Normal', False: 'Anomaly'})
            )
            .with_columns()
        )

    @staticmethod
    def plot(
        data: pl.DataFrame,
        lw: float = 2,
        la: float = 0.5,
    ):
        x = data['Te'].to_numpy()
        ypred = data['fit'].to_numpy()

        fig = Figure((32, 10, 'cm'))
        axes: tuple[Axes, ...] = fig.subplots(1, 3)

        sns.scatterplot(data, x='Te', y='eui', ax=axes[0], alpha=0.5, c='gray')
        axes[0].plot(x, ypred, alpha=la, lw=lw, c='k')

        sns.scatterplot(data, x='Te', y='resid', ax=axes[1], alpha=0.5, c='#283')
        axes[1].plot(x, data['resid_pred'].to_numpy(), alpha=la, lw=lw, c='k')

        sns.scatterplot(
            data,
            x='Te',
            y='eui',
            hue='anomaly',
            style='anomaly',
            ax=axes[2],
            alpha=0.5,
        )

        legend = axes[2].legend(title='')
        for handle in legend.legend_handles:
            if handle is None:
                continue

            handle.set_alpha(1.0)

        axes[2].plot(x, ypred, alpha=la, lw=lw, c='k')
        axes[2].plot(x, data['lower'].to_numpy(), alpha=la * 0.8, lw=lw, c='k', ls='--')
        axes[2].plot(x, data['upper'].to_numpy(), alpha=la * 0.8, lw=lw, c='k', ls='--')

        for idx, ax in enumerate(axes):
            title = ' 잔차' if idx == 1 else ''
            ylabel = ' Residual' if idx == 1 else ''
            ylabel = f'EUI{ylabel} [kWh/m²]'
            ax.set_title(f'일간 외기온 vs 사용량{title}', loc='left', weight=500)
            ax.set_xlabel('External Temperature [°C]')
            ax.set_ylabel(ylabel)

        return fig

    def __call__(self):
        data = self.calculate()
        fig = self.plot(data)

        output = self.root / DIR
        output.mkdir(exist_ok=True)
        fig.savefig(output / '01.anomaly.png')


@app.command
@dc.dataclass
class CpmByEnergy(BatchCpm):
    root: Path

    size: Literal[0, 1] = 0
    sizes: tuple[tuple[float, float], ...] = (
        (25.6, 8.0),
        (32.0, 6.5),
    )
    bldg_indices: tuple[int, ...] = (12, 23, 27, 31, 32, 33, 34, 35, 37, 39, 43, 46)

    @functools.cached_property
    def raw(self):
        return (
            super()
            .raw.with_columns(
                pl
                .col('bldg.case')
                .str.extract_groups(r'^(?P<index>\d+)\.(?:공공|다소비)\.(?P<bldg>.*)')
                .alias('group')
            )
            .unnest('group')
            .with_columns(
                pl.col('index').cast(pl.UInt16),
                pl.col('bldg').replace_strict(
                    {
                        '한국산업은행': '○○은행',
                        '한국농촌경제연구원': '○○연구원',
                        '계산현대아파트': 'G 공동주택',
                        '산본래미안하이어스아파트': 'S 공동주택',
                    },
                    default='',
                ),
            )
            .filter(pl.col('index').is_in(self.bldg_indices))
        )

    @staticmethod
    def _cpm_helper(data: pl.DataFrame, energy: Energy, ax: Axes):
        cpm = Cpm(data, energy)
        cp = cpm.optimize_result.x
        lm = cpm.linear_model

        data = data.sort('Te')
        ax.axvline(cp[0], ls=':', c='gray', alpha=0.5)
        ax.axvline(cp[1], ls=':', c='gray', alpha=0.5)
        sns.scatterplot(data, x='Te', y=f'EUI.{energy}', alpha=0.24, ax=ax)
        ax.plot(data['Te'].to_numpy(), lm.fittedvalues, '-', c='gray')

        ax.text(0.04, 0.05, f'$r^2={lm.rsquared:.4f}$', transform=ax.transAxes)
        ax.set_xlabel('External Temperature [°C]')
        ax.set_ylabel('EUI [kWh/m²]')

        bldg = data['bldg'][0]
        e = {'elec': '전력', 'heat': '열', 'total': '합계'}[energy]
        ax.set_title(f'{bldg} {e} 사용량 CPM'.strip(), loc='left', weight=600)

        return cpm

    def _cpm(self, data: pl.DataFrame, *_, **__):
        size = self.sizes[self.size]
        fig = Figure((*size, 'cm'))
        axes = fig.subplots(1, 3, sharex=True, sharey=True)

        bldg = data['bldg.case'][0]
        stats = []

        for energy, ax in zip(('elec', 'heat', 'total'), axes, strict=True):
            cpm = self._cpm_helper(data, energy, ax)
            stats.append({'bldg.case': bldg, 'energy': energy, **cpm.stats()})

        fig.get_axes()[0].set_ylim(0)
        fig.savefig(self.root / f'{DIR}/CPM/{size}_{bldg}.png')

        return stats

    def __call__(self):
        (
            utils.mpl
            .MplTheme(
                'paper',
                font={'math': 'cm'},
                font_scale=1.2,
            )
            .grid(show=False)
            .tick('xy', 'both', direction='in')
            .apply({'lines.solid_capstyle': 'butt'})
        )
        self.root.joinpath(f'{DIR}/CPM').mkdir(exist_ok=True)

        def it():
            for (c,), data in self.raw.group_by('bldg.case', maintain_order=True):
                logger.info('bldg.case=%s', c)
                if r := self._cpm(data):
                    yield r

        total = self.raw.select('bldg.case').n_unique()
        dicts = list(tqdm(it(), total=total))
        models = pl.from_dicts(itertools.chain.from_iterable(dicts))
        models.write_csv(self.root / f'{DIR}/02.cpm-energy.csv', include_bom=True)


@app.command
@dc.dataclass
class ByRegion:
    root: Path

    @functools.cached_property
    def data(self):
        region = (
            pl
            .scan_parquet(self.root / '01.bldg.parquet')
            .select('bldg.case', 'bldg.region', 'bldg.latitude', 'bldg.longitude')
            .rename(lambda x: x if x == 'bldg.case' else x.removeprefix('bldg.'))
            .collect()
        )
        return (
            pl
            .scan_parquet(self.root / '03.CPM/CPM.parquet')
            .filter(pl.col('endog') == 'total', pl.col('exog') == 'Te+Pv+I')
            .unpivot(
                ['param:Pv', 'param:I'],
                index=['bldg.case', 'r2', 'r2adj'],
                variable_name='param',
            )
            .drop_nulls('param')
            .with_columns(pl.col('param').str.strip_prefix('param:'))
            .with_columns(
                pl.format(
                    '{} 계수',
                    pl.col('param').replace_strict({'I': '$I$', 'Pv': '$P_v$'}),
                ).alias('param.eq'),
            )
            .collect()
            .join(region, on='bldg.case', how='left', validate='m:1')
        )

    @functools.cached_property
    def output(self):
        return self.root / DIR

    def plot(self, param: str):
        data = self.data.filter(pl.col('param') == param)
        eq = data['param.eq'][0]

        fig = Figure()
        ax = fig.subplots()
        sns.histplot(data, x='value', hue='region', ax=ax, stat='probability', kde=True)
        ax.set_xlabel(eq)
        fig.savefig(self.output / f'03.param.{param}.region.png')

        fig = Figure()
        ax = fig.subplots()
        sns.scatterplot(data, x='latitude', y='value', ax=ax, alpha=0.8)
        ax.set_xlabel('위도 [°]')
        ax.set_ylabel(eq)
        fig.savefig(self.output / f'03.param.{param}.latitude.png')

    def __call__(self):
        (
            utils.mpl
            .MplTheme(font={'math': 'cm'}, fig_size=(12, 9))
            .grid()
            .apply({'lines.solid_capstyle': 'butt'})
        )
        for param in ('I', 'Pv'):
            self.plot(param)


if __name__ == '__main__':
    (
        utils.mpl
        .MplTheme(font={'math': 'cm'})
        .grid()
        .apply({'lines.solid_capstyle': 'butt'})
    )
    app()

import dataclasses as dc
import functools
import itertools

import cyclopts
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure

from greenbutton import utils
from scripts.exp.ecpm.common import Config, app
from scripts.exp.ecpm.s02ecpm import (
    _Dataset,  # ruff:ignore[import-private-name]
    _Model,  # ruff:ignore[import-private-name]
    _Optimizer,  # ruff:ignore[import-private-name]
)


@cyclopts.Parameter(name='*')
@dc.dataclass
class _Config(Config):
    @functools.cached_property
    def output(self):
        d = self.dirs.root / '99.presentation'
        d.mkdir(exist_ok=True)
        return d


@app.command
def stats(conf: _Config):
    v = pl.col('variable')
    (
        pl
        .scan_parquet(conf.dirs.analysis / '02.ecpm.stats.parquet')
        .with_columns(v.replace({'season.H.r2': 'r2.h', 'season.C.r2': 'r2.c'}))
        .filter(
            v.is_in(['r.squared', 'r.squared.adj', 'BIC', 'CV(RMSE)', 'r2.h', 'r2.c']),
            (pl.col('ext').is_null() | (pl.col('ext') == 'I+Pv')),
            pl.col('tvar') != 'Tiw',
        )
        .with_columns(
            case=pl.concat_str('model', 'tvar', separator='.').replace_strict({
                'CPM.Te': 'CPM',
                'MULT.Te': 'Model 1',
                'ADD.Te': 'Model 2',
                'CPM.Te-Ti': 'Model 3',
            })
        )
        .collect()
        .pivot(
            'variable',
            index=('building', 'case', 'model', 'tvar', 'ext'),
            values='value',
        )
        .with_columns(
            pl.col('r.squared', 'r.squared.adj', 'r2.h', 'r2.c').round(3),
            pl.col('BIC').round(1),
            pl.format('{}%', pl.col('CV(RMSE)').mul(100).round(2)),
        )
        .sort('building', 'ext', 'case')
        .write_csv(conf.output / 'ecpm.stats.table.csv', include_bom=True)
    )


@app.command
@dc.dataclass
class CPM:
    conf: _Config

    @functools.cached_property
    def data(self):
        return (
            pl
            .scan_parquet([
                self.conf.dirs.database / f'DATA-{x}.parquet' for x in ['KEPCO', 'KEA']
            ])
            .filter(pl.col('holiday').not_())
            .drop_nulls(['EUI', 'Te', 'I', 'Pv'])
            .collect()
        )

    def _cpm_plot(self, optimizer: _Optimizer):
        data = (
            self.data
            .filter(pl.col('building') == optimizer.dataset.building)
            .rename({'EUI': 'Measured'})
            .with_columns(
                (pl.col('Te') - pl.col('Ti')).alias('Te-Ti'),
                pl.Series('Predicted', optimizer.linear_model.fittedvalues),
            )
            .unpivot(['Measured', 'Predicted'], index=['Te', 'Te-Ti'])
        )
        fig = Figure((12, 9, 'cm'))
        ax = fig.subplots()
        sns.scatterplot(
            data,
            x=optimizer.dataset.tvar,
            y='value',
            hue='variable',
            style='variable',
            hue_order=['Measured', 'Predicted'],
            style_order=['Measured', 'Predicted'],
            ax=ax,
            alpha=0.5,
            s=10,
        )

        cp = optimizer.optimize_result.x
        ax.axvline(cp[0], ls=':', c='gray', alpha=0.5)
        ax.axvline(cp[1], ls=':', c='gray', alpha=0.5)

        ax.set_ylim(0)

        match optimizer.dataset.tvar:
            case 'Te':
                ax.set_xlabel(r'External Temperature $(T_e)$')
            case 'Te-Ti':
                ax.set_xlabel(r'Temperature Difference $(\Delta T = T_e - T_i)$')

        ax.set_ylabel('EUI [kWh/m²]')

        ax.text(
            0.04,
            0.05,
            f'$r^2={optimizer.linear_model.rsquared:.4f}$',
            transform=ax.transAxes,
        )

        legend = ax.legend(title='', markerscale=2)
        for handle in legend.legend_handles:
            if handle is None:
                continue

            handle.set_alpha(1.0)

        return fig

    def cpm_plot(self):
        (
            utils.mpl
            .MplTheme(font={'math': 'cm'})
            .grid(show=False)
            .tick('xy', 'both', direction='in')
            .apply()
        )

        for bldg, (name, tvar, extra_weather) in itertools.product(
            ('KEPCO', 'KEA'),
            (
                ('CPM', 'Te', None),
                ('CPM+I&Pv', 'Te', 'I+Pv'),
                ('Model3+I&Pv', 'Te-Ti', 'I+Pv'),
            ),
        ):
            dataset = _Dataset(self.data, building=bldg, tvar=tvar)
            optimizer = _Optimizer(_Model.CPM, dataset, extra_weather=extra_weather)

            path = self.conf.output / f'CPM.{bldg}.{name}.txt'
            path.write_text(optimizer.summary.as_text())
            fig = self._cpm_plot(optimizer)
            fig.savefig(path.with_suffix('.png'))

    def _residual_plot(self, optimizer: _Optimizer, v: str):
        x = (
            self.data
            .filter(pl.col('building') == optimizer.dataset.building)
            .select(v)
            .to_series()
            .to_numpy()
        )
        y = optimizer.linear_model.resid

        fig = Figure((9, 9, 'cm'))
        ax = fig.subplots()

        sns.regplot(
            x=x,
            y=y,
            ax=ax,
            scatter_kws={'color': 'slategray', 'alpha': 0.25, 's': 5},
            line_kws={'color': 'darkslategray', 'alpha': 0.75},
        )

        match v:
            case 'I':
                xlabel = '일사량 ($I$, $MJ/m^2$)'
            case 'Pv':
                xlabel = '수증기 분압 ($P_v$, $Pa$)'
            case _:
                raise ValueError

        ax.set_xlabel(xlabel)
        ax.set_ylabel('CPM 잔차')

        return fig

    def residual_plot(self):
        utils.mpl.MplTheme(font={'math': 'cm'}).grid().apply()
        dataset = _Dataset(self.data, building='KEPCO', tvar='Te')
        optimizer = _Optimizer(_Model.CPM, dataset)
        for v in ('I', 'Pv'):
            fig = self._residual_plot(optimizer, v)
            fig.savefig(self.conf.output / f'residual.{v}.png')

    def __call__(self):
        self.cpm_plot()
        self.residual_plot()


if __name__ == '__main__':
    app()

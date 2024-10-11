from __future__ import annotations

import dataclasses as dc
import warnings
from typing import TYPE_CHECKING, ClassVar, Literal

import polars as pl
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from greenbutton import sensors
from scripts.config import Config, _ExpDir, sensor_location

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dc.dataclass
class PlotOption:
    height: float = 2
    aspect: float = 9 / 16
    fig_size: tuple[float, float] | None = (24, 13.5)

    linewidth: float = 2
    alpha: float = 0.8

    INCH: ClassVar[float] = 2.54

    def fig_size_inches(self):
        if not self.fig_size:
            raise ValueError

        return (self.fig_size[0] / self.INCH, self.fig_size[1] / self.INCH)


@dc.dataclass
class Experiment:
    """각 실험 공통적인 센서 데이터 추출, 시각화."""

    building: str
    date: str | None = None

    conf: Config = dc.field(default_factory=Config.read)
    loc: pl.DataFrame = dc.field(default_factory=sensor_location)
    dirs: _ExpDir = dc.field(init=False)

    def __post_init__(self):
        self.dirs = self.conf.experiment.directory(
            building=self.building, date=self.date
        )

        self.loc = self.loc.filter(
            pl.col('building') == self.building,
            (pl.col('date') == self.date) if self.date else pl.lit(1),
        )

        if self.loc.height == 0:
            msg = 'sensor location not found'
            raise ValueError(msg)

        warnings.filterwarnings(
            'ignore', category=UserWarning, module='seaborn.axisgrid'
        )

    def convert_tr7(
        self,
        *,
        write=True,
        id_pattern: str = r'.*TR(\d+).*',
    ):
        if not (paths := list(self.dirs.TR7.glob('*.csv'))):
            raise FileNotFoundError(self.dirs.TR7)

        loc = ['floor', 'point', 'space']
        dfs = (
            sensors.read_tr7(x).with_columns(pl.lit(x.name).alias('file'))
            for x in paths
        )
        df = (
            pl.concat(dfs)
            .with_columns(
                pl.col('file').str.extract(id_pattern).cast(pl.UInt8).alias('id')
            )
            .join(self.loc.select(pl.col('TR').alias('id'), *loc), on='id', how='left')
            .select(*loc, 'id', 'datetime', 'variable', 'value', 'unit')
            .sort(pl.all())
        )

        if write:
            df.write_parquet(self.dirs.ROOT / '[DATA] TR7.parquet')

        return df

    def convert_pmv(
        self,
        sensor: Literal['testo', 'deltaohm'],
        *,
        write=True,
        id_pattern: str = r'.*PMV(\d+).*',
    ):
        files = self.dirs.PMV.glob('*.csv')

        if sensor == 'testo':
            probes = sensors.TestoPMV('').probe_dict()
            dfs = (
                sensors.TestoPMV(x, probes=probes).dataframe.with_columns(
                    pl.lit(x.name).alias('file')
                )
                for x in files
            )
        else:
            dfs = (
                sensors.DeltaOhmPMV(x).dataframe.with_columns(
                    pl.lit(x.name).alias('file')
                )
                for x in files
            )

        df = (
            pl.concat(dfs)
            .with_columns(
                pl.col('file').str.extract(id_pattern).cast(pl.UInt8).alias('id')
            )
            .join(
                self.loc.select(pl.col('PMV').alias('id'), 'floor', 'point', 'space'),
                on='id',
                how='left',
            )
            .drop('file')
        )

        head = ['floor', 'point', 'space', 'id', 'datetime']
        df = df.select([*head, *(x for x in df.columns if x not in head)])

        if write:
            df.write_parquet(self.dirs.ROOT / '[DATA] PMV.parquet')

        return df

    @staticmethod
    def plot_tr7(data: pl.DataFrame, option: PlotOption | None = None):
        data = data.sort('floor', descending=True).with_columns(
            pl.format('{}층', 'floor').alias('floor')
        )
        floor_count = data.select('floor').n_unique()
        option = option or PlotOption()

        for var in ['T', 'RH']:
            grid = (
                sns.FacetGrid(
                    data.filter(pl.col('variable') == var),
                    row='floor',
                    height=option.height,
                    aspect=floor_count / option.aspect,
                    despine=False,
                )
                .map_dataframe(
                    sns.lineplot,
                    x='datetime',
                    y='value',
                    hue='space',
                    alpha=option.alpha,
                )
                .set_axis_labels('', '온도 [ºC]' if var == 'T' else '상대습도')
                .set_titles('')
                .set_titles('{row_name}', loc='left', weight='bold')
            )

            ax: Axes
            for ax in grid.axes.flat:
                ax.legend()
                if var == 'RH':
                    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

            if legend := grid.legend:
                legend.set_title('')

            grid.figure.set_layout_engine('constrained', hspace=0.05)
            if option.fig_size:
                grid.figure.set_size_inches(option.fig_size_inches())

            yield grid, var

    @staticmethod
    def plot_pmv(
        data: pl.DataFrame,
        variables=('PMV', '온도', '흑구온도', '상대습도', '기류'),
        *,
        comfort_range=True,
        option: PlotOption | None = None,
    ):
        data = (
            data.filter(pl.col('variable').is_in(variables))
            .with_columns(pl.format('{}층 {}', 'floor', 'space').alias('space'))
            .sort('space', 'datetime')
        )
        option = option or PlotOption()

        grid = (
            sns.relplot(
                data,
                x='datetime',
                y='value',
                hue='space',
                row='variable',
                row_order=variables,
                kind='line',
                alpha=option.alpha,
                height=option.height,
                aspect=len(variables) / option.aspect,
                facet_kws={'sharey': False, 'despine': False, 'legend_out': False},
            )
            .set_titles('')
            .set_xlabels('')
        )

        units = {
            r[0]: r[1] for r in data.select('variable', 'unit').unique().iter_rows()
        }

        ax: Axes
        for ax, row in zip(grid.axes.flat, grid.row_names, strict=True):
            unit = units.get(row)
            ax.set_ylabel(f'{row} [{unit}]' if unit else row)

            if comfort_range and row == 'PMV':
                lim = ax.get_ylim()
                ax.axhspan(-0.5, 0.5, color='#1E88E5', alpha=0.25, linewidth=0)
                ax.set_ylim(lim)
            elif row == '상대습도':
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

        if legend := grid.legend:
            legend.set_title('')

        grid.figure.set_layout_engine('constrained', hspace=0.05)
        if option.fig_size:
            grid.figure.set_size_inches(option.fig_size_inches())

        return grid

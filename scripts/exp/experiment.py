from __future__ import annotations

import dataclasses as dc
import functools
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from greenbutton import sensors

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.typing import ColorType

    from greenbutton.sensors import Source


@dc.dataclass
class Dirs:
    """
    실증 분석 폴더 구조.

    Building
    ├── 01.sensor
    │   ├── raw
    │   │   ├── 2000-01-01_TR01*.csv
    │   │   ├── 2000-01-01_TR02*.csv
    │   │   ├── 2000-01-01_PMV01*.csv
    │   │   ├── 2000-01-01_PMV02*.csv
    │   │   ├── TR01*.csv
    │   │   ├── PMV01*.csv
    │   │   └── ...
    │   ├── TR7.parquet
    │   ├── PMV.parquet
    │   ├── 2000-01-01_TR7.xlsx
    │   ├── 2000-01-01_PMV.xlsx
    │   └── ...
    ├── 02.database
    │   └── ...
    └── 03.analysis
        ├── 2000-01-01_TR7_T.png
        ├── 2000-01-01_TR7_RH.png
        ├── 2000-01-01_PMV.png
        └── ...
    """

    root: Path

    sensor: Path = Path('01.sensor')
    database: Path = Path('02.database')
    analysis: Path = Path('03.analysis')

    def __post_init__(self):
        self.update()

    def update(self):
        for field in (f.name for f in dc.fields(self)):
            if field == 'root':
                continue

            p = getattr(self, field)
            setattr(self, field, self.root / p)

        return self

    @property
    def sensor_raw(self):
        return self.sensor / 'raw'

    def mkdir(self):
        self.root.mkdir(exist_ok=True)

        for field in (f.name for f in dc.fields(self)):
            if field == 'root':
                continue

            p: Path = getattr(self, field)
            p.mkdir(exist_ok=True)

        self.sensor_raw.mkdir(exist_ok=True)


@dc.dataclass
class BaseConfig:
    buildings: dict[str, str] = dc.field(default_factory=dict)
    root: Path = Path()

    BUILDING: ClassVar[str]

    @functools.cached_property
    def dirs(self):
        try:
            name = self.buildings[self.BUILDING]
        except KeyError as e:
            msg = (
                f'Unknown building: {self.BUILDING}. '
                f'Available buildings: {set(self.buildings.keys())}'
            )
            raise ValueError(msg) from e

        return Dirs(root=self.root / name)

    def experiment(self):
        return Experiment(conf=self)


def sensor_location(
    path: str | Path = 'config/sensor_location.json',
    *,
    from_xlsx=False,
):
    path = Path(path)
    xlsx = path.with_suffix('.xlsx')
    schema_overrides = dict.fromkeys(['floor', 'point', 'PMV', 'TR', 'GT'], pl.UInt8)

    if from_xlsx or not path.exists() or (xlsx.stat().st_mtime > path.stat().st_mtime):
        data = (
            pl.read_excel(xlsx, schema_overrides=schema_overrides)
            .with_columns(pl.col('date'))
            .with_columns()
        )
        data.write_json(path)
    else:
        data = (
            pl.read_json(path, schema_overrides=schema_overrides)
            .with_columns(pl.col('date').str.to_date())
            .with_columns()
        )

    return data


def _read_pmv(source: Source, **kwargs):
    try:
        return sensors.TestoPMV(source, **kwargs).dataframe
    except sensors.DataFormatError:
        return sensors.DeltaOhmPMV(source, **kwargs).dataframe


@dc.dataclass
class PlotOption:
    height: float = 2
    aspect: float = 9 / 16
    fig_size: tuple[float, float] | None = (24, 13.5)

    linewidth: float = 2
    alpha: float = 0.8
    comfort_range_color: ColorType = '#42A5F5'

    INCH: ClassVar[float] = 2.54

    def fig_size_inches(self):
        if not self.fig_size:
            raise ValueError

        return (self.fig_size[0] / self.INCH, self.fig_size[1] / self.INCH)


@dc.dataclass
class Experiment:
    """각 실험 공통적인 센서 데이터 추출, 시각화."""

    conf: BaseConfig

    loc: pl.DataFrame = dc.field(default_factory=sensor_location)

    def __post_init__(self):
        self.loc = self.loc.filter(pl.col('building') == self.conf.BUILDING)

        if self.loc.height == 0:
            msg = f'Sensor location not found. building={self.conf.BUILDING}'
            raise ValueError(msg)

    def _sources(
        self,
        sources: Iterable[Source] | None,
        pattern: str,
        directory: Path | None = None,
    ):
        directory = directory or self.conf.dirs.sensor_raw

        if sources is not None:
            src = list(sources)
        else:
            p = re.compile(pattern)
            src = [x for x in directory.glob('*') if p.match(x.name)]

            if not src:
                raise FileNotFoundError(directory)

        return src

    def read_pmv(
        self,
        sources: Iterable[Source] | None = None,
        pattern: str = r'((?P<date>\d{4}\-\d{2}\-\d{2})_)?PMV(?P<id>\d+).*\.(csv|dlg)',
    ) -> pl.DataFrame:
        """
        PMV (testo, DeltaOhm) 측정 결과 읽기.

        Parameters
        ----------
        sources : Iterable[Source] | None, optional
        pattern : str, optional
            기본 형식: "2000-01-01_PMV01(...).csv", "2000-01-01_PMV02(...).dlg"

        Returns
        -------
        pl.DataFrame
        """
        src = self._sources(sources=sources, pattern=pattern)
        data = pl.concat(
            _read_pmv(x).with_columns(pl.lit(x.name).alias('file')) for x in src
        )

        columns = data.columns
        loc = ['floor', 'point', 'space']

        data = (
            data.with_columns(
                pl.col('file').str.extract_groups(pattern).alias('matches')
            )
            .with_columns(
                pl.col('matches').struct['id'].cast(pl.UInt8).alias('id'),
                pl.col('matches').struct['date'].str.to_date().alias('date'),
            )
            .join(
                self.loc.select(pl.col('PMV').alias('id'), 'date', *loc),
                on=['id', 'date'],
                how='left',
            )
        )

        return data.select('date', *loc, 'id', *columns).drop('file').sort(pl.all())

    def read_tr7(
        self,
        sources: Iterable[Source] | None = None,
        pattern: str = r'((?P<date>\d{4}\-\d{2}\-\d{2})_)?TR(?P<id>\d+).*\.csv',
    ) -> pl.DataFrame:
        """
        TR7 센서 측정 결과 (csv) 읽기.

        Parameters
        ----------
        sources : Iterable[Source] | None, optional
        pattern : str, optional
            기본 형식: "2000-01-01_TR01(...).csv"

        Returns
        -------
        pl.DataFrame
        """
        src = self._sources(sources=sources, pattern=pattern)
        data = pl.concat(
            sensors.read_tr7(x).with_columns(pl.lit(x.name).alias('file')) for x in src
        )

        columns = data.columns
        loc = ['floor', 'point', 'space']

        data = (
            data.with_columns(
                pl.col('file').str.extract_groups(pattern).alias('matches')
            )
            .with_columns(
                pl.col('matches').struct['id'].cast(pl.UInt8).alias('id'),
                pl.col('matches').struct['date'].str.to_date().alias('date'),
            )
            .join(
                self.loc.select(pl.col('TR').alias('id'), 'date', *loc),
                on=['id', 'date'],
                how='left',
            )
        )

        return data.select('date', *loc, 'id', *columns).drop('file').sort(pl.all())

    def parse_sensors(
        self,
        *,
        write_parquet: bool = True,
        write_xlsx: bool = True,
    ):
        path = self.conf.dirs.sensor
        for sensor in ['PMV', 'TR7']:
            data = self.read_pmv() if sensor == 'PMV' else self.read_tr7()

            if write_parquet:
                data.write_parquet(path / f'{sensor}.parquet')

            if write_xlsx:
                for by, df in data.group_by('date'):
                    df.write_excel(path / f'{by[0]} {sensor}.xlsx')

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
                data.to_pandas(),
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
                ax.axhspan(
                    -0.5,
                    0.5,
                    color=option.comfort_range_color,
                    alpha=0.25,
                    linewidth=0,
                )
                ax.set_ylim(lim)
            elif row == '상대습도':
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

        if legend := grid.legend:
            legend.set_title('')

        grid.figure.set_layout_engine('constrained', hspace=0.05)
        if option.fig_size:
            grid.figure.set_size_inches(option.fig_size_inches())

        return grid

    @staticmethod
    def plot_tr7(data: pl.DataFrame, option: PlotOption | None = None):
        data = (
            data.sort('floor', descending=True)
            .with_columns(pl.format('{}층', 'floor').alias('floor'))
            .with_columns(pl.format('P{} {}', 'point', 'space').alias('space'))
        )
        floor_count = data.select('floor').n_unique()
        option = option or PlotOption()

        for var in ['T', 'RH']:
            grid = (
                sns.FacetGrid(
                    data.filter(pl.col('variable') == var).to_pandas(),
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

    def _plot_pmv(self):
        data = pl.read_parquet(self.conf.dirs.sensor / 'PMV.parquet')

        for (date,), df in data.group_by('date'):
            grid = self.plot_pmv(df)
            grid.savefig(self.conf.dirs.analysis / f'{date}_PMV.png')
            plt.close(grid.figure)

    def _plot_tr7(self):
        data = pl.read_parquet(self.conf.dirs.sensor / 'TR7.parquet')

        for (date,), df in data.group_by('date'):
            for grid, var in self.plot_tr7(df):
                grid.savefig(self.conf.dirs.analysis / f'{date}_TR7_{var}.png')
                plt.close(grid.figure)

    def plot_sensors(self, *, pmv: bool = True, tr7: bool = True):
        self.conf.dirs.analysis.mkdir(exist_ok=True)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, module='seaborn.axisgrid'
            )

            if pmv:
                self._plot_pmv()

            if tr7:
                self._plot_tr7()

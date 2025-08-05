from __future__ import annotations

import contextlib
import dataclasses as dc
import datetime as dt
import functools
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import msgspec
import polars as pl
import seaborn as sns
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter

from greenbutton import misc, sensors
from greenbutton.utils.mpl import MplTheme

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

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
        data = pl.read_excel(xlsx, schema_overrides=schema_overrides)
        path.write_text(msgspec.json.format(data.write_json()), encoding='utf-8')
    else:
        data = (
            pl.read_json(path, schema_overrides=schema_overrides)
            .with_columns(pl.col('date').str.to_date())
            .with_columns()
        )

    return data


def _read_pmv(source: Source, **kwargs):
    try:
        return sensors.TestoPMV(source, **kwargs).data
    except sensors.DataFormatError:
        return sensors.DeltaOhmPMV(source, **kwargs).data


class HolidayMarker:
    DEFAULT_BORDER_STYLE: ClassVar[dict] = {
        'color': 'tab:red',
        'linestyle': '--',
        'alpha': 0.2,
    }
    DEFAULT_FILL_STYLE: ClassVar[dict] = {'color': 'tab:red', 'alpha': 0.1}

    def __init__(
        self,
        datetimes: Sequence[dt.datetime] | pl.Series,
        *,
        border: bool = False,
        fill: bool = True,
        border_style: dict | None = None,
        fill_style: dict | None = None,
    ):
        if isinstance(datetimes, pl.Series):
            datetimes = datetimes.cast(pl.Datetime)
            dtmin = datetimes.min()
            dtmax = datetimes.max()
            assert isinstance(dtmin, dt.datetime)
            assert isinstance(dtmax, dt.datetime)
        else:
            dtmin = min(datetimes)
            dtmax = max(datetimes)

        years = list(range(dtmin.year, dtmax.year + 1))
        is_holiday = misc.is_holiday(pl.col('datetime'), years=years)
        h = pl.col('holiday')

        self.dates = (
            pl.datetime_range(
                (dtmin - dt.timedelta(1)).date(),
                (dtmax + dt.timedelta(1)).date(),
                interval='12h',
                eager=True,
            )
            .alias('datetime')
            .to_frame()
            .with_columns(is_holiday.alias('holiday'))
            .with_columns(
                h.xor(h.shift()).alias('border'),
                (h | (h.shift() & ~h)).alias('fill'),
            )
        )

        self.fill = fill
        self.border = border
        self.border_style = border_style or self.DEFAULT_BORDER_STYLE
        self.fill_style = fill_style or self.DEFAULT_FILL_STYLE

    def _mark(self, ax: Axes):
        xlim = [mdates.num2date(x).replace(tzinfo=None) for x in ax.get_xlim()]
        dates = self.dates.filter(pl.col('datetime').is_between(*xlim))

        if self.border:
            for x in dates.filter(pl.col('border'))['datetime']:
                ax.axvline(x, **self.border_style)

        if self.fill:
            ylim = ax.get_ylim()
            ax.fill_between(
                x=dates['datetime'].to_numpy(),
                y1=ylim[0],
                y2=ylim[1],
                where=dates['fill'].to_list(),
                **self.fill_style,
            )
            ax.set_ylim(ylim)

    def __call__(self, axes: Axes | Iterable[Axes]):
        for ax in [axes] if isinstance(axes, Axes) else axes:
            self._mark(ax)

    mark = __call__


@dc.dataclass
class HolidayMarkerStyle:
    border: bool = False
    fill: bool = True
    border_style: dict = dc.field(
        default_factory=lambda: HolidayMarker.DEFAULT_BORDER_STYLE
    )
    fill_style: dict = dc.field(
        default_factory=lambda: HolidayMarker.DEFAULT_FILL_STYLE
    )

    def marker(self, datetimes: Sequence[dt.datetime] | pl.Series):
        return HolidayMarker(datetimes, **dc.asdict(self))


@dc.dataclass
class SensorPlotStyle:
    height: float = 2
    aspect: float = 9 / 16
    fig_size: tuple[float, float] | None = (24, 13.5)

    linewidth: float = 2
    alpha: float = 0.8

    comfort_range: bool = True
    comfort_range_color: ColorType = '#42A5F564'
    holidays: HolidayMarkerStyle | None = dc.field(default_factory=HolidayMarkerStyle)
    max_minor_ticks: int = 80

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
        sources: Iterable[str | Path] | None,
        pattern: str,
        directory: Path | None = None,
    ):
        directory = directory or self.conf.dirs.sensor_raw

        if sources is not None:
            src = [Path(x) for x in sources]
        else:
            p = re.compile(pattern)
            src = [x for x in directory.glob('*') if p.match(x.name)]

            if not src:
                raise FileNotFoundError(directory)

        return src

    def read_pmv(
        self,
        sources: Iterable[str | Path] | None = None,
        pattern: str = r'((?P<date>\d{4}\-\d{2}\-\d{2})_)?PMV(?P<id>\d+).*\.(csv|dlg)',
    ) -> pl.DataFrame:
        """
        PMV (testo, DeltaOhm) 측정 결과 읽기.

        Parameters
        ----------
        sources : Iterable[str | Path] | None, optional
        pattern : str, optional
            기본 형식: "2000-01-01_PMV01(...).csv", "2000-01-01_PMV02(...).dlg"

        Returns
        -------
        pl.DataFrame
        """
        src = self._sources(sources=sources, pattern=pattern)
        data = pl.concat(
            (_read_pmv(x).with_columns(pl.lit(x.name).alias('file')) for x in src),
            how='diagonal',
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
        sources: Iterable[str | Path] | None = None,
        pattern: str = r'((?P<date>\d{4}\-\d{2}\-\d{2})_)?TR(?P<id>\d+).*\.csv',
    ) -> pl.DataFrame:
        """
        TR7 센서 측정 결과 (csv) 읽기.

        Parameters
        ----------
        sources : Iterable[str | Path] | None, optional
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
        column_widths: int = 100,
    ):
        output = self.conf.dirs.sensor
        for sensor in ['PMV', 'TR7']:
            try:
                data = self.read_pmv() if sensor == 'PMV' else self.read_tr7()
            except FileNotFoundError as e:
                logger.warning('{} file not found in "{}"', sensor, e.args[0])
                continue

            if write_parquet:
                data.write_parquet(output / f'{sensor}.parquet')

            if write_xlsx:
                for by, df in data.group_by('date'):
                    df.write_excel(
                        output / f'{by[0]} {sensor}.xlsx',
                        column_widths=column_widths,
                    )

    @staticmethod
    def plot_pmv(
        data: pl.DataFrame,
        variables: Sequence[str] = ('PMV', '온도', '흑구온도', '상대습도', '기류'),
        *,
        style: SensorPlotStyle | None = None,
    ):
        data = (
            data.filter(pl.col('variable').is_in(variables))
            .with_columns(pl.format('{}층 {}', 'floor', 'space').alias('space'))
            .sort('space', 'datetime')
        )
        style = style or SensorPlotStyle()

        grid = (
            sns.relplot(
                data.to_pandas(),
                x='datetime',
                y='value',
                hue='space',
                row='variable',
                row_order=variables,
                kind='line',
                alpha=style.alpha,
                height=style.height,
                aspect=len(variables) / style.aspect,
                facet_kws={'sharey': False, 'despine': False, 'legend_out': False},
            )
            .set_titles('')
            .set_xlabels('')
        )

        units = {
            r[0]: r[1] for r in data.select('variable', 'unit').unique().iter_rows()
        }

        ax: Axes = grid.axes.ravel()[0]
        ax.xaxis.set_minor_locator(
            mdates.AutoDateLocator(maxticks=style.max_minor_ticks)
        )
        for row, ax in grid.axes_dict.items():
            unit = units.get(row)
            ax.set_ylabel(f'{row} [{unit}]' if unit else row)

            if style.comfort_range and row == 'PMV':
                lim = ax.get_ylim()
                ax.axhspan(-0.5, 0.5, color=style.comfort_range_color, linewidth=0)
                ax.set_ylim(lim)
            elif row == '상대습도':
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

        if legend := grid.legend:
            legend.set_title('')

        grid.figure.set_layout_engine('constrained', hspace=0.05)
        if style.fig_size:
            grid.figure.set_size_inches(style.fig_size_inches())

        if style.holidays is not None:
            marker = style.holidays.marker(data.select('datetime').to_series())
            marker(grid.axes.ravel())

        return grid

    @staticmethod
    def plot_tr7(data: pl.DataFrame, style: SensorPlotStyle | None = None):
        data = (
            data.sort('floor', descending=True)
            .with_columns(pl.format('{}층', 'floor').alias('floor'))
            .with_columns(pl.format('P{} {}', 'point', 'space').alias('space'))
        )
        floor_count = data.select('floor').n_unique()
        style = style or SensorPlotStyle()

        for var in ['T', 'RH']:
            grid = (
                sns.FacetGrid(
                    data.filter(pl.col('variable') == var).to_pandas(),
                    row='floor',
                    height=style.height,
                    aspect=floor_count / style.aspect,
                    despine=False,
                )
                .map_dataframe(
                    sns.lineplot,
                    x='datetime',
                    y='value',
                    hue='space',
                    alpha=style.alpha,
                )
                .set_axis_labels('', '온도 [°C]' if var == 'T' else '상대습도')
                .set_titles('')
                .set_titles('{row_name}', loc='left', weight='bold')
            )

            ax: Axes = grid.axes.flat[0]
            ax.xaxis.set_minor_locator(
                mdates.AutoDateLocator(maxticks=style.max_minor_ticks)
            )
            for ax in grid.axes.flat:
                ax.legend()
                if var == 'RH':
                    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

            if legend := grid.legend:
                legend.set_title('')

            grid.figure.set_layout_engine('constrained', hspace=0.05)
            if style.fig_size:
                grid.figure.set_size_inches(style.fig_size_inches())

            if style.holidays is not None:
                marker = style.holidays.marker(data['datetime'])
                marker(grid.axes.ravel())

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

    def plot_sensors(
        self,
        *,
        pmv: bool = True,
        tr7: bool = True,
        theme: bool = True,
    ):
        self.conf.dirs.analysis.mkdir(exist_ok=True)
        theme_context = (
            MplTheme('paper').grid().tick('x', 'both', color='.4').rc_context()
            if theme
            else contextlib.nullcontext()
        )

        with warnings.catch_warnings() and theme_context:
            warnings.filterwarnings(
                'ignore', category=UserWarning, module='seaborn.axisgrid'
            )

            if pmv:
                self._plot_pmv()

            if tr7:
                self._plot_tr7()

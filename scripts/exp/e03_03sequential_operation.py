"""2025-03-18 순차운휴 감지 테스트."""

from __future__ import annotations

import dataclasses as dc
from pathlib import Path
from typing import TYPE_CHECKING

import cyclopts
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import rich
import seaborn as sns
from loguru import logger
from matplotlib.dates import HourLocator
from whenever import PlainDateTime

from greenbutton import misc, utils
from greenbutton.utils.cli import App
from scripts.exp.e03_01kepco_paju import Config as _Config
from scripts.exp.e03_01kepco_paju import DBDirs  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


@dc.dataclass
class Config(_Config):
    wd: Path = Path('SequentialOperation')

    def __post_init__(self):
        super().__post_init__()
        self.wd = self.dirs.analysis / self.wd


def _path(conf: Config, table: str, database: str | Sequence[str] | None = None):
    if database is None:
        database = conf.log_db
    if isinstance(database, str):
        database = [database, 'ksem.pajoo.log']

    for db in database:
        if (path := conf.db_dirs.binary / f'{db}.dbo.{table}.parquet').exists():
            return path

    raise FileNotFoundError(table, database)


def _paths(conf: Config, table: str, database: str | Sequence[str] | None = None):
    if not table.endswith(' (0)'):
        table = f'{table} (0)'

    path = _path(conf, table, database)
    pattern = f'{path.stem.removesuffix("(0)")}(*).parquet'

    if not (paths := list(conf.db_dirs.binary.glob(pattern))):
        raise FileNotFoundError(table, database)

    return paths


# ===========================================================================

app = App(
    config=[
        cyclopts.config.Toml(f'config/{x}.toml', use_commands_as_keys=False)
        for x in ['.experiment', '.experiment_kepco_paju']
    ]
)
app.command(app_raw := App('raw', help='원본 데이터 탐색.'))


@app_raw.command
def raw_point_control(*, conf: Config):
    """T_BECO_POINT_CONTROL EHP 기동/정지."""
    path = _path(conf, table='T_BECO_POINT_CONTROL')
    logger.info('path="{}"', path)

    tag = pl.col('tag')
    data = (
        pl.scan_parquet(path)
        .rename({'[tagName]': 'tag', 'controlValue': 'control_value'})
        .filter(tag.str.starts_with('EHP'), tag.str.ends_with('기동정지'))
        .unpivot(
            ['controlRequestDate', 'controlCompleteDate'],
            index=['tag', 'control_value'],
            value_name='datetime',
        )
        .with_columns(
            pl.col('control_value').cast(pl.UInt8),
            floor=tag.str.extract(r'\.(.*\d층)\.'),
        )
        .sort('floor', 'tag', 'datetime')
        .collect()
    )

    rich.print(data)
    data.write_excel(conf.wd / '0000.point-control EHP.xlsx', column_widths=200)

    grid = (
        sns.FacetGrid(data, row='floor', height=2, aspect=16 * 5 / 9)
        .map_dataframe(sns.lineplot, x='datetime', y='control_value', hue='tag')
        .set_axis_labels('', 'Control Value')
    )

    for ax in grid.axes_dict.values():
        ax.legend()

    grid.figure.savefig(conf.wd / '0000.point-control.png')


@app_raw.command
def raw_point_control_fill(year: int = 2020, *, conf: Config):
    """T_BECO_POINT_CONTROL EHP 기동/정지 (forward fill 유사 그래프)."""
    path = _path(conf, table='T_BECO_POINT_CONTROL')
    logger.info('path="{}"', path)

    tag = pl.col('tag')
    cv = 'control_value'
    ccv = pl.col(cv)

    data = (
        pl.scan_parquet(path)
        .rename({
            '[tagName]': 'tag',
            'controlValue': cv,
            'controlCompleteDate': 'datetime',
        })
        .filter(
            pl.col('datetime').dt.year() == year,
            tag.str.starts_with('EHP'),
            tag.str.ends_with('기동정지'),
        )
        .with_columns(ccv.cast(pl.Float64))
        .sort('datetime', 'tag')
        .with_columns(prev_control=ccv.shift(1).over('tag'))
        .collect()
        .unpivot([cv, 'prev_control'], index=['datetime', 'tag'], value_name=cv)
        .select(
            'datetime',
            'tag',
            pl.col('variable')
            .replace_strict({'prev_control': -1, cv: 0})
            .alias('step'),
            cv,
        )
        .with_columns(tag.str.extract(r'\.(.*\d층)\.').alias('floor'))
        .with_columns(
            pl.when(pl.col('step') == -1)
            .then(pl.col('datetime') - pl.duration(seconds=1))
            .otherwise(pl.col('datetime'))
            .alias('datetime'),
            pl.when(
                tag.str.contains_any(['1층', '2층'])
                & tag.str.contains_any([
                    '상담실',
                    '종합봉사실',
                    '다목적실',
                    '부속실',
                    '사무실',
                    '지사장실',
                    '회의실',
                ])
            )
            .then(pl.format('{} 사무실', 'floor'))
            .otherwise(pl.col('floor'))
            .alias('floor'),
            tag.str.strip_suffix('.기동정지'),
        )
        .sort('floor', 'tag', 'datetime')
    )

    rich.print(data)

    grid = (
        sns.FacetGrid(data, row='floor', height=1.5, aspect=16 * 7 / 9)
        .map_dataframe(
            sns.lineplot, x='datetime', y='control_value', hue='tag', alpha=0.8
        )
        .set_axis_labels('', 'Control Value')
    )

    for ax in grid.axes_dict.values():
        ax.legend()

    grid.figure.savefig(conf.wd / f'0000.point-control-fill {year}.png')


@app_raw.command
def raw_elec_gshp(year: int = 2020, *, conf: Config):
    """T_BELO_ELEC_DAY 지열히트펌프전력량."""
    paths = _paths(conf, table='T_BELO_ELEC_15MIN')
    logger.info('path="{}"', paths[0])

    data = (
        pl.scan_parquet(paths, glob=False)
        .filter(
            pl.col('updateDate').dt.year() == year,
            pl.col('[tagName]').str.contains(
                r'지열\.((펌프\.히트펌프\d\.)|누적|전력\.현재)전력량'
            ),
        )
        .select(
            pl.col('updateDate').alias('datetime'),
            pl.col('tagValue').cast(pl.Float64).alias('value'),
            pl.col('[tagName]').alias('tag'),
        )
        .group_by(pl.col('datetime').dt.date(), 'tag')
        .agg(pl.sum('value'))
        .sort('tag', 'datetime')
        .collect()
    )

    rich.print(data)

    fig, ax = plt.subplots()
    sns.lineplot(data, x='datetime', y='value', hue='tag', ax=ax, alpha=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('히트펌프 전력량')

    if legend := ax.get_legend():
        legend.set_title('')

    fig.savefig(conf.wd / f'0000.elec-day gshp {year}.png')


@app_raw.command
def raw_elec_gshp_day(date: str = '2025-01-01', *, conf: Config):
    """T_BELO_ELEC_DAY 지열히트펌프전력량 하루 패턴."""
    paths = _paths(conf, table='T_BELO_ELEC_15MIN')
    logger.info('path="{}"', paths[0])

    d = PlainDateTime.parse_strptime(date, format='%Y-%m-%d')
    data = (
        pl.scan_parquet(paths, glob=False)
        .rename({
            'updateDate': 'datetime',
            '[tagName]': 'tag',
            'tagValue': 'value',
        })
        .filter(
            pl.col('datetime').dt.date() == pl.date(d.year, d.month, d.day),
            pl.col('tag').str.contains(
                r'지열\.((펌프\.히트펌프\d\.)|누적|전력\.현재)전력량'
            ),
        )
        .with_columns(
            pl.col('tag')
            .str.replace(r'\d', '')
            .str.strip_prefix('지열.')
            .alias('category'),
            cs.decimal().cast(pl.Float64),
        )
        .collect()
    )

    rich.print(data)

    grid = (
        sns.FacetGrid(data, row='category', sharey=False, aspect=16 * 3 / 9, height=2)
        .map_dataframe(sns.lineplot, x='datetime', y='value', hue='tag', alpha=0.75)
        .set_axis_labels('')
    )

    for ax in grid.axes_dict.values():
        ax.legend()

    output = conf.wd / '0102.GSHP'
    output.mkdir(exist_ok=True)

    grid.savefig(output / f'0102.GSHP-{d.date()}.png')


@app_raw.command
def raw_facility_ehp_zone(
    year: int = 2020,
    zone: str = 'EHP.2층.사무실_수요',
    *,
    conf: Config,
):
    """T_BELO_FACILITY_15MIN 존별 EHP 정보."""
    paths = _paths(conf, table='T_BELO_FACILITY_15MIN')
    logger.info('path="{}"', paths[0])

    data = (
        pl.scan_parquet(paths)
        .rename({
            'updateDate': 'datetime',
            '[tagName]': 'tag',
            'tagValue': 'value',
        })
        .filter(
            pl.col('datetime').dt.year() == year, pl.col('tag').str.starts_with(zone)
        )
        .select('tag', 'datetime', pl.col('value').cast(pl.Float64))
        .sort('tag', 'datetime')
        .collect()
    )

    def iter_month_season():
        seasons = {
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'autumn': [9, 10, 11],
            'winter': [12, 1, 2],
        }
        for season, months in seasons.items():
            for month in months:
                yield month, season

    by_time = data.filter(pl.col('tag').str.ends_with('.상태')).with_columns(
        season=pl.col('datetime').dt.month().replace_strict(dict(iter_month_season())),
        date=pl.col('datetime').dt.date(),
        dummy_datetime=pl.date(2000, 1, 1).dt.combine(pl.col('datetime').dt.time()),
    )
    grid = (
        sns.FacetGrid(by_time, row='season', height=2, aspect=4 * 16 / 9)
        .map_dataframe(
            sns.lineplot,
            x='dummy_datetime',
            y='value',
            alpha=0.25,
            units='date',
            estimator=None,
        )
        .set_axis_labels('')
    )
    grid.savefig(conf.wd / f'0000.facility-zone {zone} {year} by-time.png')
    plt.close(grid.figure)


@app_raw.command
def raw_facility_ehp_state(*, conf: Config):
    """EHP.X층.ZONE.상태 변수 빈도."""
    paths = _paths(conf, table='T_BELO_FACILITY_15MIN')
    logger.info('path="{}"', paths[0])

    data = (
        pl.scan_parquet(paths)
        .rename({'[tagName]': 'tag', 'tagValue': 'value'})
        .filter(
            pl.col('tag').str.starts_with('EHP.'), pl.col('tag').str.ends_with('.상태')
        )
        .select(pl.col('value').cast(pl.Float64))
    )

    pl.Config.set_tbl_rows(20)
    count = (
        data.group_by('value')
        .len()
        .collect()
        .sort('value')
        .with_columns(ratio=pl.col('len') / pl.sum('len'))
    )
    rich.print(count)
    # ┌───────┬─────────┬──────────┐
    # │ value ┆ len     ┆ ratio    │
    # │ ---   ┆ ---     ┆ ---      │
    # │ f64   ┆ u32     ┆ f64      │
    # ╞═══════╪═════════╪══════════╡
    # │ 0.0   ┆ 5393116 ┆ 0.863689 │
    # │ 1.0   ┆ 6454    ┆ 0.001034 │
    # │ 2.0   ┆ 6225    ┆ 0.000997 │
    # │ 3.0   ┆ 6244    ┆ 0.001    │
    # │ 4.0   ┆ 5707    ┆ 0.000914 │
    # │ 5.0   ┆ 5449    ┆ 0.000873 │
    # │ 6.0   ┆ 5406    ┆ 0.000866 │
    # │ 7.0   ┆ 5534    ┆ 0.000886 │
    # │ 8.0   ┆ 5482    ┆ 0.000878 │
    # │ 9.0   ┆ 5206    ┆ 0.000834 │
    # │ 10.0  ┆ 5196    ┆ 0.000832 │
    # │ 11.0  ┆ 5300    ┆ 0.000849 │
    # │ 12.0  ┆ 5199    ┆ 0.000833 │
    # │ 13.0  ┆ 5373    ┆ 0.00086  │
    # │ 14.0  ┆ 5791    ┆ 0.000927 │
    # │ 15.0  ┆ 772601  ┆ 0.123729 │
    # └───────┴─────────┴──────────┘

    rich.print(
        count.group_by(
            pl.col('value').replace_strict({0: '0', 15: '15'}, default='others')
        )
        .agg(pl.sum('len', 'ratio'))
        .with_columns()
    )
    # ┌────────┬─────────┬──────────┐
    # │ value  ┆ len     ┆ ratio    │
    # │ ---    ┆ ---     ┆ ---      │
    # │ str    ┆ u32     ┆ f64      │
    # ╞════════╪═════════╪══════════╡
    # │ others ┆ 78566   ┆ 0.012582 │
    # │ 0      ┆ 5393116 ┆ 0.863689 │
    # │ 15     ┆ 772601  ┆ 0.123729 │
    # └────────┴─────────┴──────────┘

    # state 0 여부로 on/off 취급 가능성


@app.command
def prep(*, conf: Config, binary_ehp_state: bool = False):
    """데이터 전처리."""
    facility = pl.scan_parquet(_paths(conf, table='T_BELO_FACILITY_15MIN'))
    elec = pl.scan_parquet(_paths(conf, table='T_BELO_ELEC_15MIN'))
    rename = {'updateDate': 'datetime', '[tagName]': 'tag'}

    # EHP state (전체 EHP 평균)
    ehp_state = 'EHP State'
    ehp = (
        facility.rename({'tagValue': ehp_state, **rename})
        .filter(
            pl.col('tag').str.starts_with('EHP.'), pl.col('tag').str.ends_with('.상태')
        )
        .with_columns(
            pl.col(ehp_state).replace_strict(0, 0, default=1, return_dtype=pl.Float64)
            if binary_ehp_state
            else pl.col(ehp_state).cast(pl.Float64) / 15
        )
        .group_by('datetime')
        .agg(pl.mean(ehp_state))
    )

    # 지열히트펌프 전력량 (총합)
    geothermal_consumption = 'Geothermal Consumption'
    geothermal = (
        elec.rename({'tagValue': geothermal_consumption, **rename})
        .filter(pl.col('tag').str.contains(r'지열.펌프.히트펌프\d.전력량'))
        .with_columns(pl.col(geothermal_consumption).cast(pl.Float64))
        .group_by('datetime')
        .agg(pl.sum(geothermal_consumption))
    )

    # 전체전력량
    total_consumption = (
        elec.rename({'tagValue': 'Total Consumption', **rename})
        .filter(pl.col('tag') == '전기.전체전력량')  # XXX 단위 불명
        .select('datetime', pl.col('Total Consumption').cast(pl.Float64))
    )

    data = (
        total_consumption.join(geothermal, on='datetime', how='full', coalesce=True)
        .join(ehp, on='datetime', how='full', coalesce=True)
        .sort('datetime')
        .collect()
    )
    data.write_parquet(conf.wd / '0100.raw.parquet')
    data.tail(1000).write_excel(conf.wd / '0100.raw-sample.xlsx', column_widths=200)


app.command(app_vis := App('vis', help='시각화.'))


@app_vis.command
def vis_raw(
    *,
    by_date: bool = False,
    corr: bool = False,
    every: str = '1h',
    conf: Config,
):
    if not any([by_date, corr]):
        raise ValueError

    data = (
        pl.scan_parquet(conf.wd / '0100.raw.parquet')
        .group_by_dynamic('datetime', every=every)
        .agg(
            pl.mean('EHP State'), pl.sum('Total Consumption', 'Geothermal Consumption')
        )
        .collect()
    )

    years = data['datetime'].dt.year().unique().to_list()
    is_holiday = misc.is_holiday(pl.col('datetime'), years=years)
    data = data.with_columns(is_holiday.alias('holiday'))

    output = conf.wd / '0101.raw'
    output.mkdir(exist_ok=True)

    if by_date:
        row_order = ['Geothermal Consumption', 'Total Consumption', 'EHP State']
        for by, grouped in data.group_by_dynamic('datetime', every='1q'):
            date = PlainDateTime.from_py_datetime(by[0]).date()  # type: ignore[arg-type]
            logger.info(date)

            df = grouped.upsample('datetime', every=every).unpivot(
                index=['datetime', 'holiday']
            )
            df = pl.concat([
                df,
                df.with_columns(pl.col('holiday').not_(), pl.lit(None).alias('value')),
            ]).sort('datetime', 'holiday', 'variable')

            grid = (
                sns.FacetGrid(
                    df,
                    row='variable',
                    row_order=row_order,
                    height=2,
                    aspect=16 * 3 / 9,
                    sharey=False,
                    hue='holiday',
                )
                .map_dataframe(utils.mpl.lineplot_break_nans, x='datetime', y='value')
                .set_axis_labels('')
                .set_titles('{row_name}')
            )
            grid.savefig(output / f'0101.raw-by-date {date.year}-{date.month:02d}.png')
            plt.close(grid.figure)

    if not corr:
        return

    for by, grouped in data.group_by_dynamic('datetime', every='1y'):
        year = by[0].year  # type: ignore[attr-defined]
        logger.info(year)
        grid = (
            sns.PairGrid(grouped.drop('datetime'), hue='holiday')
            .map_lower(sns.scatterplot, alpha=0.1)
            .map_diag(sns.histplot)
            .map_upper(sns.kdeplot)
            .add_legend()
        )
        grid.savefig(output / f'0101.raw-corr {year}.png')
        plt.close(grid.figure)


@app_vis.command
def vis_by_hour(*, conf: Config, every: str | None = None):
    """동/하절기 월별 그래프."""
    lf = pl.scan_parquet(conf.wd / '0100.raw.parquet').filter(
        pl.col('datetime').dt.month().is_in([6, 7, 8, 12, 1, 2])
    )

    if every:
        lf = lf.group_by_dynamic('datetime', every=every).agg(
            pl.mean('EHP State'),
            pl.sum('Total Consumption', 'Geothermal Consumption'),
        )

    years = lf.select(pl.col('datetime').dt.year().unique()).collect().to_series()
    data = lf.with_columns(
        misc.is_holiday(pl.col('datetime'), years=years.to_list()).alias('holiday')
    ).collect()

    output = conf.wd / '0101.raw-by-hour'
    output.mkdir(exist_ok=True)

    row_order = ['Geothermal Consumption', 'Total Consumption', 'EHP State']
    utils.mpl.MplConciseDate(zero_formats=['', '', '', '%H:%M', '', '']).apply()
    (
        utils.mpl.MplTheme(rc={'legend.fontsize': 'x-small'})
        .grid()
        .tick('x', 'both', direction='in')
        .apply()
    )
    ax: Axes

    for by, grouped in data.group_by_dynamic('datetime', every='1mo'):
        date = PlainDateTime.from_py_datetime(by[0]).date()  # type: ignore[arg-type]
        logger.info(date)

        dummy = pl.date(date.year, date.month, 1)
        df = grouped.unpivot(index=['datetime', 'holiday'])
        df = (
            pl.concat([
                df,
                df.with_columns(pl.col('holiday').not_(), pl.lit(None).alias('value')),
            ])
            .sort('datetime', 'holiday', 'variable')
            .with_columns(
                dummy_date=dummy.dt.combine(pl.col('datetime').dt.time()),
                date=pl.col('datetime').dt.date(),
            )
        )

        grid = (
            sns.FacetGrid(
                df,
                row='variable',
                row_order=row_order,
                height=2,
                aspect=16 * 3 / 9,
                sharey=False,
                hue='holiday',
            )
            .map_dataframe(
                utils.mpl.lineplot_break_nans,
                x='dummy_date',
                y='value',
                units='date',
                alpha=0.4,
            )
            .set_xlabels('')
            .set_titles('{row_name}')
        )

        for ax in grid.axes_dict.values():
            ax.xaxis.set_major_locator(HourLocator(interval=2))
            ax.xaxis.set_minor_locator(HourLocator(interval=1))

        grid.savefig(output / f'0102.raw-by-hour {date.year}-{date.month:02d}.png')
        plt.close(grid.figure)

    # -> 2021년 7월, 8월 순차운휴 추정


@app_vis.command
def vis_candidate(*, conf: Config):
    """2021년 7-8월 16:00 EHP 상태 시각화."""
    data = (
        pl.scan_parquet(conf.wd / '0100.raw.parquet')
        .filter(
            pl.col('datetime').is_between(
                pl.date(2021, 7, 1), pl.date(2021, 9, 1), closed='left'
            )
        )
        .with_columns(
            pl.col('datetime').dt.date().alias('date'),
            misc.is_holiday(pl.col('datetime'), years=2021).alias('holiday'),
        )
        .collect()
    )
    rich.print(data)

    data16 = data.filter(pl.col('datetime').dt.time() == pl.time(16, 0, 0))

    # 16시 날짜-EHP상태
    fig, ax = plt.subplots()
    sns.scatterplot(data16, x='date', y='EHP State', hue='holiday', ax=ax)
    ax.set_xlabel('')
    fig.savefig(conf.wd / '0101.candidate.png')

    # 16시 EHP상태-전력사용량
    unpivot = data16.unpivot(
        on=['Geothermal Consumption', 'Total Consumption'],
        index=['date', 'holiday', 'EHP State'],
    )
    grid = (
        sns.FacetGrid(unpivot, col='variable', hue='holiday', sharey=False)
        .map_dataframe(sns.scatterplot, x='EHP State', y='value', alpha=0.5)
        .set_titles('{col_name}')
        .add_legend()
    )
    grid.savefig(conf.wd / '0101.candidate-scatter.png')

    # 각 날짜 시간별 EHP 상태, 사용량 변화
    output = conf.wd / '0101.raw-by-hour'
    output.mkdir(exist_ok=True)
    norm = (
        data.filter(pl.col('holiday').not_())
        .unpivot(index=['datetime', 'date', 'holiday'])
        .with_columns(
            pl.col('value').truediv(pl.max('value').over(['variable', 'date'])),
            pl.format(
                '{}{}',
                'date',
                pl.col('holiday').replace_strict({True: ' (휴일)', False: ''}),
            ).alias('date'),
        )
        .sort('datetime')
    )

    (
        utils.mpl.MplTheme(rc={'legend.fontsize': 'x-small'})
        .grid()
        .tick('x', 'both', direction='in')
        .apply()
    )
    utils.mpl.MplConciseDate(zero_formats=['', '', '', '%H:%M', '', '']).apply()

    for by, grouped in norm.group_by_dynamic('datetime', every='1w'):
        d = PlainDateTime.from_py_datetime(by[0]).date()  # type: ignore[arg-type]
        dummy = pl.date(d.year, d.month, d.day)

        logger.info(d)

        df = grouped.with_columns(
            dummy_datetime=dummy.dt.combine(pl.col('datetime').dt.time())
        ).sort('datetime')
        n = df['date'].unique().len()
        grid = (
            sns.FacetGrid(
                df,
                row='date',
                hue='variable',
                hue_order=['EHP State', 'Geothermal Consumption', 'Total Consumption'],
                aspect=16 * n / 9,
                height=1.5,
            )
            .map_dataframe(sns.lineplot, x='dummy_datetime', y='value', alpha=0.8)
            .set_titles('')
            .set_titles('{row_name}', weight=500, loc='left')
            .set_axis_labels('', 'Norm. Value')
            .add_legend()
        )

        for ax in grid.axes_dict.values():
            ax.xaxis.set_minor_locator(HourLocator(interval=1))

        grid.savefig(output / f'0101.norm-{d}.png')
        plt.close(grid.figure)

    # => 순차운휴 여부에 따라 전체 전력 사용량 큰 차이 없음


if __name__ == '__main__':
    import warnings

    utils.terminal.LogHandler.set()
    utils.mpl.MplTheme(rc={'legend.fontsize': 'x-small'}).grid().apply()
    utils.mpl.MplConciseDate().apply()

    warnings.filterwarnings(
        'ignore', 'The figure layout has changed to tight', category=UserWarning
    )

    app()

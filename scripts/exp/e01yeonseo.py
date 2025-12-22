from __future__ import annotations

import dataclasses as dc
import itertools
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns
import xlsxwriter
from loguru import logger

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils.cli import App
from greenbutton.utils.terminal import Progress

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class _Experiment(exp.Experiment):
    PMV1_TIME_DELTA = pl.duration(hours=11, minutes=17)
    DATE = (2024, 2, 21)

    def read_pmv(
        self,
        sources=None,
        pattern=r'((?P<date>\d{4}\-\d{2}\-\d{2})_)?PMV(?P<id>\d+).*\.(csv|dlg)',
    ):
        data = super().read_pmv(sources, pattern)

        # PMV1 (1층) 기록 시간 보정
        return data.with_columns(
            pl
            .when(pl.col('id') == 1)
            .then(
                pl.date(*self.DATE).dt.combine(
                    (pl.col('datetime') + self.PMV1_TIME_DELTA).dt.time()
                )
            )
            .otherwise(pl.col('datetime'))
            .alias('datetime')
        )


@dc.dataclass
class DBDirs:
    raw: Path = Path('0000.raw')
    data: Path = Path('0001.data')
    anomaly_detection: Path = Path('0100.anomaly_detection')
    analysis: Path = Path('0101.analysis')


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'yeonseo'

    db_dirs: DBDirs = dc.field(default_factory=DBDirs)

    def __post_init__(self):
        for field in (f.name for f in dc.fields(self.db_dirs)):
            p = getattr(self.db_dirs, field)
            setattr(self.db_dirs, field, self.dirs.database / p)

    def experiment(self):
        return _Experiment(conf=self)


app = App(
    config=cyclopts.config.Toml('config/.experiment.toml', use_commands_as_keys=False)
)


@app.command
def init(*, conf: Config):
    conf.dirs.mkdir()


app.command(App('sensor'))


@app['sensor'].command
def sensor_parse(*, conf: Config, parquet: bool = True, xlsx: bool = True):
    exp = conf.experiment()
    exp.parse_sensors(write_parquet=parquet, write_xlsx=xlsx)


@app['sensor'].command
def sensor_plot(*, conf: Config, pmv: bool = True, tr7: bool = True):
    exp = conf.experiment()
    exp.plot_sensors(pmv=pmv, tr7=tr7)


app.command(App('db', help='제로에너지모니터링시스템 분석'))


def _read_raw(path: Path, ext: str = '.xls'):
    point = path.name
    if not (sources := list(path.rglob(f'*{ext}'))):
        raise FileNotFoundError(path)

    it = Progress.iter(sources, transient=True)
    return (
        pl
        .concat(
            pl.read_excel(s).select(pl.lit(point).alias('point'), pl.all()) for s in it
        )
        .with_columns(pl.col('검침시간').str.to_datetime())
        .drop('번호')
        .rename({'검침시간': 'datetime'})
        .with_columns(cs.integer().cast(pl.Float64))
    )


@app['db'].command
def db_convert(*, conf: Config, xlsx: bool = False):
    """
    다운받은 xls 데이터 변환.

    Parameters
    ----------
    conf : Config
    """
    dirs = conf.db_dirs
    dirs.data.mkdir(exist_ok=True)

    for path in dirs.raw.glob('*'):
        if not path.is_dir():
            continue

        name = path.name
        logger.info(name)

        data = _read_raw(path)
        data.write_parquet(dirs.data / f'{name}.parquet')

        if xlsx:
            data.write_excel(dirs.data / f'{name}.xlsx')


@app['db'].command
def db_concat(*, conf: Config, drop_zero: bool = True, diff: bool = True):
    """
    Parquet 파일 통합.

    Parameters
    ----------
    conf : Config
    drop_zero : bool, optional
    diff : bool, optional
    """
    src = conf.db_dirs.data
    dst = conf.dirs.database

    dt = 'datetime'
    variables = ['전기', '수도', '가스', '온수', '난방']

    def _scan(path):
        data = pl.scan_parquet(path).sort(dt)
        if diff:
            data = data.with_columns(pl.col(variables).diff())
        return data

    lf = (
        pl
        .concat(_scan(x) for x in src.glob('*.parquet'))
        .unpivot(variables, index=['point', dt])
        .with_columns()
    )

    if drop_zero:
        lf = lf.filter(pl.col('value') != 0)

    name = 'data_diff' if diff else 'data'

    df = lf.collect()
    df.write_parquet(dst / f'{name}.parquet')

    with xlsxwriter.Workbook(dst / f'{name}.xlsx') as wb:
        for variable in variables:
            pivot = (
                df
                .filter(pl.col('variable') == variable)
                .pivot('point', index=dt, values='value', sort_columns=True)
                .sort(dt)
            )
            if pivot.height:
                pivot.write_excel(wb, worksheet=variable)


@cyclopts.Parameter(name='detector')
@dc.dataclass
class DetectorOption:
    lags: tuple[int, ...] = (-1, -2, -6)
    threshold: float = 10

    max_value: float = 100
    interpolate: Literal['linear', 'nearest', 'forward'] | None = None
    interval: str = '10m'
    plot_interval: str = '1h'


_DEFAULT_DETECTOR_OPTION = DetectorOption()


@app['db'].command
def db_detect_anomaly(
    *,
    conf: Config,
    option: DetectorOption = _DEFAULT_DETECTOR_OPTION,
):
    from greenbutton.anomaly import tsad  # noqa: PLC0415

    utils.mpl.MplTheme().grid().apply()

    dst = conf.db_dirs.anomaly_detection
    dst.mkdir(exist_ok=True)

    data = (
        pl
        .scan_parquet(conf.dirs.database / 'data_diff.parquet')
        .filter(pl.col('variable') == '전기')  # 전기 데이터만 사용
        .with_columns(
            value=pl
            .when(pl.col('value') > option.max_value)
            .then(None)
            .otherwise(pl.col('value'))
        )
    )
    detector = tsad.Detector(
        columns=tsad.Columns(time='datetime', value='value'),
        config=tsad.DetectorConfig(
            target='normalized',
            lags=option.lags,
            threshold=option.threshold,
        ),
    )
    points: list[str] = (
        data.select(pl.col('point').unique().sort()).collect().to_series().to_list()
    )

    for point in Progress.iter(points):
        logger.info(f'{point=}')

        df = (
            data
            .filter(pl.col('point') == point)
            .sort('datetime')
            .collect()
            .upsample('datetime', every=option.interval)
        )
        od = detector(df).rename({'time': 'datetime'})

        match option.interpolate:
            case 'forward':
                od = od.with_columns(pl.col('value').forward_fill())
            case 'linear' | 'nearest':
                od = od.with_columns(pl.col('value').interpolate(option.interpolate))

        od.write_parquet(dst / f'{point}.parquet')

        plot = (
            od
            .group_by_dynamic('datetime', every=option.plot_interval)
            .agg(pl.sum('original', 'value'), pl.max('score'), pl.any('outlier'))
            .with_columns()
        )

        grid = detector.plot(plot, time='datetime')
        grid.figure.suptitle(f'{option.plot_interval} 간격 전력 사용량 [kWh]')
        grid.set_axis_labels(y_var='전력 사용량 [kWh]')

        grid.savefig(
            dst / f'{point}_interval={option.plot_interval}'
            f'_interpolate={option.interpolate}.png'
        )
        plt.close(grid.figure)


app.command(App('plot'))


def _read_energy(cases: dict[str, list[Path]]):
    for key, paths in cases.items():
        for path in paths:
            yield (
                pl
                .scan_parquet(path)
                .select(
                    pl.lit(key).alias('type'),
                    pl.lit(path.stem).alias('sensor'),
                    pl.all(),
                )
                .filter(pl.col('outlier').fill_null(value=False).not_())
            )


@app['plot'].command
def plot_energy(*, conf: Config):
    src = conf.db_dirs.anomaly_detection
    dst = conf.dirs.analysis
    dst.mkdir(exist_ok=True)

    files = list(src.glob('*.parquet'))
    texts = {
        'MAIN': 'MAIN',
        '비상계단': '비상',
        '바닥난방': '바닥난방',
        '히트펌프': 'HP',
    }
    cases = {k: [x for x in files if v in x.name] for k, v in texts.items()}
    assert sorted(itertools.chain.from_iterable(cases.values())) == sorted(files)

    df = (
        pl
        .concat(_read_energy(cases))
        .group_by_dynamic('datetime', every='1d', group_by=['type', 'sensor'])
        .agg(pl.sum('value'))
        .collect()
    )
    grid = (
        sns
        .FacetGrid(
            df, row='type', height=1.8, aspect=4 * 16 / 9, despine=False, sharey=False
        )
        .map_dataframe(sns.lineplot, x='datetime', y='value', hue='sensor', alpha=0.6)
        .set_axis_labels('', '전력사용량 [kWh]')
        .set_titles(row_template='{row_name}', weight=500)
        .tight_layout()
    )

    grid.figure.set_layout_engine('constrained')
    grid.savefig(dst / '[energy] line.png')
    plt.close(grid.figure)

    cases['저사용'] = [
        x for x in files if x.name.startswith(('09', '17', '18', '22', '23'))
    ]
    for name, source in cases.items():
        df = (
            pl
            .concat(
                pl.scan_parquet(s).select(pl.lit(s.stem).alias('sensor'), pl.all())
                for s in source
            )
            .filter(pl.col('outlier').fill_null(value=False).not_())
            .group_by_dynamic('datetime', every='1d', group_by='sensor')
            .agg(pl.sum('value'))
            .collect()
        )

        grid = (
            sns
            .relplot(
                df,
                x='datetime',
                y='value',
                col='sensor',
                col_wrap=int(utils.mpl.ColWrap(df.select('sensor').n_unique())),
                height=3,
                aspect=4 / 3,
                kind='line',
                facet_kws={'despine': False},
                alpha=0.8,
            )
            .set_axis_labels('', '전력사용량 [kWh]')
            .set_titles(col_template='{col_name}')
        )

        ax: Axes
        for ax in grid.axes.flat:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

        grid.figure.set_layout_engine('constrained')
        grid.savefig(dst / f'[energy] line_{name}.png')
        plt.close(grid.figure)


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplConciseDate().apply()
    utils.mpl.MplTheme(palette='tol:vibrant').grid().apply()

    app()

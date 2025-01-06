from __future__ import annotations

import dataclasses as dc
import functools
import itertools
import math
import operator
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple

import cyclopts
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns
import sqlalchemy
import sqlalchemy.exc
from loguru import logger
from matplotlib.layout_engine import ConstrainedLayoutEngine
from rich.progress import track

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils import App, Progress

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from polars._typing import FrameType


@dc.dataclass
class DBDirs:
    root: Path
    sample: Path = Path('01.sample')
    binary: Path = Path('02.binary')
    parsed: Path = Path('03.parsed')

    def __post_init__(self):
        self.update()

    def update(self):
        for field in (f.name for f in dc.fields(self)):
            if field == 'root':
                continue

            p = getattr(self, field)
            setattr(self, field, self.root / p)

        return self


@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING = 'kea'

    points: dict[str, list[str]] = dc.field(default_factory=dict)

    @functools.cached_property
    def db_dirs(self):
        return DBDirs(self.dirs.database)


ConfigParam = Annotated[Config, cyclopts.Parameter(name='*')]
app = App(
    config=[
        cyclopts.config.Toml('config/.experiment.toml', use_commands_as_keys=False),
        cyclopts.config.Toml(
            'config/experiment.toml', root_keys='kea', use_commands_as_keys=False
        ),
    ]
)


@app.command
def init(*, conf: ConfigParam):
    conf.dirs.mkdir()


app.command(App('sensor'))


@app['sensor'].command
def sensor_parse(*, conf: ConfigParam, parquet: bool = True, xlsx: bool = True):
    exp = conf.experiment()
    exp.parse_sensors(write_parquet=parquet, write_xlsx=xlsx)


@app['sensor'].command
def sensor_plot(*, conf: ConfigParam, pmv: bool = True, tr7: bool = True):
    exp = conf.experiment()
    exp.plot_sensors(pmv=pmv, tr7=tr7)


@dc.dataclass
class MsSql:
    database: str

    @functools.cached_property
    def engine(self):
        return self.create_engine(self.database)

    @staticmethod
    def create_engine(database: str, **kwargs):
        url = sqlalchemy.URL.create(
            'mssql+pyodbc',
            host='localhost',
            database=database,
            query={'driver': 'ODBC Driver 17 for SQL Server'},
        )

        return sqlalchemy.create_engine(url, **kwargs)

    def height(self, table: str) -> int:
        return pl.read_database(
            f'SELECT COUNT(*) FROM {table}', connection=self.engine
        ).item()

    def read_df(self, table: str):
        return pl.read_database(f'SELECT * FROM {table}', connection=self.engine)

    def iter_df(self, table: str, batch_size: int = 10**7):
        height = self.height(table)

        if height <= batch_size:
            df = pl.read_database(f'SELECT * FROM {table}', connection=self.engine)
            yield None, df
            return

        yield from enumerate(
            pl.read_database(
                f'SELECT * FROM {table}',
                connection=self.engine,
                iter_batches=True,
                batch_size=batch_size,
            )
        )


@functools.lru_cache
def _db_engine(db: str, **kwargs):
    url = sqlalchemy.URL.create(
        'mssql+pyodbc',
        host='localhost',
        database=db,
        query={'driver': 'ODBC Driver 17 for SQL Server'},
    )
    return sqlalchemy.create_engine(url, **kwargs)


app.command(App('db'))


@app['db'].command
def db_extract_tables(
    *,
    conf: ConfigParam,
    databases: tuple[str, ...] = (
        'KEAPV',
        'NMWDataLogDB20181120',
        'NMWDataLogDB20210816',
        'NMWDataLogDB20240512',
        'NMWObjectDB',
        'NMWSysLogDB20220412',
        'NMWSystemDB',
        'NWCS.Client',
    ),
):
    """테이블 목록 추출."""
    dfs: list[pl.DataFrame] = []

    for db in databases:
        engine = _db_engine(db=db)
        tables = pl.read_database(
            query='SELECT * FROM INFORMATION_SCHEMA.TABLES',
            connection=engine,
        ).sort(pl.all())

        tables.write_excel(conf.dirs.database / f'[TABLES] {db}.xlsx', autofit=True)
        logger.info('DB={}, tables={}', db, tables.height)

        dfs.append(tables)

    df = pl.concat(dfs)
    df.write_excel(conf.dirs.database / '[TABLES].xlsx', column_widths=200)
    df.write_parquet(conf.db_dirs.binary / 'TABLES.parquet')


@app['db'].command
def db_sample(
    *,
    conf: ConfigParam,
    n: int = 1000,
    fragment_n: int = 2,
):
    dirs = conf.db_dirs
    dirs.sample.mkdir(exist_ok=True)

    fragmented_prefix = (
        'EventLog',
        'TrendLog',
        'ConnectionLog',
        'ControlLog',
        'ModifyLog',
        'UserSystemLog',
    )
    fragmented_pattern = re.compile(f'^(.*)({"|".join(fragmented_prefix)}).*')
    fragmented: defaultdict[str, list[pl.DataFrame]] = defaultdict(list)

    tables = (
        pl.scan_parquet(dirs.binary / 'TABLES.parquet', glob=False)
        .filter(pl.col('TABLE_TYPE') == 'BASE TABLE')
        .with_columns(
            schema=pl.format('{}.{}', 'TABLE_SCHEMA', 'TABLE_NAME'),
            catalog=pl.format(
                '{}.{}.{}', 'TABLE_CATALOG', 'TABLE_SCHEMA', 'TABLE_NAME'
            ),
        )
        .collect()
    )

    for by, df in tables.group_by('TABLE_CATALOG', maintain_order=True):
        logger.info('TABLE_CATALOG={}', by[0])

        if 'NWCS.Client' in by[0]:  # type: ignore[operator]
            # 에러
            continue

        for row in track(df.iter_rows(named=True), total=df.height):
            is_fragmented: bool = (
                'Log' in row['TABLE_CATALOG']  ##
                and row['TABLE_NAME'].startswith(fragmented_prefix)
            )

            if not is_fragmented:
                logger.info('{}, fragmented={}', row['catalog'], is_fragmented)

            engine = _db_engine(row['TABLE_CATALOG'])
            height = pl.read_database(
                f'SELECT COUNT(*) FROM {row["catalog"]}', connection=engine
            ).item()
            sample = pl.read_database(
                f'SELECT TOP {fragment_n if is_fragmented else n} * '
                f'FROM {row["schema"]}',
                connection=engine,
            ).with_columns(cs.binary().bin.encode('hex'))

            if is_fragmented:
                name = fragmented_pattern.sub(r'\g<1>\g<2>', row['catalog'])
                fragmented[name].append(sample)
            else:
                name = f'{row["catalog"]} (n={height}).xlsx'
                if height == 0:
                    name = f'(ZERO) {name}'

                sample.write_excel(dirs.sample / name)

        for key, dfs in fragmented.items():
            logger.info(key)
            pl.concat(dfs).write_excel(dirs.sample / f'{key}.xlsx', autofit=True)

        fragmented = defaultdict(list)


@dc.dataclass
class _PointsExtractor:
    """NMWObjectDB 계측점 목록 추출."""

    catalog: str = 'NMWObjectDB'
    tables: Sequence[str] = (
        'AnalogInputObject',
        'AnalogOutputObject',
        'AnalogValueObject',
        'BinaryInputObject',
        'BinaryOutputObject',
        'BinaryValueObject',
    )

    def __iter__(self):
        db = MsSql(self.catalog)

        for table in self.tables:
            yield pl.read_database(
                'SELECT device_identifier, object_identifier, '
                f'object_name, object_type, description FROM {table}',
                connection=db.engine,
            )

    def __call__(self):
        # XXX propid??
        return pl.concat(self).rename({
            'device_identifier': 'deviceid',
            'object_identifier': 'objectid',
        })


@app['db'].command
def db_extract_points(*, conf: ConfigParam):
    output = conf.db_dirs.binary
    output.mkdir(exist_ok=True)

    data = _PointsExtractor()()
    data.write_parquet(output / 'NMWObjectDB.Points.parquet')


app.command(App('trendlog', help='NMWDataLog/TrendLog 해석.'))


class _TrendLogExtractor:
    TC = 'TABLE_CATALOG'
    TN = 'TABLE_NAME'
    FMT = '{catalog}.TrendLog.batch{idx}'

    def __init__(
        self,
        tables: Path,
        *,
        read_batch: int = 1,
        write_batch: int = 2**25,
        trace: bool = True,
        concat: bool = True,
    ):
        self.tables = (
            pl.scan_parquet(tables, glob=False)
            .select(self.TC, self.TN)
            .filter(
                pl.col(self.TC).str.starts_with('NMWDataLog'),
                pl.col(self.TN).str.starts_with('TrendLog'),
            )
            .collect()
        )
        self.read_batch = read_batch
        self.write_batch = write_batch
        self.trace = trace
        self.flag_concat = concat

    def iter_frame(self, catalog: str) -> Iterable[pl.DataFrame]:
        tables = (
            self.tables.filter(pl.col(self.TC) == catalog)
            .select(pl.col(self.TN).sort())
            .to_series()
        )
        db = MsSql(catalog)

        for table in tables:
            yield db.read_df(table).select(pl.lit(table).alias(self.TN), pl.all())

    def iter(self, catalog: str):
        it: Iterable[tuple[int, tuple[pl.DataFrame, ...]]] = enumerate(
            itertools.batched(self.iter_frame(catalog), n=self.read_batch)
        )

        if self.trace:
            total = math.ceil(
                self.tables.filter(pl.col(self.TC) == catalog).height / self.read_batch
            )
            it = Progress.trace(it, description=f'Extracting {catalog}', total=total)

        for idx, dfs in it:
            yield (
                catalog,
                idx,
                self.FMT.format(catalog=catalog, idx=f'{idx:04d}'),
                pl.concat(dfs),
            )

    def concat(self, catalog: str, directory: Path):
        fmt = self.FMT.format(catalog=catalog, idx='')
        files = list(directory.glob(f'{fmt}*.parquet'))
        logger.debug('{} files={}', catalog, files)

        for idx, df in enumerate(
            pl.scan_parquet(files).collect().iter_slices(self.write_batch)
        ):
            df.write_parquet(directory / f'{catalog}.TrendLog{idx:04d}.parquet')

    def __call__(self, directory: Path, catalogs: Sequence[str] | None = None):
        if catalogs is None:
            catalogs = (
                self.tables.select(pl.col(self.TC).unique().sort())
                .to_series()
                .to_list()
            )

        for catalog in catalogs:
            for _, _, name, data in self.iter(catalog=catalog):
                data.write_parquet(directory / f'{name}.parquet')

            if self.flag_concat:
                self.concat(catalog=catalog, directory=directory)


@app['trendlog'].command
def trendlog_extract(*, conf: ConfigParam, read_batch: int = 10):
    """Raw 데이터 parquet으로 저장."""
    dirs = conf.db_dirs
    output = dirs.binary

    extractor = _TrendLogExtractor(
        tables=dirs.binary / 'TABLES.parquet',
        read_batch=read_batch,
    )
    extractor(directory=output)


class _Point(NamedTuple):
    deviceid: int
    objectid: int
    type_: int  # object_type
    name: str

    def __str__(self):
        return f'D{self.deviceid}_P{self.objectid}_{self.name.replace("/", "-")}'


@dc.dataclass
class _TrendLogParser:
    conf: Config

    catalogs: Sequence[str] = (
        'NMWDataLogDB20181120',
        'NMWDataLogDB20210816',
        'NMWDataLogDB20240512',
    )

    dtypes: dict[int, type[pl.DataType]] = dc.field(
        default_factory=lambda: {4: pl.Float32, 6: pl.Float32, 10: pl.Float64}
    )
    endianness: Literal['big', 'little'] = 'little'

    _points: pl.LazyFrame = dc.field(init=False)

    def __post_init__(self):
        self._points = pl.scan_parquet(
            self.conf.db_dirs.binary / 'NMWObjectDB.Points.parquet'
        )

    def lazy_frames(self):
        for path in self.conf.db_dirs.binary.glob('*TrendLog*.parquet'):
            yield (
                pl.scan_parquet(path).select(
                    pl.lit(path.name.split('.')[0]).alias('TABLE_CATALOG'), pl.all()
                )
            )

    def match_points(self):
        return (
            pl.concat(self.lazy_frames())
            .select('TABLE_CATALOG', 'deviceid', 'objectid')
            .unique()
            .join(self._points, on=['deviceid', 'objectid'], how='left')
            .sort(pl.all())
        )

    def parse(self, data: FrameType, *, is_state: bool):
        data = data.with_columns(
            pl.col('datavalue')
            .map_elements(
                # x -> x[2:]
                operator.itemgetter(slice(2, None)),
                return_dtype=pl.Binary,
            )
            .alias('_datavalue')
        )

        if is_state:
            parsed = data.with_columns(
                pl.col('_datavalue')
                .replace_strict(
                    {b'\x01\x00': 1, b'\x00\x00': 0},
                    default=None,
                    return_dtype=pl.Int8,
                )
                .alias('parsed_value')
            )
        else:
            bytes_len = len(data.lazy().select('datavalue').head(1).collect().item())
            parsed = data.with_columns(
                pl.col('_datavalue')
                .bin.reinterpret(
                    dtype=self.dtypes[bytes_len], endianness=self.endianness
                )
                .cast(pl.Float64)
                .alias('parsed_value')
            )

        return parsed.drop('_datavalue')

    def _parse_point(self, point: _Point):
        data = (
            pl.concat(self.lazy_frames())
            .filter(
                pl.col('deviceid') == point.deviceid,
                pl.col('objectid') == point.objectid,
            )
            .collect()
        )

        is_state = point.type_ not in {0, 1}
        return (
            pl.concat(
                self.parse(df, is_state=is_state)
                for _, df in data.group_by('TABLE_NAME')
            )
            .rename({'timeStamp': 'datetime'})
            .sort('datetime')
            .with_columns()
        )

    def _parse_and_write(self, point: _Point, *, skip_exists: bool = True):
        output = self.conf.db_dirs.parsed
        path = output / f'TrendLog_{point}.parquet'

        if skip_exists and path.exists():
            return

        data = self._parse_point(point)

        count = (
            data.select(pl.col('parsed_value').count()).item(),
            data.select(pl.col('parsed_value').null_count()).item(),
        )
        logger.info(
            'count={} | null_count={} | null_ratio={:.6f}',
            count[0],
            count[1],
            count[1] / (count[0] + count[1]),
        )

        data.with_columns(pl.lit(point.name).alias('point')).write_parquet(path)
        data = data.drop_nulls('parsed_value').drop(cs.binary())
        pl.concat([data.head(50), data.tail(50)]).write_excel(path.with_suffix('.xlsx'))

    def parse_points(self, points: pl.DataFrame, *, skip_exists: bool = True):
        points = (
            points.select('deviceid', 'objectid', 'object_type', 'object_name')
            .unique()
            .sort(pl.all())
        )

        for row in Progress.trace(points.iter_rows(), total=points.height):
            point = _Point(*row)
            logger.info(point)
            self._parse_and_write(point, skip_exists=skip_exists)


@app['trendlog'].command
def trendlog_match_points(*, conf: ConfigParam):
    parser = _TrendLogParser(conf=conf)
    points = parser.match_points().collect()

    points.write_parquet(conf.dirs.database / 'NMWTrendLogPoints.parquet')
    points.write_excel(conf.dirs.database / 'NMWTrendLogPoints.xlsx', column_widths=200)


@app['trendlog'].command
def trendlog_parse_binary(*, conf: ConfigParam, skip_exists: bool = True):
    conf.db_dirs.parsed.mkdir(exist_ok=True)

    points = pl.read_parquet(conf.dirs.database / 'NMWTrendLogPoints.parquet')
    parser = _TrendLogParser(conf=conf)
    parser.parse_points(points, skip_exists=skip_exists)


@app['trendlog'].command
def trendlog_plot(*, conf: ConfigParam, every: str = '1h'):
    """TrendLog 각 변수 시계열 그래프 (검토용)."""
    # TODO 새로 추출한 parsed 데이터로 변경
    dirs = conf.db_dirs

    objects_list = (dirs.root / 'TrendLogObjects.txt').read_text().splitlines()

    object_name = 'object_name'
    objects = (
        pl.DataFrame({object_name: objects_list})
        .with_columns(
            pl.col(object_name).str.split(' ').list[0].alias('category'),
            pl.col(object_name)
            .str.extract('(온도|습도|수위|상태|유량)')
            .alias('variable'),
        )
        .with_columns(
            pl.col('category', 'variable').replace({'': None}).fill_null('etc')
        )
    )

    lf = pl.scan_parquet(dirs.parsed / 'TrendLog*.parquet')
    categories = objects.select(pl.col('category').unique().sort()).to_series()

    utils.MplTheme(context='paper', palette='tol:bright').grid().apply()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', category=UserWarning, module='seaborn.axisgrid'
        )

        for category in Progress.trace(categories, total=categories.len()):
            logger.info('category={}', category)

            objs = (
                objects.filter(pl.col('category') == category)
                .select(pl.col(object_name))
                .to_series()
            )
            df = (
                lf.filter(pl.col(object_name).is_in(objs))
                .select('timeStamp', 'parsedvalue', object_name)
                .join(objects.lazy(), on=object_name, how='left')
                .sort('timeStamp')
                .group_by_dynamic(
                    'timeStamp', every=every, group_by=['variable', object_name]
                )
                .agg(pl.mean('parsedvalue'))
                .collect()
            )

            grid = (
                sns.FacetGrid(df, row='variable', sharey=False, height=3, aspect=32 / 9)
                .map_dataframe(
                    sns.lineplot,
                    x='timeStamp',
                    y='parsedvalue',
                    hue=object_name,
                    alpha=0.5,
                )
                .set_axis_labels('', 'value')
            )
            for ax in grid.axes.ravel():
                ax.legend()

            grid.savefig(conf.dirs.analysis / f'TrendLog-{category}-every={every}.png')
            plt.close(grid.figure)


@app['db'].command
def db_extract_pv(
    *,
    conf: ConfigParam,
    tables: tuple[str, ...] = ('th_event', 'th_inverter', 'th_weather'),
    batch_size: int = 10**6,
):
    dirs = conf.db_dirs
    dirs.binary.mkdir(exist_ok=True)

    keapv = MsSql('KEAPV')
    total = sum(math.ceil(keapv.height(x) / batch_size) for x in tables)

    def _iter():
        for table in tables:
            for idx, df in keapv.iter_df(table=table, batch_size=batch_size):
                yield table, idx, df

    for table, idx, df in Progress.trace(_iter(), total=total):
        idx_ = '' if idx is None else f' ({idx})'
        df.write_parquet(dirs.binary / f'KEAPV.{table}{idx_}.parquet')


app.command(App('analyse'))


@app['analyse'].command
def analyse_pv_trend(*, conf: ConfigParam):
    dirs = conf.db_dirs

    inverter = (
        pl.read_parquet(list(dirs.binary.glob('KEAPV.th_inverter*.parquet')))
        .unpivot(['InverterKw', 'InverterKwh', 'InverterTodayKwh'], index='create_date')
        .with_columns()
    )

    weather = (
        pl.read_parquet(list(dirs.binary.glob('KEAPV.th_weather*.parquet')))
        .unpivot(
            ['S_radiation', 'H_radiation', 'mod_temp', 'outdoor_temp'],
            index='create_date',
        )
        .with_columns()
    )

    df = (
        pl.concat([inverter, weather])
        .sort('create_date')
        .group_by_dynamic('create_date', every='1d', group_by='variable')
        .agg(pl.mean('value'))
        .with_columns(
            kind=pl.col('variable').replace({
                'InverterKw': 'kW',
                'InverterKwh': 'kWh',
                'InverterTodayKwh': 'Today kWh',
                'S_radiation': 'Radiation',
                'H_radiation': 'Radiation',
                'mod_temp': 'Temperature',
                'outdoor_temp': 'Temperature',
            })
        )
    )

    grid = (
        sns.FacetGrid(df, row='kind', sharey=False, legend_out=False)
        .map_dataframe(
            sns.lineplot, x='create_date', y='value', hue='variable', alpha=0.8
        )
        .set_axis_labels('', '')
        .set_titles('')
        .set_titles('{row_name}', loc='left', weight=500)
    )

    for ax in grid.axes.flat:
        ax.legend()

    grid.figure.set_size_inches(tuple(x * 1.2 / 2.54 for x in (16, 9)))
    ConstrainedLayoutEngine().execute(grid.figure)

    grid.figure.savefig(conf.dirs.analysis / 'DB-PV trend.png')


if __name__ == '__main__':
    utils.LogHandler.set()
    utils.MplConciseDate().apply()
    utils.MplTheme(context='paper').grid().apply()

    app()

from __future__ import annotations

import dataclasses as dc
import itertools
import math
import os
import re
import warnings
from collections import defaultdict
from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import cyclopts
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import rich
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
    from collections.abc import Sequence


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

    def db_dirs(self):
        return DBDirs(self.dirs.database)


ConfigParam = Annotated[Config, cyclopts.Parameter(name='*')]
app = App(
    config=cyclopts.config.Toml('config/.experiment.toml', use_commands_as_keys=False)
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
    engine: sqlalchemy.Engine = dc.field(init=False)

    def __post_init__(self):
        self.engine = self.create_engine(self.database)

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


@lru_cache
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
def db_tables(
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

    pl.concat(dfs).write_parquet(conf.dirs.database / '[TABLES].parquet')


@app['db'].command
def db_sample(
    *,
    conf: ConfigParam,
    n: int = 1000,
    fragment_n: int = 2,
):
    dirs = conf.db_dirs()
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
        pl.scan_parquet(dirs.root / '[TABLES].parquet', glob=False)
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


app.command(App('trendlog', help='NMWDataLog/TrendLog 해석.'))


@dc.dataclass
class TrendLogParser:
    conf: Config

    datalogs_db: Sequence[str] = (
        'NMWDataLogDB20181120',
        'NMWDataLogDB20210816',
        'NMWDataLogDB20240512',
    )
    object_db: str = 'NMWObjectDB'
    object_tables: Sequence[str] = (
        'AnalogInputObject',
        'AnalogOutputObject',
        'AnalogValueObject',
        'BinaryInputObject',
        'BinaryOutputObject',
        'BinaryValueObject',
    )

    dtype: type[pl.DataType] = pl.Float32
    endianness: Literal['big', 'little'] = 'little'

    def _tables(self):
        return pl.scan_parquet(self.conf.dirs.database / '[TABLES].parquet', glob=False)

    def table_count(self) -> int:
        return (
            self._tables()
            .filter(
                pl.col('TABLE_CATALOG').is_in(self.datalogs_db),
                pl.col('TABLE_NAME').str.starts_with('TrendLog'),
            )
            .select('TABLE_NAME')
            .collect()
            .height
        )

    def objects(self):
        # XXX propid??
        db = MsSql(self.object_db)
        return (
            pl.concat(
                pl.read_database(
                    'SELECT device_identifier, object_identifier, '
                    f'object_name, object_type, description FROM {table}',
                    connection=db.engine,
                )
                for table in self.object_tables
            )
            .rename({'device_identifier': 'deviceid', 'object_identifier': 'objectid'})
            .with_columns()
        )

    def _parse_float(self, data: pl.DataFrame):
        values: list[bytes] = data.select('datavalue').to_series().to_list()

        if prefix := os.path.commonprefix(values):
            data = data.with_columns(
                pl.Series('datavalue', [x.removeprefix(prefix) for x in values])
            )

        return data.with_columns(
            pl.col('datavalue')
            .bin.reinterpret(dtype=self.dtype, endianness=self.endianness)
            .alias('parsedvalue')
        )

    def __iter__(self):
        objects = self.objects()
        tables_lf = self._tables()
        for database in self.datalogs_db:
            db = MsSql(database)
            tables = (
                tables_lf.filter(
                    pl.col('TABLE_CATALOG') == database,
                    pl.col('TABLE_NAME').str.starts_with('TrendLog'),
                )
                .select(pl.col('TABLE_NAME'))
                .collect()
                .to_series()
            )

            for table in tables:
                data = (
                    pl.read_database(f'SELECT * FROM {table}', connection=db.engine)
                    .join(objects, on=['deviceid', 'objectid'], how='left')
                    .with_columns()
                )
                data = self._parse_float(data)

                yield database, table, data

    def iter_dataframe(self):
        for db, table, data in self:
            yield data.select(
                pl.lit(db).alias('database'),
                pl.lit(table).alias('table'),
                pl.all(),
            )


@app['trendlog'].command
def trendlog_parse(*, conf: ConfigParam, batch_size: int = 10):
    """데이터베이스 varbinary 데이터 해석."""
    trend_log = TrendLogParser(conf=conf)

    output = conf.db_dirs().parsed
    output.mkdir(exist_ok=True)

    for idx, data in Progress.trace(
        enumerate(itertools.batched(trend_log.iter_dataframe(), n=batch_size)),
        total=math.ceil(trend_log.table_count() / batch_size),
    ):
        pl.concat(data).write_parquet(output / f'TrendLog{idx:04d}.parquet')


@app['trendlog'].command
def trendlog_objects_list(*, conf: ConfigParam):
    """계측점 리스트."""
    db_dirs = conf.db_dirs()
    lf = pl.scan_parquet(db_dirs.parsed / 'TrendLog*.parquet')
    objects = (
        lf.select(pl.col('object_name').unique().sort()).collect().to_series().to_list()
    )

    rich.print(f'{objects=}')
    with (db_dirs.root / 'TrendLogObjects.txt').open('w') as f:
        for obj in objects:
            f.write(f'{obj}\n')


@app['trendlog'].command
def trendlog_plot(*, conf: ConfigParam, every: str = '1h'):
    """TrendLog 각 변수 시계열 그래프 (검토용)."""
    # TODO

    db_dirs = conf.db_dirs()

    objects_list = (db_dirs.root / 'TrendLogObjects.txt').read_text().splitlines()

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

    lf = pl.scan_parquet(db_dirs.parsed / 'TrendLog*.parquet')
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
    dirs = conf.db_dirs()
    dirs.binary.mkdir(exist_ok=True)

    keapv = MsSql('KEAPV')
    total = sum(ceil(keapv.height(x) / batch_size) for x in tables)

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
    dirs = conf.db_dirs()

    inverter = (
        pl.read_parquet(list(dirs.binary.glob('KEAPV.th_inverter*.parquet')))
        .unpivot(['InverterKw', 'InverterKwh', 'InverterTodayKwh'], index='create_date')
        .with_columns()
    )

    weather = (
        pl.read_parquet(list(dirs.binary.glob('KEAPV.th_weather*.parquet')))
        .unpivot(
            [
                'S_radiation',
                'H_radiation',
                'mod_temp',
                'outdoor_temp',
            ],
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

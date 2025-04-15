from __future__ import annotations

import dataclasses as dc
import shutil
from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import cyclopts
import matplotlib.pyplot as plt
import polars as pl
import rich
import sqlalchemy
import sqlalchemy.exc
import whenever
from loguru import logger

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils import App, Progress

if TYPE_CHECKING:
    from collections.abc import Mapping


@dc.dataclass
class DBDirs:
    root: Path = Path()
    sample: Path = Path('0001.sample')
    binary: Path = Path('0002.binary')
    filtered: Path = Path('0003.filtered')


class _Experiment(exp.Experiment):
    def _plot_pmv(self):
        data = pl.read_parquet(self.conf.dirs.sensor / 'PMV.parquet')

        # 데이터 10개 이하 PMV 측정치
        expr = (pl.col('date') == pl.date(2024, 3, 20)) & (pl.col('floor') == 2)  # noqa: PLR2004
        data = data.filter(expr.not_())

        for (date,), df in data.group_by('date'):
            grid = self.plot_pmv(df)
            grid.savefig(self.conf.dirs.analysis / f'{date}_PMV.png')
            plt.close(grid.figure)


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'kepco_paju'

    databases: tuple[str, ...] = (
        'ksem.pajoo',
        'ksem.pajoo.log',
        'ksem.pajoo.network',
        'ksem.pajoo.raw',
    )
    log_db: str = 'ksem.pajoo.log.filtered'

    db_dirs: DBDirs = dc.field(default_factory=DBDirs)

    def __post_init__(self):
        for field in (f.name for f in dc.fields(self.db_dirs)):
            p = getattr(self.db_dirs, field)
            setattr(self.db_dirs, field, self.dirs.database / p)

    def experiment(self):
        return _Experiment(conf=self)


app = App(
    config=[
        cyclopts.config.Toml(f'config/{x}.toml', use_commands_as_keys=False)
        for x in ['.experiment', '.experiment_kepco_paju']
    ]
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
def db_tables(*, conf: Config):
    """테이블 목록 추출."""
    root = conf.dirs.database
    root.mkdir(exist_ok=True)

    dfs: list[pl.DataFrame] = []

    for db in Progress.trace(conf.databases):
        engine = _db_engine(db=db)
        tables = pl.read_database(
            query='SELECT * FROM INFORMATION_SCHEMA.TABLES',
            connection=engine,
        ).sort(pl.all())

        dfs.append(tables)
        tables.write_excel(root / f'[TABLES] {db}.xlsx', autofit=True)
        logger.info('{} (shape={})', db, tables.shape)

    pl.concat(dfs).write_parquet(root / '[TABLES].parquet')


def _db_sample(
    row: Mapping[str, Any],
    *,
    n: int,
    tail: bool,
    join_tag: bool,
    tag: pl.DataFrame,
):
    schema_table = f'{row["TABLE_SCHEMA"]}.{row["TABLE_NAME"]}'
    name = f'{row["TABLE_CATALOG"]}.{schema_table}'

    logger.info(name)

    engine = _db_engine(row['TABLE_CATALOG'])
    height = pl.read_database(
        f'SELECT COUNT(*) FROM {schema_table}', connection=engine
    ).item()

    ntop = 2 * n if (n < height <= 2 * n) else n
    top = pl.read_database(
        f'SELECT TOP {ntop} * FROM {schema_table}',
        connection=engine,
    )

    if tail and height > 2 * n:
        bottom = (
            pl.read_database(
                f'SELECT TOP {n} * FROM '
                f'(SELECT ROW_NUMBER() OVER (ORDER BY  (SELECT NULL) ) AS RowIndex'
                f', * FROM {schema_table}) AS SubQuery '
                'ORDER BY RowIndex DESC',
                connection=engine,
            )
            .sort('RowIndex')
            .drop('RowIndex')
        )
        sample = pl.concat([top, bottom])
    else:
        sample = top

    if join_tag and 'tagSeq' in sample.columns:
        sample = (
            sample.with_columns(pl.col('tagSeq').cast(pl.Int64))
            .join(tag, on='tagSeq', how='left')
            .with_columns()
        )

    return sample, name, height


@app['db'].command
def db_sample(
    *,
    conf: Config,
    n: int = 1000,
    tail: bool = True,
    join_tag: bool = True,
):
    """각 테이블 샘플 추출."""
    dirs = conf.db_dirs
    dirs.sample.mkdir(exist_ok=True)

    tables = (
        pl.scan_parquet(dirs.root / '[TABLES].parquet', glob=False)
        .filter(pl.col('TABLE_TYPE') == 'BASE TABLE')
        .collect()
    )

    tag = pl.read_database(
        'SELECT tagSeq, tagName, tagDesc FROM T_BECO_TAG',
        connection=_db_engine('ksem.pajoo'),
    ).rename({'tagName': '[tagName]', 'tagDesc': '[tagDesc]'})

    for row in Progress.trace(tables.iter_rows(named=True), total=tables.height):
        try:
            sample, name, height = _db_sample(
                row=row, n=n, tail=tail, join_tag=join_tag, tag=tag
            )
        except sqlalchemy.exc.DBAPIError as e:
            logger.error(str(e))
            continue

        sample.write_excel(dirs.sample / f'{name} (n={height}).xlsx')


def _iter_db_table(database: str, table: str, batch_size: int = 10**7):
    engine = _db_engine(database)
    height = pl.read_database(f'SELECT COUNT(*) FROM {table}', connection=engine).item()

    if height < batch_size:
        df = pl.read_database(f'SELECT * FROM {table}', connection=engine)
        yield None, df
        return

    logger.info('height={}, {} batches', height, ceil(height / batch_size))

    yield from enumerate(
        pl.read_database(
            f'SELECT * FROM {table}',
            connection=engine,
            iter_batches=True,
            batch_size=batch_size,
        )
    )


@app['db'].command
def db_db2bin(
    *,
    conf: Config,
    join_tag: bool = True,
    batch_size: int = 10**7,
    fifteen_min: bool = False,
):
    """주요 파일 parquet으로 변환."""
    dirs = conf.db_dirs
    dirs.binary.mkdir(exist_ok=True)

    tables = (
        pl.scan_parquet(dirs.root / '[TABLES].parquet', glob=False)
        .filter(pl.col('TABLE_TYPE') == 'BASE TABLE')
        .collect()
    )
    tag = (
        pl.read_database(
            'SELECT tagSeq, tagName, tagDesc FROM T_BECO_TAG',
            connection=_db_engine('ksem.pajoo'),
        )
        .rename({'tagName': '[tagName]', 'tagDesc': '[tagDesc]'})
        .with_columns()
    )
    skip = ['_POINT_', '_ELEC_', '_FACILITY_', 'T_BECO_TAG']

    for row in Progress.trace(tables.iter_rows(named=True), total=tables.height):
        table_name = row['TABLE_NAME']
        schema_table = f'{row["TABLE_SCHEMA"]}.{row["TABLE_NAME"]}'

        if not any(x in table_name for x in skip):
            continue
        if not (fifteen_min ^ ('15MIN' not in table_name)):
            continue

        logger.info(table_name)

        for idx, table in _iter_db_table(
            database=row['TABLE_CATALOG'],
            table=schema_table,
            batch_size=batch_size,
        ):
            if idx is not None:
                logger.info('idx={}', idx)

            if join_tag and 'tagSeq' in table.columns:
                data = (
                    table.with_columns(pl.col('tagSeq').cast(pl.Int64))
                    .join(tag, on='tagSeq', how='left')
                    .with_columns()
                )
            else:
                data = table

            idx_ = '' if idx is None else f' ({idx})'
            data.write_parquet(
                dirs.binary / f'{row["TABLE_CATALOG"]}.{schema_table}{idx_}.parquet'
            )


@app['db'].command
def db_extract_misc(*, conf: Config, log_db: str | None = None):
    """기타 정보 추출."""
    dirs = conf.db_dirs
    log_db = log_db or conf.log_db
    prefix = f'{log_db}.dbo.'

    # ELEC, FACILITY, POINT tag 목록
    files = [
        f'{prefix}T_BELO_ELEC_DAY',
        f'{prefix}T_BELO_FACILITY_DAY',
        f'{prefix}T_BECO_POINT_CONTROL',
    ]

    console = rich.get_console()
    for file in files:
        path = dirs.binary / f'{file}.parquet'
        logger.info(path)

        try:
            tags = (
                pl.scan_parquet(path)
                .select('tagSeq', '[tagName]', '[tagDesc]')
                .rename(lambda x: x.strip('[]'))
                .unique()
                .sort('tagSeq')
                .collect()
            )
        except OSError as e:
            logger.error(repr(e))
            continue

        console.print(f'DB: "{file}"\n{tags}')

        tags.write_excel(dirs.root / f'[tags]{file}.xlsx', column_widths=200)


@app['db'].command
def db_extract_after(*, conf: Config, date: str = '2024-07-01'):
    output = conf.db_dirs.filtered
    output.mkdir(exist_ok=True)

    d = whenever.Date.parse_common_iso(date).py_date()
    update_date = 'updateDate'

    paths = list(conf.db_dirs.binary.glob('*.parquet'))

    for path in Progress.trace(paths):
        logger.info(path.relative_to(conf.db_dirs.binary))

        lf = pl.scan_parquet(path)
        if update_date not in lf.collect_schema().names():
            shutil.copy2(path, output)
            continue

        df = lf.filter(pl.col(update_date) >= d).collect()

        if not df.height:
            continue

        df.write_parquet(output / path.name)


if __name__ == '__main__':
    utils.LogHandler.set()
    utils.MplConciseDate().apply()
    utils.MplTheme(palette='tol:vibrant').grid().apply()

    app()

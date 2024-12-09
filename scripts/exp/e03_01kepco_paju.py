from __future__ import annotations

import dataclasses as dc
from functools import lru_cache
from math import ceil
from typing import TYPE_CHECKING, Annotated, Literal

import matplotlib.pyplot as plt
import more_itertools as mi
import polars as pl
import rich
import sqlalchemy
from cyclopts import App, Parameter
from loguru import logger
from rich.progress import track

from greenbutton import utils
from scripts.sensor import Experiment

if TYPE_CHECKING:
    from pathlib import Path


DB_LIST = ('ksem.pajoo', 'ksem.pajoo.log', 'ksem.pajoo.network', 'ksem.pajoo.raw')


class KepcoPajuExperiment(Experiment):
    def plot_sensors(self, *, pmv: bool = True, tr7: bool = True):
        if pmv:
            df_pmv = pl.read_parquet(self.dirs.ROOT / '[DATA] PMV.parquet', glob=False)

            if self.date == '2024-03-20':
                # 데이터 10개 이하
                df_pmv = df_pmv.filter(pl.col('floor') != 2)  # noqa: PLR2004

            grid = self.plot_pmv(df_pmv)
            grid.savefig(self.dirs.PLOT / 'PMV.png')
            plt.close(grid.figure)

        if tr7:
            df_tr7 = pl.read_parquet(self.dirs.ROOT / '[DATA] TR7.parquet', glob=False)
            for grid, var in self.plot_tr7(df_tr7):
                grid.savefig(self.dirs.PLOT / f'TR7-{var}.png')
                plt.close(grid.figure)


@dc.dataclass
class DBDirs:
    source: dc.InitVar[Path | Experiment]

    root: Path = dc.field(init=False)
    sample: Path = dc.field(init=False)
    parquet: Path = dc.field(init=False)
    analysis: Path = dc.field(init=False)

    def __post_init__(self, source: Path | Experiment):
        r = source.dirs.DB if isinstance(source, Experiment) else source

        self.root = r
        self.sample = r / '01sample'
        self.parquet = r / '02parquet'
        self.analysis = r / '03analysis'


@dc.dataclass
class Config:
    building: str = 'kepco_paju'
    date: Literal['2024-03-20', '2024-07-11'] | None = None

    pmv: bool = True
    tr7: bool = True

    xlsx: bool = False

    def experiment(self):
        return KepcoPajuExperiment(building=self.building, date=self.date)

    def db_dirs(self):
        return DBDirs(self.experiment())


_Config = Annotated[Config, Parameter(name='*')]

app = App()
DEFAULT_CONFIG = Config()


@app.command
def parse_sensors(*, conf: _Config = DEFAULT_CONFIG):
    exp = conf.experiment()
    exp.parse_sensors(
        pmv=conf.pmv, tr7=conf.tr7, write_parquet=True, write_xlsx=conf.xlsx
    )


@app.command
def plot_sensors(*, conf: _Config = DEFAULT_CONFIG):
    exp = conf.experiment()
    exp.plot_sensors(pmv=conf.pmv, tr7=conf.tr7)


@lru_cache
def _db_engine(db: str, **kwargs):
    url = sqlalchemy.URL.create(
        'mssql+pyodbc',
        host='localhost',
        database=db,
        query={'driver': 'ODBC Driver 17 for SQL Server'},
    )
    return sqlalchemy.create_engine(url, **kwargs)


@app.command
def db_tables(
    databases: str | tuple[str, ...] = DB_LIST, *, conf: _Config = DEFAULT_CONFIG
):
    """테이블 목록 추출."""
    root = conf.experiment().dirs.DB
    root.mkdir(exist_ok=True)

    dfs: list[pl.DataFrame] = []

    for db in track(tuple(mi.always_iterable(databases))):
        engine = _db_engine(db=db)
        tables = pl.read_database(
            query='SELECT * FROM INFORMATION_SCHEMA.TABLES',
            connection=engine,
        ).sort(pl.all())

        dfs.append(tables)
        tables.write_excel(root / f'[TABLES] {db}.xlsx', autofit=True)
        logger.info('{} (shape={})', db, tables.shape)

    pl.concat(dfs).write_parquet(root / '[TABLES].parquet')


@app.command
def db_sample(
    *,
    n: int = 1000,
    tail: bool = True,
    join_tag: bool = True,
    conf: _Config = DEFAULT_CONFIG,
):
    """각 테이블 샘플 추출."""
    dirs = conf.db_dirs()
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

    for row in track(tables.iter_rows(named=True), total=tables.height):
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
            sample = sample.with_columns(pl.col('tagSeq').cast(pl.Int64)).join(
                tag, on='tagSeq', how='left'
            )

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


@app.command
def db2parquet(
    *,
    join_tag: bool = True,
    batch_size: int = 10**7,
    fifteen_min: bool = False,
    conf: _Config = DEFAULT_CONFIG,
):
    """주요 파일 parquet으로 변환."""
    dirs = conf.db_dirs()
    dirs.parquet.mkdir(exist_ok=True)

    tables = (
        pl.scan_parquet(dirs.root / '[TABLES].parquet', glob=False)
        .filter(pl.col('TABLE_TYPE') == 'BASE TABLE')
        .collect()
    )

    tag = pl.read_database(
        'SELECT tagSeq, tagName, tagDesc FROM T_BECO_TAG',
        connection=_db_engine('ksem.pajoo'),
    ).rename({'tagName': '[tagName]', 'tagDesc': '[tagDesc]'})

    for row in track(tables.iter_rows(named=True), total=tables.height):
        table = row['TABLE_NAME']
        schema_table = f'{row["TABLE_SCHEMA"]}.{row["TABLE_NAME"]}'

        if not any(
            x in table for x in ['_POINT_', '_ELEC_', '_FACILITY_', 'T_BECO_TAG']
        ):
            continue

        if not (fifteen_min ^ ('15MIN' not in table)):
            continue

        logger.info(table)

        for idx, df in _iter_db_table(
            database=row['TABLE_CATALOG'],
            table=schema_table,
            batch_size=batch_size,
        ):
            if idx is not None:
                logger.info('idx={}', idx)

            if join_tag and 'tagSeq' in df.columns:
                dftag = df.with_columns(pl.col('tagSeq').cast(pl.Int64)).join(
                    tag, on='tagSeq', how='left'
                )
            else:
                dftag = df

            idx_ = '' if idx is None else f' ({idx})'
            dftag.write_parquet(
                dirs.parquet / f'{row["TABLE_CATALOG"]}.{schema_table}{idx_}.parquet'
            )


@app.command
def db_misc(*, conf: _Config = DEFAULT_CONFIG):
    """기타 정보 추출."""
    dirs = conf.db_dirs()

    # ELEC, FACILITY, POINT tag 목록
    files = [
        'ksem.pajoo.log.dbo.T_BELO_ELEC_DAY',
        'ksem.pajoo.log.dbo.T_BELO_FACILITY_DAY',
        'ksem.pajoo.log.dbo.T_BECO_POINT_CONTROL',
    ]

    console = rich.get_console()
    for file in files:
        path = dirs.parquet / f'{file}.parquet'

        tags = (
            pl.scan_parquet(path)
            .select('tagSeq', '[tagName]', '[tagDesc]')
            .rename(lambda x: x.strip('[]'))
            .unique()
            .sort('tagSeq')
            .collect()
        )

        console.print(f'{file}\n{tags}')

        tags.write_excel(
            dirs.root / f'tags_{file.removeprefix("ksem.pajoo.log.dbo.T_")}.xlsx',
            column_widths=200,
        )


if __name__ == '__main__':
    utils.LogHandler.set()
    utils.MplConciseDate().apply()
    utils.MplTheme(palette='tol:vibrant').grid().apply()

    app()

from __future__ import annotations

import dataclasses as dc
from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import cyclopts
import matplotlib.pyplot as plt
import more_itertools as mi
import polars as pl
import rich
import sqlalchemy
import sqlalchemy.exc
from loguru import logger

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils import App, Progress

if TYPE_CHECKING:
    from collections.abc import Mapping


@dc.dataclass
class DBDirs:
    root: Path
    sample: Path = Path('01.sample')
    parquet: Path = Path('02.parquet')
    weather: Path = Path('03.weather')

    def __post_init__(self):
        self.update()

    def update(self):
        for field in (f.name for f in dc.fields(self)):
            if field == 'root':
                continue

            p = getattr(self, field)
            setattr(self, field, self.root / p)

        return self


class _Experiment(exp.Experiment):
    def _plot_pmv(self):
        data = pl.read_parquet(self.conf.dirs.sensor / 'PMV.parquet')

        data = data.filter(
            # 데이터 10개 이하 PMV 제외
            (
                (pl.col('date') == pl.date(2024, 3, 20))  ##
                & (pl.col('floor') == 2)  # noqa: PLR2004
            ).not_()
        )

        for (date,), df in data.group_by('date'):
            grid = self.plot_pmv(df)
            grid.savefig(self.conf.dirs.analysis / f'{date}_PMV.png')
            plt.close(grid.figure)


@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING = 'kepco_paju'

    def experiment(self):
        return _Experiment(conf=self)

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
    databases: str | tuple[str, ...] = (
        'ksem.pajoo',
        'ksem.pajoo.log',
        'ksem.pajoo.network',
        'ksem.pajoo.raw',
    ),
    *,
    conf: ConfigParam,
):
    """테이블 목록 추출."""
    root = conf.dirs.database
    root.mkdir(exist_ok=True)

    dfs: list[pl.DataFrame] = []

    for db in Progress.trace(tuple(mi.always_iterable(databases))):
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
        sample = sample.with_columns(pl.col('tagSeq').cast(pl.Int64)).join(
            tag, on='tagSeq', how='left'
        )

    return sample, name, height


@app['db'].command
def db_sample(
    *,
    n: int = 1000,
    tail: bool = True,
    join_tag: bool = True,
    conf: ConfigParam,
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
def db_db2parquet(
    *,
    join_tag: bool = True,
    batch_size: int = 10**7,
    fifteen_min: bool = False,
    conf: ConfigParam,
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

    for row in Progress.trace(tables.iter_rows(named=True), total=tables.height):
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


@app['db'].command
def db_misc(*, conf: ConfigParam):
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

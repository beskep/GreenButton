from __future__ import annotations

import dataclasses as dc
import functools
import itertools
import math
import operator
import re
import typing
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple

import cyclopts
import matplotlib.pyplot as plt
import pathvalidate
import polars as pl
import polars.selectors as cs
import seaborn as sns
import sqlalchemy
from loguru import logger
from matplotlib.layout_engine import ConstrainedLayoutEngine

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils.cli import App
from greenbutton.utils.terminal import Progress

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from polars._typing import FrameType


Log = Literal['TrendLog', 'EventLog', 'ControlLog', 'ModifyLog']
LogShort = Literal['trend', 'event', 'control', 'modify']


def _norm_log(v: Log | LogShort, /) -> Log:
    if v in typing.get_args(LogShort):
        return f'{v.title()}Log'  # type: ignore[return-value]

    return v  # type: ignore[return-value]


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


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'kea'

    @functools.cached_property
    def db_dirs(self):
        return DBDirs(self.dirs.database)


class Cnst:
    TC = 'TABLE_CATALOG'
    TS = 'TABLE_SCHEMA'
    TN = 'TABLE_NAME'

    DEV_ID = 'deviceid'
    OBJ_ID = 'objectid'
    IDS = (DEV_ID, OBJ_ID)

    OBJ_NAME = 'object_name'
    OBJ_TYPE = 'object_type'


app = App(
    config=cyclopts.config.Toml('config/.experiment.toml', use_commands_as_keys=False),
    result_action=['call_if_callable', 'print_non_int_sys_exit'],
)


@app.command
def init(*, conf: Config):
    conf.dirs.mkdir()


# ================================= sensor ================================


app.command(App('sensor'))


@app['sensor'].command
def sensor_parse(*, conf: Config, parquet: bool = True, xlsx: bool = True):
    exp = conf.experiment()
    exp.parse_sensors(write_parquet=parquet, write_xlsx=xlsx)


@app['sensor'].command
def sensor_plot(*, conf: Config, pmv: bool = True, tr7: bool = True):
    exp = conf.experiment()
    exp.plot_sensors(pmv=pmv, tr7=tr7)


# ================================= DB ================================


@dc.dataclass
class MsSql:
    database: str

    @functools.cached_property
    def engine(self):
        return self.create_engine(self.database)

    @staticmethod
    @functools.lru_cache
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


app.command(App('db'))


@app['db'].command
def db_extract_tables(
    *,
    conf: Config,
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
        engine = MsSql.create_engine(db=db)
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
@dc.dataclass
class Sample:
    """샘플 추출."""

    conf: Config
    n: int = 1000
    fragment_n: int = 2
    fragmented_prefix: tuple[str, ...] = (
        'EventLog',
        'TrendLog',
        'ConnectionLog',
        'ControlLog',
        'ModifyLog',
        'UserSystemLog',
    )

    @property
    def dirs(self):
        return self.conf.db_dirs

    @property
    def fragmented_pattern(self):
        return re.compile(f'^(.*)({"|".join(self.fragmented_prefix)}).*')

    def _sample(self, row: dict):
        is_fragmented: bool = (
            'Log' in row[Cnst.TC]  # fmt
            and row[Cnst.TN].startswith(self.fragmented_prefix)
        )
        fragmented: defaultdict[str, list[pl.DataFrame]] = defaultdict(list)

        if not is_fragmented:
            logger.info('{}, fragmented={}', row['catalog'], is_fragmented)

        engine = MsSql.create_engine(row[Cnst.TC])
        height = pl.read_database(
            f'SELECT COUNT(*) FROM {row["catalog"]}', connection=engine
        ).item()
        sample = pl.read_database(
            f'SELECT TOP {self.fragment_n if is_fragmented else self.n} '
            f'* FROM {row["schema"]}',
            connection=engine,
        ).with_columns(cs.binary().bin.encode('hex'))

        if is_fragmented:
            name = self.fragmented_pattern.sub(r'\g<1>\g<2>', row['catalog'])
            fragmented[name].append(sample)
        else:
            name = f'{row["catalog"]} (n={height}).xlsx'
            if height == 0:
                name = f'(ZERO) {name}'

            sample.write_excel(self.dirs.sample / name)

        return fragmented

    def __call__(self):
        self.dirs.sample.mkdir(exist_ok=True)

        tables = (
            pl
            .scan_parquet(self.dirs.binary / 'TABLES.parquet', glob=False)
            .filter(pl.col('TABLE_TYPE') == 'BASE TABLE')
            .with_columns(
                schema=pl.format('{}.{}', Cnst.TS, Cnst.TN),
                catalog=pl.format('{}.{}.{}', Cnst.TC, Cnst.TS, Cnst.TN),
            )
            .collect()
        )

        for by, df in tables.group_by(Cnst.TC, maintain_order=True):
            logger.info('TABLE_CATALOG={}', by[0])

            if 'NWCS.Client' in by[0]:
                # 에러
                continue

            fragmented: dict[str, list[pl.DataFrame]] = {}

            for row in Progress.iter(df.iter_rows(named=True), total=df.height):
                fragmented |= self._sample(row)

            for key, dfs in fragmented.items():
                logger.info(key)
                pl.concat(dfs).write_excel(
                    self.dirs.sample / f'{key}.xlsx', autofit=True
                )


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
                f'object_type, object_name, description FROM {table}',
                connection=db.engine,
            ).select(pl.lit(table).alias(Cnst.TN), pl.all())

    def __call__(self):
        return pl.concat(self).rename({
            'device_identifier': Cnst.DEV_ID,
            'object_identifier': Cnst.OBJ_ID,
        })


@app['db'].command
def db_extract_points(*, conf: Config):
    dir_ = conf.db_dirs.binary
    dir_.mkdir(exist_ok=True)

    tables = (
        pl
        .scan_parquet(conf.db_dirs.binary / 'TABLES.parquet', glob=False)
        .filter(pl.col(Cnst.TC) == 'NMWObjectDB')
        .select(Cnst.TN)
        .collect()
        .to_series()
        .to_list()
    )

    points = _PointsExtractor(tables=tables)().unique().sort(pl.all())

    path = dir_ / 'NMWObjectDB.Points.parquet'
    points.write_parquet(path)
    points.write_excel(path.with_suffix('.xlsx'), column_widths=200)


@app['db'].command
def db_extract_pv(
    *,
    conf: Config,
    tables: tuple[str, ...] = ('th_event', 'th_inverter', 'th_weather'),
    batch_size: int = 10**6,
):
    dirs = conf.db_dirs
    dirs.binary.mkdir(exist_ok=True)

    keapv = MsSql('KEAPV')
    total = sum(math.ceil(keapv.height(x) / batch_size) for x in tables)

    def it():
        for table in tables:
            for idx, df in keapv.iter_df(table=table, batch_size=batch_size):
                yield table, idx, df

    for table, idx, df in Progress.iter(it(), total=total):
        idx_ = '' if idx is None else f' ({idx})'
        df.write_parquet(dirs.binary / f'KEAPV.{table}{idx_}.parquet')


# ================================= Log ================================


app.command(App('log', help='NMWDataLog TrendLog/EventLog 해석.'))


@dc.dataclass
class _NMWLogExtractor:
    tables_path: dc.InitVar[Path]
    tables: pl.DataFrame = dc.field(init=False)

    log: Log | LogShort

    read_batch: int = 1
    write_batch: int = 2**25
    trace: bool = True
    concat_extracted: bool = True
    fmt: str = '{catalog}.{log}.batch{idx}'

    def __post_init__(self, tables_path: Path):
        self.log = _norm_log(self.log)

        match self.log:
            case 'TrendLog' | 'EventLog':
                prefix = 'NMWDataLog'
            case 'ControlLog' | 'ModifyLog':
                prefix = 'NMWSysLog'
            case _:
                raise ValueError(self.log)

        self.tables = (
            pl
            .scan_parquet(tables_path, glob=False)
            .select(Cnst.TC, Cnst.TN)
            .filter(
                pl.col(Cnst.TC).str.starts_with(prefix),
                pl.col(Cnst.TN).str.starts_with(self.log),
            )
            .collect()
        )

    def iter_frame(self, catalog: str) -> Iterable[pl.DataFrame]:
        tables = (
            self.tables
            .filter(pl.col(Cnst.TC) == catalog)
            .select(pl.col(Cnst.TN).sort())
            .to_series()
        )
        db = MsSql(catalog)

        for table in tables:
            yield db.read_df(table).select(pl.lit(table).alias(Cnst.TN), pl.all())

    def iter(self, catalog: str):
        it: Iterable[tuple[int, tuple[pl.DataFrame, ...]]] = enumerate(
            itertools.batched(self.iter_frame(catalog), n=self.read_batch, strict=False)
        )

        if self.trace:
            total = math.ceil(
                self.tables.filter(pl.col(Cnst.TC) == catalog).height / self.read_batch
            )
            it = Progress.iter(it, description=f'Extracting {catalog}', total=total)

        for idx, dfs in it:
            yield (
                catalog,
                idx,
                self.fmt.format(catalog=catalog, log=self.log, idx=f'{idx:04d}'),
                pl.concat(dfs),
            )

    def concat_and_write(self, catalog: str, directory: Path):
        fmt = self.fmt.format(catalog=catalog, log=self.log, idx='')
        files = list(directory.glob(f'{fmt}*.parquet'))
        logger.debug('{} files={}', catalog, files)

        for idx, df in enumerate(
            pl.scan_parquet(files).collect().iter_slices(self.write_batch)
        ):
            df.write_parquet(
                directory.parent / f'{catalog}.{self.log}{idx:04d}.parquet'
            )

    def __call__(self, output: Path, catalogs: Sequence[str] | None = None):
        if catalogs is None:
            catalogs = (
                self.tables
                .select(pl.col(Cnst.TC).unique().sort())
                .to_series()
                .to_list()
            )

        for catalog in catalogs:
            for _, _, name, data in self.iter(catalog=catalog):
                data.write_parquet(output / f'{name}.parquet')

            if self.concat_extracted:
                self.concat_and_write(catalog=catalog, directory=output)


@app['log'].command
def log_parse_samples(
    *,
    conf: Config,
    n: int = 200,
    seed: int = 42,
    parse_int: bool = True,
):
    """DB별 샘플링/해석 테스트."""

    def scan(path: Path):
        logger.info(path)
        log = re.findall(r'(EventLog|TrendLog)', path.stem)[0]
        return (
            pl
            .scan_parquet(path)
            .select(
                pl.lit(path.stem.split('.')[0]).alias('catalog'),
                pl.lit(log).alias('log'),
                pl.col('datavalue').sample(
                    n=n, with_replacement=True, shuffle=True, seed=seed
                ),
            )
            .unique()
            .with_columns(
                pl
                .col('datavalue')
                .map_elements(len, return_dtype=pl.Int64)
                .alias('len'),
                pl
                .col('datavalue')
                .map_elements(operator.itemgetter(0), return_dtype=pl.Int64)
                .alias('prefix'),
            )
        )

    dtypes: list[type[pl.DataType]] = [pl.Float32, pl.Float64]
    if parse_int:
        dtypes.extend([pl.Int8, pl.Int16, pl.Int32, pl.Int64])

    def convert(data: FrameType, max_len=17):
        for length, s in itertools.product([4, 8, 16], range(max_len)):
            data = data.with_columns(
                pl
                .col('datavalue')
                .map_elements(
                    operator.itemgetter(slice(s, s + length)),
                    return_dtype=pl.Binary,
                )
                .alias(f'_slice_{s}_{s + length}')
            )

        data = (
            data
            .unpivot(
                on=cs.starts_with('_slice'),
                index=['catalog', 'log', 'datavalue', 'len', 'prefix'],
                value_name='slice',
            )
            .with_columns(
                pl.col('variable').str.extract_groups(
                    r'_slice_(?<start>\d+)_(?<end>\d+)'
                )
            )
            .unnest('variable')
            .with_columns(pl.col('start', 'end').cast(pl.Int8))
        )

        for dtype in dtypes:
            try:
                data = data.with_columns(
                    pl
                    .col('slice')
                    .bin.reinterpret(dtype=dtype, endianness='little')
                    .cast(pl.Float64 if dtype is not pl.String else pl.String)
                    .alias(f'{dtype}-little')
                )
            except pl.exceptions.PolarsError:
                pass

            try:
                data = data.with_columns(
                    pl
                    .col('slice')
                    .bin.reinterpret(dtype=dtype, endianness='big')
                    .cast(pl.Float64 if dtype is not pl.String else pl.String)
                    .alias(f'{dtype}-big')
                )
            except pl.exceptions.PolarsError:
                pass

        return data.filter(
            ~pl.all_horizontal(cs.ends_with('-little', '-big').is_null())
        )

    d = conf.db_dirs.binary
    data = (
        pl
        .concat(
            [convert(scan(p)) for p in d.glob('NMWDataLog*.parquet')],
            how='diagonal',
        )
        .with_columns(
            pl.col('datavalue', 'slice').map_elements(repr, return_dtype=pl.String),
        )
        .collect()
    )
    data.write_excel(
        conf.dirs.database / 'ParsedSample.xlsx',
        column_widths=100,
    )


@app['log'].command
def log_extract(
    *,
    conf: Config,
    log: Log | LogShort,
    read_batch: int = 10,
):
    """Raw 데이터 parquet으로 저장."""
    dirs = conf.db_dirs
    output = dirs.binary / f'{log.title()}Log'
    output.mkdir(exist_ok=True)

    extractor = _NMWLogExtractor(
        tables_path=dirs.binary / 'TABLES.parquet',
        log=log,
        read_batch=read_batch,
    )
    extractor(output=output)


class _Point(NamedTuple):
    deviceid: int
    objectid: int
    type_: int  # object_type
    name: str | None

    def __str__(self):
        name = 'NULL' if self.name is None else self.name.replace('/', '-')
        return f'D{self.deviceid}_P{self.objectid}_{name}'


@dc.dataclass
class _NMWLogParser:
    conf: Config

    log: Log | LogShort
    catalogs: Sequence[str] = (
        'NMWDataLogDB20181120',
        'NMWDataLogDB20210816',
        'NMWDataLogDB20240512',
    )

    binary: str = 'datavalue'
    parsed: str = 'parsed_value'

    dtypes: dict[tuple[int, int], type[pl.DataType]] = dc.field(
        default_factory=lambda: {
            # (prefix, len): dtype
            (9, 2): pl.Int16,  # \t\00 + \x?? * 2 => on/off state
            (4, 4): pl.Float32,  # \x04\x00 + \x?? * 4 => numeric
            (5, 8): pl.Float64,  # \x05\x00 + \x?? * 8 => numeric
        }
    )
    endianness: Literal['big', 'little'] = 'little'

    _points: pl.LazyFrame = dc.field(init=False)

    def __post_init__(self):
        self.log = _norm_log(self.log)
        self._points = pl.scan_parquet(
            self.conf.db_dirs.binary / 'NMWObjectDB.Points.parquet'
        )

    def lazy_frames(self):
        for path in self.conf.db_dirs.binary.glob(f'*{self.log}*.parquet'):
            yield (
                pl.scan_parquet(path).select(
                    pl.lit(path.name.split('.')[0]).alias(Cnst.TC), pl.all()
                )
            )

    def match_points(self):
        return (
            pl
            .concat(self.lazy_frames())
            .select(Cnst.TC, *Cnst.IDS)
            .unique()
            .join(self._points, on=Cnst.IDS, how='left')
            .sort(pl.all())
        )

    def parse(self, data: FrameType):
        tail = '__tail__'
        data = data.with_columns(
            pl
            .col(self.binary)
            .map_elements(
                # x -> x[2:]
                operator.itemgetter(slice(2, None)),
                return_dtype=pl.Binary,
            )
            .alias(tail)
        )

        # (데이터 prefix, prefix 제외한 데이터 길이)
        key = tuple(
            data
            .lazy()
            .head(1)
            .select(
                pl
                .col(self.binary)
                .map_elements(operator.itemgetter(0), return_dtype=pl.Int64)
                .cast(pl.Int64),
                pl.col(tail).map_elements(len, return_dtype=pl.Int64),
            )
            .collect()
            .to_numpy()
            .ravel()
        )

        return data.with_columns(
            pl
            .col(tail)
            .bin.reinterpret(dtype=self.dtypes[key], endianness=self.endianness)
            .cast(pl.Float64)
            .alias(self.parsed)
        ).drop(tail)

    def _parse_point(self, point: _Point):
        data = (
            pl
            .concat(self.lazy_frames())
            .filter(
                pl.col(Cnst.DEV_ID) == point.deviceid,
                pl.col(Cnst.OBJ_ID) == point.objectid,
            )
            .collect()
        )

        return (
            pl
            .concat(self.parse(df) for _, df in data.group_by(Cnst.TN))
            .rename({'timeStamp': 'datetime'})
            .sort('datetime')
            .with_columns()
        )

    def _parse_and_write(self, point: _Point, *, skip_exists: bool = True):
        output = self.conf.db_dirs.parsed
        path = output / f'{self.log}_{point}.parquet'

        if skip_exists and path.exists():
            return

        data = self._parse_point(point)

        # log data/null count
        count = (
            data.select(pl.col(self.parsed).count()).item(),
            data.select(pl.col(self.parsed).null_count()).item(),
        )
        logger.info(
            'count={} | null_count={} | null_ratio={:.6f}',
            count[0],
            count[1],
            count[1] / (count[0] + count[1]),
        )

        # save
        if data.drop_nulls(self.parsed).height:
            data.with_columns(pl.lit(point.name).alias('point')).write_parquet(path)
            data = data.drop_nulls(self.parsed).drop(cs.binary())
            pl.concat([data.head(50), data.tail(50)]).write_excel(
                path.with_suffix('.xlsx')
            )

    def batch_parse(self, points: pl.DataFrame, *, skip_exists: bool = True):
        points = (
            points
            .select(Cnst.DEV_ID, Cnst.OBJ_ID, Cnst.OBJ_TYPE, Cnst.OBJ_NAME)
            .unique()
            .sort(pl.all())
        )

        for row in Progress.iter(points.iter_rows(), total=points.height):
            point = _Point(*row)
            logger.info(point)
            self._parse_and_write(point, skip_exists=skip_exists)


@app['log'].command
def log_match_points(*, conf: Config, log: Log | LogShort):
    log = _norm_log(log)
    parser = _NMWLogParser(conf=conf, log=log)
    points = parser.match_points().collect()

    points.write_parquet(conf.dirs.database / f'NMW{log}Points.parquet')
    points.write_excel(conf.dirs.database / f'NMW{log}Points.xlsx', column_widths=200)

    # -> EventLog는 매치되는 device, object id 없음


@app['log'].command
def log_parse_binary(
    *,
    conf: Config,
    log: Literal['trend', 'event'],
    skip_exists: bool = True,
):
    """varbinary로 저장된 EventLog, TrendLog 데이터 해석."""
    conf.db_dirs.parsed.mkdir(exist_ok=True)
    norm_log = _norm_log(log)

    points = pl.read_parquet(conf.dirs.database / f'NMW{norm_log}Points.parquet')
    parser = _NMWLogParser(conf=conf, log=norm_log)
    parser.batch_parse(points, skip_exists=skip_exists)


@dc.dataclass
class TrendLogPlotter:
    conf: Config
    points: dc.InitVar[str] = 'config/experiment.KEA.points.csv'
    evergy: str = '1h'

    _points: pl.DataFrame = dc.field(init=False)

    def __post_init__(self, points: str):
        name = pl.col(Cnst.OBJ_NAME)
        self._points = (
            pl
            .read_csv(points)
            .with_columns(cs.string().str.strip_chars())
            .with_columns(
                name.str.extract('(공급|배기|환기|급수|환수)').alias('var1'),
                pl
                .when(pl.col(Cnst.OBJ_TYPE).is_in([3, 4]))
                .then(pl.lit('상태'))
                .otherwise(name.str.extract('(온도|습도|수위|유량|CO2)'))
                .fill_null('etc')
                .alias('var2'),
                name.str
                .replace('ohu', 'OHU')
                .str.extract(r'((AHU|OHU|HVU|EF|OF|P))-?\d+')
                .replace({'OHU': 'OHU/HVU', 'HVU': 'OHU/HVU'})
                .alias('AH'),
                name.str.extract(r'((지하)?\d+층)').alias('floor'),
            )
            .with_columns(
                pl.concat_str(
                    ['group', 'floor', 'var1'], separator=' ', ignore_nulls=True
                ).alias('group'),
                pl.concat_str(['AH', 'var2'], separator=' ', ignore_nulls=True).alias(
                    'variable'
                ),
            )
        )

    def _iter_lf(self, group: str | None):
        src = self.conf.db_dirs.parsed
        paths = (
            pl
            .DataFrame({'path': [x.name for x in src.glob('TrendLog*.parquet')]})
            .with_columns(
                pl
                .col('path')
                .str.extract_groups(
                    r'^TrendLog_D(?P<deviceid>\d+)_P(?P<objectid>\d+)_.*'
                )
                .alias('match')
            )
            .unnest('match')
            .with_columns(pl.col(Cnst.IDS).cast(pl.Int64))
            .join(self._points, on=Cnst.IDS, how='left')
            .filter(
                pl.col('group').is_null() if group is None else pl.col('group') == group
            )
        )

        if not paths.height:
            raise FileNotFoundError(group)

        for p, n, g, v in paths.select(
            'path', Cnst.OBJ_NAME, 'group', 'variable'
        ).iter_rows():
            yield pl.scan_parquet(src / p).with_columns(
                pl.lit(p).alias('path'),
                pl.lit(n).alias(Cnst.OBJ_NAME),
                pl.lit(g).alias('group'),
                pl.lit(v).alias('variable'),
                pl.col('parsed_value').cast(pl.Float64),
            )

    def data(self, group: str | None):
        lf = pl.concat(self._iter_lf(group))
        group_by = [Cnst.DEV_ID, Cnst.OBJ_ID, Cnst.OBJ_NAME, 'variable']
        return (
            lf
            .sort('datetime')
            .group_by_dynamic('datetime', every=self.evergy, group_by=group_by)
            .agg(pl.mean('parsed_value'))
            .collect()
            .upsample('datetime', every=self.evergy, group_by=group_by)
        )

    def plot(self, group: str | None):
        data = self.data(group).sort(Cnst.OBJ_NAME, 'variable', 'datetime')

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, module='seaborn.axisgrid'
            )

            grid = (
                sns
                .FacetGrid(
                    data, row='variable', sharey=False, height=2, aspect=2.5 * 16 / 9
                )
                .map_dataframe(
                    sns.lineplot,
                    x='datetime',
                    y='parsed_value',
                    hue=Cnst.OBJ_NAME,
                    alpha=0.5,
                )
                .set_axis_labels('', 'value')
            )
            for ax in grid.axes.ravel():
                ax.legend()

        return grid

    def plot_and_save(self):
        output = self.conf.dirs.analysis / 'TrendLog'
        output.mkdir(exist_ok=True)

        groups = (
            self._points
            .select(pl.col('group').unique().sort().drop_nulls())
            .with_columns()
            .to_series()
        )

        for group in [None, *groups]:
            logger.info('group={}', group)

            try:
                grid = self.plot(group)
            except FileNotFoundError:
                logger.warning('No data in group={}', group)
                continue

            g = pathvalidate.sanitize_filename(group or 'etc', replacement_text='-')
            grid.savefig(output / f'TrendLog-{g}-every={self.evergy}.png')
            plt.close(grid.figure)


@app['log'].command
def log_plot(*, conf: Config, every: str = '1d'):
    """TrendLog 각 변수 시계열 그래프 (검토용)."""
    (
        utils.mpl
        .MplTheme(context='paper', palette='tol:bright')
        .grid()
        .apply({'legend.fontsize': 'small'})
    )
    plotter = TrendLogPlotter(conf=conf, evergy=every)
    plotter.plot_and_save()


# ================================= analyse ================================


app.command(App('analyse'))


@app['analyse'].command
def analyse_pv_trend(*, conf: Config):
    dirs = conf.db_dirs

    inverter = (
        pl
        .read_parquet(list(dirs.binary.glob('KEAPV.th_inverter*.parquet')))
        .unpivot(['InverterKw', 'InverterKwh', 'InverterTodayKwh'], index='create_date')
        .with_columns()
    )

    weather = (
        pl
        .read_parquet(list(dirs.binary.glob('KEAPV.th_weather*.parquet')))
        .unpivot(
            ['S_radiation', 'H_radiation', 'mod_temp', 'outdoor_temp'],
            index='create_date',
        )
        .with_columns()
    )

    df = (
        pl
        .concat([inverter, weather])
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
        sns
        .FacetGrid(df, row='kind', sharey=False, legend_out=False)
        .map_dataframe(
            sns.lineplot, x='create_date', y='value', hue='variable', alpha=0.8
        )
        .set_axis_labels('', '')
        .set_titles('')
        .set_titles('{row_name}', loc='left', weight=500)
    )

    for ax in grid.axes.flat:
        ax.legend()

    grid.figure.set_size_inches(tuple(x * 1.2 / 2.54 for x in (16, 9)))  # pyright: ignore[reportArgumentType]
    ConstrainedLayoutEngine().execute(grid.figure)

    grid.figure.savefig(conf.dirs.analysis / 'DB-PV trend.png')


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplConciseDate().apply()
    utils.mpl.MplTheme(context='paper').grid().apply()

    app()

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import pyarrow.csv
import rich
import seaborn as sns
from cyclopts import App
from loguru import logger

from greenbutton import misc, utils
from scripts.config import Config

if TYPE_CHECKING:
    from collections.abc import Collection

T = TypeVar('T')

cnsl = rich.get_console()
KEMC_CODE: dict[int, str] = {
    501: '상용',
    502: '공공',
    503: '아파트',
    504: '호텔',
    505: '병원',
    506: '학교',
    507: 'IDC',
    508: '연구소',
    509: '백화점',
    599: '건물기타',
}


def _sort_head_tail(
    it: Collection[T],
    head: Collection[T] = (),
    tail: Collection[T] = (),
):
    yield from (x for x in head if x in it)
    yield from (x for x in it if x not in {*head, *tail})
    yield from (x for x in tail if x in it)


app = App()


@app.command
def to_parquet(
    src: Path = Path('AMI2023.csv'),
    dst: Path | None = None,
    encoding: str = 'korean',
):
    if not src.exists():
        conf = Config.read()
        src = conf.ami.root / src

    dst = dst or src.with_suffix('.parquet')

    arrow = pyarrow.csv.read_csv(
        src, read_options=pyarrow.csv.ReadOptions(encoding=encoding)
    )
    data = pl.from_arrow(arrow)
    assert isinstance(data, pl.DataFrame)
    data.write_parquet(dst)


@app.command
def unpivot(src: Path = Path('AMI2023.parquet'), dst: Path = Path('AMI2023')):
    stem = src.stem
    if not src.exists():
        conf = Config.read()
        src = conf.ami.root / src

    dst = conf.ami.root / dst
    dst.mkdir(exist_ok=True)

    ami = pl.scan_parquet(src)
    index = ami.select(~cs.starts_with('pwr_qty')).collect_schema().names()
    head = ['datetime', 'YEAR', 'kemc', 'ente', 'KEMC_OLDX_CODE', 'cust_no', 'meter_no']

    for code in ami.select('KEMC_OLDX_CODE').unique().collect().to_series():
        kind = KEMC_CODE[int(code)]
        logger.info('KEMC CODE {} ({})', code, kind)

        lf = (
            ami.filter(pl.col('KEMC_OLDX_CODE') == code)
            .with_columns(pl.col('mr_ymd').cast(pl.String).str.to_date('%Y%m%d'))
            .unpivot(index=index, variable_name='time')
            .with_columns(pl.col('time').str.strip_prefix('pwr_qty'))
            .with_columns(
                # time, datetime 형식은 `24:00:00` 데이터를 허용하지 않음
                # 시간이 "2400"인 열의 날짜를 하루 더하고, 시간은 "0000"으로 변환
                mr_ymd=pl.when(pl.col('time') == '2400')
                .then(pl.col('mr_ymd') + pl.duration(days=1))
                .otherwise(pl.col('mr_ymd')),
                time=pl.when(pl.col('time') == '2400')
                .then(pl.lit('0000'))
                .otherwise(pl.col('time')),
            )
            .with_columns(pl.col('time').str.to_time('%H%M'))
            .with_columns(datetime=pl.col('mr_ymd').dt.combine(pl.col('time')))
            .drop('time', 'mr_ymd')
        )

        columns = _sort_head_tail(
            lf.collect_schema().names(), head=head, tail=['value']
        )
        lf.select(columns).sink_parquet(dst / f'{stem}_{code}{kind}.parquet')


@app.command
def eda(src: Path = Path('AMI2023')):
    if not src.exists():
        conf = Config.read()
        src = conf.ami.root / src

    lf = pl.scan_parquet(list(src.glob('*.parquet')))

    misc.PolarsSummary(
        lf.select('kemc', 'KEMC_OLDX_CODE', 'cust_no', 'meter_no')
        .unique()
        .with_columns(kind=pl.col('KEMC_OLDX_CODE').replace_strict(KEMC_CODE))
    ).write_excel(src / '요약.xlsx')


@app.command
def sample_plot(src: Path = Path('AMI2023'), dst: Path | None = None, idx: int = 10):
    if not src.exists():
        conf = Config.read()
        src = conf.ami.root / src

    dst = dst or src

    lf = pl.scan_parquet(next(src.glob('*.parquet')))

    meter = (
        lf.select('meter_no')
        .unique(maintain_order=True)
        .head(idx)
        .tail(1)
        .collect()
        .item()
    )
    kemc = lf.select('KEMC_OLDX_CODE').head(1).collect().item()
    logger.info('meter {}', meter)
    logger.info('KEMC CODE {}', kemc)

    df = (
        lf.filter(pl.col('meter_no') == meter, pl.col('value') != 0)
        .sort(pl.col('datetime'))
        .group_by_dynamic('datetime', every='1h')
        .agg(pl.sum('value'))
        .collect()
    )

    fig, ax = plt.subplots()
    sns.scatterplot(df, x='datetime', y='value', ax=ax, edgecolor='none', alpha=0.25)
    ax.set_xlabel('')
    ax.set_ylabel('전력사용량 [kWh]')
    fig.savefig(dst / f'{src.stem}_KEMC{kemc}_meter{meter}.png')


if __name__ == '__main__':
    utils.LogHandler.set()
    utils.MplConciseDate().apply()
    utils.MplTheme().grid().apply()

    app()

"""2026-09-01 인하대 공유 BEMS 자료 복사."""

from pathlib import Path  # ruff: ignore[typing-only-standard-library-import]
from typing import Literal

import polars as pl
import structlog
from tqdm.rich import tqdm

from greenbutton.utils.cli import App

app = App()
logger = structlog.stdlib.get_logger()


@app.command
def kepco_paju(
    table: Literal['facility', 'elec'],
    src: Path,
    dst: Path,
    years: tuple[int, int] = (2020, 2025),
):
    """한전 파주 자료 중복 제거 등."""
    tbl = 'T_BELO_FACILITY_15MIN' if table == 'facility' else 'T_BELO_ELEC_15MIN'

    sources = list(src.rglob(f'*{tbl}*.parquet'))
    lf = pl.concat(
        pl.scan_parquet(x).with_columns(pl.col('tagValue').cast(pl.Float64))
        for x in sources
    )
    rename = {
        'updateDate': 'datetime',
        '[tagName]': 'tag',
        '[tagDesc]': 'tag.description',
        'tagValue': 'value',
    }

    for year in range(years[0], years[1] + 1):
        logger.info('year=%d', year)
        df = (
            lf
            .filter(pl.col('updateDate').dt.year() == year)
            .rename(rename)
            .unique(['datetime', 'tag'])
            .select(list(rename.values()))
            .with_columns(pl.col('value').cast(pl.Float64))
            .sort('datetime', 'tag')
            .collect()
        )
        df.write_parquet(dst / f'{tbl}.{year}.parquet')


@app.command
def kea(src: Path, dst: Path):
    """KEA 자료 필수 변수만 복사."""
    sources = list(src.glob('*.parquet'))
    columns = [
        'TABLE_CATALOG',
        'TABLE_NAME',
        'datetime',
        'statusflags',
        'point',
        'parsed_value',
    ]

    for s in tqdm(sources):
        logger.info(s.as_posix())

        d = dst / s.name
        (
            pl
            .scan_parquet(s)
            .select(columns)
            .rename({'parsed_value': 'value'})
            .collect()
            .write_parquet(d)
        )


if __name__ == '__main__':
    app()

"""
2025-10-14.

냉난방도일 보정한 공공기관 에너지 절감률 평가 (SQI 성과).
"""

import dataclasses as dc
from pathlib import Path

import cyclopts
import fastexcel
import polars as pl
import rich

from greenbutton.utils.cli import App
from scripts.ami.public_institution import config as _config


@dc.dataclass
class Dirs(_config.Dirs):
    saving: Path = Path('0400.saving')


@dc.dataclass
class Config(_config.Config):
    dirs: Dirs = dc.field(default_factory=Dirs)


app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml',
        root_keys='public_institution',
        allow_unknown=True,
        use_commands_as_keys=False,
    )
)


@app.default
def degree_day(*, conf: Config):
    root = conf.dirs.saving
    cache = root / '00.weather.parquet'

    if cache.exists():
        data = pl.read_parquet(cache)
    else:
        xlsx = cache.with_suffix('.xlsx')
        sheets = fastexcel.read_excel(xlsx).sheet_names
        data = (
            pl
            .concat([pl.read_excel(xlsx, sheet_name=x) for x in sheets])
            .with_columns(date=pl.col('날짜').cast(pl.String).str.to_date('%Y%m%d'))
            .with_columns(year=pl.col('date').dt.year())
        )
        data.write_parquet(cache)

    avg = pl.col('평균온도')
    data = data.with_columns(
        pl.max_horizontal(0, pl.lit(18) - avg).alias('HDD'),
        pl.max_horizontal(0, avg - pl.lit(24)).alias('CDD'),
    )
    rich.print(data)

    def change_rate(v: str):
        return (
            (pl.col(f'{v}_2024') - pl.col(f'{v}_2023'))  # fmt
            / pl.col(f'{v}_2023')
        ).alias(f'{v} 변화율')

    def agg(data: pl.DataFrame):
        return (
            data
            .group_by('ID', 'year')
            .agg(pl.len(), pl.sum('HDD', 'CDD'))
            .with_columns(pl.sum_horizontal('HDD', 'CDD').alias('(HDD+CDD)'))
            .sort('year')
            .pivot('year', index='ID', values=['len', 'HDD', 'CDD', '(HDD+CDD)'])
            .with_columns(change_rate('HDD'), change_rate('CDD'))
            .sort(pl.all())
        )

    # 연간
    agg(data).write_excel(root / '01.degree_day.xlsx', column_widths=120)

    # 동/하절기
    agg(data.filter(pl.col('date').dt.month().is_in([12, 1, 2, 6, 7, 8]))).write_excel(
        root / '02.degree_day_seasonal.xlsx', column_widths=120
    )


if __name__ == '__main__':
    app()

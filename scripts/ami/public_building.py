from __future__ import annotations

import dataclasses as dc
from dataclasses import InitVar
from io import StringIO
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import polars as pl
import rich
import seaborn as sns
from cyclopts import App
from polars.exceptions import ComputeError
from rich.progress import track

from greenbutton import utils
from scripts.config import Config

if TYPE_CHECKING:
    from pathlib import Path

app = App()
cnsl = rich.get_console()


@dc.dataclass
class _Config:
    path: InitVar[str | Path] = 'config/config.toml'
    subdirs: InitVar[tuple[str, str, str]] = ('01raw', '02data', '03EDA')

    # working directory
    root: Path = dc.field(init=False)

    raw: Path = dc.field(init=False)
    data: Path = dc.field(init=False)
    eda: Path = dc.field(init=False)

    def __post_init__(self, path: str | Path, subdirs: tuple[str, str, str]):
        conf = Config.read(path)

        self.root = conf.ami.root
        self.raw = self.root / subdirs[0]
        self.data = self.root / subdirs[1]
        self.eda = self.root / subdirs[2]


@app.command
def building_info(src: Path | None = None, dst: Path | None = None):
    conf = _Config()
    src = src or conf.raw
    dst = dst or conf.data

    dst.mkdir(exist_ok=True)

    for p in src.glob('*.txt'):
        if p.name.startswith('kcl_'):
            # AMI 사용량 데이터
            continue

        cnsl.print(p.name)

        text = p.read_text('korean', errors='ignore')

        try:
            df = pl.read_csv(StringIO(text), separator='|')
        except ComputeError:
            df = pl.read_csv(StringIO(text), separator='|', truncate_ragged_lines=True)

        cnsl.print(df)

        df.write_parquet(dst / f'{p.stem}.parquet')
        df.write_excel(
            dst / f'{p.stem}.xlsx', column_widths=min(50, int(1500 / df.width))
        )


@app.command
def ami(src: Path | None = None, dst: Path | None = None):
    conf = _Config()
    src = src or conf.raw
    dst = dst or conf.data

    dst.mkdir(exist_ok=True)

    files = list(src.glob('kcl_*.txt'))
    cnsl.print(files)

    for p in src.glob('kcl_*.txt'):
        data = pl.scan_csv(p, separator='|')

        n = data.select(pl.col('기관ID').n_unique()).collect().item()
        cnsl.print(f'n_building={n}')

        converted = (
            data.with_columns(
                pl.format(
                    '{}-{}-{}',
                    '년도',
                    pl.col('월').cast(pl.String).str.pad_start(2, '0'),
                    pl.col('일').cast(pl.String).str.pad_start(2, '0'),
                )
                .str.to_date()
                .alias('date')
            )
            .with_columns(
                pl.when(pl.col('시간') == 24)  # noqa: PLR2004
                .then(pl.col('date') + pl.duration(days=1))
                .otherwise('date')
                .alias('date'),
                pl.col('시간').replace(24, 0),
            )
            .with_columns(pl.col('date').dt.combine(pl.time('시간')).alias('datetime'))
            .select('기관ID', 'datetime', '사용량', '보정사용량')
        )

        cnsl.print(converted.head().collect())

        year = p.stem.removeprefix('kcl_')
        converted.sink_parquet(dst / f'PublicAMI{year}.parquet')


def _ami_plot(lf: pl.LazyFrame):
    df = lf.unpivot(['사용량', '보정사용량'], index='datetime').collect()

    fig, ax = plt.subplots()
    sns.lineplot(df, x='datetime', y='value', hue='variable', ax=ax, alpha=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('사용량')
    ax.get_legend().set_title('')

    return fig


@app.command
def ami_plot(src: Path | None = None, dst: Path | None = None):
    conf = _Config()
    src = src or conf.data
    dst = dst or conf.eda

    dst.mkdir(exist_ok=True)

    lf = pl.scan_parquet(list(src.glob('PublicAMI*.parquet')))

    cnsl.print(lf.head().collect())
    buildings = lf.select(pl.col('기관ID').unique()).collect().to_series().to_list()

    utils.MplTheme().grid().apply()
    utils.MplConciseDate().apply()

    for building in track(buildings):
        fig = _ami_plot(lf.filter(pl.col('기관ID') == building))
        fig.savefig(dst / f'{building}.png')
        plt.close(fig)


if __name__ == '__main__':
    app()

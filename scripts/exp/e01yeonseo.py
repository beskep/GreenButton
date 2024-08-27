from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import polars as pl
import rich
from cyclopts import App

from greenbutton import utils
from scripts import sensor

if TYPE_CHECKING:
    from matplotlib.axes import Axes

cnsl = rich.get_console()
app = App()
exp = sensor.Experiment(building='yeonseo')

PMV1_TIME_DELTA = pl.duration(hours=11, minutes=17)
DURATION = (pl.datetime(2024, 2, 21, 9), pl.datetime(2024, 2, 26, 12))


@app.command
def convert(*, write: bool = True):
    """TR7, PMV 센서 형식 변환."""
    tr7 = exp.convert_tr7(write=write, id_pattern=r'.*온습도계(\d+).*')
    cnsl.print('TR7', tr7)

    pmv = exp.convert_pmv('testo', write=False)

    # PMV1 (1층) 기록 시간 보정
    pmv = pmv.with_columns(
        pl.when(pl.col('id') == 1)
        .then(
            pl.date(2024, 2, 21).dt.combine(
                (pl.col('datetime') + PMV1_TIME_DELTA).dt.time()
            )
        )
        .otherwise(pl.col('datetime'))
        .alias('datetime')
    )
    pmv.write_parquet(exp.dirs.ROOT / '[DATA] PMV.parquet')
    cnsl.print('PMV', pmv)


@app.command
def plot_tr7():
    df = (
        pl.scan_parquet(exp.dirs.ROOT / '[DATA] TR7.parquet', glob=False)
        .with_columns(pl.format('P{} {}', 'point', 'space').alias('space'))
        .sort('space')
        .collect()
    )

    exp.dirs.PLOT.mkdir(exist_ok=True)

    for grid, var in exp.plot_tr7(df):
        grid.savefig(exp.dirs.PLOT / f'TR7-{var}.png')
        plt.close(grid.figure)


@app.command
def plot_pmv():
    exp.dirs.PLOT.mkdir(exist_ok=True)

    df = (
        pl.scan_parquet(exp.dirs.ROOT / '[DATA] PMV.parquet', glob=False)
        .group_by_dynamic(
            'datetime',
            every='10s',
            group_by=['floor', 'point', 'space', 'variable', 'unit'],
        )
        .agg(pl.mean('value'))
        .collect()
    )
    grid = exp.plot_pmv(df)

    ax: Axes = grid.axes.flat[0]
    lim = ax.get_xlim()
    ax.set_xlim(lim[0] - 0.1 * (lim[1] - lim[0]), lim[1])

    grid.savefig(exp.dirs.PLOT / 'PMV.png')


@app.command
def sensors():
    """TR7, PMV 센서 형식 변환, 그래프."""
    convert()
    plot_tr7()
    plot_pmv()


if __name__ == '__main__':
    utils.MplConciseDate().apply()
    utils.MplTheme(palette='tol:vibrant').grid().apply()

    app()

    # TODO 에너지 데이터 변환

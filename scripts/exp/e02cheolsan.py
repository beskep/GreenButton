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
exp = sensor.Experiment(building='cheolsan')


@app.command
def convert(*, write: bool = True):
    """TR7, PMV 센서 형식 변환."""
    tr7 = exp.convert_tr7(write=write)
    cnsl.print('TR7', tr7)

    pmv = exp.convert_pmv('testo', write=write)
    cnsl.print('PMV', pmv)


@app.command
def plot_tr7():
    df = pl.read_parquet(exp.dirs.ROOT / '[DATA] TR7.parquet', glob=False)
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
    ax.set_xlim(lim[0] - 0.15 * (lim[1] - lim[0]), lim[1])

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

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import matplotlib.pyplot as plt
import polars as pl
import rich
from cyclopts import App, Parameter

from greenbutton import utils
from scripts import sensor

if TYPE_CHECKING:
    from matplotlib.axes import Axes

Exp = Annotated[sensor.Experiment, Parameter(parse=False)]

PMV1_TIME_DELTA = pl.duration(hours=11, minutes=17)
DATE = (2024, 2, 21)


cnsl = rich.get_console()
app = App()


@app.meta.default
def launcher(*tokens: Annotated[str, Parameter(show=False)]):
    command, bound = app.parse_args(tokens)

    kwargs = bound.kwargs
    if 'exp' in command.__code__.co_varnames:
        kwargs['exp'] = sensor.Experiment(building='yeonseo')

    return command(*bound.args, **kwargs)


@app.command
def convert(*, exp: Exp, write: bool = True):
    """TR7, PMV 센서 형식 변환."""
    tr7 = exp.convert_tr7(write=write, id_pattern=r'.*온습도계(\d+).*')
    cnsl.print('TR7', tr7)

    pmv = exp.convert_pmv('testo', write=False)

    # PMV1 (1층) 기록 시간 보정
    pmv = pmv.with_columns(
        pl.when(pl.col('id') == 1)
        .then(
            pl.date(*DATE).dt.combine((pl.col('datetime') + PMV1_TIME_DELTA).dt.time())
        )
        .otherwise(pl.col('datetime'))
        .alias('datetime')
    )
    pmv.write_parquet(exp.dirs.ROOT / '[DATA] PMV.parquet')
    cnsl.print('PMV', pmv)


@app.command
def plot_tr7(*, exp: Exp):
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
def plot_pmv(*, exp: Exp):
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
def sensors(*, exp: Exp):
    """TR7, PMV 센서 형식 변환, 그래프."""
    convert(exp=exp)
    plot_tr7(exp=exp)
    plot_pmv(exp=exp)


if __name__ == '__main__':
    utils.MplConciseDate().apply()
    utils.MplTheme(palette='tol:vibrant').grid().apply()

    app.meta()

    # TODO 에너지 데이터 변환

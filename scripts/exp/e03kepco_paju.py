from typing import Annotated, Literal

import matplotlib.pyplot as plt
import polars as pl
import rich
from cyclopts import App, Parameter

from greenbutton import utils
from scripts.sensor import Experiment

ExperimentDate = Literal['2024-03-20', '2024-07-11']
_Exp = Annotated[Experiment, Parameter(parse=False)]

cnsl = rich.get_console()
app = App()


@app.meta.default
def launcher(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    date: ExperimentDate = '2024-03-20',
):
    exp = Experiment(building='kepco_paju', date=date)
    cmd, bound = app.parse_args(tokens)

    if 'exp' in cmd.__code__.co_varnames:
        return cmd(*bound.args, **bound.kwargs, exp=exp)

    return cmd(*bound.args, **bound.kwargs)


@app.command
def convert(*, write: bool = True, exp: _Exp):
    """TR7, PMV 센서 형식 변환."""
    tr7 = exp.convert_tr7(write=write)
    cnsl.print('TR7', tr7)

    pmv = exp.convert_pmv('testo', write=write)
    cnsl.print('PMV', pmv)


@app.command
def plot_tr7(*, exp: _Exp):
    df = (
        pl.scan_parquet(exp.dirs.ROOT / '[DATA] TR7.parquet', glob=False)
        .with_columns(pl.format('P{} {}', 'point', 'space').alias('space'))
        .sort('space', 'datetime')
        .collect()
    )

    exp.dirs.PLOT.mkdir(exist_ok=True)

    for grid, var in exp.plot_tr7(df):
        grid.savefig(exp.dirs.PLOT / f'TR7-{var}.png')
        plt.close(grid.figure)


@app.command
def plot_pmv(*, exp: _Exp):
    df = pl.read_parquet(exp.dirs.ROOT / '[DATA] PMV.parquet', glob=False)

    if exp.date == '2024-03-20':
        df = df.filter(pl.col('floor') != 2)  # noqa: PLR2004 - 데이터 10개 이하

    exp.dirs.PLOT.mkdir(exist_ok=True)
    grid = exp.plot_pmv(df)
    grid.savefig(exp.dirs.PLOT / 'PMV.png')


@app.command
def sensors(*, exp: _Exp):
    """TR7, PMV 센서 형식 변환, 그래프."""
    convert(exp=exp)
    plot_tr7(exp=exp)
    plot_pmv(exp=exp)


if __name__ == '__main__':
    utils.MplConciseDate().apply()
    utils.MplTheme(palette='tol:vibrant').grid().apply()

    app.meta()

import matplotlib.pyplot as plt
import polars as pl
import rich
from cyclopts import App

from greenbutton import utils
from scripts import sensor

cnsl = rich.get_console()
app = App()
exp = sensor.Experiment(building='neulpum')


@app.command
def convert(*, write: bool = True):
    """TR7, PMV 센서 형식 변환."""
    tr7 = exp.convert_tr7(write=write)
    cnsl.print('TR7', tr7)

    pmv = exp.convert_pmv('testo', write=write)
    cnsl.print('PMV', pmv)


@app.command
def plot_tr7():
    df = (
        pl.read_parquet(exp.dirs.ROOT / '[DATA] TR7.parquet', glob=False)
        .with_columns(pl.format('P{} {}', 'point', 'space').alias('space'))
        .sort('space', 'datetime')
    )
    exp.dirs.PLOT.mkdir(exist_ok=True)

    for grid, var in exp.plot_tr7(df):
        grid.savefig(exp.dirs.PLOT / f'TR7-{var}.png')
        plt.close(grid.figure)


@app.command
def plot_pmv():
    exp.dirs.PLOT.mkdir(exist_ok=True)

    df = pl.read_parquet(exp.dirs.ROOT / '[DATA] PMV.parquet', glob=False)
    grid = exp.plot_pmv(df)

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

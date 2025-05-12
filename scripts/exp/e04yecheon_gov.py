from __future__ import annotations

import dataclasses as dc
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import cyclopts
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from loguru import logger

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils.cli import App

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dc.dataclass
class DBDirs:
    raw: Path = Path('0000.raw')
    binary: Path = Path('0001.binary')
    sample: Path = Path('0002.sample')


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'yecheon_gov'

    db_dirs: DBDirs = dc.field(default_factory=DBDirs)

    def __post_init__(self):
        for field in (f.name for f in dc.fields(self.db_dirs)):
            p = getattr(self.db_dirs, field)
            setattr(self.db_dirs, field, self.dirs.database / p)


app = App(
    config=cyclopts.config.Toml('config/.experiment.toml', use_commands_as_keys=False)
)


@app.command
def init(*, conf: Config):
    conf.dirs.mkdir()


app.command(App('sensor'))


@app['sensor'].command
def sensor_parse(*, conf: Config, parquet: bool = True, xlsx: bool = True):
    exp = conf.experiment()
    exp.parse_sensors(write_parquet=parquet, write_xlsx=xlsx)


@app['sensor'].command
def sensor_plot(*, conf: Config, pmv: bool = True, tr7: bool = True):
    exp = conf.experiment()
    exp.plot_sensors(pmv=pmv, tr7=tr7)


app.command(App('downloaded', help='설비 컴퓨터에서 직접 다운받은 데이터'))


@app['downloaded'].command
def downloaded_convert(*, conf: Config):
    src = conf.db_dirs.raw / '모니터 컴퓨터 다운로드'
    dst = conf.db_dirs.binary

    def read(path):
        return (
            pl.read_excel(path)
            .rename({'날짜': 'datetime'})
            .with_columns(
                pl.format('{}:00', 'datetime').str.to_datetime().alias('datetime')
            )
            .unpivot(index='datetime')
            .with_columns(pl.col('value').cast(pl.Float64))
        )

    for path in src.glob('*'):
        if not path.is_dir():
            continue

        logger.info(path.name)

        data = pl.concat(read(x) for x in path.glob('*.xlsx'))
        data.write_parquet(dst / f'downloaded-{path.name}.parquet')


@app['downloaded'].command
def downloaded_plot(*, conf: Config):
    utils.mpl.MplTheme('paper').grid().apply()
    ax: Axes

    for path in conf.db_dirs.binary.glob('downloaded-*.parquet'):
        data = (
            pl.scan_parquet(path)
            .filter(pl.col('variable').str.starts_with('[1일전]').not_())
            .with_columns(
                pl.col('variable')
                .str.replace('㎥', 'm³')
                .str.extract_groups(r'(?<group>(\(\S+\))|(\S+ -))?(?<variable>.*)')
            )
            .unnest('variable')
            .with_columns(
                pl.col('group').str.strip_chars('()- '),
                pl.col('variable').str.strip_chars(),
            )
            .sort('group', 'variable', 'datetime')
            .with_columns(pl.col('group').fill_null('Null'))
            .collect()
        )

        grid = (
            sns.FacetGrid(
                data,
                col='group',
                col_wrap=int(
                    utils.mpl.ColWrap(data.select(pl.col('group').n_unique()).item())
                ),
                sharey=False,
                height=4,
                despine=False,
            )
            .map_dataframe(
                sns.lineplot, x='datetime', y='value', hue='variable', alpha=0.75
            )
            .set_xlabels('')
            .set_ylabels('')
        )

        for ax in grid.axes.ravel():
            ax.legend()
            ax.set_yscale('asinh')
            ax.autoscale_view()

        grid.savefig(conf.dirs.analysis / f'{path.stem}.png')
        plt.close(grid.figure)


app.command(App('db', help='MySQL 백업 데이터'))  # TODO

if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplConciseDate().apply()
    utils.mpl.MplTheme(palette='tol:vibrant').grid().apply()

    app()

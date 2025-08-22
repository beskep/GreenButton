"""한국로봇산업진흥원."""

from __future__ import annotations

import dataclasses as dc
from pathlib import Path  # noqa: F401
from typing import ClassVar

import cyclopts
import matplotlib.pyplot as plt
import more_itertools as mi
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils.cli import App


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'kiria'


app = App(
    config=cyclopts.config.Toml('config/.experiment.toml', use_commands_as_keys=False),
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
def sensor_plot(*, conf: Config):
    exp = conf.experiment()
    exp.plot_sensors(pmv=True, tr7=False)


@app['sensor'].command
def sensor_plot_pmv(
    *,
    testbed: bool = False,  # 하루 측정한 테스트베드실 제외
    conf: Config,
):
    data = pl.read_parquet(conf.dirs.sensor / 'PMV.parquet')

    if not testbed:
        data = data.filter(pl.col('space').str.contains('테스트베드').not_())

    grid = conf.experiment().plot_pmv(data)
    grid.savefig(conf.dirs.analysis / f'PMV {testbed=}.png')


app.command(App('db'))


@app['db'].command
def db_parse(*, conf: Config):
    data = (
        pl.read_excel(mi.one(conf.dirs.database.glob('BEMS*.xlsx')))
        .with_columns(pl.col('시간').str.strip_suffix('.0').str.to_datetime())
        .with_columns()
    )
    data.write_parquet(conf.dirs.database / 'BEMS.parquet')


@app['db'].command
def db_plot(*, height: float = 6, every: str = '2h', conf: Config):
    """측정 기간 사용량 데이터 plot."""
    data = (
        pl.scan_parquet(conf.dirs.database / 'BEMS.parquet')
        .drop('합계')
        .unpivot(index='시간')
        .filter(pl.col('value') > 0)
        .group_by_dynamic('시간', every=every, group_by='variable')
        .agg(pl.sum('value'))
        .with_columns(
            group=pl.col('variable')
            .is_in(['급탕', '난방', '냉방'])
            .replace_strict({True: 'group1', False: 'group2'})
        )
        .collect()
        .upsample('시간', every=every, group_by=['variable', 'group'])
        .with_columns(pl.col('variable', 'group').fill_null(strategy='forward'))
        .sort('group', 'variable', '시간')
    )

    fig = Figure()
    ax = fig.subplots()
    utils.mpl.lineplot_break_nans(
        data.filter(pl.col('variable').is_in(['전열', '조명']).not_()),
        x='시간',
        y='value',
        ax=ax,
        hue='variable',
        alpha=0.8,
        linewidth=0.8,
    )
    ax.set_xlabel('')
    ax.set_ylabel('사용량 [kWh]')
    ax.legend(title='')
    fig.savefig(conf.dirs.analysis / 'BEMS.png')

    grid = (
        sns.FacetGrid(
            data,
            col='variable',
            col_wrap=3,
            height=height / 2.54,
            aspect=4 / 3,
            despine=False,
        )
        .map_dataframe(sns.lineplot, x='시간', y='value', alpha=0.8)
        .set_axis_labels('', '사용량 [kWh]')
        .set_titles('')
        .set_titles('{col_name} 사용량', loc='left', weight=500)
    )
    grid.savefig(conf.dirs.analysis / 'BEMS-grid.png')
    plt.close('all')


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplConciseDate().apply()
    utils.mpl.MplTheme(context='paper').grid().apply()

    app()

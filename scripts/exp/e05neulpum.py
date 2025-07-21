import re
from dataclasses import dataclass
from pathlib import Path  # noqa: F401
from typing import ClassVar

import cyclopts
import polars as pl
from matplotlib.figure import Figure

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils.cli import App


@cyclopts.Parameter(name='*')
@dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'neulpum'


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


app.command(App('db'))


@app['db'].command
def db_parse(*, conf: Config):
    """시험성적서 증빙을 위한 PostgreSQL export 데이터 변환."""
    src = conf.dirs.database / 'hm_hour_trend_history.csv'
    data = pl.read_csv(src)
    data.write_parquet(src.with_suffix('.parquet'))

    print(data)


@app['db'].command
def db_plot(*, conf: Config):
    src = conf.dirs.database / 'hm_hour_trend_history.parquet'

    df = (
        pl.scan_parquet(src)
        .filter(
            pl.col('tag_name').is_in([
                'BEMS_COOLHEAT_KWH',
                'BEMS_ELEC_SUM',
                'BEMS_GAS_SUM',
            ])
        )
        .with_columns(pl.col('record_date').cast(pl.String).str.to_datetime('%Y%m%d'))
        .filter(pl.col('record_date') >= pl.date(2024, 4, 1))
        .collect()
    )

    v = re.compile(r'^t\d{2}$')
    index = [x for x in df.columns if not v.match(x)]

    tidy = (
        df.unpivot(index=index, variable_name='time')
        .with_columns(
            pl.col('time').str.strip_prefix('t').cast(pl.Int8),
            pl.col('tag_name')
            .str.strip_prefix('BEMS_')
            .str.strip_suffix('_SUM')
            .str.strip_suffix('_KWH'),
        )
        .with_columns(
            datetime=pl.col('record_date') + pl.duration(hours=pl.col('time'))
        )
        .sort('datetime', 'tag_name')
        .upsample('datetime', every='1h', group_by='tag_name')
        .with_columns(pl.col('value').diff().over('tag_name'))
    )

    fig = Figure()
    ax = fig.subplots()
    utils.mpl.lineplot_break_nans(
        tidy,
        x='datetime',
        y='value',
        hue='tag_name',
        ax=ax,
        alpha=0.6,
        hue_order=['ELEC', 'GAS', 'COOLHEAT'],
    )
    ax.get_legend().set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('사용량 [kWh]')
    fig.savefig(conf.dirs.database / 'hm_hour_trend_history.png')


if __name__ == '__main__':
    utils.mpl.MplConciseDate().apply()
    utils.mpl.MplTheme().grid().apply()

    app()

import dataclasses as dc
from typing import ClassVar

import cyclopts
import more_itertools as mi
import polars as pl

import scripts.exp.experiment as exp
from greenbutton.utils.cli import App


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'energyx'


app = App(
    config=[
        cyclopts.config.Toml('config/.experiment.toml', use_commands_as_keys=False),
        cyclopts.config.Toml('config/experiment.toml', use_commands_as_keys=False),
    ],
)


@app.command
def init(*, conf: Config):
    conf.dirs.mkdir()


app.command(App('sensor'))
app.command(App('db'))


@app['sensor'].command
def sensor_parse(*, conf: Config, parquet: bool = True, xlsx: bool = True):
    exp = conf.experiment()
    exp.parse_sensors(write_parquet=parquet, write_xlsx=xlsx)


@app['sensor'].command
def sensor_plot(*, conf: Config):
    exp = conf.experiment()
    exp.plot_sensors(pmv=True, tr7=False)


@app['db'].command
def db_parse(*, conf: Config, threshold: float = 1000):
    def read(s: str):
        src = mi.one(conf.dirs.database.glob(f'*{s}*xlsx'))
        df = pl.read_excel(src).drop('(단위: kWh)', strict=False)
        df.columns = ['datetime', *df.columns[1:]]
        types = {'에너지': 'energy', '실내환경': 'environment'}
        return (
            df
            .with_columns(
                pl.col('datetime').str.to_datetime(),
                pl.all().exclude('datetime').cast(pl.Float64),
            )
            .unpivot(index='datetime')
            .select('datetime', pl.lit(types[s]).alias('type'), 'variable', 'value')
        )

    energy = read('에너지').filter(pl.col('value') < threshold)
    environment = read('실내환경')
    data = (
        pl
        .concat([energy, environment])
        .sort('datetime', 'type', 'variable')
        .drop_nulls()
    )
    data.write_parquet(conf.dirs.database / 'BEMS.parquet')

    return data


@app.command
def convert(*, conf: Config):
    """나머지 EAN BEMS 데이터와 형식 통일."""
    data = pl.read_parquet(conf.dirs.database / 'BEMS.parquet')

    (
        data
        .filter(pl.col('type') == 'energy')
        .pivot('variable', index='datetime', values='value', sort_columns=True)
        .rename({'datetime': '시간'})
        .write_parquet(conf.dirs.database / '01.EnergyX_에너지.parquet')
    )
    (
        data
        .filter(pl.col('type') == 'environment')
        .with_columns(
            pl.lit('사무실').alias('실이름'),
            pl.col('variable').replace('실내온도', '온도'),
        )
        .pivot(
            'variable', index=['datetime', '실이름'], values='value', sort_columns=True
        )
        .rename({'datetime': '시간'})
        .write_parquet(conf.dirs.database / '01.EnergyX_실내환경.parquet')
    )


if __name__ == '__main__':
    app()

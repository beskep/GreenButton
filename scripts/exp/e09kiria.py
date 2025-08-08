"""한국로봇산업진흥원."""

from __future__ import annotations

import dataclasses as dc
from pathlib import Path  # noqa: F401
from typing import ClassVar

import cyclopts

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


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplConciseDate().apply()
    utils.mpl.MplTheme(context='paper').grid().apply()

    app()

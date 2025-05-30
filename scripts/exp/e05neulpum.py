from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: F401
from typing import ClassVar

import cyclopts

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


if __name__ == '__main__':
    utils.mpl.MplConciseDate().apply()
    utils.mpl.MplTheme(palette='tol:vibrant').grid().apply()

    app()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: F401
from typing import Annotated

import cyclopts

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils import App


@dataclass
class Config(exp.BaseConfig):
    BUILDING = 'neulpum'


ConfigParam = Annotated[Config, cyclopts.Parameter(name='*')]
app = App(
    config=cyclopts.config.Toml('config/.experiment.toml', use_commands_as_keys=False)
)


@app.command
def init(*, conf: ConfigParam):
    conf.dirs.mkdir()


app.command(App('sensor'))


@app['sensor'].command
def sensor_parse(*, conf: ConfigParam, parquet: bool = True, xlsx: bool = True):
    exp = conf.experiment()
    exp.parse_sensors(write_parquet=parquet, write_xlsx=xlsx)


@app['sensor'].command
def sensor_plot(*, conf: ConfigParam, pmv: bool = True, tr7: bool = True):
    exp = conf.experiment()
    exp.plot_sensors(pmv=pmv, tr7=tr7)


if __name__ == '__main__':
    utils.MplConciseDate().apply()
    utils.MplTheme(palette='tol:vibrant').grid().apply()

    app()

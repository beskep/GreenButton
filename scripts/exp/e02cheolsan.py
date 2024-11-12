from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App, Parameter

from greenbutton import utils
from scripts import sensor


@dataclass
class Config:
    building: str = 'cheolsan'
    date: str | None = None

    pmv: bool = True
    tr7: bool = True

    xlsx: bool = False

    def experiment(self):
        return sensor.Experiment(building=self.building, date=self.date)


_Config = Annotated[Config, Parameter(name='*')]

app = App()
DEFAULT_CONFIG = Config()


@app.command
def parse_sensors(*, conf: _Config = DEFAULT_CONFIG):
    conf = conf or Config()
    exp = conf.experiment()
    exp.parse_sensors(
        pmv=conf.pmv, tr7=conf.tr7, write_parquet=True, write_xlsx=conf.xlsx
    )


@app.command
def plot_sensors(*, conf: _Config = DEFAULT_CONFIG):
    conf = conf or Config()
    exp = conf.experiment()
    exp.plot_sensors(pmv=conf.pmv, tr7=conf.tr7)


if __name__ == '__main__':
    utils.MplConciseDate().apply()
    utils.MplTheme(palette='tol:vibrant').grid().apply()

    app()

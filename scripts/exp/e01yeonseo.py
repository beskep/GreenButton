from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: F401
from typing import Annotated

import cyclopts
import polars as pl

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils import App


class _Experiment(exp.Experiment):
    PMV1_TIME_DELTA = pl.duration(hours=11, minutes=17)
    DATE = (2024, 2, 21)

    def read_pmv(
        self,
        sources=None,
        pattern=r'((?P<date>\d{4}\-\d{2}\-\d{2})_)?PMV(?P<id>\d+).*\.(csv|dlg)',
    ):
        data = super().read_pmv(sources, pattern)

        # PMV1 (1층) 기록 시간 보정
        return data.with_columns(
            pl.when(pl.col('id') == 1)
            .then(
                pl.date(*self.DATE).dt.combine(
                    (pl.col('datetime') + self.PMV1_TIME_DELTA).dt.time()
                )
            )
            .otherwise(pl.col('datetime'))
            .alias('datetime')
        )


@dataclass
class Config(exp.BaseConfig):
    BUILDING = 'yeonseo'

    def experiment(self):
        return _Experiment(conf=self)


ConfigParam = Annotated[Config, cyclopts.Parameter(name='*')]
app = App(
    config=cyclopts.config.Toml('config/experiment.toml', use_commands_as_keys=False)
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

    # TODO 에너지 데이터 변환

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import polars as pl
from cyclopts import App, Parameter

from greenbutton import utils
from scripts import sensor


class YeonseoExperiment(sensor.Experiment):
    PMV1_TIME_DELTA = pl.duration(hours=11, minutes=17)
    DATE = (2024, 2, 21)

    def parse_pmv(
        self,
        paths: sensor.Iterable[sensor.Path] | None = None,
        pattern: str = r'.*PMV(\d+).*',
    ):
        df = super().parse_pmv(paths, pattern)

        # PMV1 (1층) 기록 시간 보정
        return df.with_columns(
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
class Config:
    building: str = 'yeonseo'
    date: str | None = None

    pmv: bool = True
    tr7: bool = True

    xlsx: bool = False

    def experiment(self):
        return YeonseoExperiment(building=self.building, date=self.date)


_Config = Annotated[Config, Parameter(name='*')]

app = App()


@app.command
def parse_sensors(*, conf: _Config | None = None):
    conf = conf or Config()
    exp = conf.experiment()
    exp.parse_sensors(
        pmv=conf.pmv, tr7=conf.tr7, write_parquet=True, write_xlsx=conf.xlsx
    )


@app.command
def plot_sensors(*, conf: _Config | None = None):
    conf = conf or Config()
    exp = conf.experiment()
    exp.plot_sensors(pmv=conf.pmv, tr7=conf.tr7)


if __name__ == '__main__':
    utils.MplConciseDate().apply()
    utils.MplTheme(palette='tol:vibrant').grid().apply()

    app()

    # TODO 에너지 데이터 변환

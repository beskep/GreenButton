"""경희대학교 extended change point model 아이디어 테스트."""

import dataclasses as dc
import itertools
from typing import ClassVar, Literal

import cyclopts
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import structlog

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils.cli import App

type Building = Literal['KEPCO', 'KEA']
type Delta = Literal['day', 'hour']

BUILDINGS: tuple[Building, Building] = ('KEPCO', 'KEA')
DELTAS: tuple[Delta, Delta] = ('day', 'hour')

logger = structlog.stdlib.get_logger()
app = App(
    config=[
        cyclopts.config.Toml(f'config/{x}.toml', use_commands_as_keys=False)
        for x in ['.experiment', 'experiment']
    ],
)


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'ecpm'

    EUI_THRESHOLD: float = 0.1

    # gross floor area [m²]
    GFA: ClassVar[dict[Building, float]] = {'KEPCO': 5208.81, 'KEA': 24348.0}

    def bems(self, bldg: Building, delta: Delta):
        index = '01' if bldg == 'KEPCO' else '02'
        return self.dirs.database / f'{index}.{bldg}-BEMS-{delta}.parquet'

    def read(
        self,
        bldg: Building,
        delta: Delta = 'day',
        *,
        holiday: bool | None = False,
    ):
        lf = (
            pl
            .scan_parquet(self.bems(bldg, delta))
            .drop_nulls()
            .rename({
                'electricity_kWh': 'energy',
                'temp_external': 'te',
                'temp_internal': 'ti',
            })
            .with_columns(pl.col('energy').truediv(self.GFA[bldg]).alias('EUI'))
        )

        if delta == 'day':
            lf = lf.filter(pl.col('EUI') >= self.EUI_THRESHOLD)

        if isinstance(holiday, bool):
            lf = lf.filter(pl.col('holiday') == holiday)

        return lf.collect()


@app.command
@dc.dataclass
class GridPlot:
    conf: Config

    def plot(self, bldg: Building, delta: Delta):
        data = (
            self.conf
            .read(bldg, delta)
            .with_columns((pl.col('ti') - pl.col('te')).alias(r'$\Delta T$'))
            .rename({'te': '$T_{ext}$', 'ti': '$T_{int}$'})
        )

        if delta == 'day':
            hue = None
        else:
            hue = 'office hour'
            data = data.with_columns(
                (
                    (pl.col('datetime').dt.time() >= pl.time(9, 0))
                    & (pl.col('datetime').dt.time() <= pl.time(18, 0))
                ).alias(hue)
            )

        grid = (
            sns
            .PairGrid(data.drop('datetime', 'energy', 'holiday').to_pandas(), hue=hue)
            .map_lower(sns.scatterplot, alpha=0.25)
            .map_diag(sns.histplot)
        )

        if delta == 'hour':
            grid.add_legend()

        grid.savefig(self.conf.dirs.analysis / f'PairGrid-{bldg}-{delta}.png')
        plt.close(grid.figure)

    def __call__(self):
        for bldg, delta in itertools.product(BUILDINGS, DELTAS):
            logger.info('%s %s', bldg, delta)
            self.plot(bldg, delta)


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    app()

"""BEMS 사용량 데이터가 없어 AMI 사용량 사용."""

import dataclasses as dc
from pathlib import Path  # noqa: TC003
from typing import ClassVar

import more_itertools as mi
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure

from greenbutton import misc, utils
from scripts.validation.common import BasePrep, Config, app  # noqa: TC001


@app.command
def ami(
    src: Path,
    iid: str = 'DB_B7AE8782-40AF-8EED-E050-007F01001D51',
    *,
    conf: Config,
):
    """AMI 데이터 복사."""
    sources = list(src.glob('AMI*.parquet'))
    data = (
        pl.scan_parquet(sources)
        .filter(pl.col('기관ID') == iid)
        .sort('datetime')
        .group_by_dynamic('datetime', every='1d')
        .agg(pl.sum('사용량').alias('energy'))
        .sort('datetime')
        .select(pl.col('datetime').dt.date().alias('date'), 'energy')
        .collect()
    )
    dst = conf.path.raw / f'05.{Prep.NAME}'
    data.write_parquet(dst / 'ami.parquet')


@app.default
@dc.dataclass
class Prep(BasePrep):
    conf: Config
    threshold: float = 2000

    NAME: ClassVar[str] = 'KEA'

    def read_weather(self):
        src = mi.one(self.conf.path.raw.glob(f'*{self.NAME}/OBS_ASOS*.csv'))
        return pl.read_csv(src, encoding='korean').select(
            pl.col('지점명').alias('weather_station'),
            pl.col('일시').str.to_date().alias('date'),
            pl.col('평균기온(°C)').alias('temperature'),
        )

    def __call__(self):
        src = mi.one(self.conf.path.raw.glob(f'*{self.NAME}/ami.parquet'))
        data = (
            pl.scan_parquet(src)
            .filter(pl.col('energy') >= self.threshold)
            .collect()
            .join(self.read_weather(), on='date', how='left', validate='1:1')
        )

        years = data['date'].dt.year().unique().sort().to_list()
        data = data.with_columns(
            is_holiday=misc.is_holiday(pl.col('date'), years=years)
        )
        self.write(data)

        fig = Figure()
        ax = fig.subplots()
        sns.lineplot(data=data, x='date', y='energy', ax=ax)
        ax.set_ylim(0)
        ax.set_xlabel('')
        ax.set_ylabel('Energy [kWh]')
        fig.savefig(self.conf.path.data / f'02.{self.NAME}.png')

        return data


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    utils.mpl.MplConciseDate().apply()
    app()

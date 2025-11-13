import dataclasses as dc
from typing import ClassVar

import matplotlib.pyplot as plt
import more_itertools as mi
import polars as pl
import seaborn as sns

from greenbutton import misc, utils
from scripts.validation.common import BasePrep, Config, app  # noqa: TC001


@app.default
@dc.dataclass
class Prep(BasePrep):
    conf: Config

    threshold: tuple[float, float] = (800.0, 3000.0)

    FIELDS: ClassVar[tuple[str, ...]] = ('난방', '냉방', '전력순사용량', '발전량')
    NAME: ClassVar[str] = 'YeonseoLib'

    def read_weather(self):
        src = mi.one(self.conf.path.raw.glob(f'*{self.NAME}/OBS_AWS*.csv'))
        return pl.read_csv(src, encoding='korean').select(
            pl.col('일시').str.to_date().alias('date'),
            pl.col('평균기온(°C)').alias('temperature'),
        )

    def read_data(self):
        root = mi.one(self.conf.path.raw.glob(f'*{self.NAME}'))
        assert root.is_dir()

        src = mi.one(self.conf.path.raw.glob(f'*{self.NAME}/data.parquet'))
        return (
            pl.read_parquet(src)
            .sort('datetime')
            .upsample('datetime', every='10m', group_by=['variable', 'point'])
            .with_columns(pl.col('value').diff().over('variable'))
        )

    def __call__(self):
        hourly = self.read_data()

        daily = (
            hourly.sort('datetime', 'variable')
            .group_by_dynamic('datetime', every='1d', group_by='variable')
            .agg(pl.sum('value'))
            .rename({'datetime': 'date'})
            .with_columns(pl.col('date').dt.date())
            .pivot('variable', index='date', values='value', sort_columns=True)
            .sort('date')
            .with_columns(
                pl.col('전기')
                .clip(*self.threshold)
                .replace({self.threshold[0]: None, self.threshold[1]: None})
            )
            .with_columns(pl.col('전기').alias('energy'))
        )
        weather = self.read_weather()

        years = daily['date'].dt.year().unique().to_list()
        data = (
            daily.join(weather, on='date', how='left')
            .drop_nulls('energy')
            .with_columns(
                misc.is_holiday(pl.col('date'), years=years).alias('is_holiday')
            )
        )

        self.write(data)

        grid = (
            sns.FacetGrid(
                data.unpivot(['가스', '전기'], index='date'),
                row='variable',
                sharey=False,
                despine=False,
                aspect=2 * 16 / 9,
            )
            .map_dataframe(sns.lineplot, x='date', y='value')
            .set_titles('')
            .set_titles('{row_name}', loc='left', weight=500)
            .set_axis_labels('', '사용량')
        )
        grid.savefig(self.conf.path.data / f'02.{self.NAME}.png')
        plt.close(grid.figure)

        return data


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    utils.mpl.MplConciseDate().apply()
    app()

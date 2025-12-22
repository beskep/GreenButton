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
    threshold: float = 420

    NAME: ClassVar[str] = 'KepcoPaju'

    def read_data(self):
        src = mi.one(self.conf.path.raw.glob(f'*{self.NAME}*.parquet'))
        lf = pl.scan_parquet(src)
        years = (
            lf
            .select(pl.col('date').dt.year().unique().sort())
            .collect()
            .to_series()
            .to_list()
        )

        var = pl.col('variable')
        long = (
            lf
            .with_columns(
                var.replace({
                    '기온(ASOS)': 'temperature',
                    '전기.전체전력량': 'energy',
                })
            )
            .with_columns(
                pl
                .when(var.str.contains(r'지열.펌프.히트펌프\d+.전력량'))
                .then(pl.lit('energy_heatpump'))
                .otherwise(var)
                .alias('variable')
            )
            .collect()
        )

        return (
            pl
            .concat(
                [
                    long.filter(var.is_in(['temperature', 'energy'])),
                    long
                    .filter(var == 'energy_heatpump')
                    .group_by(['date', 'variable'])
                    .agg(pl.sum('value')),
                ],
                how='diagonal',
            )
            .pivot('variable', index='date', values='value', sort_columns=True)
            .drop_nulls('energy')
            .sort('date')
            .with_columns(
                misc.is_holiday(pl.col('date'), years=years).alias('is_holiday')
            )
        )

    @staticmethod
    def plot(data: pl.DataFrame):
        return (
            sns
            .FacetGrid(
                data.unpivot(
                    ['energy', 'energy_heatpump', 'temperature'], index='date'
                ),
                row='variable',
                sharey=False,
                aspect=3 * 16 / 9,
            )
            .map_dataframe(sns.lineplot, x='date', y='value')
            .set_xlabels('')
            .set_titles('')
            .set_titles('{row_name}', loc='left', weight=500)
        )

    def __call__(self):
        data = self.read_data()

        grid = self.plot(data)
        grid.savefig(self.conf.path.data / f'02.{self.NAME}-전처리 전.png')
        plt.close(grid.figure)

        data = data.filter(pl.col('energy') >= self.threshold)

        self.write(data)

        grid = self.plot(data)
        grid.savefig(self.conf.path.data / f'02.{self.NAME}-전처리 후.png')
        plt.close(grid.figure)

        return data


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    app()

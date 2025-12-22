import dataclasses as dc
from typing import ClassVar

import matplotlib.pyplot as plt
import more_itertools as mi
import polars as pl
import seaborn as sns
from loguru import logger

from greenbutton import misc, utils
from scripts.validation.common import BasePrep, Config, app  # noqa: TC001


@app.default
@dc.dataclass
class Prep(BasePrep):
    conf: Config
    grid_kwargs: dict = dc.field(
        default_factory=lambda: {
            'col_wrap': 3,
            'sharex': False,
            'sharey': False,
            'height': 4,
        }
    )

    NAME: ClassVar[str] = 'EAN'
    ENERGY: ClassVar[tuple[str, ...]] = (
        '급탕',
        '기타',
        '난방',
        '냉방',
        '전열',
        '조명',
        '수송',
        '환기',
    )

    def visualize(self, data: pl.DataFrame):
        return (
            sns
            .FacetGrid(data, col='building', **self.grid_kwargs)
            .map_dataframe(sns.histplot, x='energy')
            .set_titles('')
            .set_titles('{col_name}', loc='left', weight=500)
        )

    def read_weather(self):
        src = mi.one(self.conf.path.raw.glob(f'*{self.NAME}/OBS_AWS*.csv'))
        return pl.read_csv(src, encoding='korean').select(
            pl.col('지점명').alias('weather_station'),
            pl.col('일시').str.to_date().alias('date'),
            pl.col('평균기온(°C)').alias('temperature'),
        )

    def read_energy(self):
        src = mi.one(self.conf.path.raw.glob(f'*{self.NAME}/*energy*.parquet'))
        lf = pl.scan_parquet(src)
        if not (
            lf
            .fill_null(0)
            .select((pl.sum_horizontal(self.ENERGY) == pl.col('합계')).all())
            .collect()
            .item()
        ):
            logger.warning('합계 오류')

        data = (
            lf
            .rename({'시간': 'date', '건물': 'building'})
            .select('date', 'building', energy=pl.sum_horizontal(self.ENERGY))
            .group_by_dynamic('date', every='1d', group_by='building')
            .agg(pl.sum('energy'))
            .with_columns(
                pl.col('date').dt.date(),
                pl.col('building').replace({
                    '광주에너지기후진흥원': '광주기후에너지진흥원',
                    '대구로봇산업진흥원': '한국로봇산업진흥원',
                }),
            )
            .sort('building', 'date')
            .with_columns(
                pl
                .col('building')
                .replace_strict(self.conf.ean.weather_station)
                .alias('weather_station')
            )
            .collect()
            .join(self.read_weather(), on=['date', 'weather_station'], how='left')
        )
        years = data['date'].dt.year().unique().to_list()
        return data.with_columns(
            is_holiday=misc.is_holiday(pl.col('date'), years=years)
        )

    def __call__(self):
        data = self.read_energy()

        grid = self.visualize(data)
        grid.savefig(self.conf.path.data / f'02.{self.NAME}-전처리 전.png')
        plt.close(grid.figure)

        # 이상치 제거
        data = data.filter(pl.col('energy') >= 0)
        for building, threshold in self.conf.ean.threshold.items():
            data = data.filter(
                ~((pl.col('building') == building) & (pl.col('energy') >= threshold))
            )

        self.write(data)

        grid = self.visualize(data)
        grid.savefig(self.conf.path.data / f'02.{self.NAME}-전처리 후.png')
        plt.close(grid.figure)

        # temperature vs energy
        grid = (
            sns
            .FacetGrid(data, col='building', hue='is_holiday', **self.grid_kwargs)
            .map_dataframe(sns.scatterplot, x='temperature', y='energy', alpha=0.5)
            .add_legend()
            .set_titles('')
            .set_titles('{col_name}', loc='left', weight=500)
        )
        for ax in grid.axes.ravel():
            ax.set_ylim(0)
        grid.savefig(self.conf.path.data / f'03.{self.NAME} temperature-energy.png')

        return data


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    app()

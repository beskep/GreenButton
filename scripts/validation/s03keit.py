import dataclasses as dc
from typing import TYPE_CHECKING, ClassVar

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import more_itertools as mi
import polars as pl
import seaborn as sns
from matplotlib import cm
from matplotlib.figure import Figure

from greenbutton import utils
from scripts.validation.common import BasePrep, Config, app  # noqa: TC001

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@app.default
@dc.dataclass
class Prep(BasePrep):
    conf: Config

    year: tuple[int | None, int | None] = (2020, None)
    heat: bool = True
    glimpse: bool = False

    NAME: ClassVar[str] = 'KEIT'
    FIELDS: ClassVar[tuple[str, ...]] = ('난방', '냉방', '전력순사용량', '발전량')

    @property
    def fields(self):
        if self.heat:
            return self.FIELDS

        return tuple(x for x in self.FIELDS if x not in {'난방', '냉방'})

    def _glimpse(self):
        it = mi.peekable(self.conf.path.raw.glob(f'*{self.NAME}/*.parquet'))
        it.peek()

        for pq in it:
            tail = pl.scan_parquet(pq, glob=False).tail(10).collect()
            pq.with_suffix('.txt').write_text(
                tail.glimpse(return_type='string'), encoding='utf-8'
            )

    def plot_timeline(self, data: pl.DataFrame, name: str | None = None):
        fig = Figure()
        ax = fig.subplots()
        utils.mpl.lineplot_break_nans(
            data.unpivot(['energy', *self.fields], index='date'),
            x='date',
            y='value',
            hue='variable',
            ax=ax,
            alpha=0.5,
            lw=1,
        )
        ax.set_ylim(0)
        ax.set_xlabel('')
        ax.set_ylabel('Energy [kWh]')

        if legend := ax.get_legend():
            legend.set_title('')

        fig.savefig(self.conf.path.data / f'02.{name or self.NAME}.png')

    def plot_scatter(self, data: pl.DataFrame, name: str | None = None):
        data = data.with_columns(epoch=pl.col('date').dt.epoch('d'))

        fig = Figure()
        axes = fig.subplots(1, 2)

        ax: Axes
        for ax, day in zip(axes, ('workday', 'holiday'), strict=True):
            sm = cm.ScalarMappable(cmap='crest', norm=mcolors.Normalize())
            sns.scatterplot(
                data.filter(pl.col('is_holiday') == (day == 'holiday')),
                x='temperature',
                y='energy',
                ax=ax,
                hue='epoch',
                hue_norm=sm.norm,
                alpha=0.5,
                palette='crest',
                legend=False,
            )
            ax.set_ylim(0)

            cb = plt.colorbar(sm, ax=ax)
            loc = mdates.AutoDateLocator()
            cb.ax.yaxis.set_major_locator(loc)
            cb.ax.yaxis.set_major_formatter(mdates.AutoDateFormatter(loc))

            ax.set_title(day.title(), loc='left', weight=500)

        fig.savefig(self.conf.path.data / f'03.{name or self.NAME}.png')

    def __call__(self):
        name = (
            f'{self.NAME}-{self.year[0] or None}-{self.year[1] or None}'
            f'-{"all" if self.heat else "elec"}'
        )

        if self.glimpse:
            self._glimpse()

        src = mi.one(self.conf.path.raw.glob(f'*{self.NAME}/data.parquet'))
        data = (
            pl.scan_parquet(src)
            .filter(
                pl.col('date').dt.year() >= (self.year[0] or 0),
                pl.col('date').dt.year() <= (self.year[1] or 99999),
                pl.col('value') >= 0,
            )
            .collect()
            .pivot(
                'energy',
                index=['date', 'is_holiday', 'temperature'],
                values='value',
                sort_columns=True,
            )
            .sort('date')
            .drop_nulls('전력순사용량')
            .with_columns(pl.col('냉방', '난방').fill_null(0))
            .with_columns(energy=pl.sum_horizontal(self.fields, ignore_nulls=True))
        )

        self.write(data, name=name)

        self.plot_timeline(data, name=name)
        self.plot_scatter(data, name=name)

        return data


if __name__ == '__main__':
    utils.mpl.MplTheme(context=0.6).grid().apply()
    utils.mpl.MplConciseDate().apply()
    app()

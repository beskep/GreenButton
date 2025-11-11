import dataclasses as dc
from typing import ClassVar

import more_itertools as mi
import polars as pl
from matplotlib.figure import Figure

from greenbutton import utils
from scripts.validation.common import BasePrep, Config, app  # noqa: TC001


@app.default
@dc.dataclass
class Prep(BasePrep):
    conf: Config

    NAME: ClassVar[str] = 'KEIT'
    FIELDS: ClassVar[tuple[str, ...]] = ('난방', '냉방', '전력순사용량', '발전량')

    def glimpse(self):
        it = mi.peekable(self.conf.path.raw.glob(f'*{self.NAME}/*.parquet'))
        it.peek()

        for pq in it:
            tail = pl.scan_parquet(pq, glob=False).tail(10).collect()
            pq.with_suffix('.txt').write_text(
                tail.glimpse(return_type='string'), encoding='utf-8'
            )

    def __call__(self):
        self.glimpse()

        src = mi.one(self.conf.path.raw.glob(f'*{self.NAME}/data.parquet'))
        data = (
            pl.read_parquet(src)
            .filter(pl.col('value') >= 0)
            .pivot(
                'energy',
                index=['date', 'is_holiday', 'temperature'],
                values='value',
                sort_columns=True,
            )
            .sort('date')
            .drop_nulls('전력순사용량')
            .with_columns(pl.col('냉방', '난방').fill_null(0))
            .with_columns(energy=pl.sum_horizontal(self.FIELDS, ignore_nulls=True))
        )
        self.write(data)

        fig = Figure()
        ax = fig.subplots()
        utils.mpl.lineplot_break_nans(
            data.unpivot(['energy', *self.FIELDS], index='date'),
            x='date',
            y='value',
            hue='variable',
            ax=ax,
            alpha=0.5,
        )
        ax.set_ylim(0)
        ax.set_xlabel('')
        ax.set_ylabel('Energy [kWh]')

        if legend := ax.get_legend():
            legend.set_title('')

        fig.savefig(self.conf.path.data / '02.KEIT.png')

        return data


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    utils.mpl.MplConciseDate().apply()
    app()

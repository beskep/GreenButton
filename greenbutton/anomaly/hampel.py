"""Hampel filter 이상치 감지."""

from __future__ import annotations

import dataclasses as dc
import warnings
from typing import TYPE_CHECKING, Literal, overload

import polars as pl

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from polars._typing import FrameType


@dc.dataclass
class Columns:
    value: str = 'value'
    rolling_median: str = 'rolling_median'
    rolling_mad: str = 'rolling_mad'  # rolling mean absolute deviation
    diff: str = 'diff'
    threshold: str = 'threshold'
    is_outlier: str = 'is_outlier'
    filtered: str = 'filtered'


@dc.dataclass
class HampelFilter:
    window_size: int  # 짝수 권고
    min_samples: int | None = None
    center_window: bool = True

    t: float = 3.0
    scale: dc.InitVar[float | Literal['norm']] = 'norm'
    _scale: float = dc.field(init=False)

    columns: Columns = dc.field(default_factory=Columns)

    def __post_init__(self, scale: float | Literal['norm']):
        self._scale = 1.4826 if scale == 'norm' else scale

    def rolling_median(self, value: pl.Expr, min_samples: int | None = None):
        return value.rolling_median(
            window_size=self.window_size,
            min_samples=self.min_samples if min_samples is None else min_samples,
            center=self.center_window,
        )

    def rolling_mad(self, value: pl.Expr, rolling_median: pl.Expr | None = None):
        if rolling_median is None:
            rolling_median = self.rolling_median(value)

        return self.rolling_median((value - rolling_median).abs(), min_samples=0)

    @overload
    def __call__(
        self,
        data: ArrayLike | pl.DataFrame,
        value: str | pl.Expr | None = ...,
    ) -> pl.DataFrame: ...

    @overload
    def __call__(
        self,
        data: pl.LazyFrame,
        value: str | pl.Expr | None = ...,
    ) -> pl.LazyFrame: ...

    def __call__(
        self,
        data: ArrayLike | FrameType,
        value: str | pl.Expr | None = None,
    ):
        if self.window_size & 1:
            warnings.warn('window_size should be an even number.', stacklevel=2)

        c = self.columns
        rm = pl.col(c.rolling_median)

        match value:
            case str():
                v = pl.col(value)
            case pl.Expr():
                v = value
            case None:
                v = pl.col(c.value)

        if isinstance(data, pl.DataFrame | pl.LazyFrame):
            frame = data
        else:
            frame = pl.DataFrame({c.value: data})

        return (
            frame.with_columns(self.rolling_median(v).alias(c.rolling_median))
            .with_columns(self.rolling_mad(v, rolling_median=rm).alias(c.rolling_mad))
            .with_columns(
                (v - rm).abs().alias(c.diff),
                (self.t * self._scale * pl.col(c.rolling_mad)).alias(c.threshold),
            )
            .with_columns((pl.col(c.diff) > pl.col(c.threshold)).alias(c.is_outlier))
            .with_columns(
                pl.when(pl.col(c.is_outlier)).then(rm).otherwise(v).alias(c.filtered)
            )
        )

    execute = __call__  # noqa: RUF045


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import rich
    import seaborn as sns

    from greenbutton.utils.mpl import MplTheme

    MplTheme().grid().apply()

    rng = np.random.default_rng(42)
    x = np.linspace(0, 2 * np.pi, num=50)
    y = np.sin(x) + 0.5 * x + rng.normal(0, scale=0.1, size=x.size)

    y[10] = 0.2
    y[40] = 0.5

    data = pl.DataFrame({'x': x, 'y': y})
    hf = HampelFilter(window_size=4)
    c = hf.columns
    output = (
        (hf)
        .execute(data, value='y')
        .with_columns(
            pl.col(c.is_outlier).fill_null(value=False),
            lt=pl.col(c.rolling_median) - pl.col(c.threshold),
            ut=pl.col(c.rolling_median) + pl.col(c.threshold),
        )
    )

    pl.Config.set_tbl_cols(10)
    rich.print(output)

    fig, ax = plt.subplots()
    scatter = (
        (output)
        .unpivot(['y', c.filtered], index=['x', c.is_outlier])
        .with_columns(
            pl.format('{}-{}', 'variable', c.is_outlier)
            .replace_strict({
                'y-false': 'Normal',
                'y-true': 'Outlier',
                f'{c.filtered}-false': None,
                f'{c.filtered}-true': 'Interpolated',
            })
            .alias('label')
        )
        .drop_nulls('label')
    )
    sns.scatterplot(
        scatter,
        x='x',
        y='value',
        hue='label',
        style='label',
        ax=ax,
        s=50,
        markers=['o', 'X', '^'],
    )
    ax.fill_between(
        x=x,
        y1=output['lt'].to_numpy(),
        y2=output['ut'].to_numpy(),
        color='k',
        alpha=0.2,
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend(title='')

    plt.show()

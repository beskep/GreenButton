from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import holidays as hd
import polars as pl
import polars.selectors as cs
from xlsxwriter import Workbook

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from pathlib import Path

    from polars._typing import ColumnWidthsDefinition


def is_holiday[T: (pl.Expr, pl.Series)](
    date: T,
    years: int | Iterable[int] | None = None,
    *,
    weekend: bool = True,
) -> T:
    if years is None:
        if isinstance(date, pl.Series):
            years = date.dt.year().unique().to_list()
        else:
            msg = 'years must be specified if date is a Expr.'
            raise ValueError(msg)

    holidays = set(hd.country_holidays('KR', years=years).keys())
    is_holiday = date.dt.date().is_in(holidays)

    if weekend:
        is_holiday |= date.dt.weekday().is_in([6, 7])

    return is_holiday


def transpose_description(desc: pl.DataFrame, decimals: int = 4):
    cols = ('count', 'null_count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max')
    return (
        desc.with_columns(cs.float().round(decimals))
        .drop('statistic')
        .transpose(include_header=True, column_names=cols)
        .with_columns(pl.col(cols[:2]).cast(pl.Float64).cast(pl.UInt64))
        .rename({'column': 'variable'})
    )


@dataclass
class PolarsSummary:
    data: pl.DataFrame | pl.LazyFrame
    group: str | Collection[str] | None = None

    _: KW_ONLY
    transpose: bool = True
    decimals: int = 4
    max_string_category: int | None = 42

    sort: bool = True
    group_prefix: str | None = 'group:'
    omission: str = '...'

    def __post_init__(self):
        if self.group is not None:
            self.group = (
                (self.group,) if isinstance(self.group, str) else tuple(self.group)
            )

    def _describe(
        self,
        data: pl.DataFrame | pl.LazyFrame | None = None,
        selector=None,
    ):
        if data is None:
            data = self.data
        if self.group:
            data = data.drop(self.group, strict=False)

        selector = cs.numeric() | cs.boolean() if selector is None else selector
        desc = data.select(selector).describe()
        if self.transpose:
            desc = transpose_description(desc)

        return desc

    def _describe_by(self, selector=None):
        assert isinstance(self.group, tuple)
        for name, df in (
            self.data.lazy()
            .collect()
            .group_by(self.group, maintain_order=not self.sort)
        ):
            yield self._describe(df, selector=selector).select(
                *(pl.lit(n).alias(g) for n, g in zip(name, self.group, strict=True)),
                pl.all(),
            )

    def describe(self, selector=None):
        if self.group is None:
            return self._describe(selector=selector)

        df = pl.concat(self._describe_by(selector), how='vertical_relaxed')
        if self.sort:
            df = df.sort(self.group)
        if self.group_prefix:
            df = df.rename({x: f'{self.group_prefix}{x}' for x in self.group})

        return df

    def _count_string(self, data: pl.DataFrame | pl.LazyFrame | None = None):
        if data is None:
            data = self.data
        if self.group:
            data = data.drop(self.group, strict=False)

        return (
            data.lazy()
            .select(cs.string() | cs.categorical())
            .unpivot()
            .group_by('variable', 'value', maintain_order=True)
            .len('count')
            .with_columns(
                pl.col('count')
                .truediv(pl.sum('count').over('variable'))
                .alias('proportion')
            )
            .collect()
        )

    def _count_string_by(self):
        assert isinstance(self.group, tuple)
        for name, df in (
            self.data.lazy()
            .collect()
            .group_by(self.group, maintain_order=not self.sort)
        ):
            yield self._count_string(df).select(
                *(pl.lit(n).alias(g) for n, g in zip(name, self.group, strict=True)),
                pl.all(),
            )

    def count_string(self):
        if self.group is None:
            df = self._count_string()
        else:
            df = pl.concat(self._count_string_by(), how='vertical_relaxed')

        if self.sort:
            df = df.sort(pl.all())
        if self.group and self.group_prefix:
            df = df.rename({x: f'{self.group_prefix}{x}' for x in self.group})

        return df

    def _write_string_categorical(
        self,
        wb: Workbook,
        column_widths: ColumnWidthsDefinition | None = 100,
    ):
        sc = cs.string() | cs.categorical()
        if not self.data.select(sc).collect_schema().len():
            return

        if self.group:
            group = (
                [f'{self.group_prefix}{x}' for x in self.group]
                if self.group_prefix
                else self.group
            )
        else:
            group = ()

        self.describe(selector=sc).write_excel(
            wb,
            worksheet='string',
            column_widths=column_widths,
        )

        count = self.count_string()
        if self.max_string_category:
            count = count.with_columns(
                pl.when(
                    pl.col('variable').is_in(self.group or []).not_(),
                    pl.col('value').n_unique().over('variable')
                    > self.max_string_category,
                )
                .then(pl.lit(self.omission))
                .otherwise(pl.col('value'))
                .alias('value')
            )
            if self.omission in count.select('value').to_series():
                count = (
                    count.group_by(*group, 'variable', 'value', maintain_order=True)
                    .agg(pl.sum('count'))
                    .with_columns(
                        value=pl.when(pl.col('value') != self.omission)
                        .then(pl.format('"{}"', 'value'))
                        .otherwise(pl.col('value')),
                        proportion=pl.when(pl.col('value') == self.omission)
                        .then(pl.lit(None))
                        .otherwise(
                            pl.col('count').truediv(pl.sum('count').over('variable'))
                        ),
                    )
                )

        count.write_excel(
            wb,
            worksheet='string count',
            column_formats={'proportion': '0.00%'},
            conditional_formats={'proportion': {'type': 'data_bar', 'bar_solid': True}},
            column_widths=column_widths,
        )

        if group:
            (
                count.with_columns(pl.col(group).fill_null('Null'))
                .with_columns(pl.concat_str(group, separator='_').alias('__group'))
                .pivot('__group', index=['variable', 'value'], values='count')
                .sort('variable', 'value')
                .write_excel(wb, worksheet='string count (pivot)')
            )

    def write_excel(
        self,
        path: str | Path,
        column_widths: ColumnWidthsDefinition | None = 100,
    ):
        with Workbook(path) as wb:
            # numeric
            if self.data.select(cs.numeric() | cs.boolean()).collect_schema().len():
                self.describe().write_excel(
                    wb, worksheet='numeric', column_widths=column_widths
                )

            # temporal
            if self.data.select(cs.temporal()).collect_schema().len():
                self.describe(selector=cs.temporal()).write_excel(
                    wb, worksheet='temporal', column_widths=column_widths
                )

            # string, categorical
            self._write_string_categorical(wb=wb, column_widths=column_widths)

from __future__ import annotations

from typing import TYPE_CHECKING

import holidays as hd
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable


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

from __future__ import annotations

import polars as pl
import pytest

from greenbutton import misc


@pytest.mark.parametrize('weekend', [True, False])
def test_is_holiday_expr(*, weekend: bool):
    dates = [
        '2025-08-14',  # 목
        '2025-08-15',  # 금
        '2025-08-16',  # 토
    ]
    is_holiday = misc.is_holiday(pl.col('date'), years=2025, weekend=weekend)
    df = (
        pl.DataFrame({'date': dates})
        .with_columns(pl.col('date').str.to_date())
        .with_columns(is_holiday.alias('is_holiday'))
    )

    assert df['is_holiday'].to_list() == [False, True, weekend]


def test_is_holiday_expr_without_years():
    with pytest.raises(ValueError, match='years must be specified if date is a Expr'):
        misc.is_holiday(pl.col('date'), years=None)


@pytest.mark.parametrize('weekend', [True, False])
def test_is_holiday_series(*, weekend: bool):
    dates = [
        '2025-08-14',  # 목
        '2025-08-15',  # 금
        '2025-08-16',  # 토
    ]
    series = pl.Series('date', dates).str.to_date()
    is_holiday = misc.is_holiday(series, years=None, weekend=weekend)

    assert is_holiday.to_list() == [False, True, weekend]


def test_transpose_description():
    data = pl.DataFrame({
        'float': [1.0, 2.0, 3.0, 42.0],
        'int': [1, 2, 3, 42],
        'decimal': [1, 2, 3, 4],
        'temporal': [
            pl.datetime(2000, 1, 1),
            pl.datetime(2000, 1, 2),
            pl.datetime(2000, 1, 3),
            pl.datetime(2000, 1, 4),
        ],
        'str': ['spam', 'egg', 'ham', 'spam'],
    }).with_columns(pl.col('decimal').cast(pl.Decimal))
    misc.transpose_description(data.describe())


def test_polars_summary(tmp_path):
    data = pl.DataFrame({
        'float': [1.0, 2.0, 3.0, 42.0],
        'int': [1, 2, 3, 42],
        'decimal': [1, 2, 3, 4],
        'temporal': [
            pl.datetime(2000, 1, 1),
            pl.datetime(2000, 1, 2),
            pl.datetime(2000, 1, 3),
            pl.datetime(2000, 1, 4),
        ],
        'str': ['spam', 'egg', 'ham', 'spam'],
        'group': ['group1', 'group1', 'group2', 'group2'],
    }).with_columns(pl.col('decimal').cast(pl.Decimal))
    summ = misc.PolarsSummary(data, group='group')
    summ.describe()
    summ.write_excel(tmp_path / 'test.xlsx')

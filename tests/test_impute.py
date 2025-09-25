import polars as pl
import polars.testing

from greenbutton.impute import imputer


def test_count_consecutive_null_expr():
    df = pl.DataFrame({'test': [None, 4, None, None, 2, None, None, None]})
    nulls = df.select(imputer.count_consecutive_null(pl.col('test'))).to_series()
    assert list(nulls) == [1, None, 2, 2, None, 3, 3, 3]


def test_count_consecutive_null_series():
    series = pl.Series('test', [4, None, 2, None, None])
    nulls = pl.Series('len', [None, 1, None, 2, 2], dtype=pl.UInt32)
    polars.testing.assert_series_equal(
        nulls, imputer.count_consecutive_null(series), check_dtypes=False
    )

import polars as pl
import polars.testing
import rich
from whenever import PlainDateTime

from greenbutton.impute import imputer as impt
from greenbutton.impute.imputer.misc import LinearImputer


def test_count_consecutive_null_expr():
    df = pl.DataFrame({'test': [None, 4, None, None, 2, None, None, None]})
    nulls = df.select(impt.count_consecutive_null(pl.col('test'))).to_series()
    assert list(nulls) == [1, None, 2, 2, None, 3, 3, 3]


def test_count_consecutive_null_series():
    series = pl.Series('test', [4, None, 2, None, None])
    nulls = pl.Series('len', [None, 1, None, 2, 2], dtype=pl.UInt32)
    polars.testing.assert_series_equal(
        nulls, impt.count_consecutive_null(series), check_dtypes=False
    )


def test_impute():
    dts = [
        PlainDateTime(2000, 1, 1),
        PlainDateTime(2000, 1, 1, 1),
        PlainDateTime(2001, 1, 1),
        PlainDateTime(2001, 1, 1, 1),
    ]

    pl.Config.set_tbl_cols(20)
    console = rich.get_console()

    data = pl.DataFrame({
        'datetime': [x.py_datetime() for x in dts],
        'value': [2, 4, None, 6],
    })

    classes: list[type[impt.AbstractImputer]] = [
        LinearImputer,
        impt.Imputer01,
        impt.Imputer02,
        impt.Imputer03,
        impt.Imputer03KHU,
    ]
    for cls in classes:
        imputer = cls()
        imputed = (
            imputer
            .impute(data)
            .lazy()
            .filter(pl.col('imputed').is_not_null())
            .collect()
        )
        assert imputed.height > data.height

        console.print(f'class={cls.__name__}\n{imputed}\n')


if __name__ == '__main__':
    test_impute()

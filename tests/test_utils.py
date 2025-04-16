from __future__ import annotations

import polars as pl

from greenbutton import utils


def test_polars_frame_cache(tmp_path):
    path = tmp_path / 'tmp.parquet'

    @utils.pl.frame_cache(path=path, timeout='24H', loglevel='INFO')
    def fn():
        return pl.DataFrame()

    assert not path.exists()
    fn()
    assert path.exists()
    mtime = path.stat().st_mtime
    fn()
    assert mtime == path.stat().st_mtime


def test_polars_transpose_description():
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
    utils.pl.transpose_description(data.describe())


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
    summ = utils.pl.PolarsSummary(data, group='group')
    summ.describe()
    summ.write_excel(tmp_path / 'test.xlsx')

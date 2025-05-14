"""AMI csv 파일 ((D+4)hive_building*.csv)에서 특정 meter 추출."""

from __future__ import annotations

import polars as pl
import polars.selectors as cs
import pyarrow.csv


def main(src, dst, meter, encoding='korean'):
    arrow = pyarrow.csv.read_csv(
        src, read_options=pyarrow.csv.ReadOptions(encoding=encoding)
    )
    df = pl.from_arrow(arrow)
    assert isinstance(df, pl.DataFrame)

    (
        df.drop(cs.starts_with('column_', 'Unnamed:'))
        .filter(pl.col('tb_day_lp_4day_bfor_data.meter_no') == meter)
        .sort(pl.col('tb_day_lp_4day_bfor_data.mr_ymd'))
        .write_csv(dst)
    )

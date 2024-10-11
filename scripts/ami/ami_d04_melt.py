"""Wide 데이터로 제공된 AMI 전력 자료 melt."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import polars.selectors as cs


def melt_ami_d4(
    df: pl.DataFrame | pl.LazyFrame,
    remove_prefix='tb_day_lp_4day_bfor_data.',
    value_prefix='pwr_qty',
):
    rename = {
        x: x.removeprefix(remove_prefix) for x in df.columns if remove_prefix in x
    }
    df = (
        df.lazy()
        .rename(rename)  # 열 이름 중 'tb_day_lp_4day_bfor_data.' 제거
        .with_columns(
            # 날짜 데이터를 date 형식으로 변환
            pl.col('mr_ymd', 'part_key_mr_ymd').cast(str).str.to_date('%Y%m%d'),
        )
    )

    # value (전력사용량) 열 목록
    value_vars = df.select(
        cs.starts_with(value_prefix)
        & cs.matches(r'.*(\d{4})$')
        & ~(cs.string() | cs.temporal())
    ).columns
    assert value_vars, value_prefix

    # melt 시 id가 될 나머지 열 목록
    id_vars = [x for x in df.columns if x not in value_vars]

    df = (
        df.melt(id_vars=id_vars, value_vars=value_vars, variable_name='time')
        .with_columns(
            # 시간 데이터 중 숫자 4자리 추출 "pwr_qty0015" -> "0015"
            pl.col('time').str.extract(r'.*(\d{4})$')
        )
        .with_columns(
            # time, datetime 형식은 `24:00:00` 데이터를 허용하지 않음
            # 시간이 "2400"인 열의 날짜를 하루 더하고, 시간은 "0000"으로 변환
            mr_ymd=pl.when(pl.col('time') == '2400')
            .then(pl.col('mr_ymd') + pl.duration(days=1))
            .otherwise(pl.col('mr_ymd')),
            time=pl.when(pl.col('time') == '2400')
            .then(pl.lit('0000'))
            .otherwise(pl.col('time')),
        )
        .with_columns(
            # 시간 데이터를 time 형식으로 변환
            pl.col('time').str.to_time('%H%M')
        )
        .with_columns(
            # 날짜 데이터와 시간 데이터를 더한 datetime 열 생성
            datetime=pl.col('mr_ymd').dt.combine(pl.col('time'))
        )
        .sort('datetime')
    )

    # 열 순서 조정
    columns = ['cust_no', 'meter_no', 'datetime', 'mr_ymd', 'time']
    return df.select(*columns, *(x for x in df.columns if x not in columns))


if __name__ == '__main__':
    root = Path(__file__).parents[1]

    data = pl.scan_csv(root / 'data/D+04Sample.csv')
    melted = melt_ami_d4(data).collect()

    pl.Config.set_tbl_cols(10)
    print(melted)

    melted.write_excel(root / 'data/D+04SampleMelted.xlsx')
    melted.write_parquet(root / 'data/D+04SampleMelted.parquet')

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from greenbutton import imputer as impt


def read_sample(path='D+04SampleMelted.parquet'):
    return (
        pl.scan_parquet(path)
        .filter(
            pl.col('mr_ymd').is_between(  # 1월 샘플링
                pl.date(2022, 1, 1), pl.date(2022, 2, 1), closed='left'
            )
        )
        .select('meter_no', 'datetime', 'value')
        .with_columns(
            ytrue=pl.col('value'),  # 결측치 없는 `ytrue` 열 생성
            value=pl.when(  # 1월 15일 0~3시 결측치 생성
                pl.col('datetime').is_between(
                    pl.datetime(2022, 1, 15, 0, 0, 0),
                    pl.datetime(2022, 1, 15, 3, 0, 0),
                )
            )
            .then(pl.lit(None))
            .otherwise(pl.col('value')),
        )
        .collect()
    )


def rmse(df: pl.DataFrame, imputer: impt.AbstractImputer):
    return (
        imputer.impute(df)
        .lazy()
        .filter(pl.col('value').is_null())
        .select((pl.col('ypred') - pl.col('ytrue')).pow(2).mean().sqrt())
        .collect()[0, 0]
    )


def cls_rmse(df: pl.DataFrame, cls: type[impt.AbstractImputer]):
    imputer = cls(columns=impt.ColumnNames(imputed='ypred'))
    return cls.__name__, rmse(df, imputer)


if __name__ == '__main__':
    try:
        from rich import print  # noqa: A004
    except ImportError:
        pass

    pl.Config.set_tbl_cols(20)

    data = read_sample()

    print(data)

    print('1월 15일 데이터: ')
    print(data.filter(pl.col('datetime').dt.date() == pl.date(2022, 1, 15)))

    imputer01 = impt.Imputer01(
        columns=impt.ColumnNames(
            imputed='ypred'  # 보간 결과 column 이름을 'ypred'로 변경
        ),
    )

    # 방법 3의 파라미터 조정
    imputer01.method3 = impt.GroupRollingMean(['hour', 'minute'], 16, 4)
    imputed = imputer01.impute(data)

    print('\nImputer01 보간 결과')
    print(imputed.filter(pl.col('value').is_null()))

    # 상대비교
    error = dict(
        cls_rmse(df=data, cls=cls)  # type: ignore[arg-type]
        for cls in [
            impt.MeanImputer,
            impt.LinearImputer,
            impt.PchipImputer,
            impt.Imputer01,
            impt.Imputer02,
            impt.Imputer03,
            impt.Imputer03KHU,
        ]
    )

    print('RCMS=', error)

    # 상대비교 그래프
    sns.set_theme(style='whitegrid', rc={'figure.constrained_layout.use': True})

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    sns.barplot(error, orient='h', ax=ax)
    ax.set_title('RMSE')

    plt.show()

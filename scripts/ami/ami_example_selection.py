import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.model_selection import GridSearchCV

from greenbutton.imputer_selection import Imputer01


def read_sample(path='D+04SampleMelted.parquet'):
    return (
        pl.scan_parquet(path)
        .filter(
            # 1월 샘플링
            pl.col('mr_ymd').is_between(
                pl.date(2022, 1, 1), pl.date(2022, 2, 1), closed='left'
            )
        )
        .select('meter_no', 'datetime', 'value')
        .with_columns(
            ytrue=pl.col('value'),  # 결측치 없는 `ytrue` 열 생성
            value=pl.when(  # 결측치 생성
                pl.col('datetime').is_between(
                    pl.datetime(2022, 1, 15, 0, 0, 0),
                    pl.datetime(2022, 1, 15, 12, 0, 0),
                )
            )
            .then(pl.lit(None))
            .otherwise(pl.col('value')),
        )
        .collect()
    )


def main():
    pl.Config.set_tbl_cols(20)

    data = read_sample()
    print('data:')
    print(data, '\n')

    imputer = Imputer01()

    # Cross Validation 대신 전체 data를 한번에 평가하기 위한 꼼수
    # (한 fold에 전체 index 지정)
    indices = np.arange(data.height)
    cv = [(indices, indices)]

    gs = GridSearchCV(
        estimator=imputer,
        param_grid={
            'method1__group': [('month', 'day', 'hour'), ('day', 'hour')],
            'method2__window_size': [8, 24, 42],
        },
        cv=cv,
    )
    gs.fit(data)

    print('cv_results:')
    print(pl.DataFrame(gs.cv_results_).drop('params', cs.ends_with('time')), '\n')

    print('best_estimator:')
    print(gs.best_estimator_, '\n')

    print(f'best_score={gs.best_score_}')


if __name__ == '__main__':
    try:
        from rich import print  # noqa: A004
    except ImportError:
        pass

    main()

"""polars 라이브러리를 통한 AMI 데이터 보간 방법 구현."""
# ruff: noqa: PLR0913

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from ._imputer import AbstractImputer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._imputer import ColumnNames


def count_consecutive_null[T: (pl.Expr, pl.Series)](expr: T) -> T:
    """
    연속해서 나타나는 null 개수 계산.

    Parameters
    ----------
    expr : pl.Expr | pl.Series

    Returns
    -------
    pl.Expr | pl.Series

    Examples
    --------
    >>> df = pl.DataFrame({'test': [None, 4, None, None, 2, None, None, None]})
    >>> list(df.select(count_consecutive_null(pl.col('t'))).to_series())
    [1, None, 2, 2, None, 3, 3, 3]
    >>> count_consecutive_null(pl.Series('test', [4, None, 2, None, None]))
    Series: 'len' [u32]
    [
            null
            1
            null
            2
            2
    ]
    """
    is_null = expr.is_null()
    rle_id = is_null.rle_id()
    count = pl.when(is_null).then(pl.len().over(rle_id))

    if isinstance(expr, pl.Series):
        return expr.to_frame().select(count).to_series()

    return count


@dataclass
class GroupMean:
    """그룹별 평균 보간 파라미터/expression.

    Parameters
    ----------
    group: str | Iterable[str]
        평균 대상 그룹
    window_size: int | None = None
        평균 기간 (대상 전후 window 크기). None이면 전체 기간 평균.
    min_samples: int | None = 1
        window 내 최소 데이터 개수
    """

    group: str | Iterable[str]
    window_size: int | None = None  # TODO str | timedelta
    min_samples: int | None = 1

    def expr(self, column: str):
        if self.window_size is None:
            # 전체 기간 평균
            return pl.mean(column).over(self.group)

        # window 내 데이터 평균
        return (
            pl.col(column)
            .rolling_mean(
                window_size=self.window_size,
                min_samples=self.min_samples,
                center=True,
            )
            .over(self.group)
        )


@dataclass
class GroupRollingMean:
    """그룹별 rolling mean (moving average) 보간 파라미터/expression.

    Parameters
    ----------
    group: str | Iterable[str]
        평균 대상 그룹
    window_size: int
        평균을 계산할 window 크기
    min_samples: int | None
        window 내 최소 데이터 개수
    """

    group: str | Iterable[str]
    window_size: int  # TODO timedelta | str
    min_samples: int | None

    def expr(self, column: str):
        return (
            pl.col(column)
            .rolling_mean(window_size=self.window_size, min_samples=self.min_samples)
            .over(self.group)
        )


class Imputer01(AbstractImputer):
    DEFAULT_METHOD1 = GroupMean(('month', 'day', 'hour'))
    DEFAULT_METHOD2 = GroupRollingMean('is_weekend', 8, 1)
    DEFAULT_METHOD3 = GroupRollingMean(('hour', 'minute'), 8, 1)
    DEFAULT_METHOD4 = GroupMean(('month', 'weekday', 'hour', 'minute'))

    def __init__(
        self,
        columns: ColumnNames | None = None,
        interval='15m',
        *,
        method1=DEFAULT_METHOD1,
        method2=DEFAULT_METHOD2,
        method3=DEFAULT_METHOD3,
        method4=DEFAULT_METHOD4,
    ) -> None:
        super().__init__(columns=columns, interval=interval)

        # 기본 파라미터
        self.method1 = method1
        self.method2 = method2
        self.method3 = method3
        self.method4 = method4

    def _impute(self, data: pl.DataFrame | pl.LazyFrame):
        value = self._col.value
        imputed = self._col.imputed

        df = (
            data.lazy()
            .with_columns(  # 각 단계 계산
                method1=self.method1.expr(value),
                method2=self.method2.expr(value),
                method3=self.method3.expr(value),
                method4=self.method4.expr(value),
            )
            .collect()
        )

        return df.with_columns(
            pl.when(pl.col(value).is_not_null())
            .then(pl.col(value))
            .otherwise(
                # 네 단계 평균
                df.select('method1', 'method2', 'method3', 'method4').mean_horizontal()
            )
            .alias(imputed)
        )


class Imputer02(AbstractImputer):
    DEFAULT_THRESHOLD1 = 4  # method 1, 2 적용 구분 임계값
    DEFAULT_THRESHOLD2 = 96  # method 2,3 적용 구분 임계값

    # 전후 총 4일 내 시간, 분이 같은 행 5개 (대상 행 포함)
    DEFAULT_METHOD2 = GroupMean(('hour', 'minute'), 5)

    # 전후 16주 내 요일, 시간, 분이 같은 행 약 17개 (대상 행 포함)
    DEFAULT_METHOD3 = GroupMean(('weekday', 'hour', 'minute'), 17)

    def __init__(
        self,
        columns: ColumnNames | None = None,
        interval='15m',
        *,
        threshold1=DEFAULT_THRESHOLD1,
        threshold2=DEFAULT_THRESHOLD2,
        method2=DEFAULT_METHOD2,
        method3=DEFAULT_METHOD3,
    ) -> None:
        super().__init__(columns=columns, interval=interval)

        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.method2 = method2
        self.method3 = method3

        # 경희대는 3번 방법에서 결측 시점 전후 (대칭적으로) 120일이 아니라,
        # 결측치가 발생한 월 기준으로 전후 2달 고려함
        # e.g. 6월 1일 결측치 발생 -> 4월 1일~8월 31일 데이터 모두 계산에 이용

    def _impute(self, data: pl.DataFrame | pl.LazyFrame):
        value = self._col.value
        imputed = self._col.imputed

        return (
            (data)
            .with_columns(
                null_count=count_consecutive_null(pl.col(value)),
                method1=pl.col(value).interpolate('linear'),  # 선형보간
                method2=self.method2.expr(value),
                method3=self.method3.expr(value),
            )
            .with_columns(
                pl.when(pl.col(value).is_not_null())
                .then(pl.col(value))
                .when(pl.col('null_count') < self.threshold1)
                .then(pl.col('method1'))
                .when(pl.col('null_count') < self.threshold2)
                .then(pl.col('method2'))
                .otherwise(pl.col('method3'))
                .alias(imputed)
            )
        )


class Imputer03(AbstractImputer):
    DEFAULT_METHOD1 = GroupRollingMean(('year', 'month'), 16, 2)
    DEFAULT_METHOD2_THRESHOLD = 4 * 24 * 3  # 3일
    DEFAULT_METHOD2_1 = GroupMean(('weekday', 'hour', 'minute'), 17)
    DEFAULT_METHOD2_2 = GroupRollingMean(('hour', 'minute'), 40, 2)

    def __init__(
        self,
        columns: ColumnNames | None = None,
        interval='15m',
        *,
        method1=DEFAULT_METHOD1,
        method2_threshold=DEFAULT_METHOD2_THRESHOLD,
        method2_1=DEFAULT_METHOD2_1,
        method2_2=DEFAULT_METHOD2_2,
    ) -> None:
        super().__init__(columns=columns, interval=interval)

        self.method1 = method1
        self.method2_threshold = method2_threshold
        self.method2_1 = method2_1
        self.method2_2 = method2_2

    def _impute(self, data: pl.DataFrame | pl.LazyFrame):
        # random forest 생략
        value = self._col.value
        imputed = self._col.imputed

        df = (
            data.lazy()
            .with_columns(
                null_count=count_consecutive_null(pl.col(value)),
                method1=self.method1.expr(value),
                method2_1=self.method2_1.expr(value),
                method2_2=self.method2_2.expr(value),
            )
            .with_columns(
                pl.when(pl.col(value).is_not_null())
                .then(pl.col(value))
                .when(pl.col('null_count') > self.method2_threshold)
                .then(pl.col('method2_1'))
                .otherwise(pl.col('method2_2'))
                .interpolate('linear')  # 남은 결측치 선형보간
                .alias('method2')
            )
            .collect()
        )

        return df.with_columns(
            pl.when(pl.col(value).is_not_null())
            .then(pl.col(value))
            .otherwise(df.select('method1', 'method2').mean_horizontal())
            .alias(imputed)
        )


class Imputer03KHU(Imputer03):
    # 경희대 보고서는 method 3-2 Step 2에서 남은 결측치에 선형보간을 적용한다고
    # 기록했으나, 코드는 MA(40, 4) 적용

    # SEDA, 경희대 보고서는 1단계에 같은 연도, 월 데이터를 이용한다고 기록
    # 코드는 같은 월, 시간, 분 데이터 이용
    # 결과는 연도/월, 월/시간/분 데이터 적용 결과 혼재 추정
    DEFAULT_METHOD1 = GroupRollingMean(('month', 'hour', 'minute'), 16, 2)

    # 경희대 보고서는 2.2에 시간, 분 데이터 그룹, MA(40, 2)
    # 코드는 월/시간/분에 MA(40, 4) 적용
    DEFAULT_METHOD2_2 = GroupRollingMean(('month', 'hour', 'minute'), 40, 4)

    # 나머지 기본 파라미터는 동일
    DEFAULT_METHOD2_THRESHOLD = 4 * 24 * 3  # 3일
    DEFAULT_METHOD2_1 = GroupMean(('weekday', 'hour', 'minute'), 17)

    def __init__(
        self,
        columns: ColumnNames | None = None,
        interval='15m',
        *,
        method1=DEFAULT_METHOD1,
        method2_threshold=DEFAULT_METHOD2_THRESHOLD,
        method2_1=DEFAULT_METHOD2_1,
        method2_2=DEFAULT_METHOD2_2,
    ) -> None:
        super().__init__(
            columns=columns,
            interval=interval,
            method1=method1,
            method2_threshold=method2_threshold,
            method2_1=method2_1,
            method2_2=method2_2,
        )

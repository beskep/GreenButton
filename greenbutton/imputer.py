"""polars 라이브러리를 통한 AMI 데이터 보간 방법 구현."""
# ruff: noqa: PLR0913

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import pchip_interpolate
from scipy.ndimage import label


def group_size(array: ArrayLike) -> NDArray[np.uint32]:
    """
    array 중 0이 아닌 연속적인 집단에 집단 크기를 채워 반환.

    연속적인 결측치 크기를 계산하는데 이용.

    TODO polars.Expr.rle_id로 교체

    Parameters
    ----------
    array : ArrayLike

    Returns
    -------
    NDArray[np.uint32]

    Examples
    --------
    >>> group_size([0, 1, 2, 3, 0, 0])
    array([0, 3, 3, 3, 0, 0], dtype=uint32)

    >>> group_size([1, 0, 42, 1, 0, 0, 1, 2, 3, 0, 0])
    array([1, 0, 2, 2, 0, 0, 3, 3, 3, 0, 0], dtype=uint32)
    """
    labeled: NDArray
    count: int

    labeled, count = label(array)  # pyright: ignore [reportGeneralTypeIssues]

    choices = np.array(
        [0, *(np.sum(labeled == x + 1) for x in range(count))], dtype=np.uint32
    )
    return np.choose(labeled, choices)


class ImputeDataError(ValueError):
    pass


@dataclass
class ColumnNames:
    """보간 입출력 데이터의 column 이름."""

    dt: str = 'datetime'  # 날짜-시간
    value: str = 'value'  # 값 (전력사용량)
    imputed: str = 'imputed'  # 보간 결과

    # 보간 후 제거할 임시 데이터 column 목록
    drop: Collection[str] = ('year', 'month', 'day', 'hour', 'minute')

    @property
    def inputs(self):
        return (self.dt, self.value)

    @property
    def outputs(self):
        return (self.dt, self.value, self.imputed)


class AbstractImputer(ABC):
    def __init__(self, columns: ColumnNames | None = None, interval='15m') -> None:
        """
        AMI 데이터 보간을 위한 class.

        Parameters
        ----------
        columns : ColumnNames | None, optional
            입출력 데이터의 시간, 값 (전력사용량), 보간 결과 등 column 이름.
        interval : str, optional
            AMI 측정 간격. 기본 15분.
        """
        if columns is None:
            columns = ColumnNames()

        self._col = columns
        self._interval = interval

    @property
    def columns(self):
        return self._col

    def preprocess(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        """
        보간 데이터 전처리.

        측정 시간을 `interval` 간격으로 upsample하고
        행별로 연, 달, 일, 시간, 분, 요일, 주말 여부 판단.

        TODO 주말 대신 공휴일 여부 데이터 필요.

        Parameters
        ----------
        data : pl.DataFrame | pl.LazyFrame

        Returns
        -------
        pl.DataFrame
        """
        dt = pl.col(self._col.dt)

        return (
            data.lazy()
            .sort(self._col.dt)  # 시간 정렬
            .collect()
            .upsample(self._col.dt, every=self._interval)  # 시간 간격 조정
            .with_columns(
                year=dt.dt.year(),
                month=dt.dt.month(),
                day=dt.dt.day(),
                hour=dt.dt.hour(),
                minute=dt.dt.minute(),
                weekday=dt.dt.weekday(),
                is_weekend=dt.dt.weekday().replace({6: True, 7: True}, default=False),
            )
        )

    @abstractmethod
    def _impute(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        pass

    def impute(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        # 필요한 column이 존재하는지 체크
        if cols := [x for x in self._col.inputs if x not in data.columns]:
            raise ImputeDataError(sorted(cols))

        prep = self.preprocess(data)
        imputed = self._impute(prep)

        # 보간 결과가 존재하는지 체크
        if cols := [x for x in self._col.outputs if x not in imputed.columns]:
            raise ImputeDataError(sorted(cols))

        # 계산 과정에 사용한 임시 column 제거하고 return
        return imputed.drop(self._col.drop)


class MeanImputer(AbstractImputer):
    """전체 평균 보간 (비교 평가용)."""

    def _impute(self, data: pl.DataFrame | pl.LazyFrame):
        return data.with_columns(
            pl.col(self._col.value).fill_null(strategy='mean').alias(self._col.imputed)
        )


class ForwardImputer(AbstractImputer):
    """Forward 보간 (비교 평가용)."""

    def _impute(self, data: pl.DataFrame | pl.LazyFrame):
        return data.with_columns(
            pl.col(self._col.value)
            .fill_null(strategy='forward')
            .alias(self._col.imputed)
        )


class LinearImputer(AbstractImputer):
    """선형 보간 (비교 평가용)."""

    def _impute(self, data: pl.DataFrame | pl.LazyFrame):
        return data.with_columns(
            pl.col(self._col.value)
            .interpolate(method='linear')
            .alias(self._col.imputed)
        )


class PchipImputer(AbstractImputer):
    """PCHIP 보간.

    ETRI와 정확히 같은 방법인지 불확실함.

    검증 필요 (PCHIP 결과, 대량 데이터 입력 시 성능 등).

    References
    ----------
    [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pchip_interpolate.html
    """

    def _impute(self, data: pl.DataFrame | pl.LazyFrame):
        value = self._col.value
        imputed = self._col.imputed

        # datetime을 `interval` 간격으로 upsample하고 정렬했기 때문에
        # index열을 x로 적용 가능
        df = data.lazy().with_row_index().collect()

        observed = (
            df.filter(pl.col(value).is_not_null()).select('index', value).to_numpy()
        )
        na = df.filter(pl.col(value).is_null())
        x = na.select('index').to_numpy().ravel()
        y = pchip_interpolate(xi=observed[:, 0], yi=observed[:, 1], x=x)

        na = na.select('index', pl.Series(imputed, y))
        return (
            df.join(na, on='index', how='left')
            .with_columns(
                pl.when(pl.col(value).is_null())
                .then(pl.col(imputed))
                .otherwise(pl.col(value))
            )
            .drop('index')
        )


@dataclass
class GroupMean:
    """그룹별 평균 보간 파라미터/expression.

    Parameters
    ----------
    group: str | Iterable[str]
        평균 대상 그룹
    window_size: int | None = None
        평균 기간 (대상 전후 window 크기). None이면 전체 기간 평균.
    min_periods: int | None = 1
        window 내 최소 데이터 개수
    """

    group: str | Iterable[str]
    window_size: int | None = None  # TODO str | timedelta
    min_periods: int | None = 1

    def expr(self, column: str):
        if self.window_size is None:
            # 전체 기간 평균
            return pl.mean(column).over(self.group)

        # window 내 데이터 평균
        return (
            pl.col(column)
            .rolling_mean(
                window_size=self.window_size,
                min_periods=self.min_periods,
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
    min_periods: int | None
        window 내 최소 데이터 개수
    """

    group: str | Iterable[str]
    window_size: int  # TODO timedelta | str
    min_periods: int | None

    def expr(self, column: str):
        return (
            pl.col(column)
            .rolling_mean(window_size=self.window_size, min_periods=self.min_periods)
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

        is_null = data.lazy().select(pl.col(value).is_null()).collect()
        null_size = group_size(is_null.to_numpy().ravel())

        return data.with_columns(
            pl.Series('null_size', null_size),
            method1=pl.col(value).interpolate('linear'),  # 선형보간
            method2=self.method2.expr(value),
            method3=self.method3.expr(value),
        ).with_columns(
            pl.when(pl.col(value).is_not_null())
            .then(pl.col(value))
            .when(pl.col('null_size') < self.threshold1)
            .then(pl.col('method1'))
            .when(pl.col('null_size') < self.threshold2)
            .then(pl.col('method2'))
            .otherwise(pl.col('method3'))
            .alias(imputed)
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

        is_null = data.lazy().select(pl.col(value).is_null()).collect()
        null_size = group_size(is_null.to_numpy().ravel())

        df = (
            data.lazy()
            .with_columns(
                pl.Series('null_size', null_size),
                method1=self.method1.expr(value),
                method2_1=self.method2_1.expr(value),
                method2_2=self.method2_2.expr(value),
            )
            .with_columns(
                pl.when(pl.col(value).is_not_null())
                .then(pl.col(value))
                .when(pl.col('null_size') > self.method2_threshold)
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


def _impute_test():
    from whenever import LocalDateTime  # noqa: PLC0415

    dts = [
        LocalDateTime(2000, 1, 1),
        LocalDateTime(2000, 1, 1, 1),
        LocalDateTime(2001, 1, 1),
        LocalDateTime(2001, 1, 1, 1),
    ]

    pl.Config.set_tbl_cols(20)

    data = pl.DataFrame({
        'datetime': [x.py_datetime() for x in dts],
        'value': [2, 4, None, 6],
    })

    classes: list[type[AbstractImputer]] = [
        LinearImputer,
        Imputer01,
        Imputer02,
        Imputer03,
        Imputer03KHU,
    ]
    for cls in classes:
        imputer = cls(columns=ColumnNames())
        imputed = imputer.impute(data).filter(pl.col('imputed').is_not_null())
        assert imputed.height > data.height

        print(f'class={cls.__name__}\n{imputed}\n')


if __name__ == '__main__':
    from rich import print  # noqa: A004

    _impute_test()

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Collection

__all__ = ['AbstractImputer', 'ColumnNames', 'ImputeDataError']


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
    def __init__(
        self,
        columns: ColumnNames | None = None,
        interval: str = '15m',
    ) -> None:
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
    def _impute(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        pass

    def impute(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
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

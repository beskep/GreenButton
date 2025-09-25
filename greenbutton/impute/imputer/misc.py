"""비교/검증용 보간법."""

import polars as pl
from scipy.interpolate import pchip_interpolate

from ._imputer import AbstractImputer


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

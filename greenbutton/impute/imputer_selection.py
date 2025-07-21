"""sklearn.model_selection을 이용하기 위한 Imputer 클래스."""

import numpy as np
import polars as pl
import pydash as pyd
from pydash.helpers import UNSET
from sklearn.base import BaseEstimator, TransformerMixin

import greenbutton.impute.imputer as impt


def _split_param(name: str, /):
    ls = name.split('__')
    return ls[-1], None if len(ls) == 1 else ls[:-1]


class _Estimator(BaseEstimator, TransformerMixin):
    # 조정 가능한 파라미터 목록
    # attribute는 '__'로 구분함
    PARAMS: tuple[str, ...] = ()

    def __init__(self, *args, **kwargs) -> None:
        params = {k: v for k, v in kwargs.items() if k not in {'columns', 'interval'}}
        kwargs = {k: v for k, v in kwargs.items() if k not in params}

        super().__init__(*args, **kwargs)

        self.set_params(**params)
        self._col = impt.ColumnNames(dt='datetime', value='value', imputed='ypred')

    def fit(self, *_args, **_kwargs):
        return self

    def transform(self, x, **kwargs) -> pl.DataFrame:
        return self.impute(x, **kwargs)

    def iter_params(self):
        for param in self.PARAMS:
            value = pyd.get(self, param.split('__'), default=UNSET)  # type: ignore[arg-type]
            yield param, value

    def get_params(self, *_args, **_kwargs) -> dict:
        return dict(self.iter_params())

    def set_params(self, **params):
        path: list | None
        for key, value in params.items():
            name, path = _split_param(key)
            obj = self if path is None else pyd.get(self, path, default=UNSET)
            setattr(obj, name, value)

        return self

    def score(self, df: pl.DataFrame) -> float:
        """
        RMSE 계산.

        Parameters
        ----------
        df : pl.DataFrame

        Returns
        -------
        float
        """
        imputed = self.transform(df).filter(pl.col('value').is_null())

        if not imputed.height:
            # 대상 데이터에 결측 부분이 없음
            return -np.inf

        rmse = (pl.col('ypred') - pl.col('ytrue')).pow(2).mean().sqrt()
        return -imputed.select(rmse).item()


class Imputer01(_Estimator, impt.Imputer01):
    PARAMS = (
        'method1__group',
        'method1__window_size',
        'method1__min_periods',
        'method2__group',
        'method2__window_size',
        'method2__min_periods',
        'method3__group',
        'method3__window_size',
        'method3__min_periods',
        'method4__group',
        'method4__window_size',
        'method4__min_periods',
    )


class Imputer02(_Estimator, impt.Imputer02):
    PARAMS = (
        'threshold1',
        'threshold2',
        'method2__group',
        'method2__window_size',
        'method2__min_periods',
        'method3__group',
        'method3__window_size',
        'method3__min_periods',
    )


class Imputer03(_Estimator, impt.Imputer03):
    PARAMS = (
        'method1__group',
        'method1__window_size',
        'method1__min_periods',
        'method2_threshold',
        'method2_1__group',
        'method2_1__window_size',
        'method2_1__min_periods',
        'method2_2__group',
        'method2_2__window_size',
        'method2_2__min_periods',
    )

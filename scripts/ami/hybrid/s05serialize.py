"""2026-08-11 서비스 예시 데이터 직렬화."""

import dataclasses as dc
import functools
from pathlib import Path  # ruff: ignore[typing-only-standard-library-import]
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import msgspec
import numpy as np
import polars as pl
import scipy.optimize as opt
import statsmodels.api as sm
import structlog
from scipy import stats

from greenbutton.utils.cli import App
from scripts.ami.hybrid.s03cpm import SOURCE

if TYPE_CHECKING:
    from statsmodels.regression.linear_model import RegressionResults

type ModelType = Literal['hc', 'h', 'c']


logger = structlog.stdlib.get_logger()
app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml', root_keys='hybrid', use_commands_as_keys=False
    )
)


class ModelParams(msgspec.Struct):
    """CPM Parameters."""

    baseline: float
    t_h: float
    t_c: float
    beta_h: float
    beta_c: float


class ModelCore(msgspec.Struct):
    """예측값, 신뢰구간(및 prediction interval) 재구성에 필수인 값들."""

    cov_params: list[list[float]]
    df_resid: float
    use_t: bool
    scale: float  # prediction interval(개별 관측값 범위) 계산에 필요
    k_constant: int
    cov_type: str


class ModelDiagnostics(msgspec.Struct):
    """예측 계산에는 불필요, 모델 품질 보고용."""

    nobs: int
    df_model: float
    rsquared: float
    rsquared_adj: float
    aic: float
    bic: float
    llf: float
    fvalue: float | None
    f_pvalue: float | None
    mse_resid: float
    mse_model: float
    mse_total: float
    condition_number: float


class ModelDerived(msgspec.Struct):
    """core로부터 파생 가능(중복)하지만 검증·즉시 사용 편의상 포함."""

    bse: list[float]
    tvalues: list[float]
    pvalues: list[float]


class ModelRobustCov(msgspec.Struct):
    """cov_type이 'nonrobust'가 아닐 때만 채움."""

    cov_type: str
    cov_kwds: dict


class Predicted(msgspec.Struct):
    temperature: list[float]

    predicted: list[float]
    """예측 에너지 사용량."""

    ci_lower: list[float]
    """(1-alpha) 구간 하한."""

    ci_upper: list[float]
    """(1-alpha) 구간 상한."""

    alpha: float = 0.05
    kind: Literal['mean', 'obs'] = 'mean'


class Model(msgspec.Struct):
    kind: ModelType
    params: ModelParams
    core: ModelCore
    diagnostics: ModelDiagnostics
    derived: ModelDerived
    robust_cov: ModelRobustCov | None = None

    @classmethod
    def create(
        cls,
        kind: ModelType,
        cp: tuple[float, float],
        r: RegressionResults,
    ) -> Model:
        p = r.params.tolist()
        match kind:
            case 'hc':
                bh, bc = p[1:]
            case 'h':
                bh = p[1]
                bc = np.nan
            case 'c':
                bh = np.nan
                bc = p[1]

        params = ModelParams(baseline=p[0], t_h=cp[0], t_c=cp[1], beta_h=bh, beta_c=bc)
        core = ModelCore(
            cov_params=r.cov_params().tolist(),
            df_resid=float(r.df_resid),
            use_t=bool(r.use_t),
            scale=float(r.scale),
            k_constant=int(r.k_constant),
            cov_type=r.cov_type,
        )
        diagnostics = ModelDiagnostics(
            nobs=int(r.nobs),
            df_model=float(r.df_model),
            rsquared=float(r.rsquared),
            rsquared_adj=float(r.rsquared_adj),
            aic=float(r.aic),
            bic=float(r.bic),
            llf=float(r.llf),
            fvalue=None if r.fvalue is None else float(r.fvalue),
            f_pvalue=None if r.f_pvalue is None else float(r.f_pvalue),
            mse_resid=float(r.mse_resid),
            mse_model=float(r.mse_model),
            mse_total=float(r.mse_total),
            condition_number=float(r.condition_number),
        )
        derived = ModelDerived(
            bse=r.bse.tolist(),
            tvalues=r.tvalues.tolist(),
            pvalues=r.pvalues.tolist(),
        )
        robust_cov = (
            None
            if r.cov_type == 'nonrobust'
            else ModelRobustCov(cov_type=r.cov_type, cov_kwds=r.cov_kwds)
        )
        return cls(
            kind=kind,
            params=params,
            core=core,
            diagnostics=diagnostics,
            derived=derived,
            robust_cov=robust_cov,
        )

    def predict(
        self,
        temperature: np.ndarray,
        alpha: float = 0.05,
        kind: Literal['mean', 'obs'] = 'mean',
    ) -> Predicted:
        """
        에너지 사용량 예측.

        Parameters
        ----------
        temperature : np.ndarray
            온도 배열 (℃). 임의 shape 허용 — 반환값도 동일 shape.
        alpha : float, optional
            유의 수준
        kind : Literal['mean', 'obs'], optional
            "mean": 회귀선 평균에 대한 confidence interval
            "obs": 개별 관측값에 대한 prediction interval (scale 포함)

        Returns
        -------
        Predicted

        Raises
        ------
        ValueError
            When temperature.ndim != 1

        Notes
        -----
        Change-point의 불확실성은 반영하지 않고, cp는 고정값으로 취급.
        """
        if temperature.ndim != 1:
            raise ValueError
        if not self.core.k_constant:
            raise ValueError

        x = np.column_stack([
            np.ones_like(temperature),
            np.maximum(0.0, self.params.t_h - temperature),
            np.maximum(0.0, temperature - self.params.t_c),
        ])
        b = np.array([self.params.baseline, self.params.beta_h, self.params.beta_c])

        match self.kind:
            case 'h':
                x = x[:, [0, 1]]
                b = b[[0, 1]]
            case 'c':
                x = x[:, [0, 2]]
                b = b[[0, 2]]

        cov = np.asarray(self.core.cov_params)

        if x.shape[1] != b.shape[0]:
            msg = (
                f'설계행렬 열 수({x.shape[1]})가 params 길이({b.shape[0]})와 다릅니다. '
                'variables/params 순서가 [intercept, heating, cooling]인지 확인하세요.'
            )
            raise ValueError(msg)

        y_hat = x @ b
        # v^T cov v를 행 단위로 벡터화: (X @ cov) 와 X의 성분별 곱을 행 방향 합산
        var_mean = (x @ cov * x).sum(axis=1)
        var_total = var_mean + self.core.scale if kind == 'obs' else var_mean
        se = np.sqrt(var_total)

        crit = (
            stats.t.ppf(1 - alpha / 2, self.core.df_resid)
            if self.core.use_t
            else stats.norm.ppf(1 - alpha / 2)
        )
        lower = y_hat - crit * se
        upper = y_hat + crit * se

        return Predicted(
            temperature=temperature.tolist(),
            predicted=y_hat.tolist(),
            ci_lower=lower.tolist(),
            ci_upper=upper.tolist(),
            alpha=alpha,
            kind=kind,
        )


@dc.dataclass(frozen=True)
class Cpm:
    data: pl.DataFrame
    kind: ModelType

    t: str = 'temperature'
    e: str = 'energy'

    _BOUND: ClassVar[tuple[float, float]] = (5, 30)

    @functools.cached_property
    def endog(self):
        return self.data[self.e].to_numpy()

    @functools.cached_property
    def exog(self):
        match self.kind:
            case 'hc':
                exog = ['HDD', 'CDD']
            case 'h':
                exog = ['HDD']
            case 'c':
                exog = ['CDD']

        return ['ones', *exog]

    def fit_linear_model(self, cp: np.ndarray):
        match self.kind:
            case 'hc':
                th, tc = cp
            case 'h':
                th = cp[0]
                tc = 0
            case 'c':
                th = 0
                tc = cp[0]

        t = pl.col(self.t)
        data = self.data.with_columns(
            pl.lit(1.0).alias('ones'),
            pl.max_horizontal(pl.lit(0), th - t).alias('HDD'),
            pl.max_horizontal(pl.lit(0), t - tc).alias('CDD'),
        )
        x = data.select(self.exog).to_numpy()
        return sm.OLS(endog=self.endog, exog=x).fit()

    def object(self, cp: np.ndarray) -> np.ndarray:
        model = self.fit_linear_model(cp)

        p1 = 1 if cp.size == 1 else 1 + max(0, cp[0] - cp[1]) ** 2

        beta: np.ndarray = model.params[1:3]
        p2 = 1 + np.sum(np.square(np.minimum(0, beta)))

        return np.sum(np.square(model.resid)) * p1 * p2

    @functools.cached_property
    def optimize_result(self) -> opt.OptimizeResult:
        r = opt.differential_evolution(
            self.object,
            bounds=(self._BOUND, self._BOUND) if self.kind == 'hc' else (self._BOUND,),
            rng=42,
        )
        if not r.success:
            msg = 'Failed to optimize'
            raise ValueError(msg)
        return r

    def __call__(self):
        cp = np.round(self.optimize_result.x, 1)
        lm = self.fit_linear_model(cp)

        match self.kind:
            case 'hc':
                t = tuple(cp.tolist())
            case 'h':
                t = (float(cp[0]), np.inf)
            case 'c':
                t = (-np.inf, float(cp[0]))

        return Model.create(self.kind, t, lm)

    @classmethod
    def fit(cls, data: pl.DataFrame, t: str = 'temperature', e: str = 'energy'):
        def it():
            for k in ('hc', 'h', 'c'):
                try:
                    yield cls(data, kind=k, t=t, e=e)()
                except ValueError:
                    continue

        return max(it(), key=lambda x: x.diagnostics.rsquared_adj)


def _write_json(obj: object, path: Path):
    buffer = msgspec.json.encode(obj)
    buffer = msgspec.json.format(buffer)
    path.write_bytes(buffer)


@app.default
@dc.dataclass
class Fit:
    root: Path

    @functools.cached_property
    def output(self):
        d = self.root / '99.dataset'
        d.mkdir(exist_ok=True)
        return d

    @functools.cached_property
    def source(self):
        lf = pl.scan_parquet(self.root / f'{SOURCE}.parquet').filter(
            pl.col('holiday').not_(), pl.col('anomaly').not_()
        )
        heat = (
            lf
            .filter(pl.col('energy') == '열')
            .select('bldg.index', 'date', 'Te', 'eui')
            .with_columns(pl.lit('heat').alias('energy.type'))
        )
        total = (
            lf
            .group_by(['bldg.index', 'date', 'Te'])
            .agg(pl.sum('eui'))
            .with_columns(pl.lit('total').alias('energy.type'))
        )

        return (
            pl
            .concat([total, heat])
            .rename({'Te': 'temperature', 'eui': 'energy'})
            .sort('bldg.index', 'date')
            .collect()
        )

    def __call__(self, max_index: int = 25):
        temperature = np.random.default_rng(42).normal(20, 1, size=10).round(1)

        for (idx, t), data in self.source.group_by(
            ['bldg.index', 'energy.type'], maintain_order=True
        ):
            if idx > max_index:
                break

            logger.info('#%d, energy=%s', idx, t)
            model = Cpm.fit(data)
            predicted = model.predict(temperature)

            _write_json(
                data.select('date', 'temperature', 'energy').to_dict(as_series=False),
                self.output / f'CPM.{t}.{idx:02d}.input.json',
            )
            _write_json(
                model,
                self.output / f'CPM.{t}.{idx:02d}.model.json',
            )
            _write_json(
                predicted,
                self.output / f'CPM.{t}.{idx:02d}.predicted.json',
            )


if __name__ == '__main__':
    app()

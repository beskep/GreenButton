"""Time series anomaly detection."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import polars as pl
import seaborn as sns
from darts import TimeSeries
from darts import models as dm
from darts.ad.anomaly_model import ForecastingAnomalyModel
from darts.ad.scorers import NormScorer
from darts.ad.scorers.scorers import AnomalyScorer
from pandas import Timestamp

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def ts2df(ts: TimeSeries, rename: dict[str, str]):
    return (
        pl.from_pandas(ts.pd_dataframe().reset_index()).rename(rename).drop('component')
    )


class _ForecastingAnomalyModel(ForecastingAnomalyModel):
    def _predict_with_forecasting(  # noqa: PLR0913 PLR0917
        self,
        series: TimeSeries,
        past_covariates: TimeSeries | None = None,
        future_covariates: TimeSeries | None = None,
        forecast_horizon: int = 1,
        start: Timestamp | float | None = None,
        num_samples: int = 1,
    ):
        kwargs = {
            'past_covariates': past_covariates,
            'future_covariates': future_covariates,
            'forecast_horizon': forecast_horizon,
            'start': start,
            'retrain': not self.model._supports_non_retrainable_historical_forecasts,  # noqa: SLF001
            'num_samples': num_samples,
            'stride': 1,
            'last_points_only': True,
            'verbose': False,
            'show_warnings': False,
        }
        return self.model.historical_forecasts(series, **kwargs)


@dataclass
class Columns:
    dt: str = 'dt'
    value: str = 'value'


@dataclass
class DetectorConfig:
    target: Literal['value', 'normalized'] = 'normalized'
    fill_value: float = 0
    lags: int | Sequence[int] = (-1, -2)

    threshold: float = 1

    plot_scale_score: str | None = 'asinh'
    plot_scale_original: str | None = None
    plot_threshold: bool = True

    def get_lags(self):
        return self.lags if isinstance(self.lags, int) else list(self.lags)


class Detector:
    def __init__(
        self,
        forecasting_model: dm.RegressionModel | None = None,
        scorer: AnomalyScorer | Sequence[AnomalyScorer] | None = None,
        columns: Columns | None = None,
        config: DetectorConfig | None = None,
    ) -> None:
        self.cols = columns or Columns()
        self.config = config or DetectorConfig()

        if forecasting_model is None:
            forecasting_model = self.default_forecasting_model()
        if scorer is None:
            scorer = NormScorer()

        self.forecasting = forecasting_model
        self.anomaly = _ForecastingAnomalyModel(model=self.forecasting, scorer=scorer)

    def __call__(self, df: pl.DataFrame):
        return self.detect(df)

    def default_forecasting_model(self, **kwargs):
        return dm.RandomForest(lags=self.config.get_lags(), **kwargs)

    def detect(self, df: pl.DataFrame, *, original_format=False):
        dt = self.cols.dt
        value = self.cols.value
        exprv = pl.col('value')

        dfs = (
            df.sort('dt')
            .select(
                dt=pl.col(dt).dt.cast_time_unit('ns'),
                original=pl.col(value),
                value=pl.col(value).fill_null(self.config.fill_value),
            )
            .with_columns(normalized=(exprv - exprv.mean()) / exprv.std())
        )

        ts = TimeSeries.from_dataframe(
            dfs.to_pandas(), time_col='dt', value_cols=self.config.target
        )

        self.anomaly.fit(ts, start=0, allow_model_training=True, show_warnings=False)
        score, pred = self.anomaly.score(ts, start=0, return_model_prediction=True)

        score_df = ts2df(score, rename={'0': 'score'})
        pred_df = ts2df(pred, rename={self.config.target: 'predicted', 'time': 'dt'})

        detected = (
            dfs.sort('dt')
            .join(pred_df, on='dt', how='left')
            .join(score_df, on='dt', how='left')
        )

        detected = (
            detected.sort('dt')
            .with_columns(
                threshold=pl.lit(self.config.threshold),
                outlier=pl.col('score') > self.config.threshold,
            )
            .with_columns(
                value=pl.when('outlier')
                .then(pl.lit(None))
                .otherwise(pl.col('original'))
            )
        )

        if original_format:
            detected = detected.select(
                pl.col('dt').alias(dt), pl.col('value').alias(value)
            )

        return detected

    def plot(self, df: pl.DataFrame, **kwargs):
        kwargs = {
            'facet_kws': {'sharey': False, 'despine': False, 'legend_out': False},
            'height': 3,
            'aspect': 16 / 3,
            'alpha': 0.5,
            'edgecolor': None,
            'legend': False,
        } | kwargs

        melted = df.melt(
            id_vars=['dt', 'outlier'], value_vars=['original', 'value', 'score']
        )
        grid = (
            sns.relplot(
                melted,
                x='dt',
                y='value',
                hue='outlier',
                row='variable',
                row_order=['score', 'original', 'value'],
                **kwargs,
            )
            .set_axis_labels('')
            .set_titles('')
        )

        ax: Axes
        ax = grid.axes[0, 0]
        ax.set_ylabel('오차 (무차원)')
        if self.config.plot_threshold:
            ax.axhline(y=self.config.threshold, c='crimson', ls=':', lw=2)

        scale = [self.config.plot_scale_score, self.config.plot_scale_original]
        for ax, s in zip(grid.axes.flat[:-1], scale, strict=True):
            if not s:
                continue

            ax.set_yscale(s)
            ax.autoscale_view()

        for ax, title in zip(
            grid.axes.flat, ['오차', '원본', '이상치 제거'], strict=True
        ):
            ax.set_title(title, fontdict={'fontweight': 'bold'}, loc='left')

        return grid.tight_layout()

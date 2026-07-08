import dataclasses as dc
import functools
import itertools
import warnings
from pathlib import Path  # noqa: TC003
from typing import Literal

import cyclopts
import numpy as np
import polars as pl
import seaborn as sns
import statsmodels.api as sm
import structlog
from matplotlib.figure import Figure
from rich.progress import track
from scipy import stats
from skmisc import loess

from greenbutton import utils
from greenbutton.utils.cli import App

type Preprocess = Literal['none', 'log', 'log1p', 'sqrt', 'filter']


logger = structlog.stdlib.get_logger()
app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml', root_keys='hybrid', use_commands_as_keys=False
    )
)


@dc.dataclass
class _AnomalyOutput:
    root: Path
    name: str

    def __post_init__(self):
        self.root.mkdir(exist_ok=True)

    def data(self):
        return self.root / f'{self.name}.parquet'

    @functools.cached_property
    def timeline(self):
        d = self.root / f'{self.name}.timeline'
        d.mkdir(exist_ok=True)
        return d

    @functools.cached_property
    def temp(self):
        d = self.root / f'{self.name}.temp'
        d.mkdir(exist_ok=True)
        return d


@dc.dataclass
class _AnomalyDetector:
    _: dc.KW_ONLY

    max_eui: float = 5.0

    root: Path
    group: tuple[str, ...] = (
        'bldg.case',
        'bldg.index',
        'bldg.type',
        'bldg',
        'gfa',
        'asos.station',
        'energy',
    )
    plot_timeline: bool = False

    def name(self):
        return self.__class__.__name__.removeprefix('Anomaly')

    def case_names(self, data: pl.DataFrame):
        drop = {'bldg', 'bldg.index', 'bldg.type', 'gfa', 'asos.station'}
        return '.'.join(str(data[x][0]) for x in self.group if x not in drop)

    @functools.cached_property
    def output(self):
        return _AnomalyOutput(root=self.root / '02.anomaly', name=self.name())

    @functools.cached_property
    def weather(self):
        return pl.read_parquet(self.root / '01.weather.parquet')

    def _daily(self, data: pl.DataFrame):
        return (
            data
            .sort('datetime')
            .group_by_dynamic('datetime', every='1d', group_by=self.group)
            .agg(pl.sum('consumption'), pl.sum('eui'))
            .with_columns(
                pl.col('datetime').dt.date(),
                pl
                .when(pl.col('eui') > self.max_eui)
                .then(pl.lit(None))
                .otherwise(pl.col('eui'))
                .alias('eui'),
            )
            .rename({'datetime': 'date'})
            .join(self.weather, on=['date', 'asos.station'], how='left', validate='1:1')
            .drop_nulls(['Te', 'eui'])
            .sort('Te')
        )

    def _plot(self, data: pl.DataFrame):
        boolean = data['anomaly'].dtype == pl.Boolean

        if 'date' not in data.columns:
            data = (
                data
                .drop_nulls('anomaly')
                .sort('datetime')
                .group_by_dynamic('datetime', every='1d', group_by=('holiday', 'gfa'))
                .agg(pl.sum('eui'), pl.max('anomaly'))
                .with_columns(
                    pl.col('datetime').dt.date().alias('date'),
                )
            )
        if 'Te' not in data.columns:
            data = data.join(self.weather, on=['date', 'asos.station'])

        if not data.height:
            return

        def plot(x: str, xlabel: str):
            fig = Figure()
            ax = fig.subplots()
            ax.set_xlabel(xlabel)
            ax.set_ylabel('EUI [kWh/m²]')
            ax.set_yscale('asinh')

            if boolean:
                sns.scatterplot(
                    data,
                    x=x,
                    y='eui',
                    hue='anomaly',
                    hue_order=[False, True],
                    ax=ax,
                    alpha=0.25,
                )
            else:
                fig.colorbar(
                    ax.scatter(
                        x=data['date'].to_numpy(),
                        y=data['eui'].to_numpy(),
                        c=data['anomaly'].to_numpy(),
                        alpha=0.25,
                    )
                )

            return fig

        name = self.case_names(data)
        plot('Te', 'External Temperature [°C]').savefig(
            self.output.temp / f'{name}.png'
        )
        if self.plot_timeline:
            plot('date', '').savefig(self.output.timeline / f'{name}.png')

    def _detect(self, data: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    def _summarize(self, data: pl.DataFrame):
        total = data.select(
            pl.lit('total').alias('bldg.case'),
            pl.len(),
            pl.col('anomaly').sum().alias('anomaly.count'),
            pl.col('anomaly').mean().alias('anomaly.ratio'),
        )
        agg = (
            data
            .group_by(self.group)
            .agg(
                pl.len(),
                pl.col('anomaly').sum().alias('anomaly.count'),
                pl.col('anomaly').mean().alias('anomaly.ratio'),
            )
            .sort(self.group)
        )
        return (
            pl
            .concat([total, agg], how='diagonal')
            .select(*self.group, 'len', 'anomaly.count', 'anomaly.ratio')
            .with_columns()
        )

    def __call__(self):
        consumption = (
            pl
            .scan_parquet(self.root / '01.raw.parquet')
            .with_columns(pl.col('consumption').truediv('gfa').alias('eui'))
            .collect()
        )
        total = consumption.select(self.group).n_unique()
        it = consumption.group_by(self.group, maintain_order=True)

        def detect():
            for _, df in track(it, total=total, description=self.__class__.__name__):
                score = self._detect(df)
                self._plot(score)
                yield score

        detected = pl.concat(detect())
        output = self.output.data()
        detected.write_parquet(output)
        self._summarize(detected).write_csv(
            output.with_suffix('.summary.csv'), include_bom=True
        )


@app.command
@dc.dataclass
class SpactralResidual(_AnomalyDetector):
    every: str = '1d'
    score_window: int = 12
    threshold: float = 0.1

    def _detect(self, data: pl.DataFrame):
        from pyod.models.ts_spectral_residual import (  # type: ignore  # noqa: PGH003, PLC0415
            SpectralResidual,
        )

        if data['bldg'].n_unique() != 1:
            raise ValueError(data['bldg'].unique().sort().to_list())

        data = (
            data
            .sort('datetime')
            .upsample('datetime', every=self.every)
            .with_columns(pl.col('consumption').interpolate('linear').alias('value'))
        )
        detector = SpectralResidual(self.score_window).fit(data['value'].to_numpy())

        return (
            data
            .with_columns(pl.Series('anomaly', detector.decision_scores_))
            .with_columns(
                pl
                .when(pl.col('consumption').is_null())
                .then(pl.lit(None))
                .otherwise('anomaly')
                .alias('anomaly')
            )
            .with_columns()
        )


@app.command
@dc.dataclass
class LowessTukey(_AnomalyDetector):
    preprocess: Preprocess = 'none'

    _: dc.KW_ONLY

    frac: float = 0.2
    min_eui: float = 0.01
    k: float = 3

    root: Path
    group: tuple[str, ...] = (*_AnomalyDetector.group, 'holiday')

    def name(self):
        return f'{super().name()}.{self.preprocess}.k={self.k:.1f}'

    @staticmethod
    def _scale(expr: pl.Expr):
        return (
            (expr - expr.median())
            .truediv(
                expr.quantile(0.75, interpolation='linear')
                - expr.quantile(0.25, interpolation='linear')
            )
            .fill_nan(None)
        )

    def _daily(self, data: pl.DataFrame):
        c = pl.col('consumption')
        match self.preprocess:
            case 'none' | 'filter':
                p = c
            case 'log':
                p = c.log().fill_nan(None)
            case 'log1p':
                p = c.log1p()
            case 'sqrt':
                p = c.sqrt()

        return (
            data
            .sort('datetime')
            .group_by_dynamic('datetime', every='1d', group_by=self.group)
            .agg(c.sum(), pl.sum('eui'))
            .with_columns(
                pl.col('datetime').dt.date(),
                pl
                .when(pl.col('eui') > self.max_eui)
                .then(pl.lit(None))
                .otherwise(pl.col('eui'))
                .alias('eui'),
                scaled=self._scale(p).fill_nan(None),
            )
            .rename({'datetime': 'date'})
            .join(self.weather, on=['date', 'asos.station'], how='left', validate='1:1')
            .drop_nulls(['Te', 'eui'])
            .sort('Te')
            .with_row_index()
        )

    def _detect(self, data: pl.DataFrame):
        if data['bldg'].n_unique() != 1:
            raise ValueError(data['bldg'].unique().sort().to_list())

        data = self._daily(data)

        match self.preprocess:
            case 'filter':
                smoothed = data.filter(pl.col('eui') >= self.min_eui)
            case _:
                smoothed = data.drop_nulls('scaled')

        array = sm.nonparametric.lowess(
            endog=smoothed['scaled'].to_numpy(),
            exog=smoothed['Te'].to_numpy(),
            return_sorted=False,
            frac=self.frac,
        )
        smoothed = smoothed.select('index', pl.Series('smoothed', array))

        return (
            data
            .join(smoothed, on='index', how='left', validate='1:1')
            .sort('Te')
            .with_columns(
                residual=pl.col('smoothed').interpolate('linear') - pl.col('scaled')
            )
            .with_columns(self._scale(pl.col('residual')).alias('residual.scaled'))
            .with_columns(
                anomaly=pl
                .col('residual.scaled')
                .is_between(-self.k, self.k)
                .not_()
                .fill_null(value=False)
            )
        )

    def __call__(self):
        warnings.filterwarnings('ignore', message='Mean of empty slice.')
        warnings.filterwarnings(
            'ignore', message='invalid value encountered in scalar divide'
        )
        return super().__call__()


@app.command
@dc.dataclass
class Lowess(_AnomalyDetector):
    frac: float = 0.2
    confidence: float = 0.95
    min_resid_quantile: float = 0.1
    min_samples: int = 10

    group: tuple[str, ...] = (*_AnomalyDetector.group, 'holiday')

    def name(self):
        return f'{super().name()}.frac{self.frac:.2f}.conf{self.confidence:.2f}'

    def _plot(self, data: pl.DataFrame):
        fig = Figure()
        ax = fig.subplots()

        sns.scatterplot(
            data,
            x='Te',
            y='eui',
            hue='anomaly',
            hue_order=[False, True],
            ax=ax,
            alpha=0.5,
        )

        cls = self.__class__.__name__.lower()
        for x in ('fit', 'lower', 'upper'):
            ax.plot(
                data['Te'],
                data[f'{cls}.{x}'],
                alpha=0.2,
                lw=1,
                ls='-' if x == 'fit' else '--',
                c='k',
            )

        ax.set_xlabel('External Temperature [°C]')
        ax.set_ylabel('EUI [kWh/m²]')

        name = self.case_names(data)
        fig.savefig(self.output.temp / f'{name}.png')

    def _detect(self, data: pl.DataFrame):
        if data['bldg'].n_unique() != 1:
            raise ValueError(data['bldg'].unique().sort().to_list())

        data = self._daily(data)
        if data.drop_nulls('eui').height < self.min_samples:
            return data.with_columns(
                pl.col('eui').alias('lowess.fit'),
                pl.lit(None).alias('lowess.lower'),
                pl.lit(None).alias('lowess.upper'),
                pl.lit(None).alias('anomaly'),
            )

        x = data['Te'].to_numpy()
        y = data['eui'].to_numpy()

        # 이중평활 분산 추정 -- GAMLSS 참조
        ypred = sm.nonparametric.lowess(
            endog=y, exog=x, frac=self.frac, return_sorted=False
        )
        resid = np.abs(y - ypred)
        sigma = sm.nonparametric.lowess(
            endog=resid, exog=x, frac=self.frac, it=0, return_sorted=False
        )
        sigma = np.maximum(
            sigma, np.quantile(resid, self.min_resid_quantile)
        ) / np.sqrt(2 / np.pi)

        t = stats.t.ppf(1 - (1 - self.confidence) / 2, data.height - 2)

        return (
            data
            .with_columns(
                pl.Series('lowess.fit', ypred),
                pl.Series('lowess.lower', ypred - t * sigma),
                pl.Series('lowess.upper', ypred + t * sigma),
            )
            .with_columns(
                anomaly=pl.col('eui').is_between('lowess.lower', 'lowess.upper').not_()
            )
            .with_columns()
        )


@app.command
@dc.dataclass
class VisLowess(Lowess):
    pattern: str = '아파트'

    def _detect(self, data: pl.DataFrame, frac: float = 0.5):
        data = self._daily(data)
        lowess = sm.nonparametric.lowess(
            endog=data['eui'].to_numpy(),
            exog=data['Te'].to_numpy(),
            return_sorted=False,
            frac=frac,
        )

        fig = Figure()
        ax = fig.subplots()
        sns.scatterplot(data, x='Te', y='eui', ax=ax, alpha=0.5)
        sns.lineplot(x=data['Te'].to_numpy(), y=lowess, ax=ax, c='gray')
        ax.set_xlabel('$T_e$')
        ax.set_ylabel('EUI')

        name = self.case_names(data)
        fig.savefig(self.output.temp / f'{name}.{frac:.2f}.png')

    def __call__(self):
        data = (
            pl
            .scan_parquet(self.root / '01.raw.parquet')
            .filter(pl.col('bldg').str.extract(f'({self.pattern})').is_not_null())
            .with_columns(pl.col('consumption').truediv('gfa').alias('eui'))
            .collect()
        )
        total = data.select(self.group).n_unique()
        group_by = data.group_by(self.group, maintain_order=True)

        for _, df in track(group_by, total=total):
            for frac in (0.1, 0.2, 0.25):
                self._detect(df, frac=frac)


@app.command
@dc.dataclass
class Loess(Lowess):
    span: float = 0.5
    confidence: float = 0.95
    group: tuple[str, ...] = (*_AnomalyDetector.group, 'holiday')

    def name(self):
        return (
            f'{super(Lowess, self).name()}'
            f'.span{self.span:.2f}.conf{self.confidence:.2f}'
        )

    def _detect(self, data: pl.DataFrame):
        if data['bldg'].n_unique() != 1:
            raise ValueError(data['bldg'].unique().sort().to_list())

        data = self._daily(data)
        if data.drop_nulls('eui').height < self.min_samples:
            return data.with_columns(
                pl.col('eui').alias('loess.fit'),
                pl.lit(None).alias('loess.lower'),
                pl.lit(None).alias('loess.upper'),
                pl.lit(None).alias('anomaly'),
            )

        x = data.select('Te').to_numpy()
        y = data['eui'].to_numpy()

        # mean model
        mm = loess.loess(x, y)
        mm.model.span = self.span
        pred = mm.predict(x).values

        # sigma model
        resid = np.abs(y - pred)
        sm = loess.loess(x, resid)
        sm.model.span = self.span
        sigma = sm.predict(x).values
        sigma = np.maximum(
            sigma, np.quantile(resid, self.min_resid_quantile)
        ) / np.sqrt(2 / np.pi)

        t = stats.t.ppf(
            1 - (1 - self.confidence) / 2,
            data.height - mm.outputs.enp,  # dof
        )

        return (
            data
            .with_columns(
                pl.Series('loess.fit', pred),
                pl.Series('loess.lower', pred - t * sigma),
                pl.Series('loess.upper', pred + t * sigma),
            )
            .with_columns(
                anomaly=pl.col('eui').is_between('loess.lower', 'loess.upper').not_()
            )
            .with_columns()
        )


@app.command
def batch(root: Path, *, tukey: bool = False):
    for s, c in itertools.product((0.25, 0.5, 0.75), (0.95, 0.99)):
        logger.info('Lowess/Loess span=%s, confidence=%f', s, c)

        Lowess(frac=s, confidence=c, root=root)()
        Loess(span=s, confidence=c, root=root)()

    if not tukey:
        return

    for k in (1.5, 3.0):
        logger.info('LowessTukey k=%s', k)
        LowessTukey(preprocess='sqrt', root=root, k=k)()


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    utils.mpl.MplConciseDate().apply()
    app()

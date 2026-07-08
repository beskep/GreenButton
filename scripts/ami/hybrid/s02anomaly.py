import dataclasses as dc
import functools
import itertools
import warnings
from typing import TYPE_CHECKING, Literal

import cyclopts
import polars as pl
import seaborn as sns
import statsmodels.api as sm
import structlog
from matplotlib.figure import Figure
from rich.progress import track

from greenbutton import utils
from greenbutton.utils.cli import App

if TYPE_CHECKING:
    from pathlib import Path

type LowessPreprocess = Literal['none', 'log', 'log1p', 'sqrt', 'filter']


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
        return self.root / f'anomaly.{self.name}.parquet'

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

    def _plot(self, data: pl.DataFrame):
        name = self.case_names(data)
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

        plot('date', '').savefig(self.output.timeline / f'{name}.png')
        plot('Te', 'External Temperature [°C]').savefig(
            self.output.temp / f'{name}.png'
        )

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
            for _, df in track(it, total=total):
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
class AnomalySR(_AnomalyDetector):
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
class AnomalyLowess(_AnomalyDetector):
    preprocess: LowessPreprocess = 'none'

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
class VisLowess(AnomalyLowess):
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
            for frac in (0.1, 0.2, 0.25, 0.5):
                self._detect(df, frac=frac)


@app.command
def anomaly_lowess_batch(root: Path):
    for p, k in itertools.product(('none', 'sqrt', 'log1p'), (1.5, 2, 3)):
        logger.info('prep=%s, k=%f', p, k)

        AnomalyLowess(preprocess=p, k=k, root=root)()


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    utils.mpl.MplConciseDate().apply()
    app()

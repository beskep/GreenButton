import dataclasses as dc
import functools
import itertools
import typing
import warnings
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import scipy.optimize as opt
import seaborn as sns
import statsmodels.api as sm
import structlog
from matplotlib import patheffects
from matplotlib.figure import Figure
from tqdm.rich import tqdm

from greenbutton import utils
from greenbutton.utils.cli import App

if TYPE_CHECKING:
    from statsmodels.iolib.summary import Summary
    from statsmodels.regression.linear_model import RegressionResults

type Energy = Literal['elec', 'heat', 'total']
type X = Literal['Te', 'Pv', 'I', 'Mon', 'Fri']
type Xs = tuple[X, ...]

XS: Xs = typing.get_args(X.__value__)

ANOMALY_DETECTION = 'Loess.span0.50.conf0.99'

logger = structlog.stdlib.get_logger()
app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml', root_keys='hybrid', use_commands_as_keys=False
    )
)


@app.command
def weekday(root: Path):
    (
        utils.mpl
        .MplTheme(font={'math': 'cm'}, palette='tol:bright')
        .grid()
        .apply({'lines.solid_capstyle': 'butt'})
    )

    workday = {idx + 1: f'{d}요일' for idx, d in enumerate('월화수목금토일')}
    workday = {**workday, 8: '공휴일'}

    data = (
        pl
        .scan_parquet(root / f'02.anomaly/{ANOMALY_DETECTION}.parquet')
        .filter(pl.col('anomaly').not_())
        .group_by('bldg.case', 'bldg.type', 'date', 'holiday')
        .agg(pl.sum('eui'))
        .with_columns(pl.col('date').dt.weekday().alias('weekday'))
        .with_columns(
            pl
            .when(pl.col('weekday').is_in([1, 2, 3, 4, 5]) & pl.col('holiday'))
            .then(pl.lit(8))
            .otherwise(pl.col('weekday'))
            .alias('weekday')
        )
        .with_columns(
            pl
            .col('weekday')
            .replace_strict(workday, return_dtype=pl.String)
            .alias('weekday.name')
        )
        .collect()
    )
    workday = (
        data
        .filter(pl.col('holiday').not_())
        .group_by('bldg.case')
        .agg(pl.mean('eui').alias('eui.workday'))
    )
    data = (
        data
        .join(workday, on='bldg.case', how='left', validate='m:1')
        .with_columns(pl.col('eui').truediv('eui.workday').alias('eui.ratio'))
        .with_columns()
    )

    fig = Figure()
    ax = fig.subplots()

    order = (
        data
        .select('weekday', 'weekday.name')
        .unique()
        .sort('weekday')['weekday.name']
        .to_list()
    )
    sns.barplot(
        data,
        x='eui.ratio',
        y='weekday.name',
        hue='bldg.type',
        hue_order=['공공', '다소비'],
        order=order,
        ax=ax,
        linewidth=0,
    )

    ax.set_xlabel('근무일 대비 사용량 비율')
    ax.set_ylabel('')
    ax.legend(title='', loc='lower left')

    fig.savefig(root / '02.anomaly/weekday.png')

    def fmt(v: float):
        if abs(1 - v) <= 0.02:  # noqa: PLR2004
            return ''
        return f'{v:.1%}'

    for container in ax.containers:
        annotations = ax.bar_label(
            container,  # ty:ignore[invalid-argument-type]
            fmt=fmt,
            padding=12,
            c='0.25',
        )

        pe = [patheffects.withStroke(linewidth=2, foreground='white')]
        for idx, ann in enumerate(annotations):
            if idx >= 5:  # noqa: PLR2004
                # 주말, 공휴일
                ann.remove()
            else:
                ann.set_path_effects(pe)  # ty:ignore[invalid-argument-type]

    ax.set_xlim(0, 1.2)
    fig.savefig(root / '02.anomaly/weekday.label.png')


@dc.dataclass(frozen=True)
class Cpm:
    data: pl.DataFrame

    energy: Energy = 'total'
    xvars: Xs = ('Te',)

    _PENALTY: ClassVar[float] = 1e4
    _BOUND: ClassVar[tuple[float, float]] = (5, 30)

    @functools.cached_property
    def y(self):
        return self.data[f'EUI.{self.energy}'].to_numpy()

    @functools.cached_property
    def exog(self):
        return ('ones', 'hdd', 'cdd', *(x for x in self.xvars if x != 'Te'))

    def fit_linear_model(self, cp: np.ndarray):
        t = pl.col('Te')
        data = self.data.with_columns(
            pl.lit(1.0).alias('ones'),
            pl.max_horizontal(pl.lit(0), cp[0] - t).alias('hdd'),
            pl.max_horizontal(pl.lit(0), t - cp[1]).alias('cdd'),
        )
        x = data.select(self.exog).to_numpy()
        return sm.OLS(endog=self.y, exog=x).fit()

    @functools.cached_property
    def penalty(self):
        sse = np.sum(np.square(self.y - self.y.mean()))
        return sse * self._PENALTY

    def object(self, cp: np.ndarray) -> np.ndarray:
        model = self.fit_linear_model(cp)

        # t_h > t_c일 경우 패널티
        p1 = self.penalty * max(0, cp[0] - cp[1]) ** 2

        # 음수 beta 패널티
        beta: np.ndarray = model.params[1:3]
        p2 = self.penalty * np.sum(np.square(np.minimum(0, beta)))

        resid = np.append(model.resid, [p1, p2])
        return np.sum(np.square(resid))

    @functools.cached_property
    def optimize_result(self) -> opt.OptimizeResult:
        r = opt.differential_evolution(
            self.object,
            bounds=(self._BOUND, self._BOUND),
            rng=42,
        )
        if not r.success:
            msg = 'Failed to optimize'
            raise ValueError(msg)
        return r

    @functools.cached_property
    def linear_model(self) -> RegressionResults:
        return self.fit_linear_model(self.optimize_result.x)

    def summary(self):
        xvars = ['const', '(Th-Te)^+', '(Te-Tc)^+', *self.xvars]
        summary: Summary = self.linear_model.summary()
        summary.add_extra_txt([
            f'exog={xvars}',
            f'change_points={self.optimize_result.x.round(2).tolist()}',
        ])
        return summary

    def stats(self):
        lm = self.linear_model
        params = {f'param:{k}': v for k, v in zip(self.exog, lm.params, strict=True)}
        te = self.data['Te'].to_numpy()
        th, tc = self.optimize_result.x
        return {
            'r2': lm.rsquared,
            'r2adj': lm.rsquared_adj,
            'AIC': lm.aic,
            'BIC': lm.bic,
            'p-value': lm.f_pvalue,
            'n.obs': lm.nobs,
            'n.heating': np.sum(te < th),
            'n.cooling': np.sum(te > tc),
            'Th': th,
            'Tc': tc,
            **params,
        }

    def plot(self):
        cp = self.optimize_result.x
        lm = self.linear_model

        # CPM
        fig = Figure()
        ax = fig.subplots()
        scatter = (
            self.data
            .select(
                'Te',
                pl.col(f'EUI.{self.energy}').alias('Measured'),
                pl.Series('Predicted', lm.fittedvalues),
            )
            .unpivot(index='Te')
            .with_columns()
        )
        ax.axhline(lm.params[0], ls=':', c='gray', alpha=0.5)
        ax.axvline(cp[0], ls=':', c='gray', alpha=0.5)
        ax.axvline(cp[1], ls=':', c='gray', alpha=0.5)
        sns.scatterplot(
            scatter,
            x='Te',
            y='value',
            hue='variable',
            style='variable',
            alpha=0.5,
            s=10,
            ax=ax,
        )
        ax.text(0.02, 0.05, f'$r^2={lm.rsquared:.4f}$', transform=ax.transAxes)
        ax.set_ylim(0)
        ax.set_xlabel('External Temperature [°C]')
        ax.set_ylabel('EUI [kWh/m²]')
        ax.legend(title='', markerscale=2)

        return fig


@dc.dataclass
class BatchCpm:
    root: Path

    _: dc.KW_ONLY

    cases: tuple[str, ...] = (
        # energy.(X1+X2+...)
        'total.Te',
        'total.Te+Pv+I',
    )
    plot: bool = False
    output: str = 'CPM'

    anomaly: str = ANOMALY_DETECTION
    min_n: int = 10

    @functools.cached_property
    def _dir(self):
        d = self.root / '03.CPM'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _source(self):
        return self.root / f'02.anomaly/{self.anomaly}.parquet'

    @functools.cached_property
    def raw(self):
        index = ('bldg.case', 'date', 'Te', 'Pv', 'I')
        return (
            pl
            .scan_parquet(self._source())
            .filter(pl.col('holiday').not_(), pl.col('anomaly').not_())
            .with_columns(
                pl.col('energy').replace_strict({
                    '전력': 'EUI.elec',
                    '열': 'EUI.heat',
                }),
                pl.col('consumption').truediv('gfa').alias('EUI'),
            )
            .collect()
            .pivot('energy', index=index, values='EUI')
            .with_columns(pl.col('date').dt.weekday().alias('weekday'))
            .with_columns(
                pl.col('weekday').eq(1).alias('Mon'),
                pl.col('weekday').eq(5).alias('Fri'),
                (pl.col('EUI.elec') + pl.col('EUI.heat')).alias('EUI.total'),
            )
            .drop_nulls(['EUI.total', *XS])
        )

    @functools.cached_property
    def bldg(self):
        return (
            pl
            .scan_parquet(self._source())
            .group_by(cs.starts_with('bldg'))
            .agg(pl.col('anomaly').mean().alias('anomaly.ratio'))
            .collect()
        )

    def bldg_info(self, case: str):
        return self.bldg.row(by_predicate=pl.col('bldg.case') == case, named=True)

    def _cpm(self, data: pl.DataFrame, case: str):
        energy, x = case.split('.')
        xvars = x.split('+')

        if data.height < self.min_n:
            return None

        cpm = Cpm(data, energy=energy, xvars=xvars)  # ty:ignore[invalid-argument-type]
        c = data['bldg.case'][0]
        bldg = self.bldg_info(c)

        if self.plot:
            d = self._dir / self.output
            d.joinpath(f'{c}.txt').write_text(cpm.summary().as_text())
            cpm.plot().savefig(d / f'{c}.png')

        return {**bldg, 'exog': x, **cpm.stats()}

    def __call__(self):
        total = self.raw.select('bldg.case').n_unique()
        if self.plot:
            self._dir.joinpath(self.output).mkdir(exist_ok=True)

        def it():
            for _, data in self.raw.group_by('bldg.case', maintain_order=True):
                for case in self.cases:
                    if r := self._cpm(data, case=case):
                        yield r

        dicts = list(tqdm(it(), total=total * len(self.cases)))
        models = pl.from_dicts(dicts)

        path = self._dir / f'{self.output}.parquet'
        models.write_parquet(path)
        models.write_csv(path.with_suffix('.csv'), include_bom=True)
        models.describe().write_csv(path.with_suffix('.desc.csv'), include_bom=True)


@app.command
@dc.dataclass
class CompareAnomaly:
    """이상치 전처리 방법별 CPM 결과 비교."""

    root: Path
    _: dc.KW_ONLY
    run: bool = False
    plot: bool = False

    def _run(self):
        for s, c in itertools.product((0.5, 0.75), (0.95, 0.99)):
            logger.info('span=%s, conf=%s', s, c)
            anomaly = f'Loess.span{s:.2f}.conf{c:.2f}'
            BatchCpm(
                root=self.root,
                cases=('total.Te',),
                output=f'AnomalyDetection.{anomaly}',
                anomaly=anomaly,
            )()

    def _plot(self):
        d = self.root / '03.CPM'
        src = list(d.glob('AnomalyDetection*.parquet'))

        pattern = r'.*/.*?span(?P<span>[\d.]+).conf(?P<conf>[\d.]+).*.parquet'
        data = (
            pl
            .read_parquet(src, include_file_paths='path')
            .with_columns(
                pl.col('path').str.replace_all(r'\\', '/').str.extract_groups(pattern)
            )
            .unnest('path')
            .with_columns(pl.col('span', 'conf').cast(pl.Float64))
        )

        (
            sns
            .FacetGrid(data, row='conf', col='span', height=2, margin_titles=True)
            .map_dataframe(sns.histplot, x='r2adj')
            .savefig(d / 'prep.r2adj.png')
        )
        plt.close('all')

        (
            sns
            .FacetGrid(data, row='conf', col='span', height=2, margin_titles=True)
            .map_dataframe(sns.scatterplot, x='anomaly.ratio', y='r2adj', alpha=0.5)
            .savefig(d / 'prep.anomaly.png')
        )
        plt.close('all')

    def __call__(self):
        if self.run:
            self._run()
        if self.plot:
            self._plot()


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='The figure layout has changed to tight')
    (
        utils.mpl
        .MplTheme(font={'math': 'cm'})
        .grid()
        .apply({'lines.solid_capstyle': 'butt'})
    )
    app()

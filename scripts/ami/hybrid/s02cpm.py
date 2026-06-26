import dataclasses as dc
import functools
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import numpy as np
import polars as pl
import scipy.optimize as opt
import seaborn as sns
import statsmodels.api as sm
import structlog
from matplotlib.figure import Figure
from rich.progress import track

from greenbutton import utils
from greenbutton.utils.cli import App

if TYPE_CHECKING:
    from statsmodels.iolib.summary import Summary
    from statsmodels.regression.linear_model import RegressionResults

type Xvar = Literal['Te', 'Te+Pv', 'Te+I', 'Te+Pv+I']

logger = structlog.stdlib.get_logger()
app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml', root_keys='hybrid', use_commands_as_keys=False
    )
)


@dc.dataclass(frozen=True)
class _Cpm:
    data: pl.DataFrame
    xvar: Xvar = 'Te'

    _PENALTY: ClassVar[float] = 1e4

    @functools.cached_property
    def y(self):
        return self.data['EUI'].to_numpy()

    def fit_linear_model(self, cp: np.ndarray):
        xvar = ['ones', 'hdd', 'cdd', *(x for x in self.xvar.split('+') if x != 'Te')]

        t = pl.col('Te')
        data = self.data.with_columns(
            pl.lit(1.0).alias('ones'),
            pl.max_horizontal(pl.lit(0), cp[0] - t).alias('hdd'),
            pl.max_horizontal(pl.lit(0), t - cp[1]).alias('cdd'),
        )

        x = data.select(xvar).to_numpy()

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

    def _optimize(self) -> opt.OptimizeResult:
        r = opt.differential_evolution(self.object, bounds=((5, 20), (5, 20)), rng=42)
        if not r.success:
            msg = 'Failed to optimize'
            raise ValueError(msg)
        return r

    @functools.cached_property
    def optimize_result(self):
        r = self._optimize()
        if not r.success:
            logger.warning('Optimization failed', case=self)
        return r

    @functools.cached_property
    def linear_model(self) -> RegressionResults:
        return self.fit_linear_model(self.optimize_result.x)

    @functools.cached_property
    def summary(self):
        xvars = ['const', '(Th-Te)^+', '(Te-Tc)^+', *self.xvar.split('+')]
        summary: Summary = self.linear_model.summary()
        summary.add_extra_txt([
            f'exog={xvars}',
            f'change_points={self.optimize_result.x.round(2).tolist()}',
        ])
        return summary

    def plot(self):
        cp = self.optimize_result.x
        linear_model = self.linear_model

        # CPM
        fig = Figure()
        ax = fig.subplots()
        scatter = (
            self.data
            .select(
                'Te',
                pl.col('EUI').alias('Measured'),
                pl.Series('Predicted', linear_model.fittedvalues),
            )
            .unpivot(index='Te')
            .with_columns()
        )
        ax.axhline(linear_model.params[0], ls=':', c='gray', alpha=0.5)
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
        ax.text(
            0.02,
            0.05,
            f'$r^2={linear_model.rsquared:.4f}$',
            transform=ax.transAxes,
        )
        ax.set_ylim(0)
        ax.set_xlabel('External Temperature [°C]')
        ax.set_ylabel('EUI [kWh/m²]')
        ax.legend(title='', markerscale=2)

        return fig


@app.command
@dc.dataclass
class Cpm:
    root: Path
    outlier_detection: Literal['IQR', 'LOF'] = 'IQR'
    min_n: int = 10

    @functools.cached_property
    def output(self):
        output = self.root / '03.CPM'
        output.mkdir(exist_ok=True)
        return output

    @functools.cached_property
    def raw(self):
        outlier = f'outlier_{self.outlier_detection.lower()}'
        index = ('type', 'building', 'date', 'Te', 'Pv', 'I')
        return (
            pl
            .scan_parquet(self.root / '01.consumption.day.LOF.parquet')
            .filter(
                pl.col('holiday').not_(),
                pl.col(outlier).not_(),
            )
            .collect()
            .pivot('energy', index=index, values='EUI')
            .with_columns((pl.col('전력') + pl.col('열')).alias('합계'))
            .unpivot(index=index, variable_name='energy', value_name='EUI')
        )

    def _cpm(self, data: pl.DataFrame, xvar: Xvar):
        data = (
            (data)
            .filter(pl.col('energy') == '합계')
            .drop_nulls(['EUI', *xvar.split('+')])
        )

        if data.height < self.min_n:
            return

        cpm = _Cpm(data, xvar=xvar)

        type_ = data['type'][0]
        building = data['building'][0]
        name = f'{type_}.{building}.CPM.{xvar}'

        self.output.joinpath(f'{name}.txt').write_text(cpm.summary.as_text())
        cpm.plot().savefig(self.output / f'{name}.png')

    def __call__(self):
        index = ('type', 'building')
        for _, data in track(
            self.raw.group_by(index),
            total=self.raw.select(index).n_unique(),
        ):
            self._cpm(data, xvar='Te')
            self._cpm(data, xvar='Te+Pv+I')


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    app()

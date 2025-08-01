from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING, Literal

import cyclopts
import polars as pl
import seaborn as sns
from loguru import logger
from matplotlib.figure import Figure

from greenbutton import cpr, utils
from scripts.ami.energy_intensive.common import Buildings, Vars
from scripts.ami.energy_intensive.config import Config  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from scripts.ami.energy_intensive.common import BuildingInfo, InterpDay

app = utils.cli.App(
    config=cyclopts.config.Toml(
        'config/.ami.toml',
        root_keys='energy_intensive',
        use_commands_as_keys=False,
    )
)


@cyclopts.Parameter('cpr')
@dc.dataclass
class _CprOption:
    interp_day: InterpDay = None
    """한전 보간 기준. 기본(`None`): 최종 자료(post)."""

    plot: bool = True


_DEFAULT_CPR_OPTION = _CprOption()


@dc.dataclass
class _CprCalculator:
    conf: Config
    option: _CprOption

    skip_existing: bool = True

    buildings: Buildings = dc.field(init=False)

    _model_output: Path = dc.field(init=False)
    _plot_output: Path = dc.field(init=False)

    @dc.dataclass
    class _Output:
        model: Path
        error: Path
        plot: Path

    def __post_init__(self):
        self.buildings = Buildings(self.conf)

        self._model_output = self.conf.dirs.cpr / 'model'
        self._plot_output = self.conf.dirs.cpr / 'plot'

        self._model_output.mkdir(exist_ok=True)
        if self.option.plot:
            self._plot_output.mkdir(exist_ok=True)

    @staticmethod
    def _set_plot(ax: Axes):
        ax.set_ylim(0)
        ax.set_xlabel('일평균 외기온 [°C]')
        ax.set_ylabel('일간 전력 사용량 [kWh/m²]')

    def concat_models(self):
        models = pl.scan_parquet(self._model_output / '*.parquet').collect()
        models.write_parquet(self.conf.dirs.cpr / 'models.parquet')
        models.write_excel(self.conf.dirs.cpr / 'models.xlsx', column_widths=100)

    def _output(self, name: str):
        return self._Output(
            model=self._model_output / f'{name}.parquet',
            error=self._model_output / f'{name}.error',
            plot=self._plot_output / f'{name}.png',
        )

    def cpr(
        self,
        bldg: BuildingInfo,
        data: pl.DataFrame,
        *,
        is_holiday: bool,
    ):
        data = data.filter(pl.col('is_holiday') == is_holiday)
        name = f'{bldg.file_name()}_{"휴일" if is_holiday else "근무일"}'
        output = self._output(name)

        if self.skip_existing and (output.model.exists() or output.error.exists()):
            return None

        try:
            model = cpr.CprEstimator(
                data, x='temperature', y='eui', datetime='date'
            ).fit()
        except cpr.CprError as e:
            logger.warning(repr(e))
            output.error.touch()

            if self.option.plot:
                fig = Figure()
                ax = fig.add_subplot()
                sns.scatterplot(
                    data, x='temperature', y='eui', ax=ax, alpha=0.6, color='tab:red'
                )
                self._set_plot(ax)
                fig.savefig(self._plot_output / f'(ERROR) {name}.png')

            return None

        (
            model.model_frame.select(
                pl.lit(bldg.kemc).alias(Vars.KEMC_CODE),
                pl.lit(bldg.ente).alias(Vars.ENTE),
                pl.lit(bldg.name).alias(Vars.NAME),
                pl.lit(is_holiday).alias('is_holiday'),
                pl.all(),
            )
            .with_columns()
            .write_parquet(output.model)
        )

        if self.option.plot:
            fig = Figure()
            ax = fig.add_subplot()
            model.plot(
                ax=ax, style={'scatter': {'zorder': 2.1, 'alpha': 0.25, 's': 10}}
            )
            ax.text(
                0.02,
                0.98,
                f'r²={model.model_dict["r2"]:.4f}',
                transform=ax.transAxes,
                va='top',
                weight=500,
            )
            self._set_plot(ax)
            fig.savefig(output.plot)

        return model

    def __call__(self):
        for bldg in self.buildings.iter_buildings(track=True):
            logger.info(bldg)

            try:
                data = self.buildings.data(bldg, interp_day=self.option.interp_day)
            except ValueError as e:
                logger.debug(repr(e))
                continue

            self.cpr(bldg=bldg, data=data, is_holiday=False)
            self.cpr(bldg=bldg, data=data, is_holiday=True)

        self.concat_models()


@app.command
def cpr_(
    cmd: Literal['calculate', 'concat'] = 'calculate',
    *,
    option: _CprOption = _DEFAULT_CPR_OPTION,
    conf: Config,
):
    calculator = _CprCalculator(conf=conf, option=option)

    if cmd == 'calculate':
        calculator()
    else:
        calculator.concat_models()


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplTheme().grid().apply()

    app()

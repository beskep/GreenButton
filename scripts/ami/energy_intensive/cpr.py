from __future__ import annotations

from typing import TYPE_CHECKING

import cyclopts
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from loguru import logger

from greenbutton import cpr, utils
from scripts.ami.energy_intensive.common import KEMC_CODE, BuildingInfo, Buildings
from scripts.ami.energy_intensive.config import Config  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path

app = utils.cli.App(
    config=cyclopts.config.Toml(
        'config/.ami.toml',
        root_keys='energy_intensive',
        use_commands_as_keys=False,
    )
)


def _fit_cpr_model(
    data: pl.DataFrame,
    *,
    building: BuildingInfo,
    is_holiday: bool,
    model_output: Path,
    plot_output: Path | None = None,
):
    plot_output = plot_output or model_output

    data = data.filter(pl.col('is_holiday') == is_holiday)
    estimator = cpr.CprEstimator(data, x='temperature', y='eui', datetime='date')

    file_name = f'{"휴일" if is_holiday else "평일"}_{building.file_name()}'

    try:
        analysis = estimator.fit()
    except cpr.CprError as e:
        fig, ax = plt.subplots()
        sns.scatterplot(data, x='temperature', y='eui', ax=ax, alpha=0.8)
        ax.set_title(repr(e), loc='left')
        fig.savefig(plot_output / f'(ERROR) {file_name}.png')
        plt.close(fig)
        return

    analysis.model_frame.write_parquet(model_output / f'{file_name}.parquet')

    fig, ax = plt.subplots()
    analysis.plot(ax=ax, style={'scatter': {'alpha': 0.25}})
    ax.set_title(
        f'[{KEMC_CODE[building.kemc]}] {building.name} '
        f'(r²={analysis.model_dict["r2"]:.4f})',
        loc='left',
        weight=500,
    )
    ax.set_xlabel('일평균 외기온 [°C]')
    ax.set_ylabel('일간 전력 사용량 [kWh/m²]')
    ax.dataLim.update_from_data_y([0], ignore=False)
    ax.autoscale_view()
    fig.savefig(plot_output / f'{file_name}.png')
    plt.close(fig)


@app.command
def concat_cpr(*, conf: Config):
    models = (
        pl.scan_parquet(
            list(conf.dirs.cpr.glob('model/*.parquet')), include_file_paths='model'
        )
        .select(
            pl.col('model').str.extract(r'.*\\(.*)\.parquet'),
            pl.col('model').str.contains('휴일').alias('is_holiday'),
            pl.all().exclude('model'),
        )
        .collect()
    )
    models.write_excel(conf.dirs.cpr / 'models.xlsx')


@app.command
def cpr_(*, conf: Config):
    buildings = Buildings(conf=conf, electric=True)

    conf.dirs.cpr.mkdir(exist_ok=True)
    model_dir = conf.dirs.cpr / 'model'
    plot_dir = conf.dirs.cpr / 'plot'

    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    for bldg in buildings.iter_buildings():
        try:
            data = buildings.data(bldg)
        except ValueError as e:
            logger.warning(repr(e))
            continue

        logger.info(bldg)

        kwargs = {
            'data': data,
            'building': bldg,
            'model_output': model_dir,
            'plot_output': plot_dir,
        }

        _fit_cpr_model(is_holiday=False, **kwargs)
        _fit_cpr_model(is_holiday=True, **kwargs)


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplTheme().grid().apply()

    app()

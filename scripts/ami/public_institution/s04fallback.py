"""
2025-10-14.

CPM 분석 실패 유형 분류.
"""

from __future__ import annotations

import dataclasses as dc
import functools
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

import cyclopts
import polars as pl
import seaborn as sns
from loguru import logger
from matplotlib.figure import Figure

from greenbutton import cpr, utils
from greenbutton.utils.cli import App
from scripts.ami.public_institution import config as _config
from scripts.ami.public_institution.s02_01cpr import Dataset

if TYPE_CHECKING:
    from matplotlib.axes import Axes

Energy = Literal[
    'raw',  # 사용량
    'compensated',  # 보정 사용량
]


class InvalidModel(TypedDict):
    iid: str
    energy: Energy
    year: NotRequired[int]
    reason: NotRequired[str]


INVALID_MODELS: tuple[InvalidModel, ...] = (
    {'iid': 'DB_B7AE877F-B814-8EED-E050-007F01001D51', 'energy': 'compensated'},
    {'iid': 'DB_B7AE8780-CD5E-8EED-E050-007F01001D51', 'energy': 'compensated'},
    {'iid': 'DB_B7AE8780-D7F7-8EED-E050-007F01001D51', 'energy': 'compensated'},
    {'iid': 'DB_B7AE8781-136D-8EED-E050-007F01001D51', 'energy': 'compensated'},
    {'iid': 'DB_B7AE8782-3309-8EED-E050-007F01001D51', 'energy': 'compensated'},
    {'iid': 'DB_B7AE8782-3F36-8EED-E050-007F01001D51', 'energy': 'compensated'},
    {'iid': 'DB_B7AE8782-42A5-8EED-E050-007F01001D51', 'energy': 'compensated'},
    {'iid': 'DB_B7AE8782-883D-8EED-E050-007F01001D51', 'energy': 'compensated'},
    {'iid': 'DB_B7AE8782-9B08-8EED-E050-007F01001D51', 'energy': 'compensated'},
    {'iid': 'DB_B7AE8782-ADCA-8EED-E050-007F01001D51', 'energy': 'compensated'},
    {'iid': 'DB_B7AE8782-AFE0-8EED-E050-007F01001D51', 'energy': 'compensated'},
    {'iid': 'DB_B7AE8782-AFED-8EED-E050-007F01001D51', 'energy': 'compensated'},
    {'iid': 'SVR_5c323e5b-210a-4643-83eb-817095677da3', 'energy': 'compensated'},
    {'iid': 'DB_B7AE8782-3309-8EED-E050-007F01001D51', 'energy': 'raw'},
    {'iid': 'SVR_5c323e5b-210a-4643-83eb-817095677da3', 'energy': 'raw'},
    {'iid': 'DB_B7AE8782-91A1-8EED-E050-007F01001D51', 'energy': 'raw', 'year': 2022},
    {'iid': 'DB_B7AE877E-2A7D-8EED-E050-007F01001D51', 'energy': 'raw', 'year': 2023},
    {'iid': 'DB_B7AE877F-F31D-8EED-E050-007F01001D51', 'energy': 'raw', 'year': 2023},
    {'iid': 'DB_B7AE877F-F910-8EED-E050-007F01001D51', 'energy': 'raw', 'year': 2023},
    {'iid': 'DB_B7AE8782-42A5-8EED-E050-007F01001D51', 'energy': 'raw', 'year': 2023},
    {'iid': 'DB_B7AE8782-9AE2-8EED-E050-007F01001D51', 'energy': 'raw', 'year': 2023},
    {'iid': 'DB_B7AE8782-B412-8EED-E050-007F01001D51', 'energy': 'raw', 'year': 2023},
    {'iid': 'DB_B7AE8782-B770-8EED-E050-007F01001D51', 'energy': 'raw', 'year': 2023},
    {'iid': 'DB_B7AE8782-9906-8EED-E050-007F01001D51', 'energy': 'raw', 'year': 2024},
)


@dc.dataclass
class Dirs(_config.Dirs):
    cpm_fallback: Path = Path('0210.CPM-fallback')


@dc.dataclass
class Config(_config.Config):
    dirs: Dirs = dc.field(default_factory=Dirs)


app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml',
        root_keys='public_institution',
        allow_unknown=True,
        use_commands_as_keys=False,
    )
)


def _set_ax(ax: Axes):
    ax.set_xlabel('일간 평균 기온 [°C]')
    ax.set_ylabel('전력 사용량 [kWh/m²]')
    if ax.dataLim.ymin >= 0:
        ax.set_ylim(bottom=0)


@cyclopts.Parameter('*')
@dc.dataclass
class Scatter:
    energy: Energy
    conf: Config
    alpha: float = 0.25

    @functools.cached_property
    def dataset(self):
        return Dataset(
            self.conf, energy='사용량' if self.energy == 'raw' else '보정사용량'
        )

    def plot(self, data: pl.DataFrame):
        fig = Figure()
        ax = fig.subplots()
        sns.scatterplot(data, x='temperature', y='energy', ax=ax, alpha=self.alpha)
        _set_ax(ax)
        return fig

    def __call__(self):
        self.conf.dirs.cpm_fallback.mkdir(exist_ok=True)
        for model in INVALID_MODELS:
            if model['energy'] != self.energy:
                continue
            if 'area' in model.get('reason', ''):
                continue

            logger.info(model)

            _, lf = self.dataset.data(institution=model['iid'], with_holiday=False)

            year = model.get('year')
            if year is not None:
                lf = lf.filter(pl.col('datetime').dt.year() == year)

            df = lf.collect()
            estimator = cpr.CprEstimator(df)
            bc = f'{1 - estimator.data.bimodality_coefficient():.3g}'

            try:
                analysis = estimator.fit()
            except cpr.NoValidModelError as e:
                logger.info(repr(e))
                analysis = None
                pattern = None if e.max_validity is None else int(e.max_validity)
                r2 = f'{e.max_r2:.3g}' if e.max_r2 else None
            else:
                pattern = int(analysis.validity)
                r2 = f'{analysis.model_dict["r2"]:.3g}'

            stem = (
                f'{model["energy"]} {pattern=} {r2=!s} {bc=!s} {year=} {model["iid"]}'
            )

            fig = self.plot(df)
            fig.savefig(self.conf.dirs.cpm_fallback / f'{stem}.png')

            if analysis is not None:
                fig = Figure()
                ax = fig.add_subplot()
                analysis.plot(ax=ax, style={'scatter': {'alpha': self.alpha}})
                _set_ax(ax)
                fig.savefig(self.conf.dirs.cpm_fallback / f'{stem} CPM.png')


@app.default
def scatter(conf: Config):
    """CPM 분석에 실패한 케이스 시각화."""
    energy: list[Energy] = ['raw', 'compensated']
    for e in energy:
        Scatter(energy=e, conf=conf)()


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplTheme(fig_size=(16, 12)).grid().apply()
    app()

"""CPR 예시 그래프."""

from __future__ import annotations

import dataclasses as dc
import functools
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cyclopts
import matplotlib.pyplot as plt
import msgspec
import numpy as np
import polars as pl
from cmap import Colormap
from matplotlib import patches

import scripts.cpr as cli
from greenbutton import cpr, utils

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dc.dataclass
class Dataset:
    base: float = 0.42  # 기저부하
    t_h: float = 8.4  # 난방 시작 온도
    t_c: float = 15.2  # 냉방 시작 온도
    beta_h: float = 0.032  # 난방 민감도
    beta_c: float = 0.03  # 냉방 민감도

    temp_range: tuple[float, float] = (-12, 30)

    hc: Literal['h', 'c', 'hc'] = 'hc'
    n: int = 200
    noise: float = 0.05
    seed: int = 42

    temperature: np.ndarray = dc.field(init=False)
    heating: np.ndarray = dc.field(init=False)
    cooling: np.ndarray = dc.field(init=False)
    energy: np.ndarray = dc.field(init=False)
    datetime: np.ndarray = dc.field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)

        self.base = np.round(self.base, 2)
        self.temperature = rng.uniform(*self.temp_range, size=self.n)

        self.t_h = np.round(self.t_h, 1) if 'h' in self.hc else -np.inf
        self.t_c = np.round(self.t_c, 1) if 'c' in self.hc else np.inf

        zeros = np.zeros_like(self.temperature)
        self.heating = np.maximum(zeros, self.t_h - self.temperature) * self.beta_h
        self.cooling = np.maximum(zeros, self.temperature - self.t_c) * self.beta_c
        self.energy = self.base + self.heating + self.cooling

        if self.noise:
            self.energy += rng.normal(0, scale=self.noise, size=self.energy.size)

        dt = np.datetime64('2000-01-01') + rng.integers(0, 1000, size=self.energy.size)
        self.datetime = np.datetime_as_string(dt)

    def dataframe(self):
        return pl.DataFrame({
            'temperature': self.temperature,
            'energy': self.energy,
        })


app = cyclopts.App(help_on_error=True)


@cyclopts.Parameter(name='*')
@dc.dataclass
class MplExample:
    path: Path | None = None

    _: dc.KW_ONLY

    scale: float = 1.0
    figsize: tuple[float, float] = (16, 9)
    alpha: float = 0.25
    lang: Literal['kor', 'eng'] = 'eng'

    style: cpr.PlotStyle = dc.field(init=False)

    def __post_init__(self):
        self.style = {
            'scatter': {'alpha': 0.25, 'color': 'gray', 's': 12},
            'line': {'color': 'slategray', 'lw': 2, 'alpha': 0.8},
            'axvline': {'ls': '--', 'color': '#B0B0B0', 'alpha': 0.5, 'lw': 1.5},
        }

    @functools.cached_property
    def colors(self):
        # baseline, heating, cooling
        return Colormap('tol:light')([5, 6, 0])

    def output(self):
        if self.path is None:
            return None

        if self.path.is_dir():
            name = (
                f'CPM scale={self.scale} figsize={self.figsize} '
                f'alpha={self.alpha} lang={self.lang}'
            )
            return self.path / f'{name}.png'

        return self.path

    def draw_patch(self, ax: Axes, model: cpr.CprAnalysis, dataset: Dataset):
        baseline = model.model_dict['coef'][0]
        segments = model.segments(*dataset.temp_range)
        xx = segments['temperature'].to_numpy()
        y1, *_, y2 = segments['Ep'].to_list()

        if not self.alpha:
            style = self.style.get('axvline')
            assert style is not None
            ax.axhline(y=baseline, **style)
        else:
            ax.axhspan(ymin=0, ymax=baseline, color=self.colors[0], alpha=self.alpha)
            heating = patches.Polygon(
                [[xx[0], baseline], [xx[1], baseline], [xx[0], y1]],
                color=self.colors[1],
                alpha=self.alpha,
            )
            cooling = patches.Polygon(
                [[xx[2], baseline], [xx[3], baseline], [xx[3], y2]],
                color=self.colors[2],
                alpha=self.alpha,
            )

            ax.add_patch(heating)
            ax.add_patch(cooling)

        ax.set_ylim(0)
        ax.set_xlim(xx[0], xx[3])

    def __call__(self):
        (
            utils.mpl.MplTheme(self.scale, fig_size=self.figsize)
            .grid(show=False)
            .tick(direction='in')
            .apply()
        )
        dataset = Dataset()
        model = cpr.CprEstimator(dataset.dataframe()).fit(operation='hc')

        fig, ax = plt.subplots()
        model.plot(ax=ax, style=self.style)

        xy = (
            ('평균 외기 온도', '에너지 사용량')
            if self.lang == 'kor'
            else ('Average External Temperature', 'Energy Usage')
        )

        ax.set_xlabel(f'{xy[0]} [℃]')
        ax.set_ylabel(f'{xy[1]} [kWh/m²]')

        self.draw_patch(ax=ax, model=model, dataset=dataset)

        if output := self.output():
            fig.savefig(output)
        else:
            plt.show()


@app.command
def mpl(cmd: MplExample | None = None):
    cmd = cmd or MplExample()
    cmd()


@app.command
def plotly(path: str, *, operation: Literal['h', 'c', 'hc'] = 'hc'):
    dataset = Dataset(hc=operation)

    output = cli.analyze(
        data=msgspec.json.encode({
            'observations': {
                'temperature': dataset.temperature.tolist(),
                'energy': dataset.energy.tolist(),
                'datetime': dataset.datetime.tolist(),
            },
            'search_range': {'delta': 1},
        }).decode(),
        plot='html',
        mode='return',
    )

    assert output is not None
    assert output.plot is not None

    Path(path).write_text(output.plot, 'utf-8')


if __name__ == '__main__':
    app()

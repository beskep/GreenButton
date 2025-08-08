"""
2025-07-14.

SQI에서 제공한 기관-ASOS 관측소 매칭 결과로 CPM 재분석.
"""

from __future__ import annotations

import dataclasses as dc
import functools
import warnings
from collections.abc import Sequence  # noqa: TC003
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rich
import seaborn as sns
from loguru import logger
from matplotlib.figure import Figure
from scipy import optimize

from greenbutton import cpr, misc, utils
from greenbutton.utils.cli import App
from greenbutton.utils.terminal import Progress
from scripts.ami.public_institution.config import Config  # noqa: TC001

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class NotEnoughDataError(ValueError):
    pass


class _SkipError(ValueError):
    pass


class Var:
    DATETIME = 'datetime'
    IS_HOLIDAY = 'is_holiday'
    TEMPERATURE = 'temperature'
    ENERGY = 'energy'

    DT = DATETIME
    T = TEMPERATURE
    E = ENERGY

    IID = '기관ID'
    CATEGORY = '기관대분류'
    NAME = '기관명'
    USE = '건물용도'
    GFA = '연면적'

    WEATHER_STATION = '기상관측지점'


@dc.dataclass
class _Institution:
    iid: str
    name: str
    gfa: float
    station: int

    VARS: ClassVar[tuple[str, ...]] = (Var.IID, Var.NAME, Var.GFA, Var.WEATHER_STATION)


@dc.dataclass
class _Dataset:
    conf: Config
    energy: Literal['사용량', '보정사용량'] = '사용량'

    min_energy: float = 0.01  # kWh/m²
    min_gfa: float = 200

    @functools.cached_property
    def _root(self):
        parts = self.conf.root.parts
        return Path(*parts[: parts.index('AMI')])

    @functools.cached_property
    def institutions(self):
        inst = (
            pl.scan_parquet(self.conf.dirs.data / self.conf.files.institution)
            .filter(pl.col(Var.GFA) >= self.min_gfa)
            .with_columns(
                pl.when(pl.col(Var.CATEGORY).str.starts_with('국립대학병원'))
                .then(pl.lit('국립대학병원 등'))
                .otherwise(pl.col(Var.CATEGORY))
                .alias(Var.CATEGORY),
            )
            .drop('asos_code')
        )

        station = (
            pl.scan_parquet(self._root / 'WeatherStation.parquet')
            .filter(pl.col('구분') == '공공기관')
            .rename({
                '건물명': Var.NAME,
                '기상관측지점': Var.WEATHER_STATION,
            })
            .select(Var.NAME, Var.WEATHER_STATION)
        )

        return inst.join(station, on=Var.NAME, how='inner').collect()

    def iter_institution(self):
        for row in self.institutions.select(_Institution.VARS).iter_rows():
            yield _Institution(*row)

    @functools.cached_property
    def temperature(self):
        return (
            pl.scan_parquet(self._root / 'weather/weather.parquet')
            .select(
                pl.col('stnId').alias(Var.WEATHER_STATION),
                pl.col('tm').alias(Var.DT),
                pl.col('ta').alias(Var.T),
            )
            .with_columns()
        )

    @functools.cached_property
    def ami(self):
        src = list(self.conf.dirs.data.glob('AMI*.parquet'))
        return (
            pl.scan_parquet(src)
            .select(Var.DT, Var.IID, pl.col(self.energy).alias(Var.E))
            .with_columns()
        )

    def data(
        self,
        institution: _Institution,
        *,
        interval: str = '1d',
        workday_status: bool = True,
    ):
        ami = (
            self.ami.filter(pl.col(Var.IID) == institution.iid)
            .sort(Var.DT)
            .group_by_dynamic(Var.DT, every=interval)
            .agg(pl.sum(Var.ENERGY) / institution.gfa)
            .filter(pl.col(Var.ENERGY) >= self.min_energy)
            .collect()
        )
        temperature = (
            self.temperature.filter(pl.col(Var.WEATHER_STATION) == institution.station)
            .sort(Var.DT)
            .collect()
            .upsample(Var.DT, every=interval)
            .group_by_dynamic(Var.DT, every=interval)
            .agg(pl.mean(Var.T))
        )
        data = ami.join(temperature, on=Var.DT, how='inner')

        if not data.height:
            raise NotEnoughDataError(institution)

        if workday_status:
            years = (
                data.select(pl.col(Var.DT).dt.year().unique().sort())
                .to_series()
                .to_list()
            )
            data = data.with_columns(
                misc.is_holiday(pl.col(Var.DT), years=years).alias(Var.IS_HOLIDAY)
            )

        return data


@cyclopts.Parameter(name='*')
@dc.dataclass
class _Calculator:
    conf: Config

    plot: bool = True

    delta: float = 0.5
    day_type: Literal['workday', 'holiday'] | None = 'workday'
    finish: bool = False

    min_samples: int = 12
    skip_already_calculated: bool = True

    @functools.cached_property
    def output(self):
        finish = f'{"" if self.finish else "no-"}finish'
        return self.conf.dirs.cpm / f'delta{self.delta}-{finish}'

    def init_output_dir(self):
        self.output.mkdir(exist_ok=True)

        (self.output / 'model').mkdir(exist_ok=True)
        if self.plot:
            (self.output / 'plot').mkdir(exist_ok=True)

    def file_name(self, institution: _Institution):
        day = self.day_type or 'all'
        return f'{day}-{institution.iid}-{institution.name}'

    @functools.cached_property
    def dataset(self):
        return _Dataset(self.conf)

    @functools.cached_property
    def search_ranges(self):
        return (
            cpr.AbsoluteSearchRange(0, 20, self.delta),
            cpr.AbsoluteSearchRange(5, 25, self.delta),
        )

    def _institution(self, institution: str | _Institution):
        if isinstance(institution, _Institution):
            return institution

        row = (
            self.dataset.institutions.filter(pl.col(Var.IID) == institution)
            .select(_Institution.VARS)
            .row(0)
        )
        return _Institution(*row)

    def cpm(self, institution: str | _Institution):
        inst = self._institution(institution)
        data = self.dataset.data(inst)

        if data.height < self.min_samples:
            raise NotEnoughDataError(institution)

        if (
            is_holiday := {'workday': False, 'holiday': True}.get(self.day_type)  # type: ignore[arg-type]
        ) is not None:
            data = data.filter(pl.col(Var.IS_HOLIDAY) == is_holiday)

        estimator = cpr.CprEstimator(data, x=Var.T, y=Var.E, datetime=Var.DT)
        return estimator.fit(
            *self.search_ranges,
            brute_finish=optimize.fmin if self.finish else None,
            round_cp=not self.finish,
        )

    @staticmethod
    def _plot(model: cpr.CprAnalysis, ax: Axes | None = None):
        if ax is None:
            ax = plt.gca()

        model.plot(ax=ax)
        if ax.dataLim.y0 > 0:
            ax.set_ylim(bottom=0)

        ax.set_xlabel('기온 [℃]')
        ax.set_ylabel('전력사용량 [kWh/m²]')
        ax.text(
            0.02,
            0.98,
            f'r²={model.model_dict["r2"]:.4f}',
            transform=ax.transAxes,
            va='top',
            weight=500,
        )

        return ax

    def cpm_and_write(self, institution: str | _Institution):
        inst = self._institution(institution)
        fn = self.file_name(inst)
        path = self.output / f'model/{fn}.parquet'

        if (
            self.skip_already_calculated  # fmt
            and (path.exists() or path.with_suffix('.failed').exists())
        ):
            raise _SkipError

        try:
            model = self.cpm(institution)
        except cpr.CprError:
            path.with_suffix('.failed').touch()
            raise

        frame = model.model_frame.select(
            pl.lit(inst.iid).alias('id'),
            pl.lit(inst.name).alias('name'),
            pl.lit(self.day_type).alias('day_type'),
            pl.all(),
        )

        frame.write_parquet(self.output / f'model/{fn}.parquet')

        if self.plot:
            fig = Figure()
            ax = fig.subplots()
            self._plot(model, ax=ax)
            fig.savefig(self.output / f'plot/{fn}.png')

    def concat_models(self):
        data = pl.read_parquet(self.output / 'model/*.parquet')
        data.write_parquet(self.output / 'models.parquet')
        return data

    def batch_cpm(self):
        self.init_output_dir()

        for inst in Progress.iter(
            self.dataset.iter_institution(),
            total=self.dataset.institutions.height,
        ):
            try:
                self.cpm_and_write(inst)
            except _SkipError:
                pass
            except (cpr.CprError, NotEnoughDataError) as e:
                logger.warning('{}: {}', repr(e), inst.name)
            else:
                logger.info(inst)

        return self.concat_models()


app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml',
        root_keys='public_institution',
        use_commands_as_keys=False,
    )
)


@app.command
def check_stations(conf: Config):
    """공공기관 AMI 목록과 기관-관측소 파일의 이름 매칭 여부 판단."""
    institution = (
        pl.scan_parquet(conf.dirs.data / conf.files.institution)
        .select(
            pl.col('기관ID').alias('id'),
            pl.col('기관명').alias('institution'),
        )
        .collect()
    )

    parts = conf.root.parts
    root = Path(*parts[: parts.index('AMI')])

    station = (
        pl.scan_parquet(root / 'WeatherStation.parquet')
        .filter(pl.col('구분') == '공공기관')
        .rename({'건물명': 'institution', '기상관측지점': 'station'})
        .select('institution', 'station')
        .collect()
    )

    data = institution.join(station, on='institution', how='outer')

    console = rich.get_console()
    console.print(data)
    console.print(data.describe())


@app.command
def institution(conf: Config):
    """전처리 거친 기관/기상관측지점 정보 저장."""
    variables = {
        Var.IID: 'id',
        Var.CATEGORY: 'category',
        Var.USE: 'use',
        Var.GFA: 'gfa',
        Var.WEATHER_STATION: 'weather-station',
    }
    institution = (
        _Dataset(conf).institutions.rename(variables).select(list(variables.values()))
    )
    institution.write_parquet(conf.dirs.cpm / 'institution.parquet')
    institution.write_excel(conf.dirs.cpm / 'institution.xlsx')


@app.command
def cpm(iid: str = 'DB_B7AE8782-40AF-8EED-E050-007F01001D51', *, conf: Config):
    calculator = _Calculator(conf)
    calculator.init_output_dir()
    calculator.cpm_and_write(iid)


@app.command
def cpm_batch(calculator: _Calculator):
    models = calculator.batch_cpm()
    rich.print(models)


@cyclopts.Parameter(name='*')
@dc.dataclass
class _Eda:
    conf: Config

    percentiles: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

    @functools.cached_property
    def data(self):
        models = pl.read_parquet(self.conf.dirs.cpm / 'models.parquet')
        institution = _Dataset(self.conf).institutions

        models = (
            institution.select(Var.IID, Var.CATEGORY)
            .filter(pl.col(Var.CATEGORY).str.starts_with('국립대학').not_())
            .rename({Var.IID: 'id'})
            .join(models, on='id', how='left')
        )
        param_type = (
            models.drop_nulls('names')
            .filter(pl.col('names') != 'Intercept')
            .group_by('id')
            .agg(pl.col('names').len())
            .with_columns(
                pl.col('names').replace_strict({1: '3P', 2: '5P'}).alias('param_type')
            )
            .drop('names')
        )

        return param_type.join(models, on='id', how='full', coalesce=True)

    def r2_ecdf(self):
        fig = Figure()
        ax = fig.subplots()
        sns.ecdfplot(
            self.data.unique(['id', 'param_type']),
            x='r2',
            hue='param_type',
            stat='count',
            complementary=True,
            ax=ax,
        )
        ax.set_xlim(0, 1)

        if legend := ax.get_legend():
            legend.set_title('')

        fig.savefig(self.conf.dirs.cpm / 'r2.png')

    def change_points(self, min_r2: float | None = None):
        data = (
            self.data.select('id', 'names', 'change_points', 'r2')
            .drop_nulls()
            .rename({'names': 'variable', 'change_points': 'value'})
            .with_columns(pl.col('variable').replace_strict({'HDD': 'Th', 'CDD': 'Tc'}))
        )

        if min_r2 is not None:
            data = data.filter(pl.col('r2') >= min_r2)

        wide = (
            data.pivot('variable', index=['id', 'r2'], values='value')
            .with_columns((pl.col('Tc') - pl.col('Th')).alias('delta'))
            .drop_nulls()
        )

        arr = wide.select('Th', 'Tc').to_numpy()
        rich.print(
            f'mean: {np.mean(arr, axis=0)}',
            f'std: {np.std(arr, axis=0)}',
            'covariance matrix:',
            f'{np.cov(arr, rowvar=False)}',
            sep='\n',
        )

        suffix = '' if min_r2 is None else f'-r2 over {min_r2}'
        utils.pl.PolarsSummary(
            data.drop('id').rename({'variable': 'cp'}),
            group='cp',
            percentiles=self.percentiles,
        ).write_excel(self.conf.dirs.cpm / f'param-change-point{suffix}.xlsx')
        utils.pl.PolarsSummary(
            wide.unpivot(index='id', variable_name='var').drop('id'),
            group='var',
            percentiles=self.percentiles,
        ).write_excel(self.conf.dirs.cpm / f'param-change-point-wide{suffix}.xlsx')

        grid = (
            sns.JointGrid(wide, x='Th', y='Tc')
            .plot_joint(sns.scatterplot, alpha=0.5)
            .plot_marginals(sns.histplot, kde=True)
            .set_axis_labels(
                'Heating Change Point Temperature [°C]',
                'Cooling Change Point Temperature [°C]',
            )
        )
        utils.mpl.equal_scale(grid.ax_joint)
        grid.savefig(self.conf.dirs.cpm / f'param-change-point-joint{suffix}.png')
        plt.close(grid.figure)


@app.command
def eda(eda: _Eda):
    warnings.filterwarnings('ignore', message='The figure layout has changed to tight')

    eda.r2_ecdf()
    eda.change_points(min_r2=None)
    eda.change_points(min_r2=0.5)
    eda.change_points(min_r2=0.6)
    eda.change_points(min_r2=0.8)


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplTheme().grid().apply()
    app()

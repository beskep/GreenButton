"""산기평 CPR 분석, 에너지 효율 진단."""

from __future__ import annotations

import dataclasses as dc
import functools
import itertools
from collections.abc import Sequence
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import cyclopts
import matplotlib.pyplot as plt
import pingouin as pg
import pint
import polars as pl
import pydash
import rich
import seaborn as sns
from cmap import Colormap
from loguru import logger
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D

import scripts.ami.public_institution.s02_01cpr as pc
import scripts.exp.experiment as exp
from greenbutton import cpr, misc, utils
from greenbutton.utils.cli import App
from greenbutton.utils.terminal import Progress

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.typing import ColorType
    from polars._typing import FrameType


Energy = Literal['heat', 'net_elec', 'total_elec', 'net', 'total']
SummedEnergy = Literal['net', 'total']


@dc.dataclass
class _PublicInstitutionCpr:
    min_r2: float = 0.4
    max_anomaly_threshold: float = 4
    group: Literal['office', 'institution', 'public', 'none'] = 'office'

    models_source: str = 'PublicInstitutionCPR.parquet'

    _dirs: exp.Dirs = dc.field(init=False)

    def scan_models(self):
        src = self._dirs.database / self.models_source
        lf = pl.scan_parquet(src).filter(pl.col('r2') >= self.min_r2)

        match self.group:
            case 'none':
                # 전체 데이터 비교
                pass
            case 'office':
                # 용도 (업무시설) 필터
                institution = (
                    pl.scan_parquet(src.parent / 'PublicInstitution.parquet')
                    .rename({'기관ID': 'id', '건물용도': 'use'})
                    .select('id', 'use')
                )
                lf = (
                    (lf)
                    .join(institution, on='id', how='left')
                    .filter(pl.col('use') == '업무시설')
                )
            case 'institution':
                # 공공기관 대분류 (대학, 대학병원 등 제외) 필터
                lf = lf.filter(
                    pl.col('category')
                    .is_in(['국립대학병원 등', '국립대학 및 공립대학'])
                    .not_()
                )
            case 'public':
                # 공공기관 대분류 중 '공공기관'만 선택 (시장형, 시장형 이외 모두 포함)
                lf = lf.filter(pl.col('category').str.starts_with('공공기관'))
            case _:
                raise ValueError(self.categories)

        return lf


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'keit'

    pi_cpr: _PublicInstitutionCpr = dc.field(default_factory=_PublicInstitutionCpr)

    def __post_init__(self):
        self.pi_cpr._dirs = self.dirs  # noqa: SLF001


app = App(
    config=cyclopts.config.Toml('config/.experiment.toml', use_commands_as_keys=False),
)
app.command(App('weather', help='기온 데이터 처리'))
app.command(App('cpr', help='CPR 분석'))
app.command(App('report', help='보고서용 데이터·그래프 출력'))


@app['weather'].command
def weather_prep(*, conf: Config):
    """기온 전처리."""
    # 신암(860)은 2016년 1월부터 데이터 존재
    # AWS 경산(827)이 대상과 더 가까우나, 2016년 2월부터 데이터 존재
    src = max(conf.dirs.root.glob('OBS_AWS*.csv'), key=lambda x: x.stat().st_birthtime)
    logger.info('source={}', src)

    data = (
        pl.read_csv(src, encoding='korean')
        .filter(pl.col('지점명') == '경산')
        .rename({
            '일시': 'date',
            '평균기온(°C)': 'temperature',
            '최저기온(°C)': 'min_temp',
            '최고기온(°C)': 'max_temp',
        })
        .select(pl.col('date').str.to_date(), 'temperature', 'min_temp', 'max_temp')
    )
    rich.print(data)

    data.write_parquet(conf.dirs.database / '0000.temperature.parquet')


@app['weather'].command
def weather_plot(
    *,
    max_year: int = 2024,
    hue: bool = False,
    conf: Config,
):
    """기온 시각화."""
    date = pl.col('date')
    data = (
        pl.scan_parquet(conf.dirs.database / '0000.temperature.parquet')
        .select(
            date,
            date.dt.year().alias('year'),
            pl.date(2000, date.dt.month(), date.dt.day()).alias('dummy'),
            'temperature',
        )
        .filter(pl.col('year') <= max_year)
        .collect()
    )

    if hue:
        utils.mpl.MplTheme(palette='crest').tick(which='both').grid().apply()

    fig, ax = plt.subplots()
    sns.scatterplot(
        data,
        x='dummy' if hue else 'date',
        y='temperature',
        hue='year' if hue else None,
        ax=ax,
        alpha=0.4,
        s=10,
        color='tab:blue',
    )
    ax.set_xlabel('')
    ax.set_ylabel('일평균 외기온 [°C]')

    suffix = '-hue' if hue else ''
    fig.savefig(conf.dirs.analysis / f'0100.temperature{suffix}.png')


@dc.dataclass
class _CprEnergyType:
    typ: str
    name: str

    @classmethod
    def create(cls, e: Energy):
        if 'heat' in e:
            t = '열'
        elif 'elec' in e:
            t = '전력'
        else:
            t = '합계'

        n = {
            'heat': '열에너지',
            'net_elec': '순전력',
            'total_elec': '총전력',
            'net': '순에너지',
            'total': '총에너지',
        }[e]

        return cls(t, n)

    def __str__(self):
        return f'{self.typ}({self.name})'


@dc.dataclass
class _KeitCpr:
    conf: Config

    energy_unit: str = 'kWh'
    energy_types: tuple[Energy, ...] = (
        'heat',
        'net_elec',
        'total_elec',
        'net',
        'total',
    )
    eui: bool = True

    remove_outlier: bool = True

    sources: dict[str, str] = dc.field(
        default_factory=lambda: {
            'temperature': 'temperature',
            'heat': '*적산열량계',
            'electricity': '전력',
            'ami': 'AMI',
        }
    )

    ur: pint.UnitRegistry = dc.field(default_factory=pint.UnitRegistry)

    data: pl.DataFrame = dc.field(init=False)
    years: tuple[int, ...] = dc.field(init=False)
    xlim: tuple[float, float] = dc.field(init=False)

    AREA: ClassVar[float] = 12614.940  # m²

    @functools.cached_property
    def y_unit(self):
        return f'{self.energy_unit}/m²' if self.eui else self.energy_unit

    def _scan(self, src: str, **kwargs):
        return pl.scan_parquet(
            self.conf.dirs.database / f'0000.{self.sources[src]}.parquet',
            **kwargs,
        )

    def _conversion_factor(self, src: str):
        return float(self.ur.Quantity(1, src).to(self.energy_unit).m)

    def __post_init__(self):
        temp = self._scan('temperature').select('date', 'temperature').collect()
        assert temp['date'].is_unique().all()

        heat = (
            self._scan('heat', glob=True, include_file_paths='source')
            .select(
                'date',
                pl.col('source').str.extract(r'([냉난]방)').alias('energy'),
                pl.col('사용량-Gcal')
                .mul(self._conversion_factor('Gcal'))
                .alias('value'),
            )
            .collect()
        )
        elec = (
            self._scan('electricity')
            .rename({'사용량': '전력순사용량', '태양광발전량': '발전량'})
            .unpivot(['전력순사용량', '발전량'], index='date', variable_name='energy')
            .with_columns(pl.col('value') * self._conversion_factor('kWh'))
            .collect()
        )

        data = (
            pl.concat([heat, elec], how='diagonal')
            .join(temp, on='date', how='left')
            # NOTE 해당일 이전 지역난방 Gcal 단위 데이터 없음
            .filter(pl.col('date') > pl.date(2016, 8, 10))
        )
        assert data.select('date', 'energy').is_unique().all()

        if self.eui:
            data = data.with_columns(pl.col('value') / self.AREA)

        if self.remove_outlier:
            # NOTE 수동 이상치 제외 - 냉방 사용량 과도
            data = data.filter(pl.col('date') != pl.date(2024, 8, 6))

        self.years = tuple(
            data.select(pl.col('date').dt.year().unique().sort()).to_series()
        )
        self.data = data.with_columns(
            misc.is_holiday(pl.col('date'), years=self.years).alias('is_holiday')
        )

        t = self.data['temperature'].drop_nulls().to_numpy()
        self.xlim = (float(t.min()), float(t.max()))

    def cpr_data(
        self,
        *,
        energy: Energy,
        year: int | Sequence[int] | None,
        holiday: bool | None = None,
        agg: bool = True,
    ):
        data = self.data.drop_nulls(['temperature', 'value'])

        if holiday is not None:
            data = self.data.filter(pl.col('is_holiday') == holiday)

        match year:
            case None:
                pass
            case int():
                data = data.filter(pl.col('date').dt.year() == year)
            case Sequence():
                data = data.filter(pl.col('date').dt.year().is_in(year))
            case _:
                raise ValueError(year)

        energies = {
            'heat': ['난방', '냉방'],
            'net_elec': ['전력순사용량'],
            'total_elec': ['전력순사용량', '발전량'],
            'net': ['난방', '냉방', '전력순사용량'],
            'total': None,
        }
        if (e := energies[energy]) is not None:
            data = data.filter(pl.col('energy').is_in(e))

        if agg:
            data = data.group_by(['date', 'temperature']).agg(pl.sum('value'))

        return data

    def cpr_estimator(
        self,
        *,
        energy: Energy,
        year: int | Sequence[int] | None,
        holiday: bool,
    ):
        data = self.cpr_data(year=year, energy=energy, holiday=holiday)
        return cpr.CprEstimator(
            data.sample(fraction=1, shuffle=True),
            x='temperature',
            y='value',
            datetime='date',
        )

    def cpr(
        self,
        *,
        output: Path,
        energy: Energy,
        year: int | Sequence[int] | None,
        holiday: bool,
        title: bool = True,
    ):
        estimator = self.cpr_estimator(year=year, energy=energy, holiday=holiday)

        try:
            model = estimator.fit()
        except cpr.NoValidModelError as err:
            logger.warning(f'{holiday=}|{year=}|{energy=}|{err!r}')
            model = None

        fig, ax = plt.subplots()

        if model is None:
            sns.scatterplot(
                estimator.data.dataframe, x='temperature', y='energy', ax=ax, alpha=0.5
            )
            text = {'s': '분석 실패', 'color': 'tab:red'}
        else:
            model.plot(ax=ax, style={'scatter': {'alpha': 0.25, 's': 12}})
            text = {'s': f'r²={model.model_dict["r2"]:.4f}'}

        ylim1 = (
            self.cpr_data(year=None, energy=energy).select(pl.col('value').max()).item()
        )
        ax.dataLim.update_from_data_y([ylim1], ignore=False)
        ax.dataLim.update_from_data_x(self.xlim)
        ax.autoscale_view()
        ax.set_ylim(0)

        ax.text(0.02, 0.98, va='top', weight=500, transform=ax.transAxes, **text)  # type: ignore[arg-type]

        ax.set_xlabel('일간 평균 외기온 [°C]')
        ax.set_ylabel(f'일간 에너지 사용량 [{self.y_unit}]')

        h = '휴일' if holiday else '평일'
        y = '전체 기간' if year is None else f'{year}년'
        et = _CprEnergyType.create(energy)
        s = '(분석실패)' if model is None else ''

        if title:
            ax.set_title(f'{y} {h} {et.name} 사용량 분석 결과', loc='left', weight=500)
        else:
            s = f'{s}(NoTitle)'

        fig.savefig(output / f'{h}_energy={et}_year={year or "전기간"}{s}.png')
        plt.close(fig)

        return None if model is None else model.model_frame

    def _iter_cpr(self, output: Path):
        for holiday, year, energy in Progress.iter(
            itertools.product(
                [False, True],
                [None, *self.years],
                self.energy_types,
            ),
            total=2 * (len(self.years) + 1) * len(self.energy_types),
        ):
            try:
                model = self.cpr(
                    output=output,
                    holiday=holiday,
                    year=year,
                    energy=energy,
                )
            except cpr.CprError as e:
                logger.error(f'{holiday=}|{year=}|{energy=}|{e!r}')
                continue

            if model is not None:
                yield model.select(
                    pl.lit(holiday).alias('holiday'),
                    pl.lit(str(year or 'all')).alias('year'),
                    pl.lit(energy).alias('energy'),
                    pl.all(),
                )

    def run(self, *, write_excel: bool = True):
        output = self.conf.dirs.analysis / 'CPR'
        output.mkdir(exist_ok=True)

        models = pl.concat(list(self._iter_cpr(output=output)))
        models.write_parquet(self.conf.dirs.analysis / '0100.CPR.parquet')

        if write_excel:
            models.write_excel(
                self.conf.dirs.analysis / '0100.CPR.xlsx', column_widths=150
            )

    def validate(
        self,
        energy: SummedEnergy = 'total',
        year: int | None = None,
        *,
        holiday: bool = False,
    ):
        raw = self.cpr_data(year=year, energy=energy, holiday=holiday, agg=False)
        agg = (
            raw.group_by(['date', 'temperature'])
            .agg(pl.sum('value'))
            .rename({'value': 'energy', 'date': 'datetime'})
        )

        ytrue = raw.select(
            'date',
            'value',
            'energy',
            pl.lit('ytrue').alias('dataset'),
        ).filter(pl.col('energy').is_in(['난방', '냉방']))

        model_frame = (
            pl.scan_parquet(self.conf.dirs.analysis / '0100.CPR.parquet')
            .filter(
                pl.col('energy') == energy,
                pl.col('year') == str(year or 'all'),
                pl.col('holiday') == holiday,
            )
            .collect()
        )
        ypred = (
            cpr.CprModel.from_dataframe(model_frame)
            .predict(agg)
            .rename({'datetime': 'date'})
            .unpivot(['Eph', 'Epc'], index='date', variable_name='energy')
            .with_columns(
                pl.lit('ypred').alias('dataset'),
                pl.col('energy').replace_strict({'Eph': '난방', 'Epc': '냉방'}),
            )
        )

        data = (
            pl.concat([ytrue, ypred], how='diagonal')
            .drop_nulls('value')
            .pivot('dataset', index=['date', 'energy'], values='value')
            .sort(pl.all())
        )
        stat = data.group_by('energy').agg(
            (pl.col('ypred') - pl.col('ytrue'))
            .pow(2)
            .mean()
            .sqrt()
            .truediv(pl.mean('ytrue'))
            .alias('cvrmse'),
            pl.corr('ypred', 'ytrue').alias('corr'),
        )

        fig, axes = plt.subplots(1, 2)
        palette = Colormap('tol:bright-alt')([1, 0])

        ax: Axes
        for e, ax, color in zip(['난방', '냉방'], axes, palette, strict=True):
            sns.scatterplot(
                data.filter(pl.col('energy') == e),
                x='ytrue',
                y='ypred',
                ax=ax,
                color=color,
                alpha=0.2,
            )
            ax.set_aspect(1)
            ax.dataLim.update_from_data_xy(
                ax.dataLim.get_points()[:, ::-1], ignore=False
            )
            ax.autoscale_view()
            ax.axline((0, 0), slope=1, c='k', alpha=0.1, lw=0.5)

            loc = ticker.MaxNLocator(nbins=5)
            ax.xaxis.set_major_locator(loc)
            ax.yaxis.set_major_locator(loc)

            ax.set_xlabel(f'실제 사용량 [{self.y_unit}]')
            ax.set_ylabel(f'예측 사용량 [{self.y_unit}]')

            _, cvrmse, corr = stat.row(by_predicate=pl.col('energy') == e)
            if not cvrmse:
                continue

            ax.text(
                0.02,
                0.98,
                f'CV(RMSE) = {cvrmse:.4f}\nr = {corr:.4f}',
                va='top',
                weight=500,
                fontsize='small',
                transform=ax.transAxes,
            )

            ax.set_title(f'{e} 사용량', loc='left', weight=500)

        return data, fig

    def plot_by_year_grid(
        self,
        energy: Energy = 'total',
        min_year: int | None = None,
        max_year: int = 2024,
    ):
        years = (
            self.data.select(pl.col('date').dt.year().unique().sort())
            .filter(pl.col('date') >= min_year, pl.col('date') <= max_year)
            .to_series()
        )

        fig, axes = plt.subplots(len(years), 1, sharex=True, sharey=True)

        ax: Axes
        for year, ax in zip(years, axes, strict=True):
            logger.info(f'{year=}')
            model = self.cpr_estimator(year=year, energy=energy, holiday=False).fit()
            model.plot(
                ax=ax,
                style={
                    'datetime_hue': False,
                    'scatter': {'s': 10, 'c': 'tab:blue', 'alpha': 0.32},
                },
            )
            r2 = model.model_dict['r2']

            ax.set_xlabel('일간 평균 외기온 [°C]')
            ax.set_ylabel('')
            ax.text(
                0.01,
                0.94,
                f'{year}년 (r²={r2:.4f})',
                weight=500,
                va='top',
                transform=ax.transAxes,
                alpha=0.9,
            )

        fig.supylabel('일간 에너지 사용량 [kWh/m²]', fontsize='medium')
        return fig


@app['cpr'].command
def cpr_(
    *,
    run: bool = True,
    write_data: bool = True,
    write_excel: bool = True,
    conf: Config,
):
    """CPR 분석 실행."""
    utils.mpl.MplTheme(0.8).grid().apply()

    cpr = _KeitCpr(conf=conf)

    if write_data:
        path = conf.dirs.analysis / '0100.CPR data.parquet'
        cpr.data.write_parquet(path)
        cpr.data.write_excel(path.with_suffix('.xlsx'), column_widths=150)

    if run:
        _KeitCpr(conf=conf).run(write_excel=write_excel)


@app['cpr'].command
def cpr_validate(*, conf: Config):
    """CPR 예측 냉난방 사용량 vs 실제 열에너지 사용량 비교."""
    output = conf.dirs.analysis / 'CPR-validation'
    output.mkdir(exist_ok=True)

    utils.mpl.MplTheme('paper').grid(lw=0.75, alpha=0.5).apply()

    cpr = _KeitCpr(conf=conf)

    years = [
        None,
        *cpr.data.select(pl.col('date').dt.year().unique().sort()).to_series(),
    ]
    energies: list[SummedEnergy] = ['total', 'net']

    for energy, year in itertools.product(energies, years):
        logger.info(f'{energy=} | {year=}')
        _, fig = cpr.validate(energy=energy, year=year, holiday=False)
        fig.savefig(output / f'{energy=}_{year=}_holiday=False.png'.replace("'", ''))
        plt.close(fig)


@app['cpr'].command
def cpr_energy_source(*, conf: Config):
    """2025-06-18 AMI vs 자체 기록 전력/총량 CPR 비교."""
    (
        utils.mpl.MplTheme(1.2, fig_size=(16, 12))
        .grid()
        .apply({
            'lines.linewidth': 2,
            'lines.solid_capstyle': 'butt',
        })
    )

    ur = pint.UnitRegistry()
    c = float(ur.Quantity(1, 'Gcal').to('kWh').m)

    lf = (
        pl.scan_parquet(conf.dirs.analysis / '0000.사용량 비교.parquet')
        .filter(pl.col('hue') != '태양광 발전량')
        .with_columns(
            pl.when(pl.col('hue').str.contains('Gcal'))
            .then(pl.lit(c))
            .otherwise(pl.lit(1.0))
            .alias('c'),
        )
        .with_columns(pl.col('value').mul(pl.col('c')).alias('value'))
    )

    ami = (
        (lf)
        .filter(pl.col('variable') == 'AMI')
        .select('date', 'value', pl.lit('AMI').alias('variable'))
    )
    elec = (
        lf.filter(pl.col('variable') == '전력')
        .group_by('date')
        .agg(pl.sum('value'))
        .select('date', 'value', pl.lit('(자체 기록) 전력').alias('variable'))
    )
    total = (
        lf.filter(pl.col('variable') != 'AMI')
        .group_by('date')
        .agg(pl.sum('value'))
        .select('date', 'value', pl.lit('(자체 기록) 전력+열').alias('variable'))
    )

    df = (
        pl.concat([ami, elec, total])
        .collect()
        .pivot('variable', index='date', values='value')
        .drop_nulls()
        .unpivot(index='date')
    )
    (
        df.pivot('variable', index='date', values='value', sort_columns=True)
        .sort('date')
        .write_excel(conf.dirs.analysis / '0110.전력-열 비교.xlsx')
    )

    # line plot
    fig = Figure()
    ax = fig.subplots()
    sns.lineplot(
        df.group_by_dynamic(
            'date', every='1w', group_by='variable', start_by='datapoint'
        )
        .agg(pl.sum('value'))
        .with_columns(),
        x='date',
        y='value',
        hue='variable',
        ax=ax,
        alpha=0.75,
    )
    ax.legend().set_title('')
    ax.set_ylim(0)
    ax.set_xlabel('')
    ax.set_ylabel('에너지 사용량 [kWh]')
    fig.savefig(conf.dirs.analysis / '0111.line.png')

    # CPR
    temp = (
        pl.scan_parquet(conf.dirs.analysis / '0100.CPR data.parquet')
        .select('date', 'temperature', 'is_holiday')
        .unique()
        .collect()
    )
    df = df.join(temp, on='date', how='left').filter(pl.col('is_holiday').not_())
    variables = df['variable'].unique(maintain_order=True)

    fig = Figure()
    ax = fig.subplots()
    colors = sns.color_palette(n_colors=len(variables))
    for v, c in zip(variables, colors, strict=True):
        model = cpr.CprEstimator(
            df.filter(pl.col('variable') == v),
            x='temperature',
            y='value',
            datetime=None,
        ).fit()
        model.plot(
            ax=ax,
            scatter=False,
            style={
                'xmin': 0,
                'xmax': 25,
                'axvline': None,
                'line': {'c': c, 'alpha': 0.75, 'lw': 4},
            },
        )

    ax.set_xlabel('일간 평균 외기온 [°C]')
    ax.set_ylabel('에너지 사용량 [kWh]')
    ax.set_ylim(0)
    ax.legend(
        handles=[Line2D([0], [0], linewidth=4, color=c) for c in colors],
        labels=variables.to_list(),
    )
    fig.savefig(conf.dirs.analysis / '0112.CPR.png')


@app['report'].command
def report_cpr_total(*, conf: Config):
    """전체 에너지 CPR 예시."""
    utils.mpl.MplTheme().grid().apply()

    output = conf.dirs.analysis / 'CPR-example'
    output.mkdir(exist_ok=True)

    cpr = _KeitCpr(conf=conf)

    for holiday in [False, True]:
        if (
            model := cpr.cpr(
                output=output,
                holiday=holiday,
                year=None,
                energy='total',
                title=False,
            )
        ) is not None:
            model.write_excel(output / f'{holiday=}.xlsx')


@app['report'].command
def report_cpr_by_year(
    *,
    min_year: int | None = None,
    max_year: int = 2024,
    conf: Config,
):
    """연도별 총에너지 CPR 모델."""
    utils.mpl.MplTheme(fig_size=(16, 14)).grid().apply()

    cpr = _KeitCpr(conf=conf)
    fig = cpr.plot_by_year_grid(min_year=min_year, max_year=max_year)
    fig.savefig(conf.dirs.analysis / f'0100.CPR-year({min_year}-{max_year}).png')
    plt.close(fig)


@dc.dataclass
class _CprParams:
    conf: Config

    models: pl.DataFrame = dc.field(init=False)
    target: pl.DataFrame = dc.field(init=False)

    energy_rename: ClassVar[dict[Energy, str]] = {
        'total': '총사용량',
        'net': '순사용량',
        'heat': '열에너지',
        'total_elec': '전력',
    }
    units: ClassVar[dict[str, str]] = {
        '난방 균형점 온도': '[°C]',
        '냉방 균형점 온도': '[°C]',
        '기저부하': '[kWh/m²]',
        '난방 민감도': '[kWh/m²°C]',
        '냉방 민감도': '[kWh/m²°C]',
    }
    title_style: ClassVar[dict] = {
        'loc': 'left',
        'weight': 500,
        'fontsize': 'medium',
    }

    def __post_init__(self):
        self.models = self.cpr_params(self.conf.pi_cpr.scan_models()).collect()
        target = self.cpr_params(
            pl.read_parquet(self.conf.dirs.analysis / '0100.CPR.parquet')
        )

        target_id = '__target__'
        percentile = (
            pl.concat(
                [
                    target.filter(
                        pl.col('year') == 'all', pl.col('energy') == 'total'
                    ).with_columns(pl.lit(target_id).alias('id')),
                    self.models,
                ],
                how='diagonal',
            )
            .with_columns(
                (pl.col('value').rank() / pl.count('value'))
                .over('holiday', 'var')
                .alias('percentile')
            )
            .filter(pl.col('id') == target_id)
            .select('holiday', 'var', 'year', 'energy', 'percentile')
            .with_columns(
                pl.when(pl.col('var').str.starts_with('냉방 균형점 온도'))
                .then(1 - pl.col('percentile'))
                .otherwise(pl.col('percentile'))
                .alias('percentile')
            )
        )

        self.target = target.join(
            percentile, on=['holiday', 'var', 'year', 'energy'], how='left'
        )

    @staticmethod
    def cpr_params(model: FrameType):
        index = [
            x
            for x in ['id', 'category', 'holiday', 'names', 'year', 'energy']
            if x in model.collect_schema()
        ]
        return (
            model.rename({'change_points': 'cp'})
            .unpivot(['cp', 'coef'], index=index)
            .drop_nulls('value')
            .with_columns(
                pl.format('{}-{}', 'variable', 'names')
                .replace({
                    'cp-HDD': '난방 균형점 온도',
                    'cp-CDD': '냉방 균형점 온도',
                    'coef-Intercept': '기저부하',
                    'coef-HDD': '난방 민감도',
                    'coef-CDD': '냉방 민감도',
                })
                .alias('var'),
                pl.col('holiday').replace_strict(
                    {True: '휴일', False: '근무일'}, return_dtype=pl.String
                ),
            )
            .with_columns(
                iqr_dist=(
                    (pl.median('value') - pl.col('value'))
                    / (pl.quantile('value', 0.75) - pl.quantile('value', 0.25))
                ).over('var', 'holiday')
            )
        )

    def _filter_anomaly(self, threshold: float | None = None):
        threshold = threshold or self.conf.pi_cpr.max_anomaly_threshold
        iqr_dist = pl.col('iqr_dist').abs()
        pos_var = pl.col('var').str.contains_any(['기저부하', '민감도'])
        return (iqr_dist <= threshold) & ~(pos_var & (pl.col('value') < 0))

    def plot_param_grid(
        self,
        *,
        width: float = 1.5 * 16,
        drop_sensitivity: bool = True,
        holiday: bool = True,
    ):
        colors = list(Colormap('tol:bright-alt')([0, 1]))

        models = self.models.filter(self._filter_anomaly())
        if drop_sensitivity:
            models = models.filter(pl.col('var').str.contains('민감도').not_())
        if not holiday:
            models = models.filter(pl.col('holiday') == '근무일').sort(
                pl.col('var').replace_strict({
                    x: i
                    for i, x in enumerate([
                        '기저부하',
                        '난방 균형점 온도',
                        '냉방 균형점 온도',
                    ])
                })
            )

        grid = (
            sns.FacetGrid(
                models,
                col='holiday' if holiday else 'var',
                row='var' if holiday else None,
                hue='holiday' if holiday else None,
                sharex='row' if holiday else False,
                sharey=False,
                palette=colors,
                despine=False,
            )
            .map_dataframe(sns.histplot, 'value', kde=True)
            .set_titles('')
            .set_titles(
                '{col_name} {row_name} 분포' if holiday else '{col_name} 분포',
                **self.title_style,
            )
            .set_axis_labels('', None)
        )

        fig_size = utils.mpl.MplFigSize(
            width=width, height=None, aspect=9 / 16 if holiday else 4 / 16
        )
        grid.figure.set_size_inches(*fig_size.inch())
        ConstrainedLayoutEngine(h_pad=0.2).execute(grid.figure)

        target = self.target.filter(
            pl.col('year') == 'all', pl.col('energy') == 'total'
        )
        for key, ax in grid.axes_dict.items():
            k = key if holiday else (key, '근무일')  # var, holiday
            row = target.filter(pl.col('var') == k[0], pl.col('holiday') == k[1])

            ax.axvline(row['value'].item(), c='darkslategray', ls='--', alpha=0.9)
            percentile = row.select('percentile').item()
            text_on_left = row['value'].item() > sum(ax.get_xlim()) / 2.0
            ax.text(
                0.02 if text_on_left else 0.98,
                0.95,
                f'상위 {percentile:.1%}',
                transform=ax.transAxes,
                va='top',
                ha='left' if text_on_left else 'right',
                weight=500,
                color='darkslategray',
            )
            ax.set_xlabel(f'{k[0]} {self.units[k[0]]}')

            if not holiday:
                ax.margins(y=0.2)

        return grid

    def _plot_param_change_line(
        self,
        ax: Axes,
        var: str,
        holiday: str = '근무일',
        max_year: int = 2024,
    ):
        target = (
            self.target.filter(
                pl.col('year') != 'all',
                pl.col('holiday') == holiday,
                pl.col('energy').is_in(self.energy_rename.keys()),
                pl.col('var') == var,
            )
            .with_columns(
                pl.col('year').cast(pl.UInt16),
                pl.col('energy').replace(self.energy_rename),
            )
            .filter(pl.col('year') <= max_year)
            .sort('year', 'energy')
        )
        sns.lineplot(
            target,
            x='year',
            y='value',
            hue='energy',
            hue_order=list(self.energy_rename.values()),
            ax=ax,
            alpha=0.75,
            lw=2,
        )
        ax.legend().set_title('')
        ax.set_ylabel(f'{var} {self.units[var]}')

    def _plot_param_change(
        self,
        var: str,
        holiday: str = '근무일',
        max_year: int = 2024,
        *,
        compare: bool = True,
    ):
        axes: list[Axes]
        fig, ax = plt.subplots(
            1,
            2 if compare else 1,
            sharey=True,
            gridspec_kw={'width_ratios': [4, 1]} if compare else None,
        )
        axes = ax if compare else [ax]

        self._plot_param_change_line(
            ax=axes[0], var=var, holiday=holiday, max_year=max_year
        )
        axes[0].set_title(f'연도별 {var} 변화', **self.title_style)

        if compare:
            models = self.models.filter(
                pl.col('var') == var,
                pl.col('holiday') == holiday,
                self._filter_anomaly(threshold=2),
            )
            palette = sns.color_palette(n_colors=4)
            sns.histplot(models, y='value', ax=axes[1], kde=True, color=palette[-1])

            axes[0].set_xlabel('연도')
            axes[1].set_title('공공기관\n전력 모델 분포', **self.title_style)
        else:
            axes[0].set_xlabel('')
            axes[0].margins(y=0.1)
            axes[0].autoscale_view()

        if any(x in var for x in ['기저', '민감도']):
            axes[0].set_ylim(0)

        return fig

    def plot_change_points_change(
        self,
        holiday: str = '근무일',
        max_year: int = 2024,
    ):
        target = (
            self.target.filter(
                pl.col('year') != 'all',
                pl.col('holiday') == holiday,
                pl.col('energy').is_in(self.energy_rename.keys()),
                pl.col('var').str.contains('균형점 온도'),
            )
            .with_columns(
                pl.col('year').cast(pl.UInt16),
                pl.col('energy').replace(self.energy_rename),
                pl.col('names')
                .replace_strict({'HDD': '난방', 'CDD': '냉방'})
                .alias('냉난방'),
            )
            .filter(pl.col('year') <= max_year, pl.col('energy') != '전력')
            .sort('year', 'energy')
        )

        fig, ax = plt.subplots()
        sns.lineplot(
            target,
            x='year',
            y='value',
            hue='energy',
            hue_order=['총사용량', '순사용량', '열에너지'],
            style='냉난방',
            style_order=['냉방', '난방'],
            ax=ax,
            alpha=0.6,
        )
        handles, labels = ax.get_legend_handles_labels()
        labels = ['' if x == '냉난방' else x for x in labels]
        ax.legend(handles[1:], labels[1:], loc='upper left')  # 'energy' 제목 제외
        ax.set_xlabel('')
        ax.set_ylabel('균형점 온도 [°C]')

        return fig

    def plot_param_change(self, output: Path):
        fig_size = (22 / 2.54, 5 / 2.54)

        kind: Any
        for kind in ['energy', 'sensitivity', 'change-point']:
            fig = self.plot_param_change_subplots(kind)
            fig.set_size_inches(fig_size)
            fig.savefig(output / f'0102.params-by-year-{kind}.png')
            plt.close(fig)

        fig = self.plot_change_points_change()
        fig.set_size_inches(fig_size)
        fig.savefig(output / '0102.params-by-year-균형점 온도.png')
        plt.close(fig)

        variables = (
            self.target.select(pl.col('var').unique().sort())
            .filter(pl.col('var').str.starts_with('ci-').not_())
            .to_series()
        )
        for var in variables:
            fig = self._plot_param_change(var, compare=False)
            fig.set_size_inches(fig_size)
            fig.axes[0].legend(loc='lower left')
            fig.savefig(output / f'0102.params-by-year-{var}.png')
            plt.close(fig)

            fig = self._plot_param_change(var, compare=True)
            fig.savefig(output / f'0102.params-by-year-{var}-compare.png')
            plt.close(fig)

    def plot_param_change_subplots(
        self,
        kind: Literal['energy', 'sensitivity', 'change-point'],
        holiday: str = '근무일',
        year_bound: tuple[int, int] = (2019, 2024),
    ):
        match kind:
            case 'energy':
                variables = ['기저부하', '난방 민감도', '냉방 민감도']
            case 'sensitivity':
                variables = ['난방 민감도', '냉방 민감도']
            case 'change-point':
                variables = ['난방 균형점 온도', '냉방 균형점 온도']

        data = (
            self.target.filter(
                pl.col('year') != 'all',
                pl.col('holiday') == holiday,
                pl.col('energy').is_in(self.energy_rename.keys()),
                pl.col('var').is_in(variables),
            )
            .with_columns(
                pl.col('year').cast(pl.UInt16),
                pl.col('energy').replace(self.energy_rename),
            )
            .filter(pl.col('year').is_between(*year_bound))
            .sort('year', 'energy')
        )

        fig, axes = plt.subplots(1, len(variables))
        ax: Axes
        for var, ax in zip(variables, axes, strict=True):
            sns.lineplot(
                data.filter(pl.col('var') == var),
                x='year',
                y='value',
                hue='energy',
                hue_order=['총사용량', '순사용량', '열에너지', '전력'],
                ax=ax,
                alpha=0.75,
                lw=2,
            )
            ax.set_xlabel('')
            ax.set_ylabel(f'{var} {self.units[var]}')
            ax.legend(loc='lower left').set_title('')
            ax.set_title(f'연도별 {var} 변화', **self.title_style)

        return fig

    def write_models(self, path: Path):
        """비교 대상 모델 저장 (검토용)."""
        self.models.write_excel(path)

        summ = utils.pl.PolarsSummary(
            self.models.select('holiday', 'var', 'value'),
            group=['holiday', 'var'],
        )
        summ.write_excel(path.parent / f'{path.stem}-summary{path.suffix}')


@app['report'].command
def report_param_dist(*, conf: Config, holiday: bool = True):
    """타 공공기관 대비 KEIT 위치."""
    params = _CprParams(conf=conf)

    output = conf.dirs.analysis / '0101. CPR Parameters'
    output.mkdir(exist_ok=True)

    group = conf.pi_cpr.group
    suffix = f'{group=} {holiday=}'.replace("'", '')

    params.write_models(output / f'0101.PublicInstModels-{suffix}.xlsx')

    grid = params.plot_param_grid(holiday=holiday)
    grid.savefig(output / f'0101.params-{suffix}.png')
    plt.close(grid.figure)


@app['report'].command
def report_param_change(scale: float = 1.0, *, conf: Config):
    """연도별 CPR 인자 변화."""
    params = _CprParams(conf=conf)

    output = conf.dirs.analysis / '0102. CPR Parameters Change'
    output.mkdir(exist_ok=True)

    (
        utils.mpl.MplTheme(scale, palette='tol:bright', rc={'axes.xmargin': 0.02})
        .grid()
        .apply()
    )
    params.plot_param_change(output)


@dc.dataclass
class _StandardEnergyUse:
    conf: Config
    energy: SummedEnergy = 'total'

    max_year: int = 2024
    ytrue_min_year: int = 2017

    def temperature(self):
        if (
            path := self.conf.dirs.database / 'NationalAverageTemperature2018.parquet'
        ).exists():
            return pl.read_parquet(path)

        t = (
            pl.read_csv(path.with_suffix('.csv'))
            .drop('region')
            .drop_nulls('temperature')
            .with_columns(pl.col('date').str.strip_chars().str.to_date())
            .with_columns(misc.is_holiday(pl.col('date'), years=2018).alias('holiday'))
        )
        assert t['date'].is_unique().all()
        t.write_parquet(path)
        return t

    def models(self):
        return (
            pl.scan_parquet(self.conf.dirs.analysis / '0100.CPR.parquet')
            .filter(pl.col('energy') == self.energy, pl.col('year') != 'all')
            .with_columns(pl.col('year').cast(pl.UInt16))
            .filter(pl.col('year') <= self.max_year)
            .collect()
        )

    def predicted(self):
        if (
            path := self.conf.dirs.analysis / f'0103.std-{self.energy}.parquet'
        ).exists():
            return pl.read_parquet(path)

        t = self.temperature().select('date', 'temperature', 'holiday')
        models = self.models()

        def it():
            for (h, y), df in models.group_by('holiday', 'year'):
                yield (
                    cpr.CprModel.from_dataframe(df)
                    .predict(t.filter(pl.col('holiday') == h))
                    .with_columns(pl.lit(y).alias('model-year'))
                )

        pred = (
            pl.concat(it())
            .select('model-year', pl.all().exclude('model-year'))
            .sort('model-year', 'date')
        )
        pred.write_parquet(path)
        return pred

    def _actual_energy_use(self):
        lf = pl.scan_parquet(self.conf.dirs.analysis / '0100.CPR data.parquet')

        if self.energy == 'net':
            lf = lf.filter(pl.col('energy') != '발전량')

        return (
            lf.group_by(pl.col('date').dt.year().alias('year'))
            .agg(pl.sum('value'))
            .filter(
                pl.col('year') <= self.max_year,
                pl.col('year') >= self.ytrue_min_year,
            )
            .with_columns(pl.lit('ytrue').alias('variable'))
            .collect()
        )

    def __call__(self):
        ypred = self.predicted()

        prefix = '총' if self.energy == 'total' else '순'
        energies = {
            'ytrue': f'실제 {prefix}사용량',
            'Ep': f'{prefix}사용량',
            'Epb': '기저부하',
            'Eph': '난방',
            'Epc': '냉방',
        }
        ypred = (
            ypred.rename({'model-year': 'year'})
            .unpivot([x for x in energies if x != 'ytrue'], index='year')
            .with_columns(pl.col('value').fill_null(0))
            .group_by('year', 'variable')
            .agg(pl.sum('value'))
        )
        ytrue = self._actual_energy_use()

        data = (
            (pl)
            .concat([ypred, ytrue], how='diagonal')
            .with_columns(pl.col('variable').replace(energies))
        )

        palette = Colormap('tol:bright')([6, 0, 2, 4, 1]).tolist()

        fig, ax = plt.subplots()
        sns.pointplot(
            data,
            x='year',
            y='value',
            hue='variable',
            hue_order=list(energies.values()),
            ax=ax,
            palette=palette,
            markers=['x', 'o', 's', 'v', '^'],
            alpha=0.9,
        )
        ax.set_ylim(0, 125)
        ax.set_xlabel('분석 연도')
        ax.set_ylabel('연간 사용량 [kWh/m²]')
        ax.legend(
            title='', ncols=len(energies) + 1, fontsize='small', loc='upper center'
        )

        return data, fig


@app['report'].command
def report_standard_energy_use(
    scale: float = 0.8,
    font_scale: float = 1.25,
    *,
    energy: SummedEnergy = 'total',
    conf: Config,
):
    """표준 사용량 (동일 기상자료 입력)."""
    utils.mpl.MplTheme(scale, font_scale=font_scale).grid().apply()

    std = _StandardEnergyUse(conf=conf, energy=energy)
    yearly, fig = std()

    output = conf.dirs.analysis / f'0103.std-{energy=}.ext'.replace("'", '')
    (
        yearly.pivot('variable', index='year', values='value')
        .sort('year')
        .write_excel(output.with_suffix('.xlsx'), column_widths=120)
    )
    fig.savefig(output.with_suffix('.png'))


@dc.dataclass
class _CprCompare:
    conf: Config
    ids: str | Sequence[str]
    legend: bool = False

    _style: cpr.PlotStyle = dc.field(init=False)
    _public_dataset: pc.Dataset = dc.field(init=False)
    _keit_cpr: _KeitCpr = dc.field(init=False)

    def __post_init__(self):
        self._style = {
            'xmin': -5,
            'xmax': 35,
            'line': {'zorder': 2.2, 'lw': 2, 'alpha': 0.9},
            'axvline': {'ls': ':', 'color': 'gray', 'alpha': 0.4},
        }

        ami_root = self.conf.root.parent / 'AMI/PublicInstitution'
        self._public_dataset = pc.Dataset(conf=pc.Config(ami_root))
        self._keit_cpr = _KeitCpr(self.conf)

    def _public_inst_model(self, iid: str, years: Sequence[int]):
        inst, data = self._public_dataset.data(iid)
        data = data.filter(
            pl.col('is_holiday').not_(), pl.col('datetime').dt.year().is_in(years)
        )
        model = cpr.CprEstimator(data.collect()).fit()
        return inst, model

    def _cpr(self, iid: str | None, years: Sequence[int], ax: Axes, color: ColorType):
        if iid is None:
            model = self._keit_cpr.cpr_estimator(
                energy='total', year=years, holiday=False
            ).fit()
            name = '한국산업기술기획평가원'
        else:
            inst, model = self._public_inst_model(iid, years)
            name = inst.name

        style = pydash.set_(self._style, 'line.color', color)

        model.plot(ax=ax, scatter=False, style=style)
        frame = model.model_frame.with_columns(pl.lit(name).alias('institution'))

        return frame, name, model.model_dict['coef'][0]

    def __call__(self, years: Sequence[int]):
        fig = Figure()
        ax = fig.subplots()

        if isinstance(self.ids, int | float) or len(self.ids) == 1:
            colors: list[Any] = ['tab:blue', 'darkgray']
        else:
            colors = [
                Colormap('tol:vibrant-alt')([0]),
                *Colormap('tol:light')(list(range(len(self.ids)))),
            ]

        results = tuple(
            self._cpr(iid, years=years, ax=ax, color=color)
            for iid, color in zip([None, *self.ids], colors, strict=True)
        )

        for r in results:
            ax.axhline(r[2], ls=':', color='gray', alpha=0.4)

        ax.set_xlabel('일간 평균 외기온 [°C]')
        ax.set_ylabel('일간 평균 에너지 사용량 [kWh/m²]')
        ax.set_ylim(0)

        if self.legend:
            ax.legend(
                handles=[
                    Line2D([0], [0], color=c, label=r[1])
                    for r, c in zip(results, colors, strict=True)
                ],
                loc='upper left',
            )
        elif legend := ax.get_legend():
            legend.remove()

        models = (
            (pl)
            .concat([x[0] for x in results])
            .with_columns(pl.lit(','.join(str(x) for x in years)).alias('year'))
        )

        return fig, models


@app['report'].command
def report_compare_public_inst(
    *,
    ids: tuple[str, ...] = (
        'DB_B7AE8782-40AF-8EED-E050-007F01001D51',  # 한국에너지공단
        'DB_B7AE8782-AF31-8EED-E050-007F01001D51',  # 한국콘텐츠진흥원
        'DB_B7AE8782-AF63-8EED-E050-007F01001D51',  # 한국로봇산업진흥원
    ),
    conf: Config,
    scale: float = 1,
    width: float = 16,
    height: float = 9,
):
    """타 공공기관과 CPR 모델 비교."""
    output = conf.dirs.analysis / 'CPR-compare'
    output.mkdir(exist_ok=True)

    (
        utils.mpl.MplTheme(scale, fig_size=(width, height))
        .grid(show=False)
        .tick(which='both', direction='in')
        .apply()
    )

    compare = _CprCompare(conf, ids=ids)
    models: list[pl.DataFrame] = []
    for years in [[2022], [2023], [2022, 2023], [2022, 2023, 2024]]:
        logger.info('years={}', years)

        ys = str(years).replace(' ', '').strip('[]')
        fig, frames = compare(years)
        fig.savefig(output / f'비교_year={ys}.png')
        models.append(frames.with_columns(pl.lit(ys).alias('years')))

    if not models:
        return

    _ = (
        pl.concat(models)
        .select('years', 'institution', pl.all().exclude('years', 'institution'))
        .write_excel(output / '비교 모델.xlsx')
    )


@cyclopts.Parameter('*')
@dc.dataclass
class _Grid:
    conf: Config

    t_ext: str = '외부 온도 [°C]'
    t_int: str = '실내 온도 [°C]'
    energy: str = '냉방 사용량 [Gcal]'

    def _temperature(self):
        return (
            pl.read_csv(
                self.conf.dirs.database / '0000.temperature2.csv', encoding='korean'
            )
            .rename({'일시': 'date', '평균기온(°C)': self.t_ext})
            .select(pl.col('date').str.to_date(), self.t_ext)
        )

    def _energy(self):
        r = pl.date(2025, 7, 1), pl.date(2025, 7, 29)
        return (
            pl.scan_parquet(self.conf.dirs.database / '0000.냉방적산열량계.parquet')
            .filter(pl.col('date').is_between(*r))
            .select('date', pl.col('사용량-Gcal').alias(self.energy))
            .collect()
        )

    def _pmv(self):
        return (
            pl.scan_parquet(self.conf.dirs.sensor / 'PMV.parquet')
            .filter(
                pl.col('datetime').dt.hour().is_between(9, 18, closed='left'),
                pl.col('datetime').dt.weekday().is_in([6, 7]).not_(),
                pl.col('variable').is_in(['온도', 'PMV']),
            )
            .with_columns(
                pl.col('variable').replace({'온도': self.t_int}),
                date=pl.col('datetime').dt.date(),
                space=pl.format('{}층', 'floor'),
            )
            .group_by_dynamic('date', every='1d', group_by=['space', 'variable'])
            .agg(pl.mean('value'))
            .collect()
            .pivot(
                'variable', index=['date', 'space'], values='value', sort_columns=True
            )
            .sort('date')
        )

    def data(self):
        data = (
            (self)
            ._temperature()
            .join(self._energy(), on='date', how='full', coalesce=True)
        )
        return self._pmv().join(data, on='date', how='left')

    @property
    def variables(self):
        return ['PMV', self.t_int, self.t_ext]

    def regression(self, data: pl.DataFrame, output: Path):
        dfs: list[pl.DataFrame] = []
        for v in self.variables:
            lm = pg.linear_regression(data[v].to_numpy(), data[self.energy].to_numpy())
            df: pl.DataFrame = pl.from_pandas(lm)
            dfs.append(df.select(pl.lit(v).alias('variable'), pl.all()))

        pl.concat(dfs).write_excel(output / 'grid.xlsx')

    def plot(self, data: pl.DataFrame, output: Path):
        grid = (
            sns.PairGrid(data.drop('date'), hue='space', despine=False, height=2)
            .map_lower(sns.scatterplot, alpha=0.6)
            .map_upper(sns.kdeplot, alpha=0.6)
            .map_diag(sns.histplot)
            .add_legend(title='')
        )

        for idx, v in enumerate(self.variables):
            sns.regplot(
                data,
                x=v,
                y=self.energy,
                ax=grid.axes[3, idx],
                scatter=False,
                color='gray',
            )

        grid.savefig(output / 'grid.png')
        plt.close('all')

    def __call__(self):
        output = self.conf.dirs.analysis / 'grid'
        output.mkdir(exist_ok=True)

        cache = output / 'grid.parquet'
        if cache.exists():
            data = pl.read_parquet(cache)
        else:
            data = self.data()
            data.write_parquet(cache)

        data.describe().write_excel(output / 'grid-describe.xlsx')
        self.plot(data, output)
        self.regression(data.drop_nulls(self.energy), output)


@app['report'].command
def report_grid(grid: _Grid):
    """PMV, 실내온도, 외기온, 에너지 사용량 비교."""
    grid()


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplTheme().grid().apply()
    utils.mpl.MplConciseDate().apply()

    app()

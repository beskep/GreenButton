"""산기평 CPR 분석, 에너지 효율 진단."""

from __future__ import annotations

import dataclasses as dc
import functools
import itertools
from collections.abc import Sequence
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import matplotlib.pyplot as plt
import pint
import polars as pl
import pydash
import rich
import seaborn as sns
from cmap import Colormap
from loguru import logger
from matplotlib import ticker
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D

import scripts.ami.public_institution.cpr as pc
from greenbutton import cpr, misc, utils
from greenbutton.utils import App, mplutils
from greenbutton.utils.console import Progress
from scripts.exp.e07_01keit import Config  # noqa: TC001

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.typing import ColorType
    from polars._typing import FrameType


Energy = Literal['heat', 'net_elec', 'total_elec', 'net', 'total']
SummedEnergy = Literal['net', 'total']

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
        mplutils.MplTheme(palette='crest').tick(which='both').grid().apply()

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

        match e:
            case 'heat':
                n = '열에너지'
            case 'net_elec':
                n = '순전력'
            case 'total_elec':
                n = '총전력'
            case 'net':
                n = '순에너지'
            case 'total':
                n = '총에너지'
            case _:
                raise AssertionError(e)

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
    search_range: cpr.SearchRange = dc.field(default_factory=lambda: cpr.DEFAULT_RANGE)

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
            .rename({'전체사용량': '전력순사용량', '태양광발전량': '발전량'})
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
            data.select(pl.col('date').dt.year().unique().sort()).to_series().to_list()
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
            model = estimator.fit(self.search_range, self.search_range)
        except cpr.OptimizationError as err:
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
        for holiday, year, energy in Progress.trace(
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
    utils.MplTheme(0.8).grid().apply()

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

    utils.MplTheme('paper').grid(lw=0.75, alpha=0.5).apply()

    cpr = _KeitCpr(conf=conf)

    years = (
        cpr.data.select(pl.col('date').dt.year().unique().sort()).to_series().to_list()
    )
    years = [None, *years]

    energies: list[SummedEnergy] = ['total', 'net']

    for energy, year in itertools.product(energies, years):
        logger.info(f'{energy=} | {year=}')
        _, fig = cpr.validate(energy=energy, year=year, holiday=False)
        fig.savefig(output / f'{energy=}_{year=}_holiday=False.png'.replace("'", ''))
        plt.close(fig)


@app['report'].command
def report_cpr_total(*, conf: Config):
    """전체 에너지 CPR 예시."""
    utils.MplTheme().grid().apply()

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
    utils.MplTheme(fig_size=(16, 14)).grid().apply()

    cpr = _KeitCpr(conf=conf)
    fig = cpr.plot_by_year_grid(min_year=min_year, max_year=max_year)
    fig.savefig(conf.dirs.analysis / f'0101.CPR-year({min_year}-{max_year}).png')
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
    ):
        colors = list(Colormap('tol:bright-alt')([0, 1]))

        models = self.models.filter(self._filter_anomaly())
        if drop_sensitivity:
            models = models.filter(pl.col('var').str.contains('민감도').not_())

        grid = (
            sns.FacetGrid(
                models,
                col='holiday',
                row='var',
                hue='holiday',
                sharex='row',
                sharey=False,
                palette=colors,
                despine=False,
            )
            .map_dataframe(sns.histplot, 'value', kde=True)
            .set_titles('')
            .set_titles('{col_name} {row_name} 분포', loc='left', weight=500)
            .set_axis_labels('', None)
        )

        fig_size = mplutils.MplFigSize(width=width, height=None)
        grid.figure.set_size_inches(*fig_size.inch())
        ConstrainedLayoutEngine(h_pad=0.2).execute(grid.figure)

        target = self.target.filter(
            pl.col('year') == 'all', pl.col('energy') == 'total'
        )
        for (row, col), ax in grid.axes_dict.items():
            target_row = target.filter(pl.col('var') == row, pl.col('holiday') == col)
            ax.axvline(
                target_row.select('value').item(), c='darkslategray', ls='--', alpha=0.9
            )
            percentile = target_row.select('percentile').item()
            ax.text(
                0.99,
                0.95,
                f'상위 {percentile:.1%}',
                transform=ax.transAxes,
                va='top',
                ha='right',
                weight=500,
                color='darkslategray',
            )
            ax.set_xlabel(f'{row} {self.units[row]}')

        return grid

    def _plot_param_change_impl(
        self,
        var: str,
        holiday: str = '근무일',
        max_year: int = 2024,
    ):
        models = self.models.filter(
            pl.col('var') == var,
            pl.col('holiday') == holiday,
            self._filter_anomaly(threshold=2),
        )

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

        axes: list[Axes]
        fig, axes = plt.subplots(
            1, 2, sharey=True, gridspec_kw={'width_ratios': [4, 1]}
        )
        palette = sns.color_palette(n_colors=4)
        sns.lineplot(
            target,
            x='year',
            y='value',
            hue='energy',
            hue_order=list(self.energy_rename.values()),
            palette=palette,
            ax=axes[0],
            alpha=0.75,
            lw=2,
        )
        sns.histplot(models, y='value', ax=axes[1], kde=True, color=palette[-1])

        axes[0].legend().set_title('')
        axes[0].set_ylabel(f'{var} {self.units[var]}')
        axes[0].set_xlabel('연도')

        title_style: dict = {
            'loc': 'left',
            'weight': 500,
            'fontsize': 'medium',
        }
        axes[0].set_title(f'연도별 {var} 변화', **title_style)
        axes[1].set_title('공공기관\n전력 모델 분포', **title_style)

        return fig

    def plot_param_change(self):
        variables = (
            self.target.select(pl.col('var').unique().sort())
            .filter(pl.col('var').str.starts_with('ci-').not_())
            .to_series()
        )
        for var in variables:
            fig = self._plot_param_change_impl(var)
            fig.savefig(self.conf.dirs.analysis / f'0102.params-by-year-{var}.png')
            plt.close(fig)


@app['report'].command
def report_param_dist(*, conf: Config):
    """타 공공기관 대비 KEIT 위치."""
    category = conf.pi_cpr.categories
    suffix = f'{category=}'.replace("'", '')
    params = _CprParams(conf=conf)
    utils.pl.PolarsSummary(
        params.models.select('holiday', 'var', 'value'), group=['holiday', 'var']
    ).write_excel(conf.dirs.analysis / f'0101.PublicInstModels-{suffix}-summary.xlsx')
    params.models.write_excel(
        conf.dirs.analysis / f'0101.PublicInstModels-{suffix}.xlsx'
    )

    grid = params.plot_param_grid()
    grid.savefig(conf.dirs.analysis / f'0101.params-{suffix}.png')
    plt.close(grid.figure)


@app['report'].command
def report_param_change(*, conf: Config):
    """연도별 CPR 인자 변화."""
    params = _CprParams(conf=conf)

    utils.MplTheme(palette='tol:bright').grid(lw=0.75, alpha=0.5).apply()
    params.plot_param_change()


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

        data = pl.concat([ypred, ytrue], how='diagonal').with_columns(
            pl.col('variable').replace(energies)
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
        ax.legend(title='', ncols=len(energies) + 1, fontsize='small')

        return data, fig


@app['report'].command
def report_standard_energy_use(*, energy: SummedEnergy = 'total', conf: Config):
    """표준 사용량 (동일 기상자료 입력)."""
    mplutils.MplTheme('paper', font_scale=1.25).grid().apply()

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

    _style: cpr.PlotStyle = dc.field(init=False)
    _public_dataset: pc.Dataset = dc.field(init=False)
    _keit_cpr: _KeitCpr = dc.field(init=False)

    def __post_init__(self):
        self._style = {
            'xmin': -5,
            'xmax': 30,
            'line': {'zorder': 2.2, 'lw': 2, 'alpha': 0.9},
            'axvline': {'ls': '--', 'color': 'gray', 'alpha': 0.25},
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

        return frame, name

    def __call__(self, years: Sequence[int]):
        fig, ax = plt.subplots()

        colors = [
            Colormap('tol:vibrant-alt')([0]),
            *Colormap('tol:light')(list(range(len(self.ids)))),
        ]
        results = tuple(
            self._cpr(iid, years=years, ax=ax, color=color)
            for iid, color in zip([None, *self.ids], colors, strict=True)
        )

        ax.set_xlabel('일간 평균 외기온 [°C]')
        ax.set_ylabel('일간 평균 에너지 사용량 [kWh/m²]')
        ax.set_ylim(0)

        ax.legend(
            handles=[
                Line2D([0], [0], color=c, label=r[1])
                for r, c in zip(results, colors, strict=True)
            ]
        )

        models = pl.concat([x[0] for x in results]).with_columns(
            pl.lit(','.join(str(x) for x in years)).alias('year')
        )

        return fig, models


@app['report'].command
def report_compare_public_inst(
    *,
    ids: tuple[str, ...] = (
        'DB_B7AE8782-40AF-8EED-E050-007F01001D51',  # 한국에너지공단
        'DB_B7AE8782-AF31-8EED-E050-007F01001D51',  # 한국콘텐츠진흥원
    ),
    conf: Config,
):
    """타 공공기관과 CPR 모델 비교."""
    utils.MplTheme().grid(lw=0.75, alpha=0.25).apply()
    output = conf.dirs.analysis / 'CPR-compare'
    output.mkdir(exist_ok=True)

    compare = _CprCompare(conf, ids=ids)
    models: list[pl.DataFrame] = []
    for years in [[2022], [2023], [2022, 2023], [2022, 2023, 2024]]:
        logger.info('years={}', years)

        ys = str(years).replace(' ', '').strip('[]')
        fig, frames = compare(years)
        fig.savefig(output / f'비교_year={ys}.png')
        plt.close(fig)
        models.append(frames.with_columns(pl.lit(ys).alias('years')))

    if not models:
        return

    _ = (
        pl.concat(models)
        .select('years', 'institution', pl.all().exclude('years', 'institution'))
        .write_excel(output / '비교 모델.xlsx')
    )


if __name__ == '__main__':
    utils.MplTheme().grid(lw=0.75, alpha=0.5).apply()
    utils.MplConciseDate().apply()
    utils.LogHandler.set()

    app()

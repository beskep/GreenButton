"""
2024-09-11.

DB parquet 파일 추출 후 분석.
"""

from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING

import cyclopts
import matplotlib.pyplot as plt
import pingouin as pg
import polars as pl
import polars.selectors as cs
import rich
import seaborn as sns
import seaborn.objects as so
from cmap import Colormap
from matplotlib.dates import MonthLocator, YearLocator
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from greenbutton import cpr, misc, utils
from greenbutton.utils import App
from scripts.exp.e03_01kepco_paju import Config  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes


app = App(
    config=[
        cyclopts.config.Toml(f'config/{x}.toml', use_commands_as_keys=False)
        for x in ['.experiment', '.experiment_kepco_paju']
    ]
)
app.command(App('misc'))
app.command(App('cpr'))
app.command(App('report', help='2024 연차보고서 분석'))


def _tag_category(expr: pl.Expr):
    return (
        expr.str.replace(r'전력.*번\.유효전력(량?)', '전력.유효전력')
        .str.replace(r'전력.*냉방\.유효전력량', '전력.냉방.유효전력')
        .str.replace(r'전력.*난방\.유효전력량', '전력.난방.유효전력')
        .str.replace(r'스마트콘센트.*누적전력량', '스마트콘센트.누적전력량')
        .str.replace(r'전력.*번\.유효전력(량?)', '전력.유효전력')
        .str.replace(r'지열\.펌프\.히트펌프\d+\.전력량', '지열히트펌프.전력량')
        .str.replace(r'ESS\.[충방]전량', 'ESS.충방전량')
    )


@app['misc'].command
def misc_plot_elec_daily(*, conf: Config, drop_zero: bool = True):
    dirs = conf.db_dirs
    conf.dirs.analysis.mkdir(exist_ok=True)

    cat = 'tag_category'

    df = (
        pl.scan_parquet(dirs.binary / f'{conf.log_db}.dbo.T_BELO_ELEC_DAY.parquet')
        .with_columns(pl.col('tagValue').cast(pl.Float64))
        .with_columns(_tag_category(pl.col('[tagName]')).alias(cat))
        .drop_nulls('tagValue')
        .sort(cat, 'updateDate')
        .collect()
    )

    if drop_zero:
        all_zero = (
            df.group_by('[tagName]')
            .agg(pl.col('tagValue').ne(0).sum())
            .filter(pl.col('tagValue') == 0)
            .to_series()
        )
        df = df.filter(pl.col('[tagName]').is_in(all_zero).not_())

    rich.print(df)

    grid = sns.relplot(
        df.to_pandas(),
        x='updateDate',
        y='tagValue',
        col=cat,
        col_wrap=4,
        kind='line',
        units='tagSeq',
        estimator=None,
        height=3.5,
        aspect=4 / 3,
        alpha=0.4,
        facet_kws={'sharey': False},
    ).set_titles('{col_name}')

    grid.savefig(conf.dirs.analysis / '[DB] ELEC_DAY.png')
    plt.close(grid.figure)


def _read_elec_daily_vs_15min(directory: Path, prefix: str):
    fifteen = (
        pl.scan_parquet(list(directory.glob(f'{prefix}15MIN*.parquet')))
        .sort('updateDate')
        .group_by_dynamic('updateDate', every='1d', group_by='tagSeq')
        .agg(pl.sum('tagValue'))
    )
    daily = pl.scan_parquet(directory / f'{prefix}DAY.parquet')
    tags = pl.scan_parquet(directory / 'ksem.pajoo.dbo.T_BECO_TAG.parquet')

    return (
        daily.select('updateDate', 'tagSeq', 'tagValue')
        .rename({'tagValue': 'daily'})
        .join(
            fifteen.rename({'tagValue': '15min'}),
            on=['updateDate', 'tagSeq'],
            how='inner',
            validate='1:1',
        )
        .sort('updateDate', 'tagSeq')
        .join(tags.select('tagSeq', 'tagName', 'tagDesc'), on='tagSeq', how='left')
        .collect()
    )


@app['misc'].command
def misc_elec_compare(*, conf: Config):
    """일간 vs 15분 간격 데이터 합산 비교."""
    dirs = conf.db_dirs
    cache = conf.dirs.analysis / '[cache] elec daily vs 15min.parquet'

    if cache.exists():
        df = pl.read_parquet(cache, glob=False)
    else:
        df = _read_elec_daily_vs_15min(
            dirs.binary, prefix=f'{conf.log_db}.dbo.T_BELO_ELEC_'
        )
        df.write_parquet(cache)

    rich.print(df.head())

    error = (  # MSE
        df.group_by('tagSeq', 'tagName', 'tagDesc')
        .agg(
            pl.col('daily').sub('15min').cast(pl.Float64).pow(2).sum().alias('MSE'),
            pl.col('daily').cast(pl.Float64).mean().alias('avg(daily)'),
        )
        .with_columns(pl.col('MSE').sqrt().alias('RMSE'))
        .with_columns(
            pl.col('RMSE').truediv('avg(daily)').fill_nan(None).alias('CV(RMSE)')
        )
        .sort('tagSeq')
    )

    rich.print('error:', error)
    error.write_excel(
        conf.dirs.analysis / '[DB] ELEC Daily vs 15min.xlsx', column_widths=150
    )


@app['cpr'].command
def cpr_prep_weather(*, conf: Config):
    weather = conf.dirs.root / '99.weather/weather.parquet'
    data = (
        pl.concat(
            pl.read_csv(x, encoding='korean') for x in weather.parent.glob('*.csv')
        )
        .with_columns(pl.col('일시').str.to_date())
        .with_columns()
    )

    rich.print(data)

    if not data.select(pl.col('일시').is_unique().all()).item():
        msg = '일시가 unique하지 않음.'
        raise ValueError(msg)

    data.write_parquet(weather)
    data.write_excel(weather.with_suffix('.xlsx'))


@app['cpr'].command
def cpr_prep_energy(*, conf: Config):
    dirs = conf.db_dirs

    tag = pl.col('[tagName]')
    db_vars = [
        pl.col('updateDate').dt.date().alias('date'),
        pl.col('[tagName]').alias('variable'),
        pl.col('[tagDesc]').alias('description'),
        pl.col('tagValue').cast(pl.Float64).alias('value'),
    ]

    elec = (
        pl.scan_parquet(dirs.binary / f'{conf.log_db}.dbo.T_BELO_ELEC_DAY.parquet')
        .filter(
            (tag == '전기.전체전력량')
            | tag.str.extract(r'^(전력\.\d+층.[냉난]방\.유효전력량)$').is_not_null()
            | tag.str.extract(r'^(지열\.펌프\.히트펌프\d+\.전력량)$').is_not_null()
        )
        .select(db_vars, source=pl.lit('ELEC'))
    )
    facility = (
        pl.scan_parquet(dirs.binary / f'{conf.log_db}.dbo.T_BELO_FACILITY_DAY.parquet')
        .filter(tag.str.contains('실외온습도계') | (tag == 'EHP.1층.상담실.실내온도'))
        .select(db_vars, source=pl.lit('FACILITY'))
    )
    weather = (
        pl.scan_parquet(conf.dirs.root / '99.weather/weather.parquet')
        .select(
            pl.col('일시').alias('date'),
            pl.lit('기온(ASOS)').alias('variable'),
            pl.col('평균기온(°C)').alias('value'),
            pl.lit('KMA').alias('source'),
        )
        .with_columns()
    )

    years = elec.select(pl.col('date').dt.year().unique()).collect().to_series()
    data = (
        pl.concat(
            [elec.sort('variable'), facility.sort('variable'), weather], how='diagonal'
        )
        .with_columns(
            pl.col('date').dt.weekday().is_in([6, 7]).alias('weekend'),
            misc.is_holiday(pl.col('date'), years=years).alias('holiday'),
        )
        .sort('date', 'variable')
        .with_columns(
            outlier=(
                # 초반 전력 사용량이 낮은 구간 -> 이상치 판단
                (pl.col('date') < pl.date(2020, 4, 10))
                | pl.col('date').is_between(
                    # 일간 전체전력량이 일정한 구간 -> 이상치 판단
                    pl.date(2022, 7, 15),
                    pl.date(2022, 9, 1),
                )
            )
        )
        .collect()
    )

    data.write_parquet(conf.dirs.analysis / 'CPR.parquet')

    pivot = data.pivot(
        on='variable',
        index=['date', 'weekend', 'holiday', 'outlier'],
        values='value',
    ).sort('date')
    pivot.write_excel(conf.dirs.analysis / '[CPR] wide.xlsx')

    rich.print(pivot)


def _check_elec(data: pl.LazyFrame):
    # 전체, 냉/난방, 히트펌프 전력 비교
    elec = (
        data.with_columns(
            pl.col('variable')
            .replace('전기.전체전력량', '전체전력')
            .str.replace(r'지열\.펌프\.히트펌프\d+\.전력량', '지열히트펌프전력')
            .str.replace(r'전력.*냉방\.유효전력량', '냉방전력')
            .str.replace(r'전력.*난방\.유효전력량', '난방전력')
        )
        .with_columns(
            pl.col('value')
            * pl.col('variable').replace_strict(
                {'지열히트펌프전력': 1 / 24},
                default=1,
            )
        )
        .group_by('date', 'variable')
        .agg(pl.sum('value'))
        .collect()
    )

    elec = pl.concat(
        [
            elec,
            elec.filter(pl.col('variable').str.contains(r'[냉난]방'))
            .group_by('date')
            .agg(pl.sum('value'))
            .with_columns(pl.lit('냉난방전력').alias('variable')),
        ],
        how='diagonal',
    ).sort('date', 'variable')
    hc = (
        elec.pivot('variable', index='date', values='value')
        .unpivot(
            ['냉방전력', '난방전력'],
            index=['date', '전체전력', '지열히트펌프전력'],
        )
        .with_columns(pl.col('variable').str.strip_suffix('전력'))
        .filter(pl.col('value') != 0)
        .sample(fraction=1, shuffle=True)
    )

    utils.MplTheme(palette='tol:bright', fig_size=(None, 10, 0.3)).grid().apply()

    fig, axes_array = plt.subplots(1, 3, squeeze=False, sharey=True)
    axes: list[Axes] = list(axes_array.ravel())

    sns.lineplot(
        elec.filter(pl.col('variable').is_in(['냉방전력', '난방전력']).not_()),
        x='date',
        y='value',
        hue='variable',
        hue_order=['냉난방전력', '지열히트펌프전력', '전체전력'],
        ax=axes[0],
        alpha=0.75,
    )
    axes[0].tick_params(which='both', bottom=True)
    axes[0].xaxis.set_major_locator(YearLocator())
    axes[0].xaxis.set_minor_locator(MonthLocator([4, 7, 10]))

    for ax, y in zip(axes[1:], ['전체전력', '지열히트펌프전력'], strict=True):
        sns.scatterplot(
            hc,
            x='value',
            y=y,
            hue='variable',
            hue_order=['냉방', '난방'],
            ax=ax,
            alpha=0.25,
        )

    for ax, title in zip(
        axes,
        ['전력사용량', '냉난방 vs 전체 사용량', '냉난방 vs 히트펌프 사용량'],
        strict=True,
    ):
        ax.set_title(title)
        ax.set_xlabel('' if title == '전력사용량' else '냉·난방 전력사용량 (kWh?)')
        ax.set_ylabel('전력사용량 (kWh?)')
        ax.get_legend().set_title('')
        for h in ax.get_legend().legend_handles:
            if h is not None:
                h.set_alpha(0.8)

    return fig, axes


@app['cpr'].command
def cpr_check_data(*, conf: Config):
    """CPR 데이터 검토."""
    data = pl.scan_parquet(conf.dirs.analysis / 'CPR.parquet')

    fig, _ = _check_elec(data.filter(pl.col('source') == 'ELEC'))
    fig.savefig(conf.dirs.analysis / '[CPR] 사용량.png')
    plt.close(fig)

    weather = (
        data.filter(
            pl.col('variable').is_in([
                '기온(ASOS)',
                '실외온습도계.외기온도',
                '실외온습도계.실내온도',
            ])
        )
        .select('date', 'variable', 'value', 'outlier')
        .collect()
    )

    fig, axes_array = plt.subplots(1, 2, squeeze=False)
    axes: list[Axes] = list(axes_array.ravel())
    sns.lineplot(weather, x='date', y='value', hue='variable', ax=axes[0], alpha=0.6)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('온도 [°C]')
    sns.scatterplot(
        weather.filter(pl.col('variable') != '실외온습도계.실내온도').pivot(
            'variable', index=['date', 'outlier'], values='value'
        ),
        x='기온(ASOS)',
        y='실외온습도계.외기온도',
        hue='outlier',
        ax=axes[1],
        alpha=0.2,
        s=20,
    )
    axes[1].set_aspect('equal')
    axes[1].dataLim.update_from_data_xy(
        axes[1].dataLim.get_points()[:, ::-1], ignore=False
    )
    axes[1].autoscale_view()
    axes[1].axline((0, 0), slope=1, c='gray', ls='--')

    fig.savefig(conf.dirs.analysis / '[CPR] 기온비교.png')
    plt.close(fig)


@dc.dataclass
class CPR:
    source: str | Path
    holiday: bool

    data: pl.DataFrame = dc.field(init=False)
    estimator: cpr.CprEstimator = dc.field(init=False)
    _model: cpr.CprAnalysis | None = dc.field(default=None, init=False)

    def __post_init__(self):
        self.data = (
            pl.scan_parquet(self.source)
            .filter(
                ~pl.col('outlier'),
                pl.col('holiday') if self.holiday else ~pl.col('holiday'),
                pl.col('variable')
                .str.extract(
                    r'^(기온\(ASOS\)|'
                    r'전기\.전체전력량|'
                    r'지열\.펌프\.히트펌프\d+\.전력량)$',
                )
                .is_not_null(),
            )
            .with_columns(
                pl.col('variable')
                .str.replace_many(
                    ['기온(ASOS)', '전기.전체전력량'], ['일평균 외기온', '전체전력량']
                )
                .str.replace(r'지열\.펌프\.히트펌프\d+\.전력량', '지열 히트펌프 전력량')
            )
            .group_by('date', 'variable')
            .agg(pl.sum('value'))
            .collect()
            .pivot('variable', index='date', values='value', sort_columns=True)
            .drop_nulls(['전체전력량', '지열 히트펌프 전력량'])
            .sort('date')
            .rename({'일평균 외기온': 'temperature', '전체전력량': 'energy'})
        )
        self.estimator = self._estimator()

    def reset(self):
        self._model = None

    def _estimator(self):
        return cpr.CprEstimator(self.data.rename({'date': 'datetime'}))

    @property
    def model(self) -> cpr.CprAnalysis:
        if self._model is None:
            self.estimator = self._estimator()
            self._model = self.estimator.fit()
        return self._model

    def validate(self, ax: Axes | None = None):
        data = self.estimator.data.dataframe
        ypred = self.model.disaggregate(data).with_columns(
            pl.format(
                '{}{}',
                pl.col('HDD').ne(0).cast(pl.Int8),
                pl.col('CDD').ne(0).cast(pl.Int8),
            )
            .replace_strict({'10': '난방', '01': '냉방', '00': '비냉난방'})
            .alias('period'),
        )
        ytrue = data.drop(ypred.columns, strict=False)
        compare = (
            pl.concat([ytrue, ypred], how='horizontal')
            .with_columns(pl.col('지열 히트펌프 전력량') / 24)  # XXX
            .with_columns((pl.col('Epc') + pl.col('Eph')).alias('Ephc'))
        )

        if ax is not None:
            self._validate_plot(
                data=compare,
                x='지열 히트펌프 전력량',
                y='Ephc',
                xlabel='지열 히트펌프 전력 사용량 [MJ]',
                ylabel='추정 냉난방 전력 사용량 [MJ]',
                ax=ax,
            )

        return compare

    @staticmethod
    def _validate_plot(  # noqa: PLR0913
        data: pl.DataFrame,
        *,
        x: str,
        y: str,
        xlabel: str,
        ylabel: str,
        ax: Axes,
    ):
        sns.scatterplot(
            data,
            x=x,
            y=y,
            palette=[*Colormap('tol:vibrant')([3, 0]), '#555'],
            hue='period',
            hue_order=['난방', '냉방', '비냉난방'],
            ax=ax,
            alpha=0.2,
            s=25,
        )

        ax.dataLim.update_from_data_xy(ax.dataLim.get_points()[:, ::-1], ignore=False)
        ax.autoscale_view()
        ax.set_box_aspect(1)

        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.get_legend().set_title('')
        for h in ax.get_legend().legend_handles:
            if h is not None:
                h.set_alpha(0.8)

        lm = pg.linear_regression(
            data.select(x).to_numpy().ravel(),
            data.select(y).to_numpy().ravel(),
            as_dataframe=False,
        )
        assert isinstance(lm, dict)
        ax.text(
            x=0.05,
            y=0.95,
            s=f'r²={lm["r2"]:.4f}',
            va='top',
            transform=ax.transAxes,
            fontsize='large',
            weight=500,
        )


@app['cpr'].command
def cpr_execute(*, conf: Config):
    for holiday in [False, True]:
        h = '휴일' if holiday else '평일'
        cpr = CPR(source=conf.dirs.analysis / 'CPR.parquet', holiday=holiday)

        cpr.model.model_frame.write_excel(conf.dirs.analysis / f'[CPR] {h}.xlsx')

        fig, ax = plt.subplots()
        cpr.model.plot(ax=ax, style={'scatter': {'alpha': 0.25, 's': 15}})
        ax.set_xlabel('일평균 외기온 [°C]')
        ax.set_ylabel('전력 사용량 [MJ]')  # XXX unit
        ax.set_ylim(0)
        fig.savefig(conf.dirs.analysis / f'[CPR] {h}.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        val = cpr.validate(ax=ax)
        val.write_parquet(conf.dirs.analysis / f'[CPR] {h}-예측.parquet')
        val.write_excel(conf.dirs.analysis / f'[CPR] {h}-예측.xlsx')
        fig.savefig(conf.dirs.analysis / f'[CPR] {h}-검증.png')
        plt.close(fig)


@app['cpr'].command
def cpr_assess(*, conf: Config):
    """연도별 CPR, 표준 기상자료로부터 냉난방 사용량 평가."""
    cpr = CPR(source=conf.dirs.analysis / 'CPR.parquet', holiday=False)

    years = [2021, 2023]
    data = cpr.data.filter(pl.col('date').dt.year().is_between(years[0], years[1]))

    last_year = (
        data.filter(pl.col('date').dt.year() == years[1])
        .select('temperature')
        .with_columns()
    )

    dfs: list[pl.DataFrame] = []
    for year in range(years[0], years[1] + 1):
        cpr.data = data.filter(pl.col('date').dt.year() == year)
        cpr.reset()
        pred = cpr.model.predict(last_year)
        dfs.append(
            pred.select(pl.lit(year).alias('year'), pl.mean('Epb', 'Eph', 'Epc'))
        )

    df = (
        pl.concat(dfs)
        .unpivot(['Eph', 'Epc'], index='year')
        .with_columns(
            pl.format('{}년', 'year').alias('year'),
            pl.col('variable').replace_strict({
                'Eph': '난방',
                'Epc': '냉방',
            }),
        )
    )
    rich.print(df)

    utils.MplTheme(palette='tol:vibrant').grid().apply()
    fig, ax = plt.subplots()
    sns.barplot(df, x='year', y='value', hue='variable', ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f')  # type: ignore[arg-type]

    ax.get_legend().set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('추정 냉·난방 에너지 사용량 [MJ]')
    ax.set_ylim(0, 350)

    fig.savefig(conf.dirs.analysis / '기상 표준 냉난방 사용량.png')


@app['report'].command(name='ts')
def report_time_series(*, conf: Config):
    """
    시계열 분석.

    냉난방 에너지 분리 결과 그래프.
    """
    sources = [
        conf.dirs.analysis / '[CPR] 평일-예측.parquet',
        conf.dirs.analysis / '[CPR] 휴일-예측.parquet',
    ]
    data = (
        pl.scan_parquet(sources, glob=False)
        .rename({'datetime': 'date'})
        .filter(pl.col('date').dt.year() < 2024)  # noqa: PLR2004
        .select('date', cs.starts_with('Ed'))
        .rename({'Edb': '기저', 'Edh': '난방', 'Edc': '냉방'})
        .unpivot(['기저', '난방', '냉방'], index='date')
        .with_columns(
            pl.col('date').dt.year().alias('년'),
            pl.col('date').dt.month().alias('월'),
            pl.col('date').dt.week().alias('주'),
            pl.col('date').dt.weekday().alias('weekday'),
        )
        .with_columns(
            pl.col('weekday')
            .replace_strict(
                dict(zip(range(1, 8), '월화수목금토일', strict=True)),
                return_dtype=pl.String,
            )
            .alias('요일')
        )
        .collect()
    )

    rich.print(data)
    rich.print(
        data.group_by('년', 'variable')
        .agg(pl.mean('value'))
        .pivot('variable', index='년', values='value')
        .sort('년')
    )
    rich.print(
        data.group_by('date', '년')
        .agg(pl.sum('value'))
        .group_by('년')
        .agg(pl.mean('value'))
        .sort('년')
    )
    rich.print(
        data.filter(pl.col('value') != 0)
        .with_columns(
            pl.col('weekday').replace_strict({6: '주말', 7: '주말'}, default='주중')
        )
        .group_by('variable', 'weekday')
        .agg(pl.mean('value'))
        .pivot('weekday', index='variable', values='value', sort_columns=True)
        .sort('variable')
        .with_columns(pl.col('주말').truediv('주중').alias('주말/주중'))
    )

    palette = ['#444', *Colormap('tol:vibrant')([3, 0])]
    theme = utils.MplTheme(
        'paper', fig_size=(16 * 0.8, 9 * 0.8), rc={'legend.fontsize': 'small'}
    ).grid()
    theme.apply()
    so.Plot.config.theme.update(theme.rc_params())

    for delta in ['년', '월', '주', '요일']:
        df = data.sort('weekday' if delta == '요일' else delta)

        if delta == '요일':
            df = df.filter(pl.col('value') != 0)

        if delta == '년':
            df = pl.concat(
                [
                    df.group_by('date', '년')
                    .agg(pl.sum('value'))
                    .with_columns(pl.lit('전체').alias('variable')),
                    df,
                ],
                how='diagonal',
            )
            p = ['#999', *palette]
        else:
            p = palette

        plot = so.Plot(df, x=delta, y='value', color='variable')

        if delta == '년':
            plot = plot.add(so.Bar(), so.Agg(), so.Dodge()).add(
                so.Range(), so.Est(errorbar='se'), so.Dodge()
            )
        else:
            plot = plot.add(so.Line(), so.Agg()).add(so.Range(), so.Est(errorbar='se'))

        fig, ax = plt.subplots()
        plot.scale(color=p).layout(engine='constrained').on(ax).plot(pyplot=True)  # pyright: ignore[reportArgumentType]

        ax.set_xlabel('')
        ax.set_ylabel('일평균 에너지 사용량 [MJ]')

        if delta == '년':
            ax.xaxis.set_major_locator(MaxNLocator(5, steps=[2, 5, 10]))
        if delta != '요일':
            ax.xaxis.set_major_formatter(StrMethodFormatter(f'{{x:.0f}}{delta}'))

        fig.legends[0].set_title('')
        utils.mplutils.move_legend_fig_to_ax(fig, ax, 'best')

        fig.savefig(conf.dirs.analysis / f'TS-{delta}.png')
        plt.close(fig)


if __name__ == '__main__':
    utils.LogHandler.set()
    utils.MplConciseDate(bold_zero_format=False).apply()
    utils.MplTheme(palette='tol:bright').grid().apply()

    app()

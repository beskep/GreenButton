"""
2024-09-11.

DB parquet 파일 추출 후 분석.
"""

from __future__ import annotations

import dataclasses as dc
import inspect
from typing import TYPE_CHECKING, Annotated, Literal

import holidays
import matplotlib.pyplot as plt
import pingouin as pg
import polars as pl
import pyarrow.csv
import rich
import seaborn as sns
from cyclopts import App, Parameter
from matplotlib.dates import MonthLocator, YearLocator

from greenbutton import utils
from greenbutton.cpr import ChangePointModel, ChangePointRegression
from scripts.sensor import Experiment

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes


class DBDirs:
    def __init__(self, source: Path | Experiment) -> None:
        root = source.dirs.DB if isinstance(source, Experiment) else source

        self.root = root
        self.weather = root / '00weather'
        self.sample = root / '01sample'
        self.parquet = root / '02parquet'
        self.analysis = root / '03analysis'


ExperimentDate = Literal['2024-03-20', '2024-07-11', None]
_DBDirs = Annotated[DBDirs, Parameter(parse=False)]

cnsl = rich.get_console()
app = App()


@app.meta.default
def launcher(*tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)]):
    command, bound = app.parse_args(tokens)
    argspec = inspect.getfullargspec(command)
    kwargs = bound.kwargs

    if 'exp' in argspec.kwonlyargs or 'dirs' in argspec.kwonlyargs:
        exp = Experiment(building='kepco_paju', date=None)

        if 'exp' in argspec.kwonlyargs:
            kwargs['exp'] = exp
        if 'dirs' in argspec.kwonlyargs:
            kwargs['dirs'] = DBDirs(exp)

    return command(*bound.args, **kwargs)


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


@app.command
def elec_daily(*, drop_zero: bool = True, dirs: _DBDirs):
    dirs.analysis.mkdir(exist_ok=True)

    cat = 'tag_category'

    df = (
        pl.scan_parquet(dirs.parquet / 'ksem.pajoo.log.dbo.T_BELO_ELEC_DAY.parquet')
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

    cnsl.print(df)

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

    grid.savefig(dirs.analysis / 'ELEC_DAY.png')
    plt.close(grid.figure)


def _read_elec_daily_vs_15min(directory: Path):
    prefix = 'ksem.pajoo.log.dbo.T_BELO_ELEC_'

    fifteen = (
        pl.scan_parquet(list(directory.glob(f'{prefix}15MIN*.parquet')))
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


@app.command
def elec_daily_vs_15min(*, dirs: _DBDirs):
    """일간 vs 15분 간격 데이터 합산 비교."""
    cache = dirs.analysis / '[cache] elec daily vs 15min.parquet'

    if cache.exists():
        df = pl.read_parquet(cache, glob=False)
    else:
        df = _read_elec_daily_vs_15min(dirs.parquet)
        df.write_parquet(cache)

    cnsl.print(df.head())

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

    cnsl.print('error:', error)
    error.write_excel(dirs.analysis / 'ELEC Daily vs 15min.xlsx', column_widths=150)


def _read_csv(path, encoding='korean'):
    df = pl.from_arrow(
        pyarrow.csv.read_csv(
            path, read_options=pyarrow.csv.ReadOptions(encoding=encoding)
        )
    )
    if not isinstance(df, pl.DataFrame):
        raise TypeError(df)

    return df


@app.command
def prep_weather(*, dirs: _DBDirs):
    df = pl.concat(_read_csv(x) for x in dirs.weather.glob('*.csv'))

    cnsl.print(df)

    if not df.select(pl.col('일시').is_unique().all()).item():
        msg = '일시가 unique하지 않음.'
        raise ValueError(msg)

    df.write_parquet(dirs.weather / 'weather.parquet')
    df.write_excel(dirs.weather / 'weather.xlsx')


@app.command
def prep_cpr(*, dirs: _DBDirs):
    pq = dirs.parquet

    tag = pl.col('[tagName]')
    db_vars = [
        pl.col('updateDate').dt.date().alias('date'),
        pl.col('[tagName]').alias('variable'),
        pl.col('[tagDesc]').alias('description'),
        pl.col('tagValue').cast(pl.Float64).alias('value'),
    ]

    elec = (
        pl.scan_parquet(pq / 'ksem.pajoo.log.dbo.T_BELO_ELEC_DAY.parquet')
        .filter(
            (tag == '전기.전체전력량')
            | tag.str.extract(r'^(전력\.\d+층.[냉난]방\.유효전력량)$').is_not_null()
            | tag.str.extract(r'^(지열\.펌프\.히트펌프\d+\.전력량)$').is_not_null()
        )
        .select(db_vars, source=pl.lit('ELEC'))
    )
    facility = (
        pl.scan_parquet(pq / 'ksem.pajoo.log.dbo.T_BELO_FACILITY_DAY.parquet')
        .filter(tag.str.contains('실외온습도계') | (tag == 'EHP.1층.상담실.실내온도'))
        .select(db_vars, source=pl.lit('FACILITY'))
    )
    weather = pl.scan_parquet(dirs.weather / 'weather.parquet').select(
        pl.col('일시').alias('date'),
        pl.lit('기온(ASOS)').alias('variable'),
        pl.col('평균기온(°C)').alias('value'),
        pl.lit('KMA').alias('source'),
    )

    kr_holidays = list(
        holidays.country_holidays(
            'KR',
            years=elec.select(pl.col('date').dt.year().unique())
            .collect()
            .to_series()
            .to_list(),
        ).keys()
    )

    long = (
        pl.concat(
            [elec.sort('variable'), facility.sort('variable'), weather], how='diagonal'
        )
        .with_columns(pl.col('date').dt.weekday().is_in([6, 7]).alias('weekend'))
        .with_columns(
            (pl.col('weekend') | (pl.col('date').is_in(kr_holidays))).alias('holiday')
        )
        .sort('date', 'variable')
        .with_columns(
            outlier=pl.col('date').is_between(
                # 일간 전체전력량이 일정한 구간 -> 이상치 판단
                pl.date(2022, 7, 15),
                pl.date(2022, 9, 1),
            )
        )
        .collect()
    )

    long.write_parquet(dirs.analysis / 'CPR.parquet')

    wide = long.pivot(
        on='variable',
        index=['date', 'weekend', 'holiday', 'outlier'],
        values='value',
    ).sort('date')
    wide.write_excel(dirs.analysis / 'CPR-wide.xlsx')

    cnsl.print(wide)


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

    fig, _axes = plt.subplots(1, 3, squeeze=False, sharey=True)
    axes: list[Axes] = list(_axes.ravel())

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


@app.command
def check_cpr_data(*, dirs: _DBDirs):
    """CPR 데이터 검토."""
    data = pl.scan_parquet(dirs.analysis / 'CPR.parquet')

    fig, _ = _check_elec(data.filter(pl.col('source') == 'ELEC'))
    fig.savefig(dirs.analysis / '사용량.png')
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

    fig, _axes = plt.subplots(1, 2, squeeze=False)
    axes: list[Axes] = list(_axes.ravel())
    sns.lineplot(weather, x='date', y='value', hue='variable', ax=axes[0], alpha=0.75)
    sns.scatterplot(
        weather.filter(pl.col('variable') != '실외온습도계.실내온도').pivot(
            'variable', index=['date', 'outlier'], values='value'
        ),
        x='기온(ASOS)',
        y='실외온습도계.외기온도',
        hue='outlier',
        ax=axes[1],
        alpha=0.25,
    )
    axes[1].set_aspect('equal')
    axes[1].axline((0, 0), slope=1, c='gray', ls='--')

    fig.savefig(dirs.analysis / '기온비교.png')
    plt.close(fig)


def _cpr(*, holiday: bool = False, dirs: _DBDirs):
    df = (
        pl.scan_parquet(dirs.analysis / 'CPR.parquet')
        .filter(
            ~pl.col('outlier'),
            pl.col('holiday') if holiday else ~pl.col('holiday'),
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
    )

    cpr = ChangePointRegression(df, x='일평균 외기온', y='전체전력량')
    opt = cpr.optimize_multi_models(optimizer='brute')

    cnsl.print(opt)


@dc.dataclass
class CPR:
    source: str | Path
    holiday: bool

    data: pl.DataFrame = dc.field(init=False)
    cpr: ChangePointRegression = dc.field(init=False)
    _opt: ChangePointModel | None = dc.field(default=None, init=False)

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
        )
        self.cpr = ChangePointRegression(self.data, x='일평균 외기온', y='전체전력량')

    @property
    def model(self) -> ChangePointModel:
        if self._opt is None:
            self._opt = self.cpr.optimize_multi_models()
        return self._opt

    def validate(self, ax: Axes | None = None):
        x = '지열 히트펌프 전력량'
        y = '추정 냉난방 전력 사용량'
        pred = self.model.predict().with_columns(
            pl.col('pred').sub(self.model.coef()['Intercept']).alias(y)
        )

        pred = pl.concat([pred, self.data.select(x)], how='horizontal')

        if ax is not None:
            # TODO 냉/난방/기저 기간 -> hue 구분
            sns.scatterplot(pred, x=x, y=y, ax=ax, alpha=0.5)
            lm = pg.linear_regression(
                pred.select(x).to_numpy().ravel(),
                pred.select(y).to_numpy().ravel(),
                as_dataframe=False,
            )
            ax.text(
                x=0.02, y=0.98, s=f'r²={lm["r2"]:.4f}', va='top', transform=ax.transAxes
            )

        return pred


@app.command
def cpr(*, dirs: _DBDirs):
    # TODO 초반부 이상치 제거
    # TODO 탐색 범위 0.5deg 1deg 간격?
    color = sns.color_palette(n_colors=1)[0]

    for holiday in [False, True]:
        h = '휴일' if holiday else '평일'
        cpr = CPR(source=dirs.analysis / 'CPR.parquet', holiday=holiday)

        cpr.model.model_frame.write_excel(dirs.analysis / f'CPR-{h}.xlsx')

        fig, ax = plt.subplots()
        cpr.model.plot(
            ax=ax,
            style={'scatter': {'color': color, 'alpha': 0.5}},
        )
        fig.savefig(dirs.analysis / f'CPR-{h}.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        val = cpr.validate(ax=ax)
        val.write_excel(dirs.analysis / f'CPR-{h}-예측.xlsx')
        fig.savefig(dirs.analysis / f'CPR-{h}-검증.png')
        plt.close(fig)


if __name__ == '__main__':
    utils.set_logger()
    utils.MplConciseDate(bold_zero_format=False).apply()
    utils.MplTheme(palette='tol:bright').grid().apply()

    app.meta()

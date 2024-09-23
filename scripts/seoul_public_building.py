"""2024-08-07 서울시 공공건물 OR 분석 데이터 정리."""
# ruff: noqa: DOC201

import dataclasses as dc
import tomllib
from collections.abc import Iterable
from functools import cached_property
from itertools import product
from pathlib import Path
from typing import Any, ClassVar, Literal

import cmasher as cmr
import matplotlib.pyplot as plt
import more_itertools as mi
import msgspec
import pingouin as pg
import polars as pl
import polars.selectors as cs
import rich
import seaborn as sns
from cmap import Colormap
from cyclopts import App
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.layout_engine import ConstrainedLayoutEngine
from rich.progress import track
from xlsxwriter import Workbook

from greenbutton import utils
from greenbutton.cpr import DEFAULT_RANGE, ChangePointRegression, Optimized, Optimizer
from scripts.config import dec_hook

cnsl = rich.get_console()
app = App()


class Config(msgspec.Struct, dict=True):
    root: Path = msgspec.field(name='dir')

    source: str
    weather: str
    report_rating: str  # 에너지 신고등급제 기준 파일
    stats: str

    @cached_property
    def path(self):
        return self.root / self.source

    @classmethod
    def read(cls, path: str | Path = 'config/config.toml'):
        data = tomllib.loads(Path(path).read_text('UTF-8'))
        return msgspec.convert(
            data['seoul_public_building'], type=cls, dec_hook=dec_hook
        )


Unit = Literal['MJ', 'kcal', 'toe', 'kWh']
RateBy = Literal['building', 'meter']


def unit_conversion(to: Unit):
    # 에너지법 시행 규칙
    # Nm3: 도시가스(LNG) 총발열량 기준
    c: dict[str, float]

    if to == 'MJ':
        c = {'kWh': 3.6, 'kcal': 4.1868e-3, 'm3': 42.7}
    elif to in {'kcal', 'toe'}:
        # kcal
        c = {'kWh': 860.0, 'kcal': 1.0, 'm3': 10_190.0}
    elif to == 'kWh':
        c = {'kcal': 1 / 860.0, 'm3': 10_190.0 / 860.0}
    else:
        raise ValueError(to)

    if to == 'toe':
        # toe == 1e7 kcal
        c = {k: v * 1e-7 for k, v in c.items()}

    return c | {'Gcal': c['kcal'] * 1e6, '1,000m3': c['m3'] * 1e3, to: 1.0}


@dc.dataclass
class Preprocess:
    conf: Config = dc.field(default_factory=Config.read)

    cpr_unit: Unit = 'MJ'

    or_sheet: str = '★총괄표(OR)★'
    ar_sheet: str = '★총괄표(AR)★'

    @cached_property
    def operational(self):
        header: list[str] = (
            pl.read_excel(
                self.conf.path,
                sheet_name=self.or_sheet,
                read_options={'header_row': None, 'n_rows': 2},
            )
            .transpose()
            .fill_null('')
            .select(
                pl.concat_str(
                    'column_0',
                    'column_1',
                    separator='-',
                ).str.strip_chars('-')
            )
            .to_series()
            .to_list()
        )
        header = [f'{t}{i}' if '합계' in t else t for i, t in enumerate(header)]

        if duplicated := list(mi.duplicates_everseen(header)):
            raise ValueError(duplicated)

        df = pl.read_excel(
            self.conf.path, sheet_name=self.or_sheet, read_options={'skip_rows': 2}
        )
        df.columns = header

        return df.drop(cs.contains('합계')).rename({'연도': '사업연도'})

    @cached_property
    def asset(self):
        req = ['소요량', '1차소요량', '등급용1차소요량']
        index = [
            list(range(20, 26)),  # 소요량
            list(range(26, 32)),  # 1차소요량
            [48],  # 등급용1차소요량
        ]
        unit = '[kWh/m2/yr]'

        df = pl.read_excel(
            self.conf.path,
            sheet_name=self.ar_sheet,
            read_options={'skip_rows': 1},
            columns=[0, 1, 2, *mi.flatten(index), 48],
        )

        df.columns = [
            '사업연도',
            '연번',
            '건물명',
            *(f'{req[0]}_{i}' for i in index[0]),
            *(f'{req[1]}_{i}' for i in index[1]),
            f'{req[2]}{unit}',
        ]

        k = unit_conversion('toe')
        return (
            df.select(
                pl.col('사업연도').cast(pl.UInt16),
                pl.col('연번').cast(pl.Int64),
                '건물명',
                # 소요량
                pl.sum_horizontal(cs.starts_with(req[0]).fill_null(0))
                .replace({0: None})
                .alias(f'{req[0]}{unit}'),
                # 1차소요량
                pl.sum_horizontal(cs.starts_with(req[1]).fill_null(0))
                .replace({0: None})
                .alias(f'{req[1]}{unit}'),
                # 등급용1차소요량
                pl.col(f'{req[2]}{unit}').replace({0: None}),
            )
            .with_columns(
                pl.col(f'{req[2]}[kWh/m2/yr]')
                .mul(k['kWh'])
                .alias(f'{req[2]}[toe/m2/yr]')
            )
            .with_columns()
        )

    @cached_property
    def stats(self):
        root = self.conf.root / self.conf.stats

        df = pl.concat(
            pl.read_excel(x, read_options={'header_row': None, 'skip_rows': 4})
            for x in root.glob('*.xls')
        )

        values = [
            '건물동수',
            '연면적',
            '전기사용량',
            '도시가스사용량',
            '지역난방사용량',
            '사용량합계',
        ]
        df.columns = ['기준년도', '시도', '시군구', '용도', *values]
        df.write_parquet(root / '통계.parquet')

        # 용도별
        return (
            df.with_columns(
                pl.col('기준년도').cast(pl.UInt16), pl.col(values).cast(pl.Float64)
            )
            .group_by('기준년도', '용도')
            .agg(pl.col(values).sum())
            .sort('기준년도', '용도')
            .with_columns(
                pl.col('사용량합계')
                .truediv(pl.col('연면적'))
                .alias('원단위사용량[toe/m²/yr]')
            )
        )

    @cached_property
    def weather(self):
        return pl.read_excel(self.conf.root / self.conf.weather).select(
            pl.col('년월').dt.year().alias('year'),
            pl.col('년월').dt.month().alias('month'),
            pl.col('평균기온(℃)').alias('temperature'),
        )

    @cached_property
    def cpr_energy(self):
        df = self.operational

        if '분류체계 정리' in df.columns:
            df = df.with_columns(pl.col('분류체계 정리').alias('용도'))

        values = df.select(cs.matches(r'\d+월$')).collect_schema().names()
        list_cols = [
            '사업연도',
            '연번',
            '건물명',
            '도로명주소',
            '지번주소',
            '용도',
            '준공연도',
        ]

        df = (
            df.group_by(values)
            .agg(pl.col(list_cols).explode(), pl.sum('연면적'))
            .with_columns(pl.col(list_cols).list.unique())
            .unpivot(values, index=cs.exclude(values))
            .with_columns(
                pl.col('value').fill_null(0),
                pl.col('variable').str.extract_groups(
                    r'^(?<energy>\w+)\((?<unit>.+)\)-(?<year>\d+)-(?<month>\d+)월$'
                ),
            )
            .unnest('variable')
            .with_columns(
                pl.col('value').replace(0, None),
                pl.col('year', 'month').cast(pl.UInt16),
            )
            .sort(
                pl.col('연번').list.get(0),
                pl.col('건물명').list.get(0),
                'year',
                'energy',
                'month',
            )
        )

        # 전력 NA 데이터 개수
        na = (
            df.filter(pl.col('energy') == '전기')
            .with_columns(
                pl.col('value')
                .is_null()
                .sum()
                .over('건물명', 'year')
                .alias('전기NA개수')
            )
            .drop('energy', 'unit', 'value')
        )
        df = df.join(
            na, on=na.select(pl.all().exclude('전기NA개수')).columns, how='left'
        )

        # 단위 변환
        uc = unit_conversion(self.cpr_unit)
        return (
            df.rename({'unit': 'original_unit', 'value': 'original_value'})
            .with_columns(
                k=pl.col('original_unit').replace_strict(uc, return_dtype=pl.Float64)
            )
            .with_columns(
                value=pl.col('original_value').mul(pl.col('k')),
                unit=pl.lit(self.cpr_unit),
            )
            .with_columns(intensity=pl.col('value') / pl.col('연면적'))
        )

    @cached_property
    def cpr(self):
        weather = self.weather.with_columns(pl.col('year', 'month').cast(pl.UInt16))
        energy = self.cpr_energy.filter(
            pl.col('value') != 0, pl.col('연면적') != 0
        ).drop('intensity')

        data = energy.join(weather, on=['year', 'month'], how='left').select(
            pl.col('건물명').list.get(0).str.replace_all(r'[\\/\?]', '').alias('건물1'),
            pl.all(),
        )

        values = [
            'energy',
            'original_value',
            'original_unit',
            'k',
            'value',
            'unit',
            'temperature',
        ]

        return data.select(pl.all().exclude(values), *values)


@app.command
def prep():
    conf = Config.read()
    prep = Preprocess(conf=conf)

    dfor = prep.operational
    dfor.write_parquet(conf.root / 'OR.parquet')
    dfor.head(100).write_excel(conf.root / 'OR-sample.xlsx')

    dfar = prep.asset
    dfar.write_parquet(conf.root / 'AR.parquet')
    dfar.head(100).write_excel(conf.root / 'AR-sample.xlsx', column_widths=200)

    dfst = prep.stats
    dfst.write_parquet(conf.root / '통계-용도별.parquet')
    dfst.write_excel(conf.root / '통계-용도별.xlsx')

    prep.weather.write_parquet(conf.root / 'weather.parquet')

    prep.cpr.write_parquet(conf.root / 'CPR.parquet')
    prep.cpr.sample(100).write_excel(conf.root / 'CPR-sample.xlsx')


@dc.dataclass
class Rating:
    """AR-OR 평가."""

    conf: Config = dc.field(default_factory=Config.read)

    asset_suffix: str = '_AR'

    @cached_property
    def operational_wide(self):
        conf = self.conf

        energy = pl.read_parquet(conf.root / 'OR.parquet')
        if (c := '분류체계 정리') in energy.columns:
            energy = energy.with_columns(pl.col(c).alias('용도')).drop(c)

        values = energy.select(cs.matches(r'\d+월$')).columns
        energy = energy.with_columns(pl.col(values).replace(0, None)).with_row_index()

        # 사용량 같은 건물 같은 meter로 판단
        energy = energy.with_columns(
            pl.concat_str(pl.col(values).fill_null(0), separator=';').alias('_value'),
        )
        meter = (
            energy.filter(pl.all_horizontal(pl.col(values).is_null()).not_())
            .select('_value')
            .unique(maintain_order=True)
            .with_row_index('meter_index')
        )

        return (
            energy.join(meter, on='_value', how='left')
            .select('index', 'meter_index', pl.all().exclude('index', 'meter_index'))
            .with_columns(
                pl.col('meter_index').fill_null(-1),
                pl.col('사업연도').cast(pl.UInt16),
            )
            .drop('_value')
        )

    @cached_property
    def operational_energy(self):
        values = self.operational_wide.select(cs.matches(r'\d+월$')).columns

        tidy = (
            self.operational_wide.unpivot(values, index=cs.exclude(values))
            .with_columns(
                pl.col('value').fill_null(0),
                pl.col('variable').str.extract_groups(
                    r'^(?<energy>\w+)\((?<unit>.+)\)-(?<year>\d+)-(?<month>\d+)월$'
                ),
            )
            .unnest('variable')
            .with_columns(
                pl.col('value').replace(0, None),
                pl.col('year', 'month').cast(pl.UInt16),
            )
        )

        # 전력 NA 데이터 개수
        na = (
            tidy.filter(pl.col('energy') == '전기')
            .group_by('index')
            .agg(pl.col('value').is_null().sum().alias('전력NA'))
        )

        # 단위변환
        uc = unit_conversion
        kwh = pl.col('unit').replace_strict(uc('kWh'), return_dtype=pl.Float64)
        toe = pl.col('unit').replace_strict(uc('toe'), return_dtype=pl.Float64)

        return (
            tidy.join(na, on=na.select(pl.all().exclude('전력NA')).columns, how='left')
            .with_columns(
                pl.col('value').mul(kwh).truediv('연면적').alias('EUI[kWh/m2]'),
                pl.col('value').mul(toe).truediv('연면적').alias('EUI[toe/m2]'),
            )
            .with_columns()
        )

    @cached_property
    def operational_stats(self):
        return (
            pl.scan_parquet(self.conf.root / '통계-용도별.parquet')
            .select(
                pl.col('기준년도').alias('year'),
                '용도',
                pl.col('원단위사용량[toe/m²/yr]').alias('AvgEUI[toe/m2/yr]'),
            )
            .collect()
        )

    @cached_property
    def operational_monthly(self):
        rating = self.operational_energy.join(
            self.operational_stats, on=['year', '용도'], how='left'
        )

        if rating.select(pl.col('AvgEUI[toe/m2/yr]').is_null().any()).item():
            msg = 'Null in AvgEUI'
            raise ValueError(msg)

        return rating

    @cached_property
    def operational_yearly(self):
        return (
            self.operational_monthly.drop('energy', 'value', 'unit', 'month')
            .drop_nulls('EUI[toe/m2]')
            .group_by(cs.exclude('EUI[kWh/m2]', 'EUI[toe/m2]'))
            .sum()
            .rename({'EUI[kWh/m2]': 'EUI[kWh/m2/yr]', 'EUI[toe/m2]': 'EUI[toe/m2/yr]'})
            .sort('index')
            .with_columns(
                pl.col('EUI[toe/m2/yr]')
                .truediv('AvgEUI[toe/m2/yr]')
                .alias('에너지 사용량비')
            )
        )

    def energy_report_rating(self, data: pl.DataFrame):
        """에너지 신고등급 사용량 등급기준."""
        breaks = (
            pl.read_excel(self.conf.root / self.conf.report_rating)
            .group_by('use', 'area')
            .agg(pl.col('break_seoul').alias('breaks'))
            .with_columns(pl.col('breaks').list.sort())
        )
        area_breaks = (3000, 5000, 10000, 20000, 50000)

        area_cut = (
            data.with_columns(
                pl.col('연면적')
                .cut(area_breaks, left_closed=True)
                .cast(pl.String)
                .alias('area'),
                pl.col('용도')
                .replace({
                    '제1종근린생활시설': '제1,2종근린생활시설',
                    '제2종근린생활시설': '제1,2종근린생활시설',
                })
                .alias('use'),
            )
            .join(breaks, on=['use', 'area'], how='left')
            .with_columns()
        )

        def _cut(df: pl.DataFrame, grade='에너지 신고등급'):
            if df.head(1).select(pl.col('area').str.starts_with('[-inf')).item():
                return df.with_columns(pl.lit('소규모').alias(grade))

            breaks = df.select('breaks').item(0, 0)

            if breaks is None:
                return df.with_columns(pl.lit('용도 외').alias(grade))

            return df.with_columns(
                pl.col('EUI[kWh/m2/yr]')
                .cut(breaks=breaks, labels=list('ABCDE'), left_closed=True)
                .cast(pl.String)
                .alias(grade)
            )

        dfs = (_cut(df) for _, df in area_cut.group_by('use', 'area'))
        return pl.concat(dfs).sort('index')

    @cached_property
    def asset(self):
        return pl.read_parquet(self.conf.root / 'AR.parquet')

    def rating(self, by: RateBy = 'meter'):
        suffix = self.asset_suffix

        df = self.operational_yearly.join(
            self.asset.with_columns(pl.col('사업연도').cast(pl.UInt16)),
            on=['사업연도', '연번'],
            how='full',
            suffix=suffix,
        )

        # 열 정렬
        df = df.select(
            mi.unique_everseen([
                'index',
                'meter_index',
                '사업연도',
                f'사업연도{suffix}',
                '연번',
                f'연번{suffix}',
                '건물명',
                f'건물명{suffix}',
                *df.columns,
            ])
        )

        df = self.energy_report_rating(df)

        if by == 'meter':
            etc = df.drop(
                'index',
                'meter_index',
                'year',
                '연면적',
                cs.contains('EUI', '에너지 사용량비', '등급용1차소요량'),
                cs.ends_with(suffix),
            ).columns

            values = cs.contains('EUI', '등급용1차소요량')
            df = (
                df.group_by('meter_index', 'year')
                .agg(
                    pl.col(etc).explode(),
                    pl.col('연면적').sum(),  # sum(Ai)
                    values.mul('연면적').sum(),  # sum(xi * Ai)
                )
                .with_columns(
                    pl.col(etc).list.unique(),
                    values.truediv('연면적'),  # sum(xi * Ai) / sum(Ai)
                )
                .with_columns(
                    pl.col('EUI[toe/m2/yr]')
                    .truediv('AvgEUI[toe/m2/yr]')
                    .alias('에너지 사용량비')
                )
                .sort('meter_index', 'year')
            )

        return df


@app.command
def rate():
    conf = Config.read()
    rating = Rating(conf=conf)

    dst = conf.root / '03Rating'
    dst.mkdir(exist_ok=True)

    rating.operational_monthly.write_parquet(dst / 'OperationalMonthly.parquet')
    (
        rating.operational_monthly.drop_nulls(cs.contains('EUI'))
        .head(100)
        .write_excel(dst / 'OperationalMonthly-sample.xlsx')
    )

    yearly = rating.energy_report_rating(rating.operational_yearly)
    yearly.write_parquet(dst / 'OperationalYearly.parquet')
    yearly.write_excel(dst / 'OperationalYearly.xlsx')

    bldg = rating.rating('building')
    bldg.write_parquet(dst / 'Rating-building.parquet')
    bldg.write_excel(dst / 'Rating-building.xlsx', column_widths=80)

    meter = rating.rating('meter')
    meter.write_parquet(dst / 'Rating-meter.parquet')
    meter.write_excel(dst / 'Rating-meter.xlsx', column_widths=80)


@app.command
def rating_plot(
    by: RateBy = 'meter',
    *,
    logx: bool = False,
    logy: bool = False,
    ymax: float | None = None,
):
    conf = Config.read()
    src = conf.root / f'03Rating/Rating-{by}.parquet'

    df = (
        pl.scan_parquet(src)
        .filter(pl.col('등급용1차소요량[kWh/m2/yr]') != 0)
        .drop_nulls(cs.contains('EUI', '등급용'))
        .select('등급용1차소요량[kWh/m2/yr]', '에너지 사용량비', 'year')
        .collect()
    )

    utils.MplTheme().grid().tick(direction='in').apply()
    fig, ax = plt.subplots()
    sns.scatterplot(
        df,
        x='등급용1차소요량[kWh/m2/yr]',
        y='에너지 사용량비',
        ax=ax,
        hue='year',
        palette='crest',
        alpha=0.5,
    )

    ax.invert_yaxis()
    ax.get_legend().set_title('연도')

    if logx:
        ax.set_xscale('log')
    if ymax:
        ax.dataLim.y1 = ymax  # type: ignore[misc]
        ax.autoscale_view()
    if logy:
        ax.set_yscale('log')

    ax.set_xlabel('등급용1차소요량 [kWh/m²yr]')

    dst = (
        src.parent / f'{src.stem}'
        f'{'_logx' if logx else ''}'
        f'{'_logy' if logy else ''}'
        f'{f'_ymax{ymax}' if ymax else ''}.png'
    )
    fig.savefig(dst)


@dc.dataclass
class RatingPlot:
    data: pl.DataFrame

    year: int = 2022
    line_ar: dc.InitVar[float | Iterable[float]] = 260
    line_or: dc.InitVar[float | Iterable[float]] = (1.5, 2.0)
    max_or: float = 10
    shade: bool = True
    height: float = 9

    lar: tuple[float, ...] = dc.field(init=False)
    lor: tuple[float, ...] = dc.field(init=False)

    def __post_init__(self, line_ar, line_or):
        self.lar = tuple(mi.always_iterable(line_ar))
        self.lor = tuple(mi.always_iterable(line_or))

    def _shade(self, ax: Axes):
        lar = self.lar
        lor = self.lor
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # mint, light yellow, orange
        palette = Colormap('tol:light')([2, 5, 6])
        kwargs: dict[str, Any] = {'zorder': 0.5, 'alpha': 0.35}

        ax.fill_between(
            x=[lar[0], xlim[1]],
            y1=lor[0],
            y2=ylim[1],
            facecolor=palette[0],
            **kwargs,
        )
        ax.fill_between(
            x=[xlim[0], lar[0]],
            y1=lor[0],
            y2=ylim[1],
            facecolor=palette[1],
            **kwargs,
        )
        ax.fill_between(
            x=[lar[0], xlim[1]],
            y1=ylim[0],
            y2=lor[0],
            facecolor=palette[1],
            **kwargs,
        )
        ax.fill_between(
            x=[xlim[0], lar[0]],
            y1=ylim[0],
            y2=lor[0],
            facecolor=palette[2],
            **kwargs,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def plot(self):
        (
            utils.MplTheme(palette='tol:bright', fig_size=(self.height, self.height))
            .grid(show=not self.shade)
            .tick(direction='in')
            .apply()
        )

        fig, ax = plt.subplots()
        sns.scatterplot(
            self.data,
            x='등급용1차소요량[kWh/m2/yr]',
            y='에너지 사용량비',
            ax=ax,
            alpha=0.6,
        )

        for x in self.lar:
            ax.axvline(x, ls='--', c='gray', alpha=0.9)
        for y in self.lor:
            ax.axhline(y, ls='--', c='gray', alpha=0.9)

        ax.set_xlim(0, 600)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_box_aspect(1)
        ax.set_xlabel('등급용1차소요량 [kWh/m²yr]')

        if self.shade:
            self._shade(ax=ax)

        return fig, ax


@app.command
def rating_plot2(  # noqa: PLR0913
    *,
    year: int = 2022,
    line_ar: float | Iterable[float] = 260,
    line_or: float | Iterable[float] = (1.5, 2.0),
    max_or: float = 10,
    shade: bool = True,
    height: float = 9,
    save_data: bool = False,
):
    conf = Config.read()

    lar = tuple(mi.always_iterable(line_ar))
    lor = tuple(mi.always_iterable(line_or))

    src = conf.root / '03Rating/Rating-building.parquet'
    dst = (
        conf.root / '03Rating/AR-OR Plot/'
        f'AR{'&'.join(str(x) for x in lar)}_OR{'&'.join(str(x) for x in lor)}'
        f'_MaxOR{max_or}_height{height}{'_shade' if shade else ''}.png'
    )
    dst.parent.mkdir(exist_ok=True)

    df = (
        pl.scan_parquet(src)
        .filter(
            pl.col('year') == year,
            pl.col('등급용1차소요량[kWh/m2/yr]') != 0,
        )
        .drop_nulls(cs.contains('EUI', '등급용'))
        .select(
            'index', '건물명', 'year', '등급용1차소요량[kWh/m2/yr]', '에너지 사용량비'
        )
        .with_columns(
            pl.format(
                '{}{}',
                pl.col('등급용1차소요량[kWh/m2/yr]')
                .gt(next(mi.always_iterable(line_ar)))
                .cast(pl.Int16),
                pl.col('에너지 사용량비')
                .gt(next(mi.always_iterable(line_or)))
                .cast(pl.Int16),
            )
            .replace_strict({
                '00': '1사분면',
                '10': '2사분면',
                '11': '3사분면',
                '01': '4사분면',
            })
            .alias('사분면')
        )
        .collect()
    )

    rp = RatingPlot(
        data=df.filter(pl.col('에너지 사용량비') <= max_or),
        year=year,
        line_ar=line_ar,
        line_or=line_or,
        max_or=max_or,
        shade=shade,
        height=height,
    )
    fig, _ = rp.plot()
    fig.savefig(dst)
    plt.close(fig)

    if save_data:
        df.write_excel(dst.with_suffix('.xlsx'))
        cnsl.print(df.group_by('사분면').len().sort('사분면'))


@app.command
def rating_plot_batch():
    for first, _, (lor, height, shade) in mi.mark_ends(
        product(
            [(1.5,), (1.5, 2.0)],
            [8, 10, 12],
            [True, False],
        )
    ):
        rating_plot2(line_or=lor, height=height, shade=shade, save_data=first)


@dc.dataclass
class ReportRating:
    data: pl.DataFrame
    requirement: Literal['소요량', '1차소요량', '등급용1차소요량']

    var_or: str = 'EUI[kWh/m2/yr]'

    palette: Any = dc.field(
        default_factory=lambda: cmr.take_cmap_colors(
            cmr.lavender_r, 5, cmap_range=(0.1, 0.8)
        )
    )

    @property
    def var_ar(self):
        return f'{self.requirement}[kWh/m2/yr]'

    def plot_hue(self):
        fig, ax = plt.subplots()
        order = [*'ABCDE', '소규모', '용도 외']
        sns.scatterplot(
            self.data.to_pandas(),
            x=self.var_ar,
            y=self.var_or,
            hue='에너지 신고등급',
            hue_order=order,
            palette=[*self.palette, '0.5', '0.2'],
            style='에너지 신고등급',
            style_order=order,
            markers=list('ooooovX'),
            ax=ax,
            alpha=0.8,
        )

        ax.set_box_aspect(1)
        ax.set_xlabel(f'{self.requirement} [kWh/m²yr]')
        ax.set_ylabel('에너지 사용량 [kWh/m²yr]')

        return fig, ax

    def plot_grid(self):
        grid = (
            sns.relplot(
                self.data.filter(pl.col('에너지 신고등급').is_in(list('ABCDE')))
                .sort('use')
                .to_pandas(),
                x=self.var_ar,
                y=self.var_or,
                hue='에너지 신고등급',
                hue_order='ABCDE',
                col='use',
                col_wrap=3,
                palette=self.palette,
                height=3,
                aspect=0.8,
                facet_kws={'despine': False},
                alpha=0.8,
            )
            .set_axis_labels(
                f'{self.requirement} [kWh/m²yr]', '에너지 사용량 [kWh/m²yr]'
            )
            .set_titles('')
            .set_titles('{col_name}', weight=500)
        )
        for ax in grid.axes.flat:
            ax.set_box_aspect(1)
            ax.tick_params(labelbottom=True)

        ConstrainedLayoutEngine().execute(grid.figure)
        utils.mplutils.move_grid_legend(grid)

        return grid

    def _corr(self, arg):
        df: pl.DataFrame
        by, df = arg
        x = df.select(self.var_ar).to_series()
        y = df.select(self.var_or).to_series()
        return pl.from_pandas(pg.corr(x, y)).select(
            pl.lit(by[0]).alias('use'), pl.all()
        )

    def corr(self):
        corr_all = pl.concat(
            mi.map_except(self._corr, self.data.group_by('use'), ValueError),
            how='vertical_relaxed',
        )

        corr_grade = pl.concat(
            mi.map_except(
                self._corr,
                self.data.filter(
                    pl.col('에너지 신고등급').is_in(list('ABCDE'))
                ).group_by('use'),
                ValueError,
            ),
            how='vertical_relaxed',
        )

        return corr_all, corr_grade


@app.command
def report_rating(
    year: int = 2022,
    max_eui: float | None = 2000,
):
    conf = Config.read()
    if not max_eui:
        max_eui = None

    output = conf.root / '03Rating/corr'
    output.mkdir(exist_ok=True)

    fmt = '에너지신고등급-MaxEUI{max_eui}-{plot}-{req}{grade}{suffix}'

    df = (
        pl.scan_parquet(conf.root / '03Rating/Rating-building.parquet')
        .drop_nulls('소요량[kWh/m2/yr]')
        .filter(
            pl.col('year') == year,
            (pl.col('EUI[kWh/m2/yr]') <= max_eui) if max_eui is not None else pl.lit(1),
        )
        .with_columns(pl.col('use').replace('문화및집회시설', '문화 및 집회시설'))
        .collect()
    )

    utils.MplTheme(fig_size=(12, 12)).grid().apply()

    rr = ReportRating(data=df, requirement='소요량')

    for req in ['소요량', '1차소요량', '등급용1차소요량']:
        rr.requirement = req  # type: ignore[assignment]

        fig, _ = rr.plot_hue()
        fig.savefig(
            output
            / fmt.format(req=req, plot='hue', max_eui=max_eui, grade='', suffix='.png')
        )
        plt.close(fig)

        grid = rr.plot_grid()
        grid.savefig(
            output
            / fmt.format(req=req, plot='grid', max_eui=max_eui, grade='', suffix='.png')
        )
        plt.close(grid.figure)

        c1, c2 = rr.corr()
        c1.write_excel(
            output
            / fmt.format(
                req=req,
                plot='corr',
                max_eui=max_eui,
                grade='(등급외 포함)',
                suffix='.xlsx',
            )
        )
        c2.write_excel(
            output
            / fmt.format(
                req=req,
                plot='corr',
                max_eui=max_eui,
                grade='',
                suffix='.xlsx',
            )
        )


@app.command
def report_rating_compare(
    year: int = 2022,
    max_eui: float | None = 2000,
):
    """소요량, 1차, 등급용 1차 비교 그래프."""
    conf = Config.read()
    if not max_eui:
        max_eui = None

    src = conf.root / '03Rating/Rating-building.parquet'
    dst = conf.root / '03Rating/corr'
    dst.mkdir(exist_ok=True)

    rr = '에너지 신고등급'
    eui = 'EUI[kWh/m2/yr]'
    req = ['소요량', '1차소요량', '등급용1차소요량']

    df = (
        pl.scan_parquet(src)
        .drop_nulls([f'{x}[kWh/m2/yr]' for x in req])
        .filter(
            pl.col('year') == year,
            (pl.col(eui) <= max_eui) if max_eui is not None else pl.lit(1),
        )
        .with_columns(pl.col('use').replace('문화및집회시설', '문화 및 집회시설'))
        .collect()
    )
    req_range = (
        df.select([f'{x}[kWh/m2/yr]' for x in req])
        .unpivot()
        .select(vmin=pl.min('value'), vmax=pl.max('value'))
        .to_numpy()
        .ravel()
    )

    utils.MplTheme(fig_size=(12, 12)).grid().apply()
    palette = cmr.take_cmap_colors(cmr.lavender_r, 5, cmap_range=(0.1, 0.8))
    order = [*'ABCDE', '소규모', '용도 외']

    for r in req:
        fig, ax = plt.subplots()
        sns.scatterplot(
            df.to_pandas(),
            x=f'{r}[kWh/m2/yr]',
            y=eui,
            hue=rr,
            hue_order=order,
            palette=[*palette, '0.5', '0.2'],
            style='에너지 신고등급',
            style_order=order,
            markers=list('ooooovX'),
            ax=ax,
            alpha=0.8,
        )
        ax.dataLim.update_from_data_x(req_range)
        ax.autoscale_view()

        ax.set_box_aspect(1)
        ax.set_xlabel(f'{r} [kWh/m²yr]')
        ax.set_ylabel('에너지 사용량 [kWh/m²yr]')

        fig.savefig(dst / f'rating-에너지신고등급-비교-MaxEUI{max_eui}-{r}.png')
        plt.close(fig)


@dc.dataclass
class CprRunner:
    data: pl.DataFrame
    x: str = 'temperature'
    y: str = 'intensity'
    energy: Literal['all', 'total'] = 'total'

    unit: Literal['MJ', 'kcal', 'toe'] = 'MJ'  # TODO
    optimizer: Optimizer = 'brute'

    palette: Any = 'crest_r'

    optimized: dict[str, Optimized] = dc.field(default_factory=dict)

    ENERGY: ClassVar[tuple[str, ...]] = ('전기', '도시가스', '열', '합계')

    def __post_init__(self):
        assert self.data.select(pl.col('energy').is_in(self.ENERGY[:-1]).all()).item()

        index = self.data.drop('value', 'energy').columns
        total = (
            self.data.group_by(index)
            .agg(pl.sum('value'))
            .with_columns(pl.lit(self.ENERGY[-1]).alias('energy'))
        )

        if total.height >= self.data.height:
            raise ValueError

        self.data = pl.concat([self.data, total], how='diagonal').with_columns(
            pl.col('value').truediv('연면적').alias('intensity')
        )

    def calculate(self):
        self.optimized = {}

        energy = self.ENERGY[-1:] if self.energy == 'total' else self.ENERGY
        for e in energy:
            if not (data := self.data.filter(pl.col('energy') == e)).height:
                continue

            try:
                cpr = ChangePointRegression(data, x=self.x, y=self.y)
                optimized = cpr.optimize_multi_models(optimizer=self.optimizer)
            except ValueError as error:
                logger.warning('{}: {}', type(error).__name__, error)
            else:
                self.optimized[e] = optimized

    def optimized_dataframe(self):
        if not self.optimized:
            raise ValueError

        return pl.concat(
            opt.dataframe.select(pl.lit(e).alias('energy'), pl.all())
            for e, opt in self.optimized.items()
        )

    def plot(self):
        if not self.optimized:
            raise ValueError

        unit = self.data.select('unit').item(0, 0)

        rc = (2, 2) if len(self.optimized) >= 3 else (1, len(self.optimized))  # noqa: PLR2004
        fig, axes = plt.subplots(*rc, squeeze=False, sharey=True)

        ax: Axes
        for (energy, optimized), ax in zip(
            self.optimized.items(), axes.flat, strict=False
        ):
            cpr = ChangePointRegression(
                self.data.filter(pl.col('energy') == energy), x=self.x, y=self.y
            )
            cpr.plot(
                optimized,
                ax=ax,
                style={'scatter': {'hue': 'year', 'palette': self.palette}},
            )

            ax.set_xlabel('기온 [℃]')
            ax.set_ylabel(f'에너지 사용량 [{unit}/m²]')
            ax.set_title(
                f'{energy} (r²={optimized.model['r2']:.4f})', loc='left', weight='bold'
            )

        for ax in axes.flat:
            if not ax.has_data():
                ax.set_axis_off()

        return fig, axes

    def model_select(self):
        data = self.data.filter(pl.col('energy') == '합계')

        fig, axes = plt.subplots(1, 3, squeeze=False)

        dfs: list[pl.DataFrame] = []
        ax: Axes
        for hc, ax in zip(['h', 'hc', 'c'], axes.flat, strict=True):
            cpr = ChangePointRegression(data, x=self.x, y=self.y)
            optimized = cpr.optimize(
                heating=DEFAULT_RANGE if 'h' in hc else None,
                cooling=DEFAULT_RANGE if 'c' in hc else None,
                optimizer=self.optimizer,
            )
            cpr.plot(
                optimized,
                ax=ax,
                style={
                    'scatter': {'hue': 'year', 'palette': self.palette, 'alpha': 0.8}
                },
            )
            ax.set_xlabel('기온 [℃]')
            ax.set_ylabel('에너지 사용량 [MJ/m²]')
            ax.set_title(
                {'h': '난방 모델', 'hc': '냉난방 모델', 'c': '냉방 모델'}[hc],
                loc='left',
                weight='bold',
            )

            dfs.append(optimized.dataframe.select(pl.lit(hc).alias('hc'), pl.all()))

        for _, last, ax in mi.mark_ends(axes.flat):
            if not last and (legend := ax.get_legend()):
                legend.remove()

        fig.set_layout_engine('constrained', wspace=0.05)

        return fig, pl.concat(dfs)


@app.command
def cpr(*, plot: bool = True):
    conf = Config.read()

    src = conf.root / 'CPR.parquet'
    dst = conf.root / '04CPR'
    dst.mkdir(exist_ok=True)

    (dst / 'model').mkdir(exist_ok=True)
    if plot:
        (dst / 'plot').mkdir(exist_ok=True)

    data = (
        pl.scan_parquet(src, glob=False).drop(cs.starts_with('original'), 'k').collect()
    )
    usage = data.select('건물1', '건물명', '용도').unique()
    cnsl.print(usage)

    utils.MplTheme(fig_size=(24, None, 3 / 4)).grid().apply()

    models: list[pl.DataFrame] = []
    for (bldg1,), df in track(
        data.group_by('건물1', maintain_order=True),
        total=data.select('건물1').n_unique(),
    ):
        logger.info(bldg1)

        try:
            cr = CprRunner(df.drop_nulls('value'), energy='all', optimizer='brute')
            cr.calculate()
        except ValueError as e:
            logger.warning(e)
            continue

        if not cr.optimized:
            logger.warning('not optimized')
            continue

        model = cr.optimized_dataframe()
        models.append(
            model.select(pl.lit(bldg1).alias('건물1'), pl.all()).join(
                usage, on='건물1', how='left'
            )
        )

        with Workbook(dst / f'model/{bldg1}.xlsx') as wb:
            model.write_excel(wb, worksheet='model')
            cr.data.write_excel(wb, worksheet='data')

        if plot:
            fig, _ = cr.plot()
            fig.savefig(dst / f'plot/{bldg1}.png')
            plt.close(fig)

    model = pl.concat(models)
    model = model.select(
        mi.unique_everseen(['건물1', '건물명', '용도', 'energy', *model.columns])
    )
    model.write_excel(dst / 'models.xlsx')


@app.command
def cpr_model_select():
    """CPR 냉난방 모델 선택 예시."""
    conf = Config.read()

    src = conf.root / 'CPR.parquet'
    dst = conf.root / '04CPR-ModelSelect'
    dst.mkdir(exist_ok=True)

    data = (
        pl.scan_parquet(src, glob=False).drop(cs.starts_with('original'), 'k').collect()
    )

    utils.MplTheme(fig_size=(None, 8, 1.1 / 4)).grid().apply()

    for (bldg1,), df in track(
        data.group_by('건물1', maintain_order=True),
        total=data.select('건물1').n_unique(),
    ):
        logger.info(bldg1)

        try:
            cr = CprRunner(df.drop_nulls('value'), energy='total', optimizer='brute')
            fig, model = cr.model_select()
        except ValueError as e:
            logger.warning('{}: {}', type(e).__name__, e)
        else:
            fig.savefig(dst / f'{bldg1}.png')
            model.write_excel(dst / f'{bldg1}.xlsx', column_widths=100)


if __name__ == '__main__':
    utils.set_logger()

    app()

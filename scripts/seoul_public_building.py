"""2024-08-07 서울시 공공건물 OR 분석 데이터 정리."""
# ruff: noqa: DOC201 DOC501

from __future__ import annotations

import dataclasses as dc
import functools
import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import cmasher as cmr
import cyclopts
import matplotlib.pyplot as plt
import more_itertools as mi
import numpy as np
import pathvalidate
import pingouin as pg
import polars as pl
import polars.selectors as cs
import rich
import seaborn as sns
from cmap import Colormap
from loguru import logger
from matplotlib.dates import DateFormatter, YearLocator
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
from xlsxwriter import Workbook

from greenbutton import cpr, utils
from greenbutton.utils import App

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes


Unit = Literal['MJ', 'kcal', 'toe', 'kWh']
RateBy = Literal['building', 'meter']


@dc.dataclass
class DirConfig:
    root: Path
    input: Path = Path('01Input')
    stats: Path = Path('02NationalStatistics')
    rating: Path = Path('03Rating')
    cpr: Path = Path('04CPR')
    etc: Path = Path('99etc')

    def __post_init__(self):
        for field in (f.name for f in dc.fields(self)):
            if field != 'root':
                setattr(self, field, self.root / getattr(self, field))


@dc.dataclass
class FileConfig:
    energy: str = ''  # AR-OR 총괄표 엑셀
    weather: str = ''  # 기상자료
    energy_report_rating: str = ''  # 에너지 신고등급 사용량등급기준


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config:
    """
    Config.

    Example
    -------
    ```
    [directory]
    root = 'ROOT PATH'
    input = '01Input'
    stats = '02NationalStatistics'
    rating = '03Rating'
    cpr = '04CPR'
    etc = '99etc'

    [file]
    energy = 'FILE NAME'
    weather = 'FILE NAME'
    energy_report_rating = 'FILE NAME'

    [cpr_group]
    'group 1' = [1, 2, 3, 4]
    'group 2' = [5, 6]
    ```
    """

    directory: DirConfig
    file: FileConfig
    cpr_group: dict[str, list[int | str]] = dc.field(default_factory=dict)


def _name_trsf(name: str, prefix: str):
    return name.removeprefix(f'{prefix}_').replace('_', '-')


app = App(
    config=cyclopts.config.Toml(
        'config/.seoul_public_building.toml', use_commands_as_keys=False
    )
)
for a, h in [
    ['rate', 'AR-OR 평가'],
    ['err', '에너지 신고등급'],
    ['report', '보고서·발표자료'],
]:
    app.command(
        utils.App(a, help=h, name_transform=functools.partial(_name_trsf, prefix=a))
    )


def _unit_conversion(to: Unit):
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
    conf: Config

    cpr_unit: Unit = 'MJ'

    or_sheet: str = 'OR 총괄표'
    ar_sheet: str = 'AR 총괄표'

    @functools.cached_property
    def operational(self):
        source = self.conf.directory.input / self.conf.file.energy
        header: list[str] = (
            pl.read_excel(
                source,
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
            source, sheet_name=self.or_sheet, read_options={'skip_rows': 2}
        )
        df.columns = header

        return df.drop(cs.contains('합계')).rename({'연도': '사업연도'})

    @functools.cached_property
    def asset(self):
        req = ['소요량', '1차소요량', '등급용1차소요량']
        index = [
            list(range(20, 26)),  # 소요량
            list(range(26, 32)),  # 1차소요량
            [48],  # 등급용1차소요량
        ]
        unit = '[kWh/m2/yr]'

        df = pl.read_excel(
            self.conf.directory.input / self.conf.file.energy,
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

        k = _unit_conversion('toe')
        return (
            df.select(
                pl.col('사업연도').cast(pl.UInt16),
                pl.col('연번').cast(pl.Int64),
                '건물명',
                # 소요량
                pl.sum_horizontal(cs.starts_with(req[0]).cast(pl.Float64).fill_null(0))
                .replace({0: None})
                .alias(f'{req[0]}{unit}'),
                # 1차소요량
                pl.sum_horizontal(cs.starts_with(req[1]).cast(pl.Float64).fill_null(0))
                .replace({0: None})
                .alias(f'{req[1]}{unit}'),
                # 등급용1차소요량
                pl.col(f'{req[2]}{unit}').cast(pl.Float64).replace({0: None}),
            )
            .with_columns(
                pl.col(f'{req[2]}[kWh/m2/yr]')
                .mul(k['kWh'])
                .alias(f'{req[2]}[toe/m2/yr]')
            )
            .with_columns()
        )

    @functools.cached_property
    def stats(self):
        """
        Stats

        References
        ----------
        https://greentogether.go.kr/sta/stat-data.do
        """  # noqa: D400
        root = self.conf.directory.stats

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

    @functools.cached_property
    def weather(self):
        return pl.read_excel(self.conf.directory.input / self.conf.file.weather).select(
            pl.col('년월').alias('date'),
            pl.col('년월').dt.year().alias('year'),
            pl.col('년월').dt.month().alias('month'),
            pl.col('평균기온(℃)').alias('temperature'),
            pl.col('평균최저기온(℃)').alias('avg_min_temperature'),
            pl.col('평균최고기온(℃)').alias('avg_max_temperature'),
        )

    @functools.cached_property
    def cpr_energy(self):
        df = self.operational.with_columns(
            pl.format('{}-{}', '사업연도', '연번').alias('연번')
        )

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
        uc = _unit_conversion(self.cpr_unit)
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

    @functools.cached_property
    def cpr(self):
        weather = self.weather.with_columns(pl.col('year', 'month').cast(pl.UInt16))
        energy = (
            self.cpr_energy.filter(pl.col('value') != 0, pl.col('연면적') != 0)
            .drop('intensity')
            .with_columns()
        )

        data = energy.join(weather, on=['year', 'month'], how='left').select(
            pl.col('건물명')
            .list.sort()
            .list.get(0)
            .str.replace_all(r'[\\/\?]', '')
            .alias('건물1'),
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


@app.command(sort_key=0)
def prep(*, conf: Config):
    """전처리."""
    root = conf.directory.root
    prep = Preprocess(conf=conf)

    dfor = prep.operational
    dfor.write_parquet(root / 'OR.parquet')
    dfor.head(100).write_excel(root / 'OR-sample.xlsx')

    dfar = prep.asset
    dfar.write_parquet(root / 'AR.parquet')
    dfar.head(100).write_excel(root / 'AR-sample.xlsx', column_widths=200)

    dfst = prep.stats
    dfst.write_parquet(root / '통계-용도별.parquet')
    dfst.write_excel(root / '통계-용도별.xlsx')

    prep.weather.write_parquet(root / 'weather.parquet')
    prep.weather.write_excel(root / 'weather.xlsx')

    prep.cpr.write_parquet(root / 'CPR.parquet')
    prep.cpr.sample(100).write_excel(root / 'CPR-sample.xlsx')


@dc.dataclass
class Rating:
    """AR-OR 평가."""

    conf: Config

    asset_suffix: str = '_AR'

    @functools.cached_property
    def operational_wide(self):
        conf = self.conf

        energy = pl.read_parquet(conf.directory.root / 'OR.parquet')
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

    @functools.cached_property
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
        uc = _unit_conversion
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

    @functools.cached_property
    def operational_stats(self):
        return (
            pl.scan_parquet(self.conf.directory.root / '통계-용도별.parquet')
            .select(
                pl.col('기준년도').alias('year'),
                '용도',
                pl.col('원단위사용량[toe/m²/yr]').alias('AvgEUI[toe/m2/yr]'),
            )
            .collect()
        )

    @functools.cached_property
    def operational_monthly(self):
        years = self.operational_stats.select(pl.col('year').unique()).to_series()
        rating = (
            self.operational_energy.filter(pl.col('year').is_in(years))
            .join(self.operational_stats, on=['year', '용도'], how='left')
            .with_columns()
        )

        if (null := rating.filter(pl.col('AvgEUI[toe/m2/yr]').is_null())).height:
            msg = 'Null in AvgEUI'
            null.write_excel(r'D:\wd\greenbutton\SeoulPublicBuilding\tmp.xlsx')
            raise ValueError(msg, null)

        return rating

    @functools.cached_property
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
            pl.read_excel(
                self.conf.directory.input / self.conf.file.energy_report_rating
            )
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

    @functools.cached_property
    def asset(self):
        return pl.read_parquet(self.conf.directory.root / 'AR.parquet')

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


@app['rate'].command
def rate(*, conf: Config):
    """AR-OR 평가."""
    rating = Rating(conf=conf)

    dst = conf.directory.rating
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

    arr = (
        bldg.select('에너지 사용량비', '등급용1차소요량[kWh/m2/yr]')
        .drop_nulls(pl.all())
        .to_numpy()
    )
    rich.print(pl.from_pandas(pg.corr(arr[:, 0], arr[:, 1])))

    # 빌딩 pivot
    index = ['index', 'meter_index', '사업연도', '건물명']
    (
        bldg.drop_nulls('EUI[kWh/m2/yr]')
        .unpivot(
            [
                'EUI[kWh/m2/yr]',
                'EUI[toe/m2/yr]',
                '등급용1차소요량[kWh/m2/yr]',
                '등급용1차소요량[toe/m2/yr]',
            ],
            index=[*index, 'year'],
        )
        .with_columns(
            pl.col('variable').str.replace_many({
                '등급용1차소요량': '등급1차',
                '[kWh/m2/yr]': 'kWh',
                '[toe/m2/yr]': 'toe',
            })
        )
        .with_columns(pl.format('{}_{}', 'variable', 'year').alias('columns'))
        .pivot('columns', index=index, values='value', sort_columns=True)
        .sort('index')
        .write_excel(dst / 'Rating-building-pivot.xlsx')
    )


@app['rate'].command
def rate_plot(
    *,
    conf: Config,
    by: RateBy = 'meter',
    logx: bool = False,
    logy: bool = False,
    ymax: float | None = None,
):
    src = conf.directory.rating / f'Rating-{by}.parquet'

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
        f'{"_logx" if logx else ""}'
        f'{"_logy" if logy else ""}'
        f'{f"_ymax{ymax}" if ymax else ""}.png'
    )
    fig.savefig(dst)


@dc.dataclass
class RatingPlot:
    data: pl.DataFrame

    year: int = 2022
    line_ar: dc.InitVar[float | Iterable[float]] = 260
    line_or: dc.InitVar[float | Iterable[float]] = (1.5, 2.0)

    max_or: float = 10
    max_ar: float = 700

    shade: bool = True
    text: bool = True
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
        kwargs: dict[str, Any] = {'zorder': 0.5, 'alpha': 0.25}

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

    @staticmethod
    def _text(ax: Axes):
        kwargs: dict = {
            'transform': ax.transAxes,
            'fontsize': 'large',
            'fontweight': 500,
            'bbox': {'boxstyle': 'circle', 'pad': 0.2, 'ec': '#0004', 'fc': '#0001'},
        }
        ax.text(0.94, 0.95, 'A', va='top', ha='right', **kwargs)
        ax.text(0.07, 0.95, 'C', va='top', ha='left', **kwargs)
        ax.text(0.94, 0.05, 'B', va='bottom', ha='right', **kwargs)
        ax.text(0.07, 0.05, 'D', va='bottom', ha='left', **kwargs)

    def plot(
        self,
        scatter: dict | None = None,
        line: dict | None = None,
        *,
        set_theme: bool = True,
    ):
        if set_theme:
            (
                utils.MplTheme(
                    palette='tol:bright', fig_size=(self.height, self.height)
                )
                .grid(show=not self.shade)
                .tick(direction='in')
                .apply()
            )

        scatter = {'alpha': 0.6} | (scatter or {})
        line = {'ls': '--', 'c': 'gray', 'alpha': 0.9} | (line or {})

        fig, ax = plt.subplots()
        sns.scatterplot(
            self.data,
            x='등급용1차소요량[kWh/m2/yr]',
            y='에너지 사용량비',
            ax=ax,
            **scatter,
        )

        for x in self.lar:
            ax.axvline(x, **line)
        for y in self.lor:
            ax.axhline(y, **line)

        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_box_aspect(0.95)

        ax.set_ylabel('에너지 사용량비 (ECR)')
        ax.set_xlabel('등급용 1차E소요량 [kWh/m²]')

        points = ax.dataLim.get_points()  # [[x0, y0], [x1, y1]]
        ax.dataLim.set_points(np.array([[0, points[0, 1]], [self.max_ar, self.max_or]]))
        ax.autoscale_view()

        if self.shade:
            self._shade(ax=ax)
        if self.text:
            self._text(ax=ax)

        return fig, ax


@app['rate'].command
def rate_plot2(  # noqa: PLR0913
    *,
    conf: Config,
    year: int = 2022,
    line_ar: float | tuple[float, ...] = 260,
    line_or: float | tuple[float, ...] = (1.5, 2.0),
    max_or: float = 10,
    shade: bool = True,
    height: float = 9,
    save_data: bool = False,
):
    lar = tuple(mi.always_iterable(line_ar))
    lor = tuple(mi.always_iterable(line_or))

    src = conf.directory.rating / 'Rating-building.parquet'
    dst = (
        conf.directory.rating / 'AR-OR Plot/'
        f'AR{"&".join(str(x) for x in lar)}_OR{"&".join(str(x) for x in lor)}'
        f'_MaxOR{max_or}_height{height}{"_shade" if shade else ""}.png'
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
        rich.print(df.group_by('사분면').len().sort('사분면'))


@app['rate'].command
def rate_batch_plot(*, conf: Config):
    for first, _, (lor, height, shade) in mi.mark_ends(
        itertools.product(
            [(1.5,), (1.5, 2.0)],
            [8, 10, 12],
            [True, False],
        )
    ):
        rate_plot2(line_or=lor, height=height, shade=shade, save_data=first, conf=conf)


@app['rate'].command
def rate_report_plot(
    *,
    conf: Config,
    year: int = 2022,
    business_year: int = 2024,
):
    dst = conf.directory.rating / 'Report AR-OR'
    dst.mkdir(exist_ok=True)

    data = (
        pl.scan_parquet(conf.directory.rating / 'Rating-building.parquet')
        .filter(
            pl.col('year') == year,
            pl.col('사업연도') == business_year,
            pl.col('등급용1차소요량[kWh/m2/yr]') != 0,
        )
        .drop_nulls(cs.contains('EUI', '등급용'))
        .select(
            'index', '건물명', 'year', '등급용1차소요량[kWh/m2/yr]', '에너지 사용량비'
        )
        .sort('index')
        .collect()
    )

    rp = RatingPlot(
        data, year=year, line_ar=260, line_or=1.5, max_or=10, shade=True, height=8
    )

    (
        utils.MplTheme(0.8, palette='tol:bright', fig_size=(8, 8))
        .grid(show=False)
        .tick(direction='in')
        .apply({
            'figure.constrained_layout.h_pad': 0.1,
            'figure.constrained_layout.w_pad': 0.1,
        })
    )
    for row in utils.Progress.trace(data.iter_rows(named=True), total=data.height):
        fig, ax = rp.plot(scatter={'c': 'k', 'alpha': 0.12}, set_theme=False)
        sns.scatterplot(
            x=[row['등급용1차소요량[kWh/m2/yr]']],
            y=[row['에너지 사용량비']],
            ax=ax,
            s=125,
            c='tab:red',
            marker='X',
            zorder=3,
        )

        building = pathvalidate.sanitize_filename(row['건물명'])
        fig.savefig(dst / f'{row["index"]:04d}-{building}.png')
        plt.close(fig)


@app['rate'].command
def rate_report_additional_plot(
    *,
    conf: Config,
    year: int = 2022,
    business_year: int = 2024,
):
    dst = conf.directory.rating / 'Report Additional AR-OR'
    dst.mkdir(exist_ok=True)

    data = (
        pl.scan_parquet(conf.directory.rating / 'Rating-building.parquet')
        .filter(
            pl.col('year') == year,
            pl.col('사업연도') == business_year,
            pl.col('등급용1차소요량[kWh/m2/yr]') != 0,
        )
        .drop_nulls(cs.contains('EUI', '등급용'))
        .select(
            'index', '건물명', 'year', '등급용1차소요량[kWh/m2/yr]', '에너지 사용량비'
        )
        .sort('index')
        .collect()
    )

    rp = RatingPlot(
        data,
        year=year,
        line_ar=260,
        line_or=1.5,
        max_or=10,
        shade=True,
        height=8,
        text=False,
    )

    (
        utils.MplTheme(0.8, palette='tol:bright', fig_size=(8, 8))
        .grid(show=False)
        .tick(direction='in')
        .apply({
            'figure.constrained_layout.h_pad': 0.1,
            'figure.constrained_layout.w_pad': 0.1,
            'legend.fontsize': 'small',
        })
    )

    gap = (20, 0.2)  # AR, OR

    yy: list[float]
    for x, yy, quadrant in [  # type: ignore[assignment]
        [191.5, [0.522, 0.641, 1.446, 1.436], 'A'],
        [261.7, [0.730, 0.867, 0.737, 0.586], 'A'],
        [177.50, [3.637, 4.977, 4.034, None], 'B'],
        [204.20, [29.917, 20.276, 7.206, 7.600], 'B'],
        [290.60, [0.624, 0.203, 0.195, 0.169], 'C'],
        [337.90, [1.026, 1.063, 1.004, 1.015], 'C'],
        [431.30, [1.741], 'D'],
        [421.70, [3.400], 'D'],
    ]:
        fig, ax = rp.plot(scatter={'c': 'k', 'alpha': 0.12}, set_theme=False)

        hue = list(range(2020, 2020 + len(yy))) if len(yy) > 1 else None
        sns.scatterplot(
            x=[x for _ in yy],
            y=yy,
            hue=hue,
            ax=ax,
            s=125,
            c='tab:red',
            marker='X',
            zorder=3,
            palette='flare' if hue else None,
            alpha=0.75,
        )

        fig.savefig(dst / f'{x=}, {yy=}.png')

        if quadrant == 'A':
            ax.set_xlim(rp.lar[0] + gap[0], 0)
            ax.set_ylim(rp.lor[0] + gap[1], 0)
        elif quadrant == 'B':
            ax.set_xlim(rp.lar[0] + gap[0], 0)
            ax.set_ylim(None, rp.lor[0] - gap[1])
        elif quadrant == 'C':
            ax.set_ylim(None, rp.lar[0] - gap[0])
            ax.set_ylim(rp.lor[0] + gap[1], 0)
        elif quadrant == 'D':
            ax.set_xlim(None, rp.lar[0] - gap[0])
            ax.set_ylim(None, rp.lor[0] - gap[1])

        fig.savefig(dst / f'{x=}, {yy=}, {quadrant=}.png')
        plt.close(fig)


@dc.dataclass
class EnergyReportRating:
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
            .set_titles('{col_name}', weight=500)
        )

        for ax in grid.axes.flat:
            ax.set_box_aspect(1)
            ax.tick_params(labelbottom=True)

        if grid.legend is not None:
            grid.legend.set_title('에너지\n신고등급')
            grid.legend.set_alignment('left')

        ConstrainedLayoutEngine(rect=(0, 0, 0.9, 1)).execute(grid.figure)

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


@app['err'].command
def err_plot(
    *,
    conf: Config,
    year: int = 2022,
    max_eui: float | None = 2000,
):
    """에너지 신고등급 그래프."""
    if not max_eui:
        max_eui = None

    output = conf.directory.rating / 'corr'
    output.mkdir(exist_ok=True)

    fmt = '에너지신고등급-MaxEUI{max_eui}-{plot}-{req}{grade}{suffix}'

    df = (
        pl.scan_parquet(conf.directory.rating / 'Rating-building.parquet')
        .drop_nulls('소요량[kWh/m2/yr]')
        .filter(
            pl.col('year') == year,
            (pl.col('EUI[kWh/m2/yr]') <= max_eui) if max_eui is not None else pl.lit(1),
        )
        .with_columns(pl.col('use').replace('문화및집회시설', '문화 및 집회시설'))
        .collect()
    )

    utils.MplTheme(fig_size=(12, 12)).grid().apply()

    rr = EnergyReportRating(data=df, requirement='소요량')

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


@app['err'].command
def err_plot_compare(
    *,
    conf: Config,
    year: int = 2022,
    max_eui: float | None = 2000,
):
    """소요량, 1차, 등급용 1차 비교 그래프."""
    if not max_eui:
        max_eui = None

    src = conf.directory.rating / 'Rating-building.parquet'
    dst = conf.directory.rating / 'corr'
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
class CprCalculator:
    data: pl.DataFrame
    x: str = 'temperature'
    y: str = 'intensity'
    energy: Literal['all', 'total'] = 'total'

    unit: Literal['MJ', 'kcal', 'toe'] = 'MJ'  # TODO
    method: cpr.Method = 'brute'

    palette: Any = 'crest_r'

    models: dict[str, cpr.CprAnalysis] = dc.field(default_factory=dict)

    ENERGY: ClassVar[tuple[str, ...]] = ('전기', '도시가스', '열', '합계')

    def __post_init__(self):
        assert self.data.select(pl.col('energy').is_in(self.ENERGY[:-1]).all()).item()

        index = self.data.drop('value', 'energy').columns
        total = (
            self.data.group_by(index)
            .agg(pl.sum('value'))
            .with_columns(pl.lit(self.ENERGY[-1]).alias('energy'))
        )

        if total.height > self.data.height:
            raise ValueError

        self.data = pl.concat([self.data, total], how='diagonal').with_columns(
            pl.col('value').truediv('연면적').alias('intensity')
        )

    def calculate(self):
        self.models = {}

        energy = self.ENERGY[-1:] if self.energy == 'total' else self.ENERGY
        for e in energy:
            if not (data := self.data.filter(pl.col('energy') == e)).height:
                continue

            try:
                model = cpr.CprEstimator(data.rename({'intensity': 'energy'})).fit()
            except ValueError as error:
                logger.warning('{}: {} ({})', type(error).__name__, error, e)
                raise
            else:
                self.models[e] = model

    def model_dataframe(self):
        if not self.models:
            raise ValueError

        return pl.concat(
            opt.model_frame.select(pl.lit(e).alias('energy'), pl.all())
            for e, opt in self.models.items()
        )

    def plot(self):
        if not self.models:
            raise ValueError

        unit = self.data.select('unit').item(0, 0)

        rc = (2, 2) if len(self.models) >= 3 else (1, len(self.models))  # noqa: PLR2004
        fig, axes = plt.subplots(*rc, squeeze=False, sharey=True)

        ax: Axes
        for (energy, model), ax in zip(self.models.items(), axes.flat, strict=False):
            model.plot(
                ax=ax, style={'scatter': {'hue': 'year', 'palette': self.palette}}
            )
            ax.set_xlabel('기온 [℃]')
            ax.set_ylabel(f'에너지 사용량 [{unit}/m²]')
            ax.set_title(
                f'{energy} (r²={model.model_dict["r2"]:.4f})',
                loc='left',
                weight='bold',
            )

        for ax in axes.flat:
            if not ax.has_data():
                ax.set_axis_off()

        return fig, axes

    def model_select(self):
        conf = cpr.CprConfig()
        data = self.data.filter(pl.col('energy') == '합계')

        fig, axes = plt.subplots(1, 3, squeeze=False)
        style = {'scatter': {'hue': 'year', 'palette': self.palette, 'alpha': 0.8}}

        dfs: list[pl.DataFrame] = []
        ax: Axes
        for op, ax in zip(['h', 'hc', 'c'], axes.flat, strict=True):
            model = cpr.CprEstimator(
                data.rename({'intensity': 'energy'}), conf=conf
            ).fit(method=self.method, operation=op)
            model.plot(ax=ax, style=style)
            ax.set_xlabel('기온 [℃]')
            ax.set_ylabel('에너지 사용량 [MJ/m²]')
            ax.set_title(
                {'h': '난방 모델', 'hc': '냉난방 모델', 'c': '냉방 모델'}[op],
                loc='left',
                weight='bold',
            )

            dfs.append(model.model_frame.select(pl.lit(op).alias('hc'), pl.all()))

        for _, last, ax in mi.mark_ends(axes.flat):
            if not last and (legend := ax.get_legend()):
                legend.remove()

        fig.set_layout_engine('constrained', wspace=0.05)

        return fig, pl.concat(dfs)


@app.command
def cpr_(*, conf: Config, plot: bool = True):
    """Change Point Regression."""
    src = conf.directory.root / 'CPR.parquet'
    dst = conf.directory.cpr
    dst.mkdir(exist_ok=True)

    (dst / 'model').mkdir(exist_ok=True)
    if plot:
        (dst / 'plot').mkdir(exist_ok=True)

    data = (
        pl.scan_parquet(src, glob=False)
        .drop(cs.starts_with('original'), 'k')
        .filter(((pl.col('year') == 2020) & pl.col('month').is_in([1, 2, 3])).not_())  # noqa: PLR2004
        .filter(pl.len().over('건물1') >= 4)  # noqa: PLR2004
        .sort('건물1')
        .collect()
    )
    usage = data.select('건물1', '건물명', '용도').unique()
    rich.print(usage)

    utils.MplTheme(fig_size=(24, None, 3 / 4)).grid().apply()

    for (bldg1,), df in utils.Progress.trace(
        data.group_by('건물1', maintain_order=True),
        total=data.select('건물1').n_unique(),
    ):
        logger.info(bldg1)

        try:
            cc = CprCalculator(df.drop_nulls('value'), energy='all', method='brute')
            cc.calculate()
        except ValueError as e:
            logger.warning(e)
            continue

        if not cc.models:
            logger.warning('not optimized')
            continue

        model = (
            cc.model_dataframe()
            .select(pl.lit(bldg1).alias('건물1'), pl.all())
            .join(usage, on='건물1', how='left')
            .select(pl.col(usage.columns), pl.all().exclude(usage.columns))
        )

        model.write_parquet(dst / f'model/model_{bldg1}.parquet')

        if '합계' not in cc.models:
            continue

        disaggregated = cc.models['합계'].disaggregate()
        disaggregated.write_parquet(dst / f'model/disaggregate_{bldg1}.parquet')

        with Workbook(dst / f'model/data_{bldg1}.xlsx') as wb:
            model.write_excel(wb, worksheet='model')
            disaggregated.write_excel(wb, worksheet='data')

        if plot:
            fig, _ = cc.plot()
            fig.savefig(dst / f'plot/{bldg1}.png')
            plt.close(fig)

    for name in ['model', 'disaggregate']:
        df = pl.read_parquet(list(dst.glob(f'model/{name}_*.parquet')))
        df.write_parquet(dst / f'{name}.parquet')
        df.write_excel(dst / f'{name}.xlsx')


@app['report'].command
def report_cpr_select(*, conf: Config):
    """CPR 냉난방 모델 선택 예시."""
    src = conf.directory.root / 'CPR.parquet'
    dst = conf.directory.cpr / 'ModelSelect'
    dst.mkdir(exist_ok=True)

    data = (
        pl.scan_parquet(src, glob=False).drop(cs.starts_with('original'), 'k').collect()
    )

    utils.MplTheme(fig_size=(None, 8, 1.1 / 4)).grid().apply()

    for (bldg1,), df in utils.Progress.trace(
        data.group_by('건물1', maintain_order=True),
        total=data.select('건물1').n_unique(),
    ):
        logger.info(bldg1)

        try:
            cr = CprCalculator(df.drop_nulls('value'), energy='total', method='brute')
            fig, model = cr.model_select()
        except ValueError as e:
            logger.warning('{}: {}', type(e).__name__, e)
        else:
            fig.savefig(dst / f'{bldg1}.png')
            model.write_excel(dst / f'{bldg1}.xlsx', column_widths=100)


@app['report'].command
def report_cpr_compare(
    *,
    conf: Config,
    max_intensity: float = np.inf,
):
    """그룹별 CPR 결과 비교."""
    groups = {
        k: [f'2024-{x}' if isinstance(x, int) else x for x in v]
        for k, v in conf.cpr_group.items()
    }

    lf = (
        pl.scan_parquet(conf.directory.root / 'CPR.parquet')
        .filter(
            ((pl.col('year') == 2020) & pl.col('month').is_in([1, 2, 3])).not_()  # noqa: PLR2004
        )
        .with_columns()
    )
    output_dir = conf.directory.cpr / 'compare'
    output_dir.mkdir(exist_ok=True)

    rich.print(groups)

    for usage, indices in groups.items():
        utils.MplTheme(fig_size=(12 * 3 / len(indices), 9)).grid().apply()
        group_data = (
            lf.filter(
                pl.any_horizontal(
                    pl.col('연번').list.contains(index) for index in indices
                )
            )
            .drop_nulls('value')
            .group_by('연번', '건물1', '연면적', 'date', 'temperature')
            .agg(pl.sum('value'))
            .with_columns(pl.col('value').truediv('연면적').alias('intensity'))
            .filter(pl.col('intensity') <= max_intensity)
            .collect()
        )
        max_y = group_data.select(pl.max('intensity')).item()

        for index in indices:
            data = (
                group_data.filter(pl.col('연번').list.contains(index))
                .drop_nulls('value')
                .with_columns()
            )
            logger.info('building={}', data.select('건물1').item(0, 0))

            try:
                model = cpr.CprEstimator(data.rename({'intensity': 'energy'})).fit()
            except ValueError as e:
                logger.error(e)
                continue

            if not model.is_valid:
                logger.warning('not optimized')
                continue

            output = (
                output_dir / f'{usage}-{index}-{data.select("건물1").item(0, 0)}.ext'
            )
            model.model_frame.write_excel(output.with_suffix('.xlsx'))

            fig, ax = plt.subplots()
            model.plot(ax=ax)
            ax.dataLim.update_from_data_y([0, max_y])
            ax.autoscale_view()
            ax.set_xlabel('월평균 기온 [°C]')
            ax.set_ylabel('에너지 사용량 [MJ/m²]')
            ax.text(
                0.5,
                0.95,
                f'$r^2={model.model_dict["r2"]:.3g}$',
                va='top',
                ha='center',
                transform=ax.transAxes,
                bbox={'boxstyle': 'square', 'ec': '#FFFA', 'fc': '#FFFA'},
            )
            fig.savefig(output.with_suffix('.png'))
            plt.close(fig)


@app['report'].command
def report_cpr_param(*, conf: Config, r2: float = 0.0):
    """용도별 CPR 분석."""
    root = conf.directory.cpr

    data = (
        pl.scan_parquet(root / 'model.parquet')
        .filter(
            pl.col('energy') == '합계',
            pl.col('r2') >= r2,
            pl.col('용도').list.len() == 1,  # 단일 용도 계량기만 사용
        )
        .with_columns(pl.col('건물명', '용도').list.get(0))
        .drop('건물1')
        .collect()
    )
    n = data.filter(pl.col('names') == 'Intercept').height
    suffix = f'MinR²={r2}_n={n}'

    rich.print(data.head())
    utils.MplTheme().grid().apply()

    # 균형점 온도
    fig, ax = plt.subplots()
    sns.pointplot(
        data.drop_nulls('change_point')
        .sort('용도', 'change_point')
        .with_columns(pl.col('names').replace({'HDD': '난방', 'CDD': '냉방'})),
        x='change_point',
        y='용도',
        hue='names',
        hue_order=['냉방', '난방'],
        estimator='median',
        linestyles='none',
        ax=ax,
        marker='D',
        alpha=0.8,
        seed=42,
        errorbar=('ci', 90),
    )
    ax.set_xlabel('냉·난방 균형점 온도 [°C]')
    ax.set_ylabel('')
    ax.get_legend().set_title('')
    sns.despine(ax=ax, bottom=False, left=True)
    fig.savefig(root / f'냉난방 균형점 온도_{suffix}.png')
    plt.close(fig)

    # 민감도
    fig, ax = plt.subplots()
    sns.barplot(
        data.filter(pl.col('names') != 'Intercept').with_columns(
            pl.col('names').replace({'HDD': '난방', 'CDD': '냉방'})
        ),
        x='coef',
        y='용도',
        hue='names',
        hue_order=['냉방', '난방'],
        estimator='median',
        ax=ax,
        seed=42,
        errorbar=('ci', 90),
    )
    ax.set_xlabel('냉·난방 민감도 [MJ/m²℃]')
    ax.set_ylabel('')
    ax.get_legend().set_title('')
    fig.savefig(root / f'냉난방 민감도_{suffix}.png')
    plt.close(fig)


@app['report'].command
def report_hist_ar_or(*, conf: Config, year: int = 2022):
    """AR, OR 분포도."""
    output = conf.directory.etc
    output.mkdir(exist_ok=True)

    grade = ['1+++', '1++', '1+', *(str(x) for x in range(1, 8)), '등급 외']
    df = (
        pl.scan_parquet(conf.directory.rating / 'Rating-building.parquet')
        .filter(pl.col('year') == year)
        .with_columns(
            grade=pl.col('등급용1차소요량[kWh/m2/yr]').cut(
                breaks=[80, 140, 200, 260, 350, 380, 450, 520, 610, 700],
                labels=grade,
                left_closed=True,
            ),
        )
        .select('grade', '에너지 사용량비')
        .collect()
    )

    inch = 2.54  # cm
    utils.MplTheme(context='paper', palette='tol:bright').grid().apply()

    fig, ax = plt.subplots()
    sns.barplot(
        df.group_by('grade').len(),
        x='grade',
        y='len',
        ax=ax,
        width=1,
        alpha=0.8,
    )
    ax.set_xlabel('')
    ax.set_ylabel('표본 수')

    ax.autoscale_view()
    ax.set_ylim(0, 65)
    ax.bar_label(ax.containers[0])  # type: ignore[arg-type]

    fig.set_size_inches(11 / inch, 5.5 / inch)
    fig.savefig(output / '등급.png')
    plt.close(fig)

    fig, ax = plt.subplots()

    eer = '에너지 사용량비'
    dfeer = (
        df.with_columns(
            pl.col(eer).cut([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], include_breaks=True)
        )
        .unnest(eer)
        .with_columns(pl.col('category').cast(pl.String))
        .group_by('breakpoint', 'category')
        .len()
        .sort('breakpoint', descending=True)
    )

    sns.barplot(dfeer.to_pandas(), x='len', y='category', ax=ax, width=1, alpha=0.8)
    ax.bar_label(ax.containers[0], padding=1)  # type: ignore[arg-type]
    ax.set_xlabel('표본 수')
    ax.set_ylabel('EER 범위')
    ax.set_xlim(0, 90)

    fig.set_size_inches(11 / inch, 4.2 / inch)
    fig.savefig(output / 'EER.png')
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.boxplot(df.filter(pl.col(eer) < 30), x=eer, ax=ax)  # noqa: PLR2004
    fig.set_size_inches(11 / inch, 4.2 / inch)
    fig.savefig(output / 'EER boxplot.png')
    plt.close(fig)


@app['report'].command
def report_temperature(*, conf: Config):
    temp = (
        pl.read_csv(
            next(conf.directory.input.glob('extremum*csv')),
            encoding='korean',
            skip_rows=10,
        )
        .rename(lambda x: x.strip().removesuffix('(℃)'))
        .with_columns(
            pl.col('지점번호').str.strip_chars().cast(pl.UInt16, strict=False),
            pl.col('일시').str.to_date('%Y-%m'),
        )
        .drop_nulls('지점번호')
        .filter(pl.col('일시').dt.year() >= 2020)  # noqa: PLR2004
    )

    (
        utils.MplTheme(palette='tol:bright')
        .grid()
        .tick('x', 'both', direction='in')
        .apply()
    )

    fig, ax = plt.subplots()
    ax.fill_between(
        x=temp.select('일시').to_numpy().ravel(),
        y1=temp.select('최저기온').to_numpy().ravel(),
        y2=temp.select('최고기온').to_numpy().ravel(),
        alpha=0.2,
        color='slategray',
        lw=0,
    )
    ax.fill_between(
        x=temp.select('일시').to_numpy().ravel(),
        y1=temp.select('평균최저기온').to_numpy().ravel(),
        y2=temp.select('평균최고기온').to_numpy().ravel(),
        alpha=0.25,
    )
    sns.lineplot(temp, x='일시', y='평균기온', ax=ax)

    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y년'))
    ax.set_xlabel('')
    ax.set_ylabel('기온 [℃]')

    fig.savefig(conf.directory.etc / '기온 범위.png')
    plt.close(fig)


@app['report'].command
def report_coef(*, conf: Config, min_count: int = 10):
    output = conf.directory.etc
    output.mkdir(exist_ok=True)

    model = (
        pl.scan_parquet(conf.directory.cpr / 'model.parquet')
        .filter(pl.col('용도').list.len() == 1, pl.col('energy') == '합계')
        .with_columns(pl.col('용도').list.get(0))
        .unpivot(
            ['change_point', 'coef', 'r2'],
            index=['건물1', '용도', 'names'],
        )
        .filter(
            ((pl.col('names') != 'Intercept') & (pl.col('variable') == 'r2')).not_()
        )
        .collect()  # XXX lazy 상태에서 아래 replace 오류
        .with_columns(
            pl.col('names').replace_strict({
                'Intercept': '기저부하',
                'HDD': '난방',
                'CDD': '냉방',
            })
        )
        .drop_nulls('value')
        .sort(
            '용도',
            pl.col('names').replace_strict(
                {'기저부하': 0, '냉방': 1, '난방': 2}, return_dtype=pl.Int8
            ),
            'names',
            'variable',
        )
    )

    utils.MplTheme(palette='tol:bright').grid().apply()

    for var, name in [
        ['coef', '모델 계수'],
        ['change_point', '냉난방 시작 온도 [°C]'],
        ['baseline', '기저부하 [MJ/m²]'],
        ['r2', '결정계수'],
    ]:
        fig, ax = plt.subplots()

        if var == 'change_point':
            sns.pointplot(
                model.filter(pl.col('variable') == var)
                .filter(pl.len().over('용도') >= min_count)
                .filter(
                    pl.col('용도').is_in(['문화및집회시설', '자동차관련시설']).not_()
                ),
                x='value',
                y='용도',
                hue='names',
                errorbar=('ci', 90),
                linestyles='none',
                dodge=False,
                ax=ax,
                alpha=0.8,
            )
        elif var == 'baseline':
            sns.barplot(
                model.filter(
                    pl.col('names') == '기저부하',
                    pl.col('variable') == 'coef',
                    pl.col('용도').is_in([
                        '교육연구시설',
                        '노유자시설',
                        '업무시설',
                        '의료시설',
                    ]),
                ),
                x='value',
                y='용도',
                ax=ax,
                estimator='median',
                errorbar=('ci', 90),
            )
        else:
            sns.barplot(
                model.filter(pl.col('variable') == var),
                x='value',
                y='용도',
                hue=None if var == 'r2' else 'names',
                ax=ax,
            )

        if legend := ax.get_legend():
            legend.set_title('')

        ax.set_xlabel(name)
        ax.set_ylabel('')
        fig.savefig(output / f'coef-{var}.png')
        plt.close(fig)


@app['report'].command
def report_monthly_r2(
    *,
    conf: Config,
    y: Literal['norm', 'standardization'] = 'norm',
):
    energy = (
        pl.scan_parquet(conf.directory.root / 'CPR.parquet')
        .group_by('건물1', 'date', 'year', 'month')
        .agg(pl.sum('value'))
    )
    model = (
        pl.scan_parquet(conf.directory.cpr / 'model.parquet')
        .filter(pl.col('names') == 'Intercept', pl.col('energy') == '합계')
        .select('건물1', 'r2')
    )

    df = (
        energy.join(model, on='건물1', how='full')
        .with_columns(
            vmin=pl.min('value').over('건물1', 'year'),
            vmax=pl.max('value').over('건물1', 'year'),
            avg=pl.mean('value').over('건물1', 'year'),
            std=pl.mean('value').over('건물1', 'year'),
        )
        .with_columns(
            norm=(pl.col('value') - pl.col('vmin')) / (pl.col('vmax') - pl.col('vmin')),
            standardization=(pl.col('value') - pl.col('avg')) / pl.col('std'),
            unit=pl.format('{}{}', '건물1', 'year'),
            r2round=pl.col('r2').mul(5).round().truediv(5).round(4),
        )
        .collect()
    )

    utils.MplTheme().grid().apply()
    ykr = '정규화' if y == 'norm' else '표준화'

    fig, ax = plt.subplots()
    sns.lineplot(
        df,
        x='month',
        y=y,
        hue='r2',
        units='unit',
        estimator=None,
        ax=ax,
        alpha=0.1,
        palette=Colormap('crameri:vik_r').to_mpl(),
    )

    ax.set_xlabel('')
    ax.set_ylabel(f'에너지 사용량 ({ykr})')
    ax.xaxis.set_major_locator(MultipleLocator())
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}월'))

    legend = ax.get_legend()
    legend.set_title('$r^2$')
    for h in legend.legend_handles:
        if h is not None:
            h.set_alpha(0.8)

    fig.savefig(conf.directory.etc / f'monthly-r2-{y}.png')
    plt.close(fig)

    grid = (
        sns.relplot(
            df,
            x='month',
            y=y,
            hue='r2',
            col='r2round',
            col_wrap=3,
            kind='line',
            height=2.5,
            units='unit',
            estimator=None,
            palette='crest',
            alpha=0.1,
        )
        .set_titles('$r^2 ≃ {col_name}$')
        .set_axis_labels('', f'에너지 사용량 ({ykr})')
    )

    for ax in grid.axes.ravel():
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}월'))

    if (legend := grid.legend) is not None:
        legend.set_title('$r^2$')
        for h in legend.legend_handles:
            if h is not None:
                h.set_alpha(0.8)

    grid.savefig(conf.directory.etc / f'monthly-r2-grid-{y}.png')
    plt.close(grid.figure)


@app['report'].command
def report_ar_or(
    *,
    conf: Config,
    year: int = 2022,
    max_or: float = 10,
    shade: bool = True,
    height: float = 8,
):
    src = conf.directory.rating / 'Rating-building.parquet'

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
                pl.col('등급용1차소요량[kWh/m2/yr]').gt(260).cast(pl.Int16),
                pl.col('에너지 사용량비').gt(1.5).cast(pl.Int16),
            )
            .replace_strict({
                '00': '1사분면',
                '10': '2사분면',
                '11': '3사분면',
                '01': '4사분면',
            })
            .alias('사분면'),
            highlight=pl.col('건물명').str.contains('서대문소방서'),
        )
        .filter(pl.col('에너지 사용량비') <= max_or)
        .collect()
    )

    rp = RatingPlot(
        df,
        year=year,
        line_ar=260,
        line_or=1.5,
        max_or=max_or,
        shade=shade,
        height=height,
    )

    fig, ax = rp.plot(scatter={'c': 'gray', 'alpha': 0.3})
    sns.scatterplot(
        df.filter(pl.col('highlight')),
        x='등급용1차소요량[kWh/m2/yr]',
        y='에너지 사용량비',
        ax=ax,
        s=100,
        marker='X',
        zorder=3,
    )

    fig.savefig(conf.directory.etc / 'ar-or-example.png')
    plt.close(fig)


if __name__ == '__main__':
    utils.LogHandler.set()

    app()

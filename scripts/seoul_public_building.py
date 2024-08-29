"""2024-08-07 서울시 공공건물 OR 분석 데이터 정리."""

import dataclasses as dc
import tomllib
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import matplotlib.pyplot as plt
import more_itertools as mi
import msgspec
import polars as pl
import polars.selectors as cs
import rich
import seaborn as sns
from cyclopts import App
from loguru import logger
from rich.progress import track
from xlsxwriter import Workbook

from greenbutton import utils
from greenbutton.cpr import ChangePointRegression, Optimized, Optimizer
from scripts.config import dec_hook

if TYPE_CHECKING:
    from matplotlib.axes import Axes

cnsl = rich.get_console()
app = App()


class Config(msgspec.Struct, dict=True):
    root: Path = msgspec.field(name='dir')
    source: str

    @cached_property
    def path(self):
        return self.root / self.source

    @classmethod
    def read(cls, path: str | Path = 'config/config.toml'):
        data = tomllib.loads(Path(path).read_text('UTF-8'))
        return msgspec.convert(
            data['seoul_public_building'], type=cls, dec_hook=dec_hook
        )


Unit = Literal['MJ', 'kcal', 'toe']
RateBy = Literal['building', 'meter']


def unit_conversion(to: Unit):
    # 에너지법 시행 규칙
    # Nm3: 도시가스(LNG) 총발열량 기준

    if to == 'MJ':
        c = {'kWh': 3.6, 'kcal': 4.1868e-3, 'm3': 42.7}
    else:
        # kcal
        c = {'kWh': 860.0, 'kcal': 1.0, 'm3': 10_190.0}

    if to == 'toe':
        c = {k: v * 1e-7 for k, v in c.items()}

    return c | {'Gcal': c['kcal'] * 1e6, '1,000m3': c['m3'] * 1e3, to: 1.0}


@dc.dataclass
class Preprocess:
    conf: Config = dc.field(default_factory=Config.read)

    cpr_unit: Unit = 'MJ'

    or_sheet: str = '★총괄표(OR)★'
    ar_sheet: str = '★총괄표(AR)★'
    stats_dir: str = 'GreenTogether'
    weather_path: str = 'ta_20240808133553.xlsx'

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
        df = pl.read_excel(
            self.conf.path,
            sheet_name=self.ar_sheet,
            read_options={'skip_rows': 1},
            columns=[0, 1, 2, 48],
        )

        var = '등급용1차소요량'
        df.columns = ['사업연도', '연번', '건물명', f'{var}[kWh/m2/yr]']

        k = unit_conversion('toe')
        return df.with_columns(
            pl.col(f'{var}[kWh/m2/yr]').replace({0: None})
        ).with_columns(
            pl.col('연번').cast(pl.Int64),
            pl.col(f'{var}[kWh/m2/yr]').mul(k['kWh']).alias(f'{var}[toe/m2/yr]'),
        )

    @cached_property
    def stats(self):
        root = self.conf.root / self.stats_dir

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
        src = self.conf.root / self.weather_path
        return pl.read_excel(src).select(
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
    dfar.head(100).write_excel(conf.root / 'AR-sample.xlsx')

    dfst = prep.stats
    dfst.write_parquet(conf.root / '통계-용도별.parquet')
    dfst.write_excel(conf.root / '통계-용도별.xlsx')

    prep.weather.write_parquet(conf.root / 'weather.parquet')

    prep.cpr.write_parquet(conf.root / 'CPR.parquet')
    prep.cpr.sample(100).write_excel(conf.root / 'CPR-sample.xlsx')


@dc.dataclass
class Rating:
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
        uc = pl.col('unit').replace_strict(
            unit_conversion('toe'), return_dtype=pl.Float64
        )

        return (
            tidy.join(na, on=na.select(pl.all().exclude('전력NA')).columns, how='left')
            .with_columns(k=uc)
            .with_columns(
                pl.col('value').mul('k').truediv('연면적').alias('EUI[toe/m2]'),
            )
            .drop('k')
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
            .group_by(cs.exclude('EUI[toe/m2]'))
            .sum()
            .rename({'EUI[toe/m2]': 'EUI[toe/m2/yr]'})
            .sort('index')
            .with_columns(
                pl.col('EUI[toe/m2/yr]')
                .truediv('AvgEUI[toe/m2/yr]')
                .alias('에너지 사용량비')
            )
        )

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

    dst = conf.root / 'Rating'
    dst.mkdir(exist_ok=True)

    rating.operational_monthly.write_parquet(dst / 'OperationalMonthly.parquet')
    rating.operational_monthly.drop_nulls(cs.contains('EUI')).head(100).write_excel(
        dst / 'OperationalMonthly-sample.xlsx'
    )

    rating.operational_yearly.write_parquet(dst / 'OperationalYearly.parquet')
    rating.operational_yearly.write_excel(dst / 'OperationalYearly.xlsx')

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
    src = conf.root / f'Rating/Rating-{by}.parquet'

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


@app.command
def cpr(*, plot: bool = True):
    conf = Config.read()

    src = conf.root / 'CPR.parquet'
    dst = conf.root / 'CPR'
    dst.mkdir(exist_ok=True)

    data = (
        pl.scan_parquet(src, glob=False).drop(cs.starts_with('original'), 'k').collect()
    )
    usage = data.select('건물1', '건물명', '용도')

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

        with Workbook(dst / f'[model] {bldg1}.xlsx') as wb:
            model.write_excel(wb, worksheet='model')
            cr.data.write_excel(wb, worksheet='data')

        if plot:
            fig, _ = cr.plot()
            fig.savefig(dst / f'[plot] {bldg1}.png')
            plt.close(fig)

    model = pl.concat(models)
    model = model.select(
        mi.unique_everseen(['건물1', '건물명', '용도', 'energy', *model.columns])
    )
    model.write_excel(conf.root / 'CPR/(models).xlsx')


if __name__ == '__main__':
    utils.set_logger()

    app()

import dataclasses as dc
import functools
import re
from itertools import starmap
from pathlib import Path
from typing import Literal

import cyclopts
import more_itertools as mi
import polars as pl
import rich
import seaborn as sns
import structlog
from matplotlib.figure import Figure
from rich.progress import track
from sklearn.neighbors import LocalOutlierFactor

from greenbutton import misc, utils
from greenbutton.utils import tqdm
from greenbutton.utils.cli import App

type OutlierDetection = Literal['LOF', 'IQR'] | None

P_FILENAME = re.compile(r'(?P<building>.*?)_(?P<type>공공|다소비)')
P_ENERGY = re.compile(r'^(전력|열)_/d+')

logger = structlog.stdlib.get_logger()
app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml', root_keys='hybrid', use_commands_as_keys=False
    )
)


@dc.dataclass
class _Building:
    src: str | Path
    datetime: str = '검침일시'

    def _filename(self):
        stem = Path(self.src).stem
        if (m := P_FILENAME.match(stem)) is None:
            raise ValueError(self.src)

        return m.groupdict()

    def _sheet(self, name: str, df: pl.DataFrame):
        if (m := P_ENERGY.match(name)) is None:
            raise ValueError(name)

        energy = m.group(1)
        consumption = f'{energy}사용량(kWh)'
        return df.select(
            pl.col(self.datetime).alias('datetime'),
            pl.lit(energy).alias('energy'),
            pl.col(consumption).alias('consumption').cast(pl.Float64),
        )

    def __call__(self):
        name = self._filename()
        return (
            pl
            .concat(starmap(self._sheet, pl.read_excel(self.src, sheet_id=0).items()))
            .select(*(pl.lit(v).alias(k) for k, v in name.items()), pl.all())
            .with_columns(
                pl
                .col('datetime')
                .str.extract_groups(r'(?P<date>/d{8})(?P<hour>/d{2})')
                .alias('group')
            )
            .unnest('group')
            .with_columns(
                pl.col('date').str.to_date('%Y%m%d'),
                pl.col('hour').cast(pl.Int8),
            )
            .with_columns(
                date=pl.col('date')
                + pl.col('hour').replace_strict({24: 1}, default=0)
                * pl.duration(days=1),
                hour=pl.col('hour').replace({24: 0}),
            )
            .with_columns(pl.col('date').dt.combine(pl.time('hour')).alias('datetime'))
            .group_by('building', 'type', 'datetime', 'energy')
            .agg(pl.sum('consumption'))
            .sort('datetime', 'energy')
        )


@dc.dataclass(frozen=True)
class _Paths:
    public_institution: Path
    energy_intensive: Path


@app.command
@dc.dataclass
class Prep:
    root: Path

    k: float = 5.0
    min_eui: float = 0.05  # 이상치 판단을 위한 최소치

    building_info: _Paths = _Paths(
        public_institution=Path('../PublicInstitution/0001.data/1.기관.parquet'),
        energy_intensive=Path('../EnergyIntensive/0001.data/building.parquet'),
    )
    asos: _Paths = _Paths(
        public_institution=Path('../ASOS/공공기관.xlsx'),
        energy_intensive=Path('../ASOS/다소비사업장.xlsx'),
    )

    def read_raw(self):
        if (cache := self.root / '00.raw.parquet').exists():
            return pl.read_parquet(cache)

        src = list(self.root.glob('00.raw/consumption/*.xlsx'))

        data = pl.concat(_Building(it)() for it in tqdm(src))
        data.write_parquet(cache)

        return data

    def read_weather(self):
        if (cache := self.root / '00.weather.parquet').exists():
            return pl.read_parquet(cache)

        src = mi.one(self.root.glob('00.raw/OBS_ASOS_*.csv'))
        cols = {
            '지점': 'asos-station-code',
            '일시': 'date',
            '평균기온(°C)': 'Te',
            '평균 증기압(hPa)': 'Pv',
            '합계 일사량(MJ/m2)': 'I',
        }
        return (
            pl
            .read_csv(src, encoding='korean')
            .rename(cols)
            .select(*cols.values())
            .with_columns(
                pl.col('date').str.to_date(),
                pl.col('asos-station-code').cast(pl.UInt16),
            )
        )

    def read_building_info(self):
        if (cache := self.root / '00.bldg.parquet').exists():
            data = pl.read_parquet(cache)
        else:
            cols = {
                '기관명': 'building',
                '기관대분류': 'subtype',
                '건물용도': 'use',
                '연면적': 'GFA',
            }
            pi = (
                pl
                .scan_parquet(self.root / self.building_info.public_institution)
                .rename(cols)
                .select(cols.values())
                .unique()
                .with_columns(pl.lit('공공').alias('type'))
                .collect()
            )
            cols = {'업체명': 'building', 'KEMC_KOR': 'subtype', '연면적(m²)': 'GFA'}
            ei = (
                pl
                .scan_parquet(self.root / self.building_info.energy_intensive)
                .rename(cols)
                .select(cols.values())
                .unique()
                .with_columns(pl.lit('다소비').alias('type'))
                .collect()
            )

            data = (
                pl
                .concat([pi, ei], how='diagonal')
                .group_by(['type', 'subtype', 'use', 'building'])
                .agg(pl.median('GFA'))  # NOTE: 신고 연도별로 연면적이 다른 경우 있음
            )
            data.write_parquet(cache)
            data.write_excel(cache.with_suffix('.xlsx'))

        return data

    def read_asos_station(self):
        if (cache := self.root / '00.asos-station.parquet').exists():
            data = pl.read_parquet(cache)
        else:
            src = dc.asdict(self.asos)
            cols = {
                '건물명': 'building',
                '기상청 지점 코드': 'asos-station-code',
                '기상청 지점명': 'asos-station',
            }
            data = (
                pl
                .concat(
                    [
                        pl.read_excel(self.root / v).with_columns(
                            pl.lit(k).alias('type')
                        )
                        for k, v in src.items()
                    ],
                    how='diagonal',
                )
                .rename(cols)
                .select('type', *cols.values())
                .with_columns(
                    pl.col('type').replace_strict({
                        'public_institution': '공공',
                        'energy_intensive': '다소비',
                    }),
                    pl.col('asos-station-code').cast(pl.Int64),
                )
                .unique()
            )
            data.write_parquet(cache)
            data.write_excel(cache.with_suffix('.xlsx'))

        # NOTE 누락 건물 수동 지정
        manual = (
            {
                'type': '공공',
                'building': '국민연금공단',
                # 전라북도 전주시 덕진구 기지로 180 (만성동)
                'asos-station': '전주(146)',
                'asos-station-code': 146,
            },
            {
                'type': '다소비',
                'building': '이지스제103호전문투자형사모부동산투자'
                '유한회사(G.Square빌딩',
                # 서울특별시 영등포구 여의공원로 115 (여의도동)
                'asos-station': '서울(108)',
                'asos-station-code': 108,
            },
        )
        return (
            pl
            .concat([data, pl.from_dicts(manual)], how='diagonal')
            .with_columns(pl.col('asos-station-code').cast(pl.UInt16))
            .with_columns()
        )

    def detect_outlier(self, data: pl.DataFrame):
        group = ['type', 'building', 'energy']
        transformed = '_transformed'
        expr = pl.col(transformed)
        return (
            data
            .with_columns(
                pl
                .when(pl.col('EUI') < self.min_eui)
                .then(pl.lit(None))
                .otherwise(pl.col('EUI').sqrt())
                .alias(transformed)
            )
            .with_columns(
                _median=expr.median().over(group),
                _iqr=(
                    expr.quantile(0.75, interpolation='linear')
                    - expr.quantile(0.25, interpolation='linear')
                ).over(group),
                _k=pl
                .col('building')
                .str.contains('아파트')
                .replace_strict(
                    {True: 2 * self.k, False: self.k}, return_dtype=pl.Float64
                ),
            )
            .with_columns(
                outlier_iqr=(pl.col('EUI') >= self.min_eui)
                & expr.is_between(
                    pl.col('_median') - pl.col('_k') * pl.col('_iqr'),
                    pl.col('_median') + pl.col('_k') * pl.col('_iqr'),
                ).not_()
            )
        )

    def __call__(self):
        cols = (
            'type',
            'subtype',
            'building',
            'use',
            'GFA',
            'asos-station',
            'asos-station-code',
            'datetime',
            'holiday',
            'energy',
            'consumption',
        )
        index = ('type', 'building')
        data = (
            self
            .read_raw()
            .with_columns(
                pl.col('building').replace({
                    '서울시설공단 서울월드컵경기장': '서울시설관리공단서울월드컵경기장',
                })
            )
            .join(self.read_building_info(), on=index, how='left')
            .join(self.read_asos_station(), on=index, how='left')
        )

        years = data['datetime'].dt.year().unique().to_list()
        data = (
            (data)
            .with_columns(
                misc.is_holiday(pl.col('datetime'), years=years).alias('holiday')
            )
            .select(*cols, pl.all().exclude(cols))
        )

        data.write_parquet(self.root / '01.consumption.hour.parquet')

        # daily
        weather = self.read_weather()
        group = set(data.columns) - {'datetime', 'consumption'}
        daily = (
            data
            .group_by_dynamic('datetime', every='1d', group_by=group)
            .agg(pl.sum('consumption'))
            .rename({'datetime': 'date'})
            .with_columns(
                pl.col('date').dt.date(),
                pl.col('consumption').truediv('GFA').alias('EUI'),
            )
            .join(weather, on=['asos-station-code', 'date'], how='left', validate='m:1')
        )
        daily = self.detect_outlier(daily)
        daily.write_parquet(self.root / '01.consumption.day.parquet')

        bldg = (
            data
            .drop('datetime', 'holiday', 'energy', 'consumption')
            .unique()
            .sort(pl.all())
        )

        if (
            d := bldg.filter(pl.concat_str(index, separator=':').is_duplicated())
        ).height:
            logger.warning('Duplicated buildings')
            rich.print(d)

        bldg.write_parquet(self.root / '01.bldg.parquet')
        bldg.write_excel(self.root / '01.bldg.xlsx', column_widths=120)
        (
            bldg
            .drop('asos-station')
            .rename({'GFA': 'GFA [m²]', 'asos-station-code': 'ASOS station'})
            .rename(lambda x: f'[{x}]')
            .with_columns(pl.format('[{}]', pl.all().fill_null('-')))
            .write_csv(self.root / '01.bldg.table.csv', include_bom=True)
        )


@app.command
@dc.dataclass
class LOF:
    root: Path
    kwargs: dict = dc.field(default_factory=dict)

    def detect(self, data: pl.DataFrame):

        def z(s: str):
            expr = pl.col(s)
            return ((expr - expr.mean()) / expr.std()).alias(s)

        array = (
            (data)
            .select(
                z('EUI'),
                (
                    pl
                    .col('date')
                    .dt.month()
                    .replace_strict({6: 1, 7: 1, 8: 1, 12: 2, 1: 2, 2: 2}, default=0)
                    + 0.01 * pl.col('date').dt.day()
                ),
                pl.col('energy').replace_strict(
                    {'전력': 0, '열': 1}, return_dtype=pl.Float64
                ),
                pl.col('holiday').cast(pl.Float64),
            )
            .to_numpy()
        )
        label = LocalOutlierFactor(**self.kwargs).fit_predict(array)
        return data.with_columns(pl.Series('outlier_lof', label.ravel() == -1))

    def __call__(self):
        data = (
            pl
            .scan_parquet(self.root / '01.consumption.day.parquet')
            .drop_nulls(['EUI', 'Te', 'holiday'])
            .collect()
        )
        data = (
            pl
            .concat([self.detect(df) for _, df in data.group_by('type', 'building')])
            .sort('type', 'building', 'date')
            .with_columns()
        )
        data.write_parquet(self.root / '01.consumption.day.LOF.parquet')


@app.command
@dc.dataclass
class Plot:
    root: Path

    @functools.cached_property
    def output_line(self):
        output = self.root / '02.lineplot'
        output.mkdir(exist_ok=True)
        return output

    @functools.cached_property
    def output_scatter(self):
        output = self.root / '02.scatter'
        output.mkdir(exist_ok=True)
        return output

    def lineplot(
        self,
        data: pl.DataFrame,
        value: Literal['consumption', 'EUI'],
        *,
        outlier_detection: OutlierDetection = None,
    ):
        data = data.drop_nulls(value)

        if outlier_detection is not None:
            data = data.filter(pl.col(f'outlier_{outlier_detection.lower()}').not_())

        if not data.height:
            return

        type_ = data['type'][0]
        building = data['building'][0]

        fig = Figure()
        ax = fig.subplots()
        utils.mpl.lineplot_break_nans(
            data,
            x='date',
            y=value,
            hue='energy',
            hue_order=['전력', '열'],
            ax=ax,
            alpha=0.8,
        )

        if outlier_detection is None:
            sns.scatterplot(
                data.filter(pl.col('outlier_detection') != 'Inlier'),
                x='date',
                y=value,
                hue='energy',
                hue_order=['전력', '열'],
                style='holiday',
                ax=ax,
            )
            ax.set_yscale('symlog')

        ax.set_title(f'[{type_}] {building}', loc='left', weight=500)
        ax.legend(title='')
        ax.set_xlabel('')
        ax.set_ylabel('Consumption [kWh]' if value == 'consumption' else 'EUI [kWh/m²]')

        fig.savefig(
            self.output_line / f'{outlier_detection}.{value}.{type_}.{building}.png'
        )

    def scatter(
        self,
        data: pl.DataFrame,
        *,
        outlier_detection: OutlierDetection = None,
    ):
        data = data.filter(pl.col('holiday').not_()).drop_nulls(('EUI', 'Te'))

        if outlier_detection is not None:
            data = data.filter(pl.col(f'outlier_{outlier_detection.lower()}').not_())

        if not data.height:
            return

        type_ = data['type'][0]
        building = data['building'][0]

        fig = Figure()
        ax = fig.subplots()
        if outlier_detection is None:
            ax.set_yscale('symlog')

        sns.scatterplot(
            data,
            x='Te',
            y='EUI',
            hue='energy',
            hue_order=['전력', '열'],
            style='outlier_detection' if outlier_detection is None else None,
            style_order=['Inlier', 'IQR+LOF', 'IQR', 'LOF']
            if outlier_detection is None
            else None,
            ax=ax,
            alpha=0.5,
        )

        ax.set_title(f'[{type_}] {building}', loc='left', weight=500)
        ax.legend(title='')
        ax.set_xlabel('External Temperature [°C]')
        ax.set_ylabel('EUI [kWh/m²]')

        fig.savefig(self.output_scatter / f'{outlier_detection}.{type_}.{building}.png')

    def __call__(self):
        utils.mpl.MplTheme('paper').grid().apply()
        utils.mpl.MplConciseDate().apply()

        data = (
            pl
            .scan_parquet(self.root / '01.consumption.day.LOF.parquet')
            .with_columns(
                pl
                .concat_str(
                    pl.col('outlier_iqr').replace_strict({False: '', True: 'IQR'}),
                    pl.col('outlier_lof').replace_strict({False: '', True: 'LOF'}),
                    separator='+',
                )
                .str.strip_chars('+')
                .replace({'': 'Inlier'})
                .alias('outlier_detection')
            )
            .collect()
        )

        # EUI dist
        fig = Figure()
        ax = fig.subplots()
        sns.histplot(data.filter(pl.col('EUI') > 0), x='EUI', ax=ax, log_scale=True)
        fig.savefig(self.root / '02.EUI-dist.png')

        total = data.select('type', 'building').unique().height

        for _, df in track(data.group_by('type', 'building'), total=total):
            for od in ('IQR', 'LOF', None):
                self.scatter(df, outlier_detection=od)

            self.lineplot(df, 'EUI', outlier_detection='IQR')
            self.lineplot(df, 'EUI', outlier_detection='LOF')


if __name__ == '__main__':
    app()

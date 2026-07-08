import dataclasses as dc
import functools
import itertools
import re
from itertools import starmap
from pathlib import Path

import cyclopts
import more_itertools as mi
import polars as pl
import seaborn as sns
import structlog
from matplotlib.figure import Figure

from greenbutton import misc, utils
from greenbutton.utils import tqdm
from greenbutton.utils.cli import App

P_FILENAME = re.compile(r'(?P<building>.*?)_(?P<type>공공|다소비)')
P_ENERGY = re.compile(r'^(전력|열)_\d+')

# 누락 건물 수동 지정
ASOS_STATION = (
    {
        'bldg.type': '공공',
        'bldg': '국민연금공단',
        # 전라북도 전주시 덕진구 기지로 180 (만성동)
        'asos.station': 146,
    },
    {
        'bldg.type': '다소비',
        'bldg': '이지스제103호전문투자형사모부동산투자유한회사(G.Square빌딩',
        # 서울특별시 영등포구 여의공원로 115 (여의도동)
        'asos.station': 108,
    },
)

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
        sheets = pl.read_excel(self.src, sheet_id=0).items()
        return (
            pl
            .concat(starmap(self._sheet, sheets))
            .select(*(pl.lit(v).alias(k) for k, v in name.items()), pl.all())
            .with_columns(
                pl
                .col('datetime')
                .str.extract_groups(r'(?P<date>\d{8})(?P<hour>\d{2})')
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
class Parse:
    """원본 데이터 해석, parquet 변환."""

    root: Path

    building_info: _Paths = _Paths(
        public_institution=Path('../PublicInstitution/0001.data/1.기관.parquet'),
        energy_intensive=Path('../EnergyIntensive/0001.data/building.parquet'),
    )
    asos: _Paths = _Paths(
        public_institution=Path('../ASOS/공공기관.xlsx'),
        energy_intensive=Path('../ASOS/다소비사업장.xlsx'),
    )

    def read_consumption(self):
        if (cache := self.root / '00.consumption.parquet').exists():
            return pl.read_parquet(cache)

        src = list(self.root.glob('00.raw/consumption/*.xlsx'))

        r = {
            '서울시설공단 서울월드컵경기장': '서울시설관리공단서울월드컵경기장',
        }
        data = (
            pl
            .concat(_Building(it)() for it in tqdm(src))
            .with_columns(pl.col('building').replace(r))
            .with_columns()
        )
        data.write_parquet(cache)

        return data

    def read_weather(self):
        if (cache := self.root / '00.weather.parquet').exists():
            return pl.read_parquet(cache)

        src = mi.one(self.root.glob('00.raw/OBS_ASOS_*.csv'))
        cols = {
            '지점': 'asos.station',
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
                pl.col('asos.station').cast(pl.UInt16),
            )
        )

    def read_building_info(self):
        if (cache := self.root / '00.bldg.parquet').exists():
            data = pl.read_parquet(cache)
        else:
            cols = {
                '기관명': 'bldg',
                '기관대분류': 'bldg.subtype',
                '건물용도': 'use',
                '연면적': 'gfa',
            }
            pi = (
                pl
                .scan_parquet(self.root / self.building_info.public_institution)
                .rename(cols)
                .select(cols.values())
                .unique()
                .with_columns(pl.lit('공공').alias('bldg.type'))
                .collect()
            )
            cols = {'업체명': 'bldg', 'KEMC_KOR': 'bldg.subtype', '연면적(m²)': 'gfa'}
            ei = (
                pl
                .scan_parquet(self.root / self.building_info.energy_intensive)
                .rename(cols)
                .select(cols.values())
                .unique()
                .with_columns(pl.lit('다소비').alias('bldg.type'))
                .collect()
            )

            data = (
                pl
                .concat([pi, ei], how='diagonal')
                .group_by(['bldg.type', 'bldg.subtype', 'use', 'bldg'])
                .agg(pl.median('gfa'))  # NOTE: 신고 연도별로 연면적이 다른 경우 있음
            )
            data.write_parquet(cache)
            data.write_excel(cache.with_suffix('.xlsx'))

        return data

    def read_asos_station(self):
        if (cache := self.root / '00.asos.station.parquet').exists():
            data = pl.read_parquet(cache)
        else:
            src = dc.asdict(self.asos)
            cols = {
                '건물명': 'bldg',
                '기상청 지점 코드': 'asos.station',
                '기상청 지점명': 'asos.agency',
            }
            data = (
                pl
                .concat(
                    [
                        pl.read_excel(self.root / v).with_columns(
                            pl.lit(k).alias('bldg.type')
                        )
                        for k, v in src.items()
                    ],
                    how='diagonal',
                )
                .rename(cols)
                .select('bldg.type', *cols.values())
                .with_columns(
                    pl.col('bldg.type').replace_strict({
                        'public_institution': '공공',
                        'energy_intensive': '다소비',
                    }),
                    pl.col('asos.station').cast(pl.UInt16),
                )
                .unique()
            )
            data.write_parquet(cache)
            data.write_csv(cache.with_suffix('.csv'))

        manual = (
            pl
            .from_dicts(ASOS_STATION)
            .with_columns(pl.col('asos.station').cast(pl.UInt16))
            .with_columns()
        )
        return pl.concat([data, manual], how='diagonal')

    def __call__(self):
        index = ('bldg.type', 'bldg')
        data = (
            self
            .read_consumption()
            .rename({'building': 'bldg', 'type': 'bldg.type'}, strict=False)
            .join(self.read_building_info(), on=index, how='left')
            .join(self.read_asos_station(), on=index, how='left')
            .sort(index)
        )

        cases = (
            data
            .select(index)
            .unique()
            .sort(pl.all())
            .with_row_index('bldg.index')
            .with_columns(
                pl.concat_str(
                    pl.col('bldg.index').cast(pl.String).str.zfill(4),
                    'bldg.type',
                    'bldg',
                    separator='.',
                ).alias('bldg.case')
            )
        )

        cols = (
            'bldg.index',
            'bldg.type',
            'bldg.subtype',
            'bldg',
            'bldg.case',
            'use',
            'gfa',
            'asos.station',
            'asos.agency',
            'datetime',
            'holiday',
            'energy',
            'consumption',
        )
        years = data['datetime'].dt.year().unique().to_list()
        data = (
            data
            .join(cases, on=index, how='left')
            .with_columns(
                misc.is_holiday(pl.col('datetime'), years=years).alias('holiday')
            )
            .select(*cols, pl.all().exclude(cols))
        )

        data.write_parquet(self.root / '01.raw.parquet')

        weather = self.read_weather()
        weather.write_parquet(self.root / '01.weather.parquet')

        group = tuple(x for x in data.columns if x not in {'datetime', 'consumption'})
        (
            data
            .sort('datetime')
            .group_by_dynamic('datetime', every='1d', group_by=group)
            .agg(pl.sum('consumption'))
            .with_columns(
                pl.col('datetime').dt.date(),
                pl.col('consumption').truediv('gfa').alias('eui'),
            )
            .rename({'datetime': 'date'})
            .join(weather, on=['date', 'asos.station'], how='left', validate='m:1')
            .sort(pl.all())
            .write_parquet(self.root / '01.raw.daily.parquet')
        )


@app.command
@dc.dataclass
class RawDist:
    """시간별 데이터 분포, 이상치 제거 기준 확인."""

    root: Path

    @functools.cached_property
    def data(self):
        return (
            pl
            .scan_parquet(self.root / '01.raw.parquet')
            .with_columns(
                pl.col('datetime').dt.weekday().alias('weekday'),
                pl.col('datetime').dt.hour().alias('hour'),
                pl.col('consumption').truediv('gfa').alias('eui'),
            )
            .with_columns(
                pl
                .col('consumption')
                .median()
                .over(['bldg.case', 'weekday', 'hour'])
                .alias('median'),
            )
            .with_columns(pl.col('consumption').truediv('median').alias('ratio'))
            .collect()
        )

    @functools.cached_property
    def output(self):
        output = self.root / '01.raw.dist'
        output.mkdir(exist_ok=True)
        return output

    def plot(self, e: str, v: str, *, ylog: bool):
        data = self.data.filter(pl.col('energy') == e)

        fig = Figure()
        ax = fig.subplots()

        sns.histplot(data, x=v, ax=ax, log_scale=True)

        if ylog:
            ax.set_yscale('symlog')

        fig.savefig(self.output / f'{e}.{v}.{ylog=}.png')

    def __call__(self):
        utils.mpl.MplTheme().grid().tick(direction='in').apply()
        for e, v, yl in itertools.product(
            ('전력', '열'),
            ('consumption', 'eui', 'ratio'),
            (False, True),
        ):
            self.plot(e, v, ylog=yl)


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    utils.mpl.MplConciseDate().apply()
    app()

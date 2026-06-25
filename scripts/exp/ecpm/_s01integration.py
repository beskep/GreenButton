"""eCPM 분석을 위한 실증지 데이터 통합.

- 2026-03-25 한전파주, 한국에너지공단, 산기평 세 건물 우선 통합
- 2026-06-25 deprecated
"""

import dataclasses as dc
import functools
import itertools
from pathlib import Path
from typing import ClassVar, Literal

import cyclopts
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import structlog
from matplotlib.figure import Figure

import scripts.exp.experiment as exp
from greenbutton import misc, utils
from greenbutton.utils.cli import App

type Delta = Literal['day', 'hour']
type Building = Literal['KEPCO', 'KEA']
type Period = Literal['summer', 'winter', 'spring-autumn'] | None

BUILDINGS: tuple[Building, ...] = ('KEPCO', 'KEA')
PERIODS: tuple[Period, ...] = ('summer', 'winter', 'spring-autumn', None)

logger = structlog.stdlib.get_logger()
app = App(
    config=[
        cyclopts.config.Toml(f'config/{x}.toml', use_commands_as_keys=False)
        for x in ['.experiment', 'experiment']
    ],
)


@dc.dataclass
class Dirs(exp.Dirs):
    db_raw: Path = Path('02.database/raw')
    pmv: Path = Path('04.PMV')


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'ecpm'

    @functools.cached_property
    def dirs(self):
        return Dirs(self.root / self.buildings[self.BUILDING])

    def bldg_dirs(self, bldg: str):
        return exp.Dirs(root=self.root / self.buildings[bldg])


@app.command
@dc.dataclass(frozen=True)
class Kepco:
    conf: Config

    def get_db_data(self, delta: Delta):
        d = self.conf.bldg_dirs('kepco_paju').database / '0002.binary'

        src = [
            *d.glob(f'**/*T_BELO_FACILITY_{delta.upper()}*.parquet'),
            *d.glob(f'**/*T_BELO_ELEC_{delta.upper()}*.parquet'),
        ]
        lfs = [
            pl
            .scan_parquet(x)
            .with_columns(pl.col('tagValue').cast(pl.Float64))
            .select(
                pl.col('updateDate').alias('datetime'),
                pl.col('[tagName]').alias('variable'),
                pl.col('tagValue').cast(pl.Float64).alias('value'),
            )
            for x in src
        ]

        data = (
            pl
            .concat(lfs)
            .filter(
                pl.col('variable').is_in([
                    '실외온습도계.외기온도',
                    '실외온습도계.실내온도',
                    '전기.전체전력량',
                ])
            )
            .with_columns(
                pl.col('variable').replace_strict({
                    '실외온습도계.외기온도': 'temp_external',
                    '실외온습도계.실내온도': 'temp_internal',
                    '전기.전체전력량': 'electricity_kWh',
                })
            )
            .unique(['datetime', 'variable'])
            .collect()
        )
        holiday = misc.is_holiday(pl.col('datetime'), years=list(range(2020, 2027)))
        return (
            (data)
            .pivot('variable', index='datetime', values='value', sort_columns=True)
            .sort('datetime')
            .with_columns(holiday.alias('holiday'))
        )

    def __call__(self):
        self.conf.dirs.mkdir()

        for d in ('hour', 'day'):
            logger.info('delta=%s', d)

            data = self.get_db_data(d)
            path = self.conf.dirs.database / f'01.KEPCO-BEMS-{d}.parquet'
            data.write_parquet(path)
            data.write_excel(path.with_suffix('.xlsx'), column_widths=150)


@app.command
@dc.dataclass(frozen=True)
class Kea:
    conf: Config
    iid: str = 'DB_B7AE8782-40AF-8EED-E050-007F01001D51'

    TI_EXCLUDE: ClassVar[tuple[str, ...]] = (
        '지하',
        '계단',
        '서고',
        '체력단련실',
        '탁구실',
        '6층',
        '7층',
    )

    def energy(self):
        return (
            (pl)
            .scan_parquet(self.conf.dirs.db_raw / f'AMI_{self.iid}.parquet')
            .select('datetime', pl.col('사용량').alias('electricity_kWh'))
        )

    def environment(self):
        d = self.conf.bldg_dirs('kea').database / '03.parsed'
        files = [*d.glob('*실내온도.parquet'), *d.glob('*실내습도.parquet')]
        return (
            pl
            .scan_parquet(files, include_file_paths='path')
            .select(
                'datetime',
                pl
                .col('path')
                .str.extract(r'.*?_(.* 실내[온습]도)\.parquet')
                .alias('variable'),
                pl.col('parsed_value').alias('value'),
            )
            .with_columns(
                pl.col('variable').str.extract_groups(
                    r'(?<location>.*?)\s+(?<variable>실내[온습]도)'
                )
            )
            .unnest('variable')
        )

    @classmethod
    def ti(cls, data: pl.DataFrame):
        return data.filter(
            pl.col('variable') == '실내온도',
            pl.col('value') > 1,
            ~pl.col('location').str.contains_any(list(cls.TI_EXCLUDE)),
        )

    def weather(self):
        paths = list(self.conf.dirs.db_raw.glob('ASOS_울산/*.csv'))
        return (
            pl
            .concat(
                pl.read_csv(x, encoding='korean', infer_schema=False) for x in paths
            )
            .select(
                pl.col('일시').str.to_datetime().alias('datetime'),
                pl.col('기온(°C)').cast(pl.Float64).alias('temp_external'),
            )
            .sort('datetime')
        )

    def __call__(self):
        d = self.conf.dirs.database

        environment = self.environment()
        raw = (
            environment
            .sort('datetime')
            .group_by_dynamic('datetime', every='1h', group_by=['location', 'variable'])
            .agg(pl.median('value'))
            .collect()
        )
        raw.write_parquet(d / '02.KEA-BEMS-raw-hourly.parquet')

        ti = self.ti(environment)
        energy = self.energy().collect()
        weather = self.weather()

        for delta in ('hour', 'day'):
            logger.info('delta=%s', delta)

            if delta == 'hour':
                e = energy
                w = weather
            else:
                e = (
                    (energy)
                    .group_by_dynamic('datetime', every='1d')
                    .agg(pl.all().sum())
                )
                w = (
                    (weather)
                    .group_by_dynamic('datetime', every='1d')
                    .agg(pl.mean('temp_external'))
                )

            t = (
                ti
                .sort(pl.col('datetime'))
                .group_by_dynamic(
                    'datetime',
                    every={'day': '1d', 'hour': '1h'}[delta],
                )
                .agg(pl.col('value').median().alias('temp_internal'))
                .collect()
            )

            holiday = misc.is_holiday(pl.col('datetime'), years=list(range(2019, 2027)))
            data = (
                e
                .join(t, on='datetime', how='full', coalesce=True)
                .join(w, on='datetime', how='left', coalesce=True)
                .sort('datetime')
                .with_columns(holiday.alias('holiday'))
            )
            data.write_parquet(d / f'02.KEA-BEMS-{delta}.parquet')
            data.write_excel(d / f'02.KEA-BEMS-{delta}.xlsx', column_widths=150)


@app.command
@dc.dataclass(frozen=True)
class Keit:
    conf: Config

    def temperature(self):
        return (
            pl
            .read_csv(
                self.conf.dirs.db_raw / 'OBS_ASOS_DD_대구_143.csv',
                encoding='korean',
                infer_schema=False,
            )
            .select(
                pl.col('일시').str.to_date().alias('date'),
                pl.col('평균기온(°C)').cast(pl.Float64).alias('temp_external'),
            )
            .sort('date')
        )

    def _heat(self, t: Literal['cooling', 'heating'], /):
        kr = {'cooling': '냉방', 'heating': '난방'}[t]
        return (
            pl
            .scan_parquet(self.conf.dirs.db_raw / f'KEIT_{kr}적산열량계.parquet')
            .select(
                'date',
                pl.col('금일지침-Gcal').alias(f'{t}_Gcal'),
                pl.col('금일지침-톤').alias(f'{t}_ton'),
            )
            .collect()
        )

    def energy(self):
        return (
            pl
            .scan_parquet(self.conf.dirs.db_raw / 'KEIT_전력.parquet')
            .drop_nulls('사용량')
            .select(
                'date',
                pl.sum_horizontal('사용량', '태양광발전량').alias('electricity_kWh'),
            )
            .collect()
            .join(self._heat('cooling'), on='date', how='full', coalesce=True)
            .join(self._heat('heating'), on='date', how='full', coalesce=True)
        )

    def __call__(self):
        holiday = misc.is_holiday(pl.col('date'), years=list(range(2016, 2027)))
        data = (
            self
            .energy()
            .join(self.temperature(), on='date', how='left')
            .with_columns(holiday.alias('holiday'))
        )
        data.write_parquet(self.conf.dirs.database / '03.KEIT.parquet')
        data.write_excel(self.conf.dirs.database / '03.KEIT.xlsx', column_widths=150)


@app.command
def energy(conf: Config):
    Kepco(conf)()
    Kea(conf)()
    Keit(conf)()


@app.command
@dc.dataclass(frozen=True)
class Pmv:
    conf: Config

    @staticmethod
    def _prep(path: Path):
        output = path.parent
        stem = path.stem

        date = 'measurement_start_date'
        data = pl.read_parquet(path).rename({'date': date})

        for (d,), df in data.group_by(date):
            logger.info(d)

            df.write_excel(output / f'{stem}_{d}.xlsx', column_widths=125)
            (
                df
                .with_columns(
                    col=pl.format(
                        '{} [{}]', 'variable', pl.col('unit').fill_null('')
                    ).str.strip_suffix(' []')
                )
                .pivot(
                    'col',
                    index=['space', 'floor', 'datetime'],
                    values='value',
                    sort_columns=True,
                )
                .drop_nulls('PMV')
                .sort(['space', 'floor', 'datetime'])
                .write_excel(output / f'{stem}_{d}_wide.xlsx', column_widths=125)
            )

    def __call__(self):
        for path in self.conf.dirs.pmv.glob('*.parquet'):
            logger.info(path)
            self._prep(path)


@app.command
@dc.dataclass
class PrepDataset:
    """일간 ASOS 기온, 습도, 일사와 KEPCO, KEA 데이터 전처리."""

    conf: Config

    EUI_THRESHOLD: float = 0.1

    KEPCO_GFA: ClassVar[float] = 5208.81  # TODO 수정
    KEA_GFA: ClassVar[float] = 24348.0

    @staticmethod
    def is_working_hour(name: str = 'date'):
        return (pl.col(name).dt.hour() >= 9) & (pl.col(name).dt.hour() <= 18)  # noqa: PLR2004

    def bldg_kepco(self):
        root = self.conf.bldg_dirs('kepco_paju').database / '0002.binary'
        ud = 'updateDate'

        def scan(pattern: str):
            return pl.concat(
                pl.scan_parquet(x).with_columns(pl.col('tagValue').cast(pl.Float64))
                for x in root.rglob(pattern)
            )

        energy = (
            scan('*T_BELO_ELEC_DAY.parquet')
            .filter(pl.col('[tagName]') == '전기.전체전력량')
            .select(ud, pl.col('tagValue').alias('energy'))
            .unique(ud)
            .collect()
        )
        facility = (
            scan('*T_BELO_FACILITY_HOUR*.parquet')
            .filter(pl.col('[tagName]') == '실외온습도계.실내온도')
            .select(ud, 'tagValue')
            .unique(ud)
            .sort(ud)
            .collect()
        )
        ti = (
            facility
            .group_by_dynamic(ud, every='1d')
            .agg(pl.mean('tagValue').alias('Ti'))
            .with_columns()
        )
        tiw = (
            facility
            .filter(self.is_working_hour(ud))
            .group_by_dynamic(ud, every='1d')
            .agg(pl.mean('tagValue').alias('Tiw'))
        )

        return (
            energy
            .join(ti, on=ud, how='full', validate='1:1', coalesce=True)
            .join(tiw, on=ud, how='full', validate='1:1', coalesce=True)
            .select(
                pl.col(ud).alias('date'),
                pl.lit('KEPCO').alias('building'),
                pl.lit('파주').alias('location'),
                'energy',
                pl.col('energy').truediv(self.KEPCO_GFA).alias('EUI'),
                'Ti',
                'Tiw',
            )
        )

    def bldg_kea(self):
        # TODO 새벽 ESS 충전 문제 체크
        energy = (
            pl
            .scan_parquet(self.conf.dirs.db_raw / f'AMI_{Kea.iid}.parquet')
            .rename({'datetime': 'date', '사용량': 'energy'})
            .select('date', 'energy')
            .sort('date')
            .group_by_dynamic('date', every='1d')
            .agg(pl.sum('energy'))
            .collect()
        )

        src = list(
            self.conf.bldg_dirs('kea').database.glob('03.parsed/*실내온도.parquet')
        )
        exclude_space = ['지하', '계단', '서고', '체력단련실', '탁구실', '6층', '7층']
        bems = (
            pl
            .scan_parquet(src)
            .select('datetime', 'point', 'parsed_value')
            .rename({'datetime': 'date', 'parsed_value': 'value'})
            .filter(
                pl.col('value') > 1,  # NOTE 이상치 처리
                pl.col('point').str.contains_any(list(exclude_space)).not_(),
            )
            .sort('date')
            .collect()
        )
        logger.info(
            'KEA 실내온도 측정 지점: %s', bems['point'].unique().sort().to_list()
        )
        ti = (
            bems
            .group_by_dynamic('date', every='1d')
            .agg(pl.median('value').alias('Ti'))
            .with_columns()
        )
        tiw = (
            bems
            .filter(self.is_working_hour())
            .group_by_dynamic('date', every='1d')
            .agg(pl.median('value').alias('Tiw'))
        )

        return (
            energy
            .join(ti, on='date', how='full', validate='1:1', coalesce=True)
            .join(tiw, on='date', how='full', validate='1:1', coalesce=True)
            .select(
                'date',
                pl.lit('KEA').alias('building'),
                pl.lit('울산').alias('location'),
                'energy',
                pl.col('energy').truediv(self.KEA_GFA).alias('EUI'),
                'Ti',
                'Tiw',
            )
        )

    def weather(self):
        v = {
            '일시': 'date',
            '지점명': 'location',
            '평균기온(°C)': 'Te',
            '평균 상대습도(%)': 'RH',
            '평균 증기압(hPa)': 'Pv',
            '합계 일사량(MJ/m2)': 'I',
        }
        return (
            pl
            .concat(
                (
                    pl.read_csv(x, encoding='korean', infer_schema_length=None)
                    for x in self.conf.dirs.db_raw.glob('DAILY_OBS_ASOS*.csv')
                ),
                how='vertical_relaxed',
            )
            .rename(v)
            .select(list(v.values()))
            .with_columns(pl.col('date').str.to_date())
        )

    @staticmethod
    def plot(data: pl.DataFrame):
        data = (
            data
            .filter(pl.col('holiday').not_())
            .drop('location', 'holiday')
            .unpivot(index=['date', 'building', 'energy', 'RH'])
        )

        nvar = data.select(pl.col('variable').n_unique()).item()

        return (
            sns
            .FacetGrid(
                data,
                row='variable',
                hue='building',
                aspect=nvar * 16 / 9,
                sharey=False,
                height=2,
                despine=False,
            )
            .map_dataframe(sns.lineplot, x='date', y='value', alpha=0.8)
            .set_titles('{row_name}')
            .set_axis_labels('', '')
            .add_legend()
        )

    def __call__(self):
        data = (
            pl
            .concat([self.bldg_kepco(), self.bldg_kea()])
            .with_columns(pl.col('date').dt.date())
            .join(self.weather(), on=['date', 'location'], how='left', validate='1:1')
            .drop_nulls('energy')
            .filter(pl.col('EUI') >= self.EUI_THRESHOLD)
            .sort('date', 'location')
        )

        years = data.select(pl.col('date').dt.year()).to_series().unique().to_list()
        data = data.insert_column(
            1, misc.is_holiday(pl.col('date'), years=years).alias('holiday')
        )

        data.write_parquet(self.conf.dirs.database / 'extended.parquet')
        data.write_excel(self.conf.dirs.database / 'extended.xlsx')

        (
            utils.mpl
            .MplTheme()
            .grid()
            .tick(axis='y', which='both')
            .apply({'axes.ymargin': 0.1})
        )
        grid = self.plot(data)
        grid.savefig(self.conf.dirs.database / 'extended.png')

        return data


@app.command
@dc.dataclass
class ByWeekday:
    conf: Config

    EUI_THRESHOLD: float = 0.1

    def data(self):
        weekday = {idx + 1: f'{d}요일' for idx, d in enumerate('월화수목금토일')}
        weekday = {**weekday, 8: '공휴일'}
        return (
            pl
            .read_parquet(self.conf.dirs.database / 'extended.parquet')
            .filter(pl.col('EUI') >= self.EUI_THRESHOLD)
            .with_columns(pl.col('date').dt.weekday().alias('weekday-index'))
            .with_columns(
                pl
                .when(
                    pl.col('weekday-index').is_in([1, 2, 3, 4, 5]) & pl.col('holiday')
                )
                .then(pl.lit(8))
                .otherwise(pl.col('weekday-index'))
                .alias('weekday-index')
            )
            .with_columns(
                weekday=pl.col('weekday-index').replace_strict(
                    weekday, return_dtype=pl.String
                )
            )
            .sort('building', 'weekday-index')
        )

    def __call__(self):
        (
            utils.mpl
            .MplTheme(palette='tol:bright')
            .grid()
            .apply({'lines.solid_capstyle': 'butt'})
        )
        data = self.data()

        fig = Figure()
        ax = fig.subplots()

        sns.barplot(data, x='weekday', y='EUI', hue='building', ax=ax, linewidth=0)
        ax.set_xlabel('')
        ax.set_ylabel('EUI [kWh/m²]')
        ax.legend(title='')

        fig.savefig(self.conf.dirs.analysis / '00.EDA.EUI-weekday.png')
        fig.savefig(self.conf.dirs.analysis / '00.EDA.EUI-weekday.svg')


@app.command
@dc.dataclass
class GridPlot:
    conf: Config
    _: dc.KW_ONLY
    by_period: bool = False
    alpha: float = 0.25

    @staticmethod
    def _months(period: Period):
        match period:
            case None:
                return ()
            case 'summer':
                return (6, 7, 8)
            case 'winter':
                return (12, 1, 2)
            case 'spring-autumn':
                return (3, 4, 5, 9, 10, 11)

    def plot(self, bldg: Building, period: Period):
        data = (
            pl
            .scan_parquet(self.conf.dirs.database / 'extended.png')
            .filter(pl.col('building') == bldg)
            .with_columns((pl.col('Te') - pl.col('Ti')).alias('Te-Ti'))
            .drop('Ti', 'RH')
            .collect()
        )

        if period is not None:
            months = self._months(period)
            data = data.filter(pl.col('date').dt.month().is_in(months))

        grid = (
            sns
            .PairGrid(data.drop('date', 'holiday', 'energy').to_pandas(), height=2)
            .map_lower(sns.scatterplot, alpha=self.alpha)
            .map_upper(sns.scatterplot, alpha=self.alpha)
            .map_diag(sns.histplot)
        )

        p = '' if period is None else f'-{period}'
        grid.savefig(self.conf.dirs.analysis / f'02.EDA.WeatherPairGrid-{bldg}{p}.png')
        grid.savefig(self.conf.dirs.analysis / f'02.EDA.WeatherPairGrid-{bldg}{p}.svg')
        plt.close(grid.figure)

    def __call__(self):
        utils.mpl.MplTheme().grid().apply()

        for bldg, period in itertools.product(
            BUILDINGS, PERIODS if self.by_period else [None]
        ):
            self.plot(bldg, period)


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply({'lines.solid_capstyle': 'butt'})
    utils.mpl.MplConciseDate().apply()

    # TODO 면적 수정
    app()

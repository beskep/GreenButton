import dataclasses as dc
import functools
import warnings
from typing import ClassVar

import matplotlib.pyplot as plt
import more_itertools as mi
import polars as pl
import seaborn as sns
import structlog
from matplotlib.figure import Figure

from greenbutton import misc, utils
from scripts.exp.ecpm.common import Config, app

logger = structlog.stdlib.get_logger()


def is_working_hour(name: str = 'date', start: float = 9, end: float = 18):
    return (pl.col(name).dt.hour() >= start) & (pl.col(name).dt.hour() <= end)


@dc.dataclass
class _PrepBldg:
    conf: Config

    LOCATION: ClassVar[str] = '__LOC__'  # ASOS 기상 관측 지점
    GFA: ClassVar[float] = 0.0

    VARS: ClassVar[tuple[str, ...]] = (
        'building',
        'location',
        'date',
        'weekday',
        'holiday',
        'consumption',
        'generation',
        'Te',
        'Ti',
        'Tiw',
        'I',
        'Pv',
    )
    ASOS_VARS: ClassVar[dict[str, str]] = {
        '일시': 'date',
        '지점명': 'location',
        '평균기온(°C)': 'Te',
        '평균 증기압(hPa)': 'Pv',
        '합계 일사량(MJ/m2)': 'I',
    }

    @functools.cached_property
    def name(self):
        return self.__class__.__name__

    def _read_weather(self, location: str):
        pattern = f'raw/OBS_ASOS*_{location}.csv'

        try:
            src = mi.one(self.conf.dirs.database.glob(pattern))
        except ValueError as e:
            raise ValueError(self.conf.dirs.database, pattern) from e

        return (
            pl
            .read_csv(src, encoding='korean', infer_schema_length=None)
            .rename(self.ASOS_VARS)
            .select(list(self.ASOS_VARS.values()))
            .with_columns(
                pl.col('date').str.to_date(),
                pl.col('I', 'Pv').cast(pl.Float64),
            )
        )

    def _weather(self):
        return self._read_weather(self.LOCATION)

    def _prep(self) -> pl.DataFrame:
        # consumption, generation은 kWh/m² 반환
        raise NotImplementedError

    def _plot_grid(self, data: pl.DataFrame, name: str | None = None):
        season = {
            12: 'W',
            1: 'W',
            2: 'W',
            3: 'S&A',
            4: 'S&A',
            5: 'S&A',
            6: 'S',
            7: 'S',
            8: 'S',
            9: 'S&A',
            10: 'S&A',
            11: 'S&A',
        }
        data = data.with_columns(
            pl.col('date').dt.month().replace_strict(season).alias('season'),
            (pl.col('Te') - pl.col('Tiw')).alias('Te-Ti'),
        )
        for type_, variables in [
            ('temperature', ('Te', 'Ti', 'Tiw', 'Te-Ti')),
            ('weather', ('I', 'Pv')),
        ]:
            vars_ = [
                x
                for x in ['consumption', 'generation', *variables]
                if x in data.columns
            ]
            grid = (
                sns
                .PairGrid(
                    data,
                    hue='season',
                    hue_order=['W', 'S', 'S&A'],
                    x_vars=vars_,
                    y_vars=vars_,
                    height=2,
                    despine=False,
                )
                .map_lower(sns.scatterplot, alpha=0.25)
                .map_upper(sns.scatterplot, alpha=0.25)
                .map_diag(sns.histplot)
                .add_legend()
            )
            grid.savefig(
                self.conf.dirs.analysis / f'00.EDA.pair.{name or self.name}.{type_}.png'
            )
            plt.close(grid.figure)

    def _plot_timeline(self, data: pl.DataFrame, name: str | None = None):
        data = (
            data
            .with_columns((pl.col('Te') - pl.col('Tiw')).alias('Te-Ti'))
            .unpivot(
                ['Te', 'Tiw', 'Te-Ti', 'I', 'Pv', 'consumption', 'generation'],
                index='date',
            )
            .with_columns(
                pl
                .col('variable')
                .replace({
                    'Te': 'Temperature',
                    'Tiw': 'Temperature',
                    'Te-Ti': 'Temperature',
                    'consumption': 'Energy',
                    'generation': 'Energy',
                })
                .alias('type')
            )
            .sort('type', 'variable', 'date')
        )

        grid = (
            sns
            .FacetGrid(data, row='type', height=2, aspect=4 * 16 / 9, sharey=False)
            .map_dataframe(sns.lineplot, x='date', y='value', hue='variable')
            .set_axis_labels('', '')
            .set_titles('{row_name}')
        )

        for ax in grid.axes_dict.values():
            ax.legend(title='')

        grid.savefig(self.conf.dirs.analysis / f'00.EDA.time.{name or self.name}.png')

    def __call__(self):
        data = self._prep()
        weather = self._weather()

        if 'building' not in data.columns:
            assert 'location' not in data.columns
            data = data.with_columns(
                pl.lit(self.name).alias('building'),
                pl.lit(self.LOCATION).alias('location'),
            )
        if 'generation' not in data.columns:
            data = data.with_columns(pl.lit(None).alias('generation'))

        years = data['date'].dt.year().unique().sort().to_list()
        data = (
            data
            .join(weather, on=['date', 'location'], how='left', validate='m:1')
            .with_columns(
                weekday=pl.col('date').dt.weekday(),
                holiday=misc.is_holiday(pl.col('date'), years=years),
            )
            .select(*self.VARS, pl.all().exclude(self.VARS))
        )

        data.write_parquet(self.conf.dirs.database / f'DATA-{self.name}.parquet')

        workday = data.filter(pl.col('holiday').not_())
        for (bldg,), df in workday.group_by('building'):
            self._plot_timeline(df, bldg)

            with utils.mpl.MplTheme('paper').grid().rc_context():
                try:
                    self._plot_grid(df, bldg)
                except ValueError, RuntimeError:
                    logger.exception('PlotGrid failed')


@app.command
@dc.dataclass
class KEPCO(_PrepBldg):
    min_eui: float = 0.1

    LOCATION: ClassVar[str] = '파주'
    GFA: ClassVar[float] = 5208.81

    def _prep(self):
        root = self.conf.bldg_dirs('kepco_paju').database / '0002.binary'
        ud = 'updateDate'
        tn = '[tagName]'
        tv = 'tagValue'

        def scan(pattern: str):
            return pl.concat(
                pl.scan_parquet(x).with_columns(pl.col('tagValue').cast(pl.Float64))
                for x in root.rglob(pattern)
            )

        variables = {
            '전기.전체전력량': 'consumption',
            'ESS.태양광.발전량': 'generation',
        }
        energy = (
            scan('*T_BELO_ELEC_DAY.parquet')
            .filter(pl.col(tn).is_in(list(variables.keys())))
            .unique([ud, tn])
            .with_columns(pl.col(tn).replace_strict(variables))
            .collect()
            .pivot(tn, index=ud, values=tv, sort_columns=True)
            .with_columns(
                pl.col(ud).dt.date(),
                pl.col('consumption').truediv(self.GFA),
                pl.col('generation').truediv(self.GFA),
            )
            .filter(pl.col('consumption') > self.min_eui)
        )
        facility = (
            scan('*T_BELO_FACILITY_HOUR*.parquet')
            .filter(pl.col(tn) == '실외온습도계.실내온도')
            .select(ud, tv)
            .unique(ud)
            .sort(ud)
            .collect()
        )
        ti = (
            facility
            .group_by_dynamic(ud, every='1d')
            .agg(pl.mean(tv).alias('Ti'))
            .with_columns(pl.col(ud).dt.date())
        )
        tiw = (
            facility
            .filter(is_working_hour(ud))
            .group_by_dynamic(ud, every='1d')
            .agg(pl.mean(tv).alias('Tiw'))
            .with_columns(pl.col(ud).dt.date())
        )

        return (
            energy
            .join(ti, on=ud, how='full', validate='m:1', coalesce=True)
            .join(tiw, on=ud, how='full', validate='m:1', coalesce=True)
            .rename({ud: 'date'})
        )


@app.command
@dc.dataclass
class KEA(_PrepBldg):
    min_eui: float = 0.1
    exclude: tuple[str, ...] = (
        '지하',
        '계단',
        '서고',
        '체력단련실',
        '탁구실',
        '6층',
        '7층',
    )
    iid: str = 'DB_B7AE8782-40AF-8EED-E050-007F01001D51'

    # ASOS 울산 2022년 이전 자료 없음
    LOCATION: ClassVar[str] = '울산'
    GFA: ClassVar[float] = 24348.0

    def _prep(self):
        # XXX 새벽 ESS 충전 문제
        consumption = (
            pl
            .scan_parquet(self.conf.dirs.database / f'raw/AMI_{self.iid}.parquet')
            .rename({'datetime': 'date', '사용량': 'consumption'})
            .select('date', 'consumption')
            .sort('date')
            .group_by_dynamic('date', every='1d')
            .agg(pl.sum('consumption'))
            .with_columns(
                pl.col('date').dt.date(),
                pl.col('consumption').truediv(self.GFA),
            )
            .filter(pl.col('consumption') > self.min_eui)
            .collect()
        )

        db_dirs = self.conf.bldg_dirs('kea').database

        # 발전량 영향 낮음
        generation = (
            pl
            .scan_parquet(list(db_dirs.glob('02.binary/KEAPV.th_inverter*.parquet')))
            .rename({'create_date': 'date', 'InverterTodayKwh': 'generation'})
            .sort('date')
            .group_by_dynamic('date', every='1d')
            .agg(pl.max('generation').truediv(self.GFA))
            .with_columns(pl.col('date').dt.date())
            .collect()
        )

        bems = (
            pl
            .scan_parquet(list(db_dirs.glob('03.parsed/*실내온도.parquet')))
            .select('datetime', 'point', 'parsed_value')
            .rename({'datetime': 'date', 'parsed_value': 'value'})
            .filter(
                pl.col('value') > 1,  # 이상치 처리
                pl.col('point').str.contains_any(list(self.exclude)).not_(),
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
            .with_columns(pl.col('date').dt.date())
        )
        tiw = (
            bems
            .filter(is_working_hour())
            .group_by_dynamic('date', every='1d')
            .agg(pl.median('value').alias('Tiw'))
            .with_columns(pl.col('date').dt.date())
        )

        return (
            consumption
            .join(generation, on='date', how='full', validate='1:1', coalesce=True)
            .join(ti, on='date', how='full', validate='1:1', coalesce=True)
            .join(tiw, on='date', how='full', validate='1:1', coalesce=True)
            .with_columns()
        )


@app.command
@dc.dataclass
class EanBems(_PrepBldg):
    # 아래 두 건물 분석에서 제외
    # - 개포중: 실내 온도 급식실, 복도에서 측정 (5°C 이하 데이터 존재)
    # - 한울권: 상시 사용하는 건물 아님
    # 분석 결과
    # - GCEA, KIRIA, 개원초는 CPM 정확도가 매우 낮고 분석에 부적합

    exclude: tuple[str, ...] = ('개원초:서측 복도', 'GCEA:1층 서버실')
    BUILDINGS: ClassVar[dict[str, str]] = {
        'EnergyX': 'EnergyX',
        '개원초': '개원초',
        '광주기후에너지진흥원': 'GCEA',
        '대구로봇산업진흥원': 'KIRIA',
    }
    GFA_MAP: ClassVar[dict[str, float]] = {
        'EnergyX': 3274.87,
        '개원초': 13636.76,
        'GCEA': 1070.95,
        'KIRIA': 3780.05,
    }
    LOCATION_MAP: ClassVar[dict[str, str]] = {
        'EnergyX': '서울',
        '개원초': '서울',
        'GCEA': '광주',
        'KIRIA': '영천',
    }

    def _prep(self):
        root = self.conf.bldg_dirs('eanbems').root
        buildings = list(self.BUILDINGS)

        bldg = pl.col('building')
        consumption = pl.col('consumption')

        energy = (
            pl
            .scan_parquet(root / '02.에너지.parquet')
            .rename({
                '건물': 'building',
                '시간': 'date',
                '합계': 'consumption',
            })
            .filter(bldg.is_in(buildings))
            .sort('date', 'building')
            .group_by_dynamic('date', every='1d', group_by='building')
            .agg(pl.sum('consumption'))
            .with_columns(bldg.replace_strict(self.BUILDINGS))
            .with_columns(
                pl.col('date').dt.date(),
                pl
                .col('building')
                .replace_strict(self.GFA_MAP, return_dtype=pl.Float64)
                .alias('GFA'),
            )
            .filter(consumption > 0)
            .with_columns(consumption.truediv('GFA'))
            .collect()
        )
        env = (
            pl
            .scan_parquet(root / '02.실내환경.parquet')
            .rename({
                '건물': 'building',
                '시간': 'date',
                '온도': 'temperature',
            })
            .filter(bldg.is_in(buildings))
            .with_columns(bldg.replace_strict(self.BUILDINGS))
            .filter(
                pl
                .concat_str('building', '실이름', separator=':')
                .is_in(self.exclude)
                .not_()
            )
            .sort('date', 'building')
            .collect()
        )
        ti = (
            env
            .group_by_dynamic('date', every='1d', group_by='building')
            .agg(pl.mean('temperature').alias('Ti'))
            .with_columns(pl.col('date').dt.date())
        )
        tiw = (
            env
            .filter(is_working_hour())
            .group_by_dynamic('date', every='1d', group_by='building')
            .agg(pl.mean('temperature').alias('Tiw'))
            .with_columns(pl.col('date').dt.date())
        )

        return (
            energy
            .join(
                ti, on=['date', 'building'], how='full', validate='1:1', coalesce=True
            )
            .join(
                tiw, on=['date', 'building'], how='full', validate='1:1', coalesce=True
            )
            .with_columns(
                bldg.replace_strict(self.LOCATION_MAP).alias('location'),
            )
            .filter(
                (
                    (bldg == 'EnergyX') & ((consumption < 0.06) | (pl.col('Tiw') < 16))  # noqa: PLR2004
                ).not_(),
                ((bldg == '개원초') & (consumption > 0.8)).not_(),  # noqa: PLR2004
            )
        )

    def _weather(self):
        return pl.concat(self._read_weather(x) for x in set(self.LOCATION_MAP.values()))


@app.command
def prep_all(conf: Config):
    for cls in (KEPCO, KEA, EanBems):
        logger.info(cls.__name__)
        cls(conf)()


@app.command
def eui_weekday(conf: Config):
    (
        utils.mpl
        .MplTheme('paper', palette='tol:light')
        .grid()
        .apply({'lines.solid_capstyle': 'butt'})
    )

    weekday = {idx + 1: f'{d}요일' for idx, d in enumerate('월화수목금토일')}
    weekday = {**weekday, 8: '공휴일'}

    data = (
        pl
        .scan_parquet(list(conf.dirs.database.glob('DATA-*.parquet')))
        .filter(pl.col('building').str.contains('EnergyX|KEA|KEPCO'))
        .with_columns(pl.col('date').dt.weekday().alias('weekday-index'))
        .with_columns(
            pl
            .when(pl.col('weekday-index').is_in([1, 2, 3, 4, 5]) & pl.col('holiday'))
            .then(pl.lit(8))
            .otherwise(pl.col('weekday-index'))
            .alias('weekday-index')
        )
        .with_columns(
            weekday=pl.col('weekday-index').replace_strict(
                weekday, return_dtype=pl.String
            )
        )
        .collect()
    )

    monday = (
        data
        .filter(pl.col('weekday') == '월요일')
        .group_by('building')
        .agg(pl.mean('EUI'))
        .sort('EUI')
        .with_row_index()
    )
    data = (
        (data)
        .join(monday.select('building', 'index'), on='building', how='left')
        .sort('weekday-index', 'index')
    )

    fig = Figure()
    ax = fig.subplots()
    sns.barplot(
        data, x='weekday', y='EUI', hue='building', ax=ax, linewidth=0, width=0.9
    )

    ax.set_xlabel('')
    ax.set_ylabel('EUI [kWh/m²]')
    ax.legend(title='')

    fig.savefig(conf.dirs.analysis / '01.EDA.EUI-weekday.png')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='The figure layout has changed to tight')

    utils.mpl.MplTheme().grid().apply()
    utils.mpl.MplConciseDate().apply()

    app()

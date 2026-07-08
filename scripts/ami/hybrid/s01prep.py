import dataclasses as dc
import functools
import itertools
import re
import warnings
from itertools import starmap
from pathlib import Path
from typing import Literal

import cyclopts
import more_itertools as mi
import polars as pl
import seaborn as sns
import statsmodels.api as sm
import structlog
from matplotlib.figure import Figure
from rich.progress import track

from greenbutton import misc, utils
from greenbutton.utils import tqdm
from greenbutton.utils.cli import App

type LowessPreprocess = Literal['none', 'log', 'log1p', 'sqrt', 'filter']

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


@dc.dataclass
class _AnomalyOutput:
    root: Path
    name: str

    def __post_init__(self):
        self.root.mkdir(exist_ok=True)

    def data(self):
        return self.root / f'anomaly.{self.name}.parquet'

    @functools.cached_property
    def timeline(self):
        d = self.root / f'{self.name}.timeline'
        d.mkdir(exist_ok=True)
        return d

    @functools.cached_property
    def temp(self):
        d = self.root / f'{self.name}.temp'
        d.mkdir(exist_ok=True)
        return d


@dc.dataclass
class _AnomalyDetector:
    _: dc.KW_ONLY

    root: Path
    group: tuple[str, ...] = (
        'bldg.case',
        'bldg.index',
        'bldg.type',
        'bldg',
        'gfa',
        'asos.station',
        'energy',
    )

    def name(self):
        return self.__class__.__name__.removeprefix('Anomaly')

    def case_names(self, data: pl.DataFrame):
        drop = {'bldg', 'bldg.index', 'bldg.type', 'gfa', 'asos.station'}
        return '.'.join(str(data[x][0]) for x in self.group if x not in drop)

    @functools.cached_property
    def output(self):
        return _AnomalyOutput(root=self.root / '02.anomaly', name=self.name())

    @functools.cached_property
    def weather(self):
        return pl.read_parquet(self.root / '01.weather.parquet')

    def _plot(self, data: pl.DataFrame):
        name = self.case_names(data)
        boolean = data['anomaly'].dtype == pl.Boolean

        if 'date' not in data.columns:
            data = (
                data
                .drop_nulls('anomaly')
                .sort('datetime')
                .group_by_dynamic('datetime', every='1d', group_by=('holiday', 'gfa'))
                .agg(pl.sum('eui'), pl.max('anomaly'))
                .with_columns(
                    pl.col('datetime').dt.date().alias('date'),
                )
            )
        if 'Te' not in data.columns:
            data = data.join(self.weather, on=['date', 'asos.station'])

        if not data.height:
            return

        def plot(x: str, xlabel: str):
            fig = Figure()
            ax = fig.subplots()
            ax.set_xlabel(xlabel)
            ax.set_ylabel('EUI [kWh/m²]')
            ax.set_yscale('asinh')

            if boolean:
                sns.scatterplot(
                    data,
                    x=x,
                    y='eui',
                    hue='anomaly',
                    hue_order=[False, True],
                    ax=ax,
                    alpha=0.25,
                )
            else:
                fig.colorbar(
                    ax.scatter(
                        x=data['date'].to_numpy(),
                        y=data['eui'].to_numpy(),
                        c=data['anomaly'].to_numpy(),
                        alpha=0.25,
                    )
                )

            return fig

        plot('date', '').savefig(self.output.timeline / f'{name}.png')
        plot('Te', 'External Temperature [°C]').savefig(
            self.output.temp / f'{name}.png'
        )

    def _detect(self, data: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    def _summarize(self, data: pl.DataFrame):
        total = data.select(
            pl.lit('total').alias('bldg.case'),
            pl.len(),
            pl.col('anomaly').sum().alias('anomaly.count'),
            pl.col('anomaly').mean().alias('anomaly.ratio'),
        )
        agg = (
            data
            .group_by(self.group)
            .agg(
                pl.len(),
                pl.col('anomaly').sum().alias('anomaly.count'),
                pl.col('anomaly').mean().alias('anomaly.ratio'),
            )
            .sort(self.group)
        )
        return (
            pl
            .concat([total, agg], how='diagonal')
            .select(*self.group, 'len', 'anomaly.count', 'anomaly.ratio')
            .with_columns()
        )

    def __call__(self):
        consumption = (
            pl
            .scan_parquet(self.root / '01.raw.parquet')
            .with_columns(pl.col('consumption').truediv('gfa').alias('eui'))
            .collect()
        )
        total = consumption.select(self.group).n_unique()
        it = consumption.group_by(self.group, maintain_order=True)

        def detect():
            for _, df in track(it, total=total):
                score = self._detect(df)
                self._plot(score)
                yield score

        detected = pl.concat(detect())
        output = self.output.data()
        detected.write_parquet(output)
        self._summarize(detected).write_csv(
            output.with_suffix('.summary.csv'), include_bom=True
        )


@app.command
@dc.dataclass
class AnomalySR(_AnomalyDetector):
    every: str = '1d'
    score_window: int = 12
    threshold: float = 0.1

    def _detect(self, data: pl.DataFrame):
        from pyod.models.ts_spectral_residual import (  # type: ignore  # noqa: PGH003, PLC0415
            SpectralResidual,
        )

        if data['bldg'].n_unique() != 1:
            raise ValueError(data['bldg'].unique().sort().to_list())

        data = (
            data
            .sort('datetime')
            .upsample('datetime', every=self.every)
            .with_columns(pl.col('consumption').interpolate('linear').alias('value'))
        )
        detector = SpectralResidual(self.score_window).fit(data['value'].to_numpy())

        return (
            data
            .with_columns(pl.Series('anomaly', detector.decision_scores_))
            .with_columns(
                pl
                .when(pl.col('consumption').is_null())
                .then(pl.lit(None))
                .otherwise('anomaly')
                .alias('anomaly')
            )
            .with_columns()
        )


@app.command
@dc.dataclass
class AnomalyLowess(_AnomalyDetector):
    preprocess: LowessPreprocess = 'none'

    _: dc.KW_ONLY

    frac: float = 0.2
    min_eui: float = 0.01
    k: float = 3

    root: Path
    group: tuple[str, ...] = (*_AnomalyDetector.group, 'holiday')

    def name(self):
        return f'{super().name()}.{self.preprocess}.k={self.k:.1f}'

    @staticmethod
    def _scale(expr: pl.Expr):
        return (
            (expr - expr.median())
            .truediv(
                expr.quantile(0.75, interpolation='linear')
                - expr.quantile(0.25, interpolation='linear')
            )
            .fill_nan(None)
        )

    def _daily(self, data: pl.DataFrame):
        c = pl.col('consumption')
        match self.preprocess:
            case 'none' | 'filter':
                p = c
            case 'log':
                p = c.log().fill_nan(None)
            case 'log1p':
                p = c.log1p()
            case 'sqrt':
                p = c.sqrt()

        return (
            data
            .sort('datetime')
            .group_by_dynamic('datetime', every='1d', group_by=self.group)
            .agg(c.sum(), pl.sum('eui'))
            .with_columns(
                pl.col('datetime').dt.date(),
                scaled=self._scale(p).fill_nan(None),
            )
            .rename({'datetime': 'date'})
            .join(self.weather, on=['date', 'asos.station'], how='left', validate='1:1')
            .drop_nulls(['Te', 'eui'])
            .sort('Te')
            .with_row_index()
        )

    def _detect(self, data: pl.DataFrame):
        if data['bldg'].n_unique() != 1:
            raise ValueError(data['bldg'].unique().sort().to_list())

        data = self._daily(data)

        match self.preprocess:
            case 'filter':
                smoothed = data.filter(pl.col('eui') >= self.min_eui)
            case _:
                smoothed = data.drop_nulls('scaled')

        array = sm.nonparametric.lowess(
            endog=smoothed['scaled'].to_numpy(),
            exog=smoothed['Te'].to_numpy(),
            return_sorted=False,
            frac=self.frac,
        )
        smoothed = smoothed.select('index', pl.Series('smoothed', array))

        return (
            data
            .join(smoothed, on='index', how='left', validate='1:1')
            .sort('Te')
            .with_columns(
                residual=pl.col('smoothed').interpolate('linear') - pl.col('scaled')
            )
            .with_columns(self._scale(pl.col('residual')).alias('residual.scaled'))
            .with_columns(
                anomaly=pl
                .col('residual.scaled')
                .is_between(-self.k, self.k)
                .not_()
                .fill_null(value=False)
            )
        )

    def __call__(self):
        warnings.filterwarnings('ignore', message='Mean of empty slice.')
        warnings.filterwarnings(
            'ignore', message='invalid value encountered in scalar divide'
        )
        return super().__call__()


@app.command
@dc.dataclass
class VisLowess(AnomalyLowess):
    pattern: str = '아파트'

    def _detect(self, data: pl.DataFrame, frac: float = 0.5):
        data = self._daily(data)
        lowess = sm.nonparametric.lowess(
            endog=data['eui'].to_numpy(),
            exog=data['Te'].to_numpy(),
            return_sorted=False,
            frac=frac,
        )

        fig = Figure()
        ax = fig.subplots()
        sns.scatterplot(data, x='Te', y='eui', ax=ax, alpha=0.5)
        sns.lineplot(x=data['Te'].to_numpy(), y=lowess, ax=ax, c='gray')
        ax.set_xlabel('$T_e$')
        ax.set_ylabel('EUI')

        name = self.case_names(data)
        fig.savefig(self.output.temp / f'{name}.{frac:.2f}.png')

    def __call__(self):
        data = (
            pl
            .scan_parquet(self.root / '01.raw.parquet')
            .filter(pl.col('bldg').str.extract(f'({self.pattern})').is_not_null())
            .with_columns(pl.col('consumption').truediv('gfa').alias('eui'))
            .collect()
        )
        total = data.select(self.group).n_unique()
        group_by = data.group_by(self.group, maintain_order=True)

        for _, df in track(group_by, total=total):
            for frac in (0.1, 0.2, 0.25, 0.5):
                self._detect(df, frac=frac)


@app.command
def anomaly_lowess_batch(root: Path):
    for p, k in itertools.product(('none', 'sqrt', 'log1p'), (1.5, 2, 3)):
        logger.info('prep=%s, k=%f', p, k)

        AnomalyLowess(preprocess=p, k=k, root=root)()


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    utils.mpl.MplConciseDate().apply()
    app()

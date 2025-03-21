"""순차운휴 감지."""

from __future__ import annotations

import dataclasses as dc
import functools
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import whenever
from loguru import logger
from matplotlib import cm

import greenbutton.anomaly.hampel as _hampel
from greenbutton import misc, utils
from greenbutton.utils import App, Progress
from scripts.ami.public_institution.config import Config  # noqa: TC001
from scripts.utils import MetropolitanGov

if TYPE_CHECKING:
    from pathlib import Path

type SeasonLiteral = Literal['summer', 'winter']


class EmptyDataError(ValueError):
    pass


class SeqOpHours:
    COOLING: ClassVar[dict[tuple[str, ...], int]] = {
        ('서울', '인천', '경북', '경상북도', '대구'): 15,
        ('전라', '전남', '전북', '광주', '경남', '경상남도', '부산', '울산'): 16,
        ('경기', '대전', '세종', '충청', '강원', '제주'): 17,
    }
    HEATING: ClassVar[dict[tuple[str, ...], int]] = {
        ('경기', '세종', '서울', '인천', '강원'): 10,
        (
            '경상',
            '경북',
            '경남',
            '부산',
            '대구',
            '울산',
            '제주',
            '충청',
            '충북',
            '충남',
            '대전',
            '전라',
            '전북',
            '전남',
            '광주',
        ): 17,
    }

    @staticmethod
    def _op_hour(region: str, hours: dict[tuple[str, ...], int]):
        for key, value in hours.items():
            if any(k in region for k in key):
                return value

        raise ValueError(region)

    @classmethod
    def hour(cls, region: str, season: SeasonLiteral):
        match season:
            case 'summer':
                return cls._op_hour(region, cls.COOLING)
            case 'winter':
                return cls._op_hour(region, cls.HEATING)


@dc.dataclass
class HampelConfig:
    window_size: int = 4
    t: float = 3

    def hampel_filter(self):
        return _hampel.HampelFilter(window_size=self.window_size, t=self.t)


@dc.dataclass(frozen=True)
class SeasonConfig:
    year: int = 2022
    year_max: int | None = None
    one_month: bool = False
    show_holiday: bool = True

    def __post_init__(self):
        if self.year_max is not None and (self.year >= self.year_max):
            msg = 'year >= year_max'
            raise ValueError(msg, self)

    @functools.cached_property
    def years_str(self):
        if self.year_max is None:
            return str(self.year)

        return f'{self.year}-{self.year_max}'

    def is_in(self, expr: pl.Expr):
        years = (
            (self.year,)
            if self.year_max is None
            else tuple(range(self.year, self.year_max + 1))
        )
        return expr.dt.offset_by('1mo').dt.year().is_in(years)  # 지난 12월 포함

    def _months(self, s: SeasonLiteral):
        months = (6, 7, 8) if s == 'summer' else (12, 1, 2)
        if self.one_month:
            return (months[1],)
        return months

    def iter(self, s: SeasonLiteral):
        months = self._months(s)
        for y in [self.year] if isinstance(self.year, int) else self.year:
            for m in months:
                yield y - 1 if m == 12 else y, m  # 12월 -> 작년  # noqa: PLR2004


@dc.dataclass
class Cols:
    iid: str = '기관ID'
    name: str = '기관명'
    area: str = '연면적'
    address: str = '주소'
    value: str = '보정사용량'


@dc.dataclass(frozen=True)
class PublicInstitution:
    iid: str
    name: str
    area: float
    address: str
    ami: pl.LazyFrame

    kind: Literal['EU', 'EUI'] = 'EUI'

    @functools.cached_property
    def file_name(self):
        return (
            f'{self.region or "(지역인식실패)"}_{self.name}'
            f'{"" if self.kind == "EUI" else "_EU"}'
        )

    @functools.cached_property
    def region(self):
        return MetropolitanGov.search(self.address)

    @classmethod
    def create(cls, conf: Config, iid: str, **kwargs):
        if not kwargs:
            cols = {
                k: v for k, v in dc.asdict(Cols()).items() if k not in {'iid', 'value'}
            }

            inst = (
                pl.scan_parquet(conf.dirs.data / conf.files.institution)
                .filter(pl.col(Cols.iid) == iid)
                .select(cols.values())
                .collect()
                .to_dicts()
            )
            assert len(inst) == 1

            kwargs = {k: inst[0][v] for k, v in cols.items()}

        if kwargs['area']:
            value = pl.col(Cols.value).truediv(kwargs['area'])
        else:
            logger.warning('area=0 ({}, {})', kwargs['name'], iid)
            value = pl.col(Cols.value)
            kwargs['kind'] = 'EU'

        ami = (
            pl.scan_parquet(conf.dirs.data / conf.files.ami)
            .filter(pl.col(Cols.iid) == iid)
            .select('datetime', value.alias('value'))
        )

        return cls(iid=iid, ami=ami, **kwargs)

    @classmethod
    def iter(cls, conf: Config, *, elec_only: bool = True):
        cols = dc.asdict(Cols())
        cols.pop('value')

        inst = (
            pl.scan_parquet(conf.dirs.data / conf.files.institution)
            .select(cols.values())
            .collect()
        )

        if elec_only:
            # 전전화 건물
            elec = (
                pl.scan_parquet(conf.dirs.data / '냉난방방식-전기식용량비율.parquet')
                .filter(pl.col('전기식용량비율') == 1)
                .select(Cols.iid)
                .collect()
                .to_series()
            )
            inst = inst.filter(pl.col(Cols.iid).is_in(elec))

        for row in Progress.trace(inst.iter_rows(named=True), total=inst.height):
            yield cls.create(conf=conf, **{k: row[v] for k, v in cols.items()})


@dc.dataclass
class SequentialOperation:
    conf: Config
    institution: PublicInstitution
    hampel: HampelConfig
    season: SeasonConfig

    def apply_hampel_filter(self):
        dt = pl.col('datetime')

        ami = (
            self.institution.ami.filter(self.season.is_in(dt))
            .with_columns(
                misc.is_holiday(dt, years=self.season.year).alias('is_holiday')
            )
            .sort(dt)
            .collect()
        )

        if not ami.height:
            raise EmptyDataError(self.institution.name)

        hf = self.hampel.hampel_filter()
        return hf(ami.upsample('datetime', every='1h'))

    def plot(
        self,
        region: str | None,
        season: SeasonLiteral,
        data: pl.DataFrame,
        path: str | Path | None = None,
    ):
        hour = SeqOpHours.hour(region, season) if region else None
        months = [f'{y}-{m}' for y, m in self.season.iter(season)]
        dummy = pl.date(2000, 1, 1)
        dt = pl.col('datetime').dt

        data = (
            data.sort('datetime')
            .with_columns((pl.col('value') <= pl.col('value').shift()).alias('reduced'))
            .filter(pl.format('{}-{}', dt.year(), dt.month()).is_in(months))
            .with_columns(dummy.dt.combine(dt.time()).alias('time'))
            .with_columns(dt.epoch('d').alias('epoch'))
        )

        if not self.season.show_holiday:
            data = data.filter(pl.col('is_holiday').not_())

        if not data.height:
            raise EmptyDataError(self.institution.name, season)

        sm = cm.ScalarMappable(cmap='crest', norm=mcolors.Normalize())

        palette = sns.color_palette()
        palette = [(*palette[0], 0.5), (*palette[1], 0.9)]

        fig, ax = plt.subplots()
        sns.lineplot(
            data,
            x='time',
            y='value',
            hue='epoch',
            hue_norm=sm.norm,
            palette='crest',
            units='epoch',
            style='is_holiday' if self.season.show_holiday else None,
            estimator=None,
            ax=ax,
            alpha=0.12,
            legend=False,
        )
        sns.scatterplot(
            data.filter('is_outlier'),
            x='time',
            y='value',
            hue='reduced',
            style='reduced',
            ax=ax,
            s=40,
            markers=['.', 'X'],
            marker='X',
            palette=palette,
            legend=False,
        )

        if hour is not None:
            ax.axvline(
                whenever.LocalDateTime(2000, 1, 1, hour).py_datetime(),  # type: ignore[arg-type]
                c='slategray',
                ls='--',
                alpha=0.8,
            )

        ax.set_ylim(0)

        # colorbar
        cbar = plt.colorbar(sm, ax=ax)
        loc = mdates.AutoDateLocator()
        cbar.ax.yaxis.set_major_locator(loc)
        cbar.ax.yaxis.set_major_formatter(mdates.AutoDateFormatter(loc))

        ax.set_xlabel('')
        ax.set_ylabel(Cols.value)
        ax.xaxis.set_minor_locator(mdates.HourLocator())

        if path is not None:
            fig.savefig(path)
            plt.close(fig)
            return None

        return fig

    def __call__(
        self,
        output: Path,
        *,
        save_dataframe: bool = False,
        skip_exist: bool = True,
    ):
        inst = self.institution
        region = inst.region
        name = f'{inst.file_name}_{self.season.years_str}'

        if skip_exist and any(True for _ in output.glob(f'{name}*.png')):
            logger.debug('{} exists', inst.name)
            return

        data = (
            self.apply_hampel_filter()
            .drop_nulls(['datetime', 'value'])
            .filter(pl.col('value').is_finite())
        )

        if not data.height:
            raise EmptyDataError(self.institution.name)

        if save_dataframe:
            data.write_parquet(output / f'{name}.parquet')

        self.plot(
            region=region,
            season='summer',
            data=data,
            path=output / f'summer-{name}.png',
        )
        self.plot(
            region=region,
            season='winter',
            data=data,
            path=output / f'winter-{name}.png',
        )


# --------------------------------- App --------------------------------

_HAMPEL_CONF = HampelConfig()
_SEASON_CONF = SeasonConfig()

app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml',
        root_keys='public_institution',
        use_commands_as_keys=False,
    )
)


@app.command
def detect(
    *,
    conf: Config,
    hc: HampelConfig = _HAMPEL_CONF,
    sc: SeasonConfig = _SEASON_CONF,
    iid: str = 'DB_B7AE8782-40AF-8EED-E050-007F01001D51',
):
    output = conf.dirs.analysis / 'hampel'
    output.mkdir(exist_ok=True)

    (
        utils.MplTheme('paper', palette='tol:bright')
        .grid()
        .tick('x', 'both', direction='in')
        .apply()
    )
    utils.MplConciseDate(
        zero_formats=('', '%Y-%m', '%m-%d', '%H:%M', '%H:%M', '%H:%M'),
        show_offset=False,
    ).apply()

    inst = PublicInstitution.create(conf=conf, iid=iid)
    seq_op = SequentialOperation(conf=conf, institution=inst, hampel=hc, season=sc)
    seq_op(output)


@app.command
def batch_detect(
    *,
    conf: Config,
    hc: HampelConfig = _HAMPEL_CONF,
    sc: SeasonConfig = _SEASON_CONF,
):
    output = conf.dirs.analysis / 'hampel'
    output.mkdir(exist_ok=True)

    (
        utils.MplTheme(0.75, palette='tol:bright')
        .grid()
        .tick('x', 'both', direction='in')
        .apply()
    )
    utils.MplConciseDate(
        zero_formats=('', '%Y-%m', '%m-%d', '%H:%M', '%H:%M', '%H:%M'),
        show_offset=False,
    ).apply()

    for inst in PublicInstitution.iter(conf):
        try:
            seq_op = SequentialOperation(
                conf=conf, institution=inst, hampel=hc, season=sc
            )
            seq_op(output)
        except EmptyDataError as e:
            logger.error(repr(e))
        else:
            logger.info('{} | {}', inst.region, inst.name)


if __name__ == '__main__':
    utils.LogHandler.set(10)

    app()

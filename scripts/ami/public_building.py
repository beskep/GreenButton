from __future__ import annotations

import dataclasses as dc
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple

import matplotlib.pyplot as plt
import polars as pl
import rich
import seaborn as sns
from cyclopts import App, Parameter
from loguru import logger
from polars.exceptions import ComputeError

from greenbutton import cpr, utils
from scripts.config import Config

if TYPE_CHECKING:
    from collections.abc import Sequence


class EmptyDataError(ValueError):
    pass


@dc.dataclass
class _Config:
    path: dc.InitVar[str | Path] = 'config/config.toml'
    subdirs: dc.InitVar[Sequence[str]] = (
        '01raw',
        '02data',
        '03EDA',
        '20CPR',
        '99weather',
    )

    # working directory
    root: Path = dc.field(init=False)

    raw: Path = dc.field(init=False)
    data: Path = dc.field(init=False)
    eda: Path = dc.field(init=False)
    cpr: Path = dc.field(init=False)
    weather: Path = dc.field(init=False)

    def __post_init__(self, path: str | Path, subdirs: Sequence[str]):
        conf = Config.read(Path(path))
        self.root = conf.ami.root / 'Public'

        attributes = ['raw', 'data', 'eda', 'cpr', 'weather']
        for attr, subdir in zip(attributes, subdirs, strict=True):
            setattr(self, attr, self.root / subdir)


app = App()


@app.command
def building_info(src: Path | None = None, dst: Path | None = None):
    conf = _Config()
    src = src or conf.raw
    dst = dst or conf.data
    dst.mkdir(exist_ok=True)
    cnsl = rich.get_console()

    for p in src.glob('*.txt'):
        if p.name.startswith('kcl_'):
            # AMI 사용량 데이터
            continue

        cnsl.print(p.name)

        text = p.read_text('korean', errors='ignore')

        try:
            df = pl.read_csv(StringIO(text), separator='|')
        except ComputeError:
            df = pl.read_csv(StringIO(text), separator='|', truncate_ragged_lines=True)

        cnsl.print(df)

        df.write_parquet(dst / f'{p.stem}.parquet')
        df.write_excel(
            dst / f'{p.stem}.xlsx', column_widths=min(50, int(1500 / df.width))
        )


@app.command
def address():
    """
    주소 표준화 자료 검토, 저장.

    https://www.juso.go.kr/CommonPageLink.do?link=/support/AddressTransformThousand
    """
    conf = _Config()
    src = conf.data / '1.기관-주소변환.xlsx'

    asos_code = {
        '강원특별자치도': '강원',
        '경기도': '서울경기',
        '경상남도': '경남',
        '경상북도': '경북',
        '광주광역시': '전남',
        '대구광역시': '경북',
        '대전광역시': '충남',
        '부산광역시': '경남',
        '서울특별시': '서울경기',
        '세종특별자치시': '충남',
        '울산광역시': '경남',
        '인천광역시': '서울경기',
        '전라남도': '전남',
        '전북특별자치도': '전북',
        '제주특별자치도': '제주',
        '충청남도': '충남',
        '충청북도': '충북',
    }

    data = pl.read_excel(src).with_columns(
        pl.col('주소-도로명')
        .str.split(' ')
        .list[0]
        .replace_strict(asos_code)
        .alias('asos_code')
    )
    data.write_excel(conf.data / f'{src.stem}-코드.xlsx')
    data.write_parquet(src.with_suffix('.parquet'))

    region = (
        data.select(pl.col('주소-도로명').str.split(' ').list[0].unique().sort())
        .to_series()
        .to_list()
    )

    rich.print('지역=', region, sep='')


@app.command
def ami(src: Path | None = None, dst: Path | None = None):
    conf = _Config()
    src = src or conf.raw
    dst = dst or conf.data
    dst.mkdir(exist_ok=True)
    files = list(src.glob('kcl_*.txt'))

    cnsl = rich.get_console()
    cnsl.print(files)

    for p in src.glob('kcl_*.txt'):
        data = pl.scan_csv(p, separator='|')

        n = data.select(pl.col('기관ID').n_unique()).collect().item()
        cnsl.print(f'n_building={n}')

        converted = (
            data.with_columns(
                pl.format(
                    '{}-{}-{}',
                    '년도',
                    pl.col('월').cast(pl.String).str.pad_start(2, '0'),
                    pl.col('일').cast(pl.String).str.pad_start(2, '0'),
                )
                .str.to_date()
                .alias('date')
            )
            .with_columns(
                pl.when(pl.col('시간') == 24)  # noqa: PLR2004
                .then(pl.col('date') + pl.duration(days=1))
                .otherwise('date')
                .alias('date'),
                pl.col('시간').replace(24, 0),
            )
            .with_columns(pl.col('date').dt.combine(pl.time('시간')).alias('datetime'))
            .select('기관ID', 'datetime', '사용량', '보정사용량')
        )

        cnsl.print(converted.head().collect())

        year = p.stem.removeprefix('kcl_')
        converted.sink_parquet(dst / f'PublicAMI{year}.parquet')


def _ami_plot(lf: pl.LazyFrame):
    df = lf.unpivot(['사용량', '보정사용량'], index='datetime').collect()

    fig, ax = plt.subplots()
    sns.lineplot(df, x='datetime', y='value', hue='variable', ax=ax, alpha=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('사용량')
    ax.get_legend().set_title('')

    return fig


@app.command
def ami_plot(src: Path | None = None, dst: Path | None = None):
    conf = _Config()
    src = src or conf.data
    dst = dst or conf.eda

    dst.mkdir(exist_ok=True)

    lf = pl.scan_parquet(list(src.glob('PublicAMI*.parquet')))

    rich.print(lf.head().collect())
    buildings = lf.select(pl.col('기관ID').unique()).collect().to_series().to_list()

    utils.MplTheme().grid().apply()
    utils.MplConciseDate().apply()

    with utils.Progress() as p:
        for building in p.track(buildings):
            fig = _ami_plot(lf.filter(pl.col('기관ID') == building))
            fig.savefig(dst / f'{building}.png')
            plt.close(fig)


class Building(NamedTuple):
    idx: str
    id_: str
    name: str
    area: float
    region: str

    def __str__(self):
        return (
            f'{type(self).__name__}'
            f'(idx={self.idx}, name={self.name!r}, region={self.region!r})'
        )


@dc.dataclass
class PublicAMI:
    energy: Literal['사용량', '보정사용량'] = '보정사용량'
    institution: str | None = None

    conf: _Config = dc.field(default_factory=_Config)
    _file_bldg: str = '1.기관-주소변환.parquet'
    _file_temp: str = 'temperature.parquet'

    def buildings(self):
        lf = (
            pl.scan_parquet(self.conf.data / self._file_bldg)
            .with_row_index()
            .with_columns(pl.col('index').cast(pl.String).str.pad_start(4, '0'))
        )

        if self.institution:
            lf = lf.filter(pl.col('기관대분류') == self.institution)
        return lf

    def building_count(self) -> int:
        return self.buildings().select(pl.len()).collect().item()

    def iter_building(self):
        for row in (
            self.buildings()
            .select('index', '기관ID', '기관명', '연면적', 'asos_code')
            .sort('asos_code')
            .collect()
            .iter_rows()
        ):
            yield Building(*row)

    def ami(self, iid: str):
        return (
            pl.scan_parquet(list(self.conf.data.glob('PublicAMI*.parquet')))
            .filter(pl.col('기관ID') == iid)
            .with_columns()
        )

    def temperature(self, region: str):
        return (
            pl.scan_parquet(self.conf.weather / self._file_temp)
            .filter(pl.col('region2') == region)
            .select('datetime', pl.col('ta').alias('temperature'))
        )

    def data(self, building: Building, temperature: pl.DataFrame | None = None):
        # XXX LocalOutlierFactor 이상치 제거?
        temp = (
            self.temperature(building.region).collect()
            if temperature is None
            else temperature
        )
        ami = (
            self.ami(building.id_)
            .select(
                'datetime',
                pl.col(self.energy).truediv(building.area).alias('energy'),
            )
            .collect()
        )
        data = ami.join(temp, on='datetime', how='inner').drop_nulls()

        return data.select(
            'datetime',
            pl.col('datetime').dt.year().alias('year'),
            pl.col('datetime').dt.weekday().is_in([6, 7]).alias('is_weekend'),
            pl.col('datetime').dt.hour().alias('hour'),
            'temperature',
            'energy',
        )

    def __iter__(self):
        @lru_cache
        def read_temp(region: str):
            return self.temperature(region=region).collect()

        for bldg in self.iter_building():
            temp = read_temp(region=bldg.region)
            yield bldg, self.data(building=bldg, temperature=temp)

    def track(self):
        with utils.Progress() as p:
            yield from p.track(self, total=self.building_count())


@dc.dataclass
class PublicAmiCpr:
    ami: PublicAMI

    week: Literal['weekday', 'weekend', 'week'] = 'week'
    hour: Literal['day', '24h'] = 'day'
    day_bound: tuple[int, int] = (9, 18)

    style: cpr.PlotStyle = dc.field(  # type: ignore[assignment]
        default_factory=lambda: {
            'scatter': {'alpha': 0.20, 'hue': 'year', 'palette': 'crest'}
        }
    )

    def suffix(self):
        hour = (
            f'{self.hour}({self.day_bound[0]}-{self.day_bound[1]})'
            if self.hour == 'day'
            else self.hour
        )
        return f'{self.week}_{hour}'

    def path(self, building: Building):
        return (
            self.ami.conf.cpr / f'{building.idx}_{building.name}_'
            f'{self.ami.energy}_{self.suffix()}.ext'
        )

    def cpr(self, building: Building, data: pl.DataFrame | None = None):
        if building.area == 0:
            raise ZeroDivisionError

        if data is None:
            data = self.ami.data(building=building)

        if self.week is not None:
            data = data.filter(
                pl.when(self.week == 'weekend')
                .then(pl.col('is_weekend'))
                .otherwise(pl.col('is_weekend').not_())
            )
        if self.hour == 'day':
            data = data.filter(pl.col('hour').is_between(*self.day_bound))

        if not data.height:
            raise EmptyDataError

        return cpr.ChangePointRegression(data).optimize_multi_models()

    def save_cpr(
        self,
        building: Building,
        data: pl.DataFrame | None = None,
        *,
        skip_exists: bool = False,
    ):
        path = self.path(building=building)
        if skip_exists and path.with_suffix('.png').exists():
            return None

        model = self.cpr(building=building, data=data)

        fig, ax = plt.subplots()
        model.plot(ax=ax, style=self.style)
        ax.set_title(
            f'{building.name} {self.suffix()} (r²={model.model_dict["r2"]:.4g})',
            loc='left',
        )
        ax.update_datalim([[0, 0]], updatex=False, updatey=True)
        ax.autoscale_view()
        ax.set_xlabel('기온 [℃]')
        ax.set_ylabel('전력사용량 [kWh/m²]')

        for h in ax.get_legend().legend_handles:
            if h is not None:
                h.set_alpha(0.8)

        if not model.is_valid:
            ax.set_facecolor('coral')

        fig.savefig(path.with_suffix('.png'))
        plt.close(fig)

        return model.model_frame

    def batch_cpr(self, *, skip_exists: bool = False):
        dfs: list[pl.DataFrame] = []

        for bldg, data in self.ami:
            logger.info(bldg)

            try:
                df = self.save_cpr(building=bldg, data=data, skip_exists=skip_exists)
            except EmptyDataError:
                logger.warning('Empty Data')
                continue

            if df is not None:
                dfs.append(
                    df.select(
                        pl.lit(bldg.idx).alias('index'),
                        pl.lit(bldg.id_).alias('id'),
                        pl.lit(bldg.name).alias('name'),
                        pl.all(),
                    )
                )

        pl.concat(dfs).write_excel(
            self.ami.conf.cpr / f'[model] {self.ami.energy}_{self.suffix()}.xlsx'
        )


@app.command
def public_cpr(
    energy: Literal['사용량', '보정사용량'] = '사용량',
    institution: Annotated[str, Parameter(negative='--all')] = '정부청사관리',
    week: Literal['weekday', 'weekend', 'week'] = 'week',
    hour: Literal['day', '24h'] = 'day',
    day_bound: tuple[int, int] = (9, 18),
):
    ami = PublicAMI(energy=energy, institution=institution)
    pa = PublicAmiCpr(ami=ami, week=week, hour=hour, day_bound=day_bound)
    pa.ami.conf.cpr.mkdir(exist_ok=True)
    pa.batch_cpr()


if __name__ == '__main__':
    utils.set_logger()
    utils.MplTheme('paper').grid().apply()
    app()

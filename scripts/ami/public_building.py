from __future__ import annotations

import dataclasses as dc
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import matplotlib.pyplot as plt
import polars as pl
import rich
import seaborn as sns
from cyclopts import App
from loguru import logger
from polars.exceptions import ComputeError
from rich.progress import track

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

    for building in track(buildings):
        fig = _ami_plot(lf.filter(pl.col('기관ID') == building))
        fig.savefig(dst / f'{building}.png')
        plt.close(fig)


class Building(NamedTuple):
    id_: str
    name: str
    area: float
    region: str


@dc.dataclass
class PublicAmiCpr:
    energy: Literal['사용량', '보정사용량'] = '보정사용량'

    conf: _Config = dc.field(default_factory=_Config)
    building_file: str = '1.기관-주소변환.parquet'
    temperature_file: str = 'temperature.parquet'

    def building_count(self):
        return (
            pl.scan_parquet(self.conf.data / self.building_file)
            .select(pl.len())
            .collect()
            .item()
        )

    def iter_building(self):
        """
        Iterate `Building`.

        Yields
        ------
        Building
        """
        for row in (
            pl.scan_parquet(self.conf.data / self.building_file)
            .select('기관ID', '기관명', '연면적', 'asos_code')
            .collect()
            .iter_rows()
        ):
            yield Building(*row)

    def ami(self, institution_id: str):
        return (
            pl.scan_parquet(list(self.conf.data.glob('PublicAMI*.parquet')))
            .filter(pl.col('기관ID') == institution_id)
            .with_columns()
        )

    def temperature(self, region: str):
        return (
            pl.scan_parquet(self.conf.weather / self.temperature_file)
            .filter(pl.col('region2') == region)
            .select('datetime', 'ta')
        )

    def path(
        self,
        building: Building,
        weekday: Literal['weekday', 'weekend', None] = None,
    ):
        w = '' if weekday is None else f' {weekday}'
        name = f'{building.id_}({building.name}){w}'
        return self.conf.cpr / self.energy / f'{name}.ext'

    def cpr(
        self,
        building: Building,
        weekday: Literal['weekday', 'weekend', None] = None,
    ):
        if building.area == 0:
            raise ZeroDivisionError

        ami = self.ami(building.id_).select(
            'datetime', pl.col(self.energy).truediv(building.area).alias('energy')
        )
        temperature = self.temperature(building.region).rename({'ta': 'temperature'})
        data = (
            ami.join(temperature, on='datetime', how='inner')
            .with_columns(pl.col('datetime').dt.year().alias('year'))
            .drop_nulls()
            .collect()
        )

        if not data.height:
            raise EmptyDataError

        if weekday is not None:
            days = [1, 2, 3, 4, 5] if weekday == 'weekday' else [6, 7]
            data = data.filter(pl.col('datetime').dt.weekday().is_in(days))

        reg = cpr.ChangePointRegression(data)
        return reg.optimize_multi_models()

    def cpr_and_write(
        self,
        building: Building,
        weekday: Literal['weekday', 'weekend', None] = None,
        *,
        skip_exists: bool = False,
    ):
        path = self.path(building=building, weekday=weekday)
        if skip_exists and path.with_suffix('.xlsx').exists():
            return

        model = self.cpr(building, weekday=weekday)
        model.model_frame.write_excel(path.with_suffix('.xlsx'))

        fig, ax = plt.subplots()
        model.plot(
            ax=ax, style={'scatter': {'alpha': 0.25, 'hue': 'year', 'palette': 'crest'}}
        )

        ax.update_datalim([[0, 0]], updatex=False, updatey=True)
        ax.autoscale_view()
        ax.set_xlabel('기온 [℃]')
        ax.set_ylabel('전력사용량 [kWh/m²]')

        for h in ax.get_legend().legend_handles:
            if h is not None:
                h.set_alpha(0.8)

        fig.savefig(path.with_suffix('.png'))
        plt.close(fig)

    def _iter(self):
        with utils.Progress() as p:
            yield from p.track(self.iter_building(), total=self.building_count())

    def batch_cpr(self):
        bldg = next(self.iter_building())
        path = self.path(bldg, 'weekday')
        path.parent.mkdir(exist_ok=True)

        for building in self._iter():
            logger.info(building)

            kwargs = {'building': building, 'skip_exists': True}
            try:
                self.cpr_and_write(weekday='weekday', **kwargs)
                self.cpr_and_write(weekday='weekend', **kwargs)
            except (EmptyDataError, ZeroDivisionError) as e:
                logger.warning(e.__class__.__name__)
                continue


@app.command
def public_cpr(energy: Literal['사용량', '보정사용량'] = '보정사용량'):
    pa = PublicAmiCpr(energy=energy)
    pa.conf.cpr.mkdir(exist_ok=True)
    pa.batch_cpr()


if __name__ == '__main__':
    utils.set_logger()
    utils.MplTheme().grid().apply()
    app()

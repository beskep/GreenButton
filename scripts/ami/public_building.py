from __future__ import annotations

import dataclasses as dc
import functools
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple

import matplotlib.pyplot as plt
import pingouin as pg
import polars as pl
import rich
import seaborn as sns
from cmap import Colormap
from cyclopts import Parameter
from loguru import logger

from greenbutton import cpr, utils
from greenbutton.outlier import HampelFilter
from scripts.config import Config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


@dc.dataclass
class _Config:
    path: dc.InitVar[str | Path] = 'config/config.toml'
    subdirs: dc.InitVar[Sequence[str]] = (
        '01raw',
        '02data',
        '03EDA',
        '20CPR',
        '40analysis',
        '99weather',
    )

    # working directory
    root: Path = dc.field(init=False)

    raw: Path = dc.field(init=False)
    data: Path = dc.field(init=False)
    eda: Path = dc.field(init=False)
    cpr: Path = dc.field(init=False)
    analysis: Path = dc.field(init=False)
    weather: Path = dc.field(init=False)

    def __post_init__(self, path: str | Path, subdirs: Sequence[str]):
        conf = Config.read(Path(path))
        self.root = conf.ami.root / 'Public'

        attributes = ['raw', 'data', 'eda', 'cpr', 'analysis', 'weather']
        for attr, subdir in zip(attributes, subdirs, strict=True):
            setattr(self, attr, self.root / subdir)


def _name_trsf(name: str, prefix: str):
    return name.removeprefix(f'{prefix}_').replace('_', '-')


app = utils.App()
for sub_app in ['ami', 'cpr', 'report']:
    app.command(
        utils.App(sub_app, name_transform=functools.partial(_name_trsf, prefix=sub_app))
    )


@app['ami'].command
def ami_building_info(src: Path | None = None, dst: Path | None = None):
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
        except pl.exceptions.ComputeError:
            df = pl.read_csv(StringIO(text), separator='|', truncate_ragged_lines=True)

        cnsl.print(df)

        df.write_parquet(dst / f'{p.stem}.parquet')
        df.write_excel(
            dst / f'{p.stem}.xlsx', column_widths=min(50, int(1500 / df.width))
        )


@app['ami'].command
def ami_address():
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


@app['ami'].command
def ami_elec_equipment():
    """전기식 설비 통계."""
    conf = _Config()

    bldg = (
        pl.scan_parquet(conf.data / '1.기관-주소변환.parquet')
        .select('기관ID', '기관명', '기관대분류', '건물용도', 'asos_code')
        .collect()
    )
    equipment = pl.read_csv(
        conf.raw / '3.냉난방방식.txt',
        separator='|',
        encoding='korean',
    )
    cols = equipment.columns
    equipment = (
        equipment.drop_nulls()
        .drop(cols[3])  # 계산 오류
        .filter(pl.col(cols[1]) != 0)
        .with_columns(
            (
                pl.col(cols[1])  # br
                / (pl.col(cols[1]) + pl.col(cols[2]))
            ).alias('전기식용량비율')
        )
        .sort('전기식용량비율', descending=True)
        .join(bldg, on='기관ID', how='left')
    )

    equipment.write_parquet(conf.data / '냉난방방식.parquet')
    equipment.write_excel(conf.data / '냉난방방식.xlsx')
    rich.print(equipment.filter(pl.col('전기식용량비율') == 1))


@app['ami'].command
def ami_parquet():
    conf = _Config()
    src = conf.raw
    dst = conf.data
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


@app['ami'].command
def ami_plot():
    conf = _Config()
    src = conf.data
    dst = conf.eda

    dst.mkdir(exist_ok=True)

    lf = pl.scan_parquet(list(src.glob('PublicAMI*.parquet')))

    rich.print(lf.head().collect())
    buildings = lf.select(pl.col('기관ID').unique()).collect().to_series().to_list()

    utils.MplTheme().grid().apply()
    utils.MplConciseDate().apply()

    for building in utils.Progress.trace(buildings):
        fig = _ami_plot(lf.filter(pl.col('기관ID') == building))
        fig.savefig(dst / f'{building}.png')
        plt.close(fig)


class Building(NamedTuple):
    idx: str
    id_: str
    name: str
    area: float
    region: str
    elect_ratio: float | None

    def __str__(self):
        return (
            f'{type(self).__name__}'
            f'(idx={self.idx}, name={self.name!r}, region={self.region!r})'
        )


@dc.dataclass
class PublicAMI:
    energy: Literal['사용량', '보정사용량'] = '보정사용량'
    institution: str | None = None
    min_elec_ratio: float | None = None

    conf: _Config = dc.field(default_factory=_Config)

    _file_bldg: str = '1.기관-주소변환.parquet'
    _file_temp: str = 'temperature.parquet'
    _file_equipment: str = '냉난방방식.parquet'

    def buildings(self):
        equipment = pl.scan_parquet(self.conf.data / self._file_equipment).select(
            '기관ID', '전기식용량비율'
        )
        lf = (
            pl.scan_parquet(self.conf.data / self._file_bldg)
            .with_columns(
                pl.when(pl.col('기관대분류').str.starts_with('국립대학병원'))
                .then(pl.lit('국립대학병원 등'))
                .otherwise(pl.col('기관대분류'))
                .alias('기관대분류'),
            )
            .with_row_index()
            .with_columns(pl.col('index').cast(pl.String).str.pad_start(4, '0'))
            .join(equipment, on='기관ID', how='left')
        )

        if self.institution:
            lf = lf.filter(pl.col('기관대분류') == self.institution)
        if self.min_elec_ratio is not None:
            lf = lf.filter(pl.col('전기식용량비율') >= self.min_elec_ratio)

        return lf

    def building_count(self) -> int:
        return self.buildings().select(pl.len()).collect().item()

    def iter_building(self):
        for row in (
            self.buildings()
            .select(
                'index', '기관ID', '기관명', '연면적', 'asos_code', '전기식용량비율'
            )
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

    def data(
        self,
        building: Building,
        *,
        temperature: pl.DataFrame | bool = True,
        group_by_dynamic: str | None = None,
    ):
        if temperature is True:
            temp = self.temperature(building.region).collect()
        elif temperature is False:
            temp = None
        else:
            temp = temperature

        ami = (
            self.ami(building.id_)
            .collect()
            .select(
                'datetime',
                pl.col(self.energy).truediv(building.area).alias('energy'),
            )
        )

        data = ami
        if temp is not None:
            data = ami.join(temp, on='datetime', how='inner')
            values = ['temperature', 'energy']
        else:
            values = ['energy']

        data = data.drop_nulls().select(
            'datetime',
            pl.col('datetime').dt.year().alias('year'),
            pl.col('datetime').dt.month().alias('month'),
            pl.col('datetime').dt.weekday().is_in([6, 7]).alias('is_weekend'),
            pl.col('datetime').dt.hour().alias('hour'),
            *values,
        )

        if group_by_dynamic:
            data = (
                data.group_by_dynamic(
                    'datetime',
                    every=group_by_dynamic,
                    group_by=['year', 'month', 'is_weekend'],
                )
                .agg(pl.mean('temperature'), pl.sum('energy'))
                .with_columns()
            )

        return data

    def __iter__(self):
        @functools.lru_cache
        def read_temp(region: str):
            return self.temperature(region=region).collect()

        for bldg in self.iter_building():
            temp = read_temp(region=bldg.region)
            yield bldg, self.data(building=bldg, temperature=temp)

    def track(self):
        yield from utils.Progress.trace(self, total=self.building_count())


@dc.dataclass
class Palettes:
    sequential: str = 'seaborn:crest'
    cyclic: str = 'colorcet:CET_CBTC1'

    def to_mpl(self, p: Literal['sequential', 'cyclic'], /):
        name = self.sequential if p == 'sequential' else self.cyclic
        c = Colormap(name)

        if name == 'colorcet:CET_CBTC1':
            array = c.color_stops.color_array
            array[:, 0:3] *= 0.8
            c = Colormap(array)

        return c.to_mpl()


@dc.dataclass
class PublicAmiCprConf:
    period: Literal['daily', 'hourly', 'daytime'] = 'daily'
    daytime_bound: tuple[int, int] = (9, 18)
    week: Literal['weekday', 'weekend'] | None = 'weekday'

    year: int | None = None
    min_energy: float = 0.001
    min_elect: float | None = None

    plot: bool = False
    plot_share_lim: bool = False

    def suffix(self):
        year = '' if self.year is None else f'{self.year} '

        match self.period:
            case 'daily':
                period = '1일'
            case 'hourly':
                period = '1시간'
            case 'daytime':
                period = f'낮({self.daytime_bound[0]}-{self.daytime_bound[1]})'

        week = {None: '', 'weekday': ' 주중', 'weekend': ' 주말'}[self.week]
        min_elect = '' if self.min_elect is None else f'_MinElec{self.min_elect}'

        return f'{year}{period}{week}{min_elect}'


@dc.dataclass
class PublicAmiCpr:
    ami: PublicAMI
    conf: PublicAmiCprConf

    style: cpr.PlotStyle = dc.field(  # type: ignore[assignment]
        default_factory=lambda: {'scatter': {'s': 10, 'alpha': 0.25}}
    )
    palettes: Palettes = dc.field(default_factory=Palettes)

    def __post_init__(self):
        self.update()

    def update(self):
        if 'scatter' not in self.style:
            self.style['scatter'] = {}
        self.style['scatter']['hue'] = (
            'month' if self.conf.period == 'daily' else 'hour'
        )
        self.style['scatter']['palette'] = self.palettes.to_mpl(
            'sequential' if self.conf.period == 'daytime' else 'cyclic'
        )
        self.ami.min_elec_ratio = self.conf.min_elect
        return self

    def path(self, building: Building):
        return (
            self.ami.conf.cpr / f'{building.idx}_{building.name}_'
            f'{self.ami.energy}_{self.conf.suffix()}.ext'
        )

    def _data(self, building: Building):
        return self.ami.data(
            building=building,
            group_by_dynamic='1d' if self.conf.period == 'daily' else None,
        )

    def cpr(self, building: Building, data: pl.DataFrame | None = None):
        if building.area == 0:
            raise ZeroDivisionError

        if data is None:
            data = self._data(building)

        conf = self.conf

        if conf.min_energy:
            data = data.filter(pl.col('energy') >= conf.min_energy)
        if conf.year is not None:
            data = data.filter(pl.col('year') == conf.year)

        if conf.week is not None:
            data = data.filter(
                pl.when(conf.week == 'weekend')
                .then(pl.col('is_weekend'))
                .otherwise(pl.col('is_weekend').not_())
            )
        if conf.period == 'daytime':
            data = data.filter(pl.col('hour').is_between(*conf.daytime_bound))

        if not data.height > 3:  # noqa: PLR2004
            raise cpr.NotEnoughDataError(required=3, given=data.height)

        return cpr.ChangePointRegression(data).optimize_multi_models()

    def plot_cpr(
        self,
        building: Building,
        model: cpr.ChangePointModel,  # type: ignore[name-defined]
        ax: Axes | None = None,
    ):
        if ax is None:
            ax = plt.gca()

        model.plot(ax=ax, style=self.update().style)

        ax.update_datalim([[0, 0]], updatex=False, updatey=True)
        if self.conf.plot_share_lim:
            data = self._data(building).select('temperature', 'energy').to_numpy()
            ax.update_datalim(data, updatey=False)
        ax.autoscale_view()

        ax.set_title(
            f'{building.name} ({self.conf.suffix()}) r²={model.model_dict["r2"]:.4g}',
            loc='left',
            weight=500,
        )
        ax.set_xlabel('기온 [℃]')
        ax.set_ylabel('전력사용량 [kWh/m²]')

        handles, labels = ax.get_legend_handles_labels()

        if self.conf.period == 'daily':
            labels = [f'{x}월' for x in labels]
        else:
            labels = [f'{x:0>2}:00' for x in labels]

        ax.legend(handles=handles, labels=labels, markerscale=1.25, fontsize='small')

        for h in ax.get_legend().legend_handles:
            if h is not None:
                h.set_alpha(0.8)

        if not model.is_valid:
            ax.set_facecolor('coral')

        return ax

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

        if self.conf.plot:
            fig, ax = plt.subplots()
            self.plot_cpr(building=building, model=model, ax=ax)
            fig.savefig(path.with_suffix('.png'))
            plt.close(fig)

        return model.model_frame

    def batch_cpr(self, *, skip_exists: bool = False):
        dfs: list[pl.DataFrame] = []

        for bldg, data in self.ami.track():
            logger.debug(bldg)

            try:
                df = self.save_cpr(building=bldg, data=data, skip_exists=skip_exists)
            except (cpr.CprError, ZeroDivisionError) as e:
                logger.error(f'{e!r} | {bldg}')
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

        df = pl.concat(dfs)
        path = (
            self.ami.conf.cpr / f'[model] {self.ami.institution or "전체"} '
            f'{self.ami.energy}_{self.conf.suffix()}.ext'
        )
        df.write_parquet(path.with_suffix('.parquet'))
        df.write_excel(path.with_suffix('.xlsx'))


DEFAULT_CPR_CONF = PublicAmiCprConf()


@app['cpr'].command
def cpr_analyze(
    energy: Literal['사용량', '보정사용량'] = '사용량',
    institution: str | None = '정부청사관리',
    *,
    all_institution: bool = False,
    conf: Annotated[PublicAmiCprConf, Parameter(name='conf')] = DEFAULT_CPR_CONF,
):
    if all_institution:
        institution = None

    ami = PublicAMI(energy=energy, institution=institution)
    pa = PublicAmiCpr(ami=ami, conf=conf)
    pa.ami.conf.cpr.mkdir(exist_ok=True)
    pa.batch_cpr()


@app['cpr'].command
def cpr_batch_analyze(
    energy: Literal['사용량', '보정사용량'] = '사용량',
    institution: str | None = '정부청사관리',
    *,
    all_institution: bool = False,
    conf: Annotated[PublicAmiCprConf, Parameter(name='conf')] = DEFAULT_CPR_CONF,
):
    from itertools import product  # noqa: PLC0415

    if all_institution:
        institution = None

    for period, week in product(['daily', 'hourly', 'daytime'], ['weekday', 'weekend']):
        logger.info(f'{period=} | {week=}')

        conf.period = period  # type:ignore[assignment]
        conf.week = week  # type:ignore[assignment]
        cpr_analyze(energy=energy, institution=institution, conf=conf)


@app['report'].command
def report_cpr_coef(
    estimator: Literal['median', 'mean'] = 'median',
    min_r2: float = 0.2,
    max_beta: float = 5,  # Wh/m²
):
    """
    건물 유형별 CPR 모델 파라미터 그래프.

    Parameters
    ----------
    estimator : Literal['median', 'mean'], optional
    """
    conf = _Config()
    cpr = conf.cpr

    buildings = pl.scan_parquet(conf.data / '1.기관-주소변환.parquet').select(
        pl.col('기관ID').alias('id'),
        pl.when(pl.col('기관대분류').str.starts_with('국립대학병원'))
        .then(pl.lit('국립대학병원 등'))
        .otherwise(pl.col('기관대분류'))
        .alias('기관대분류'),
    )
    models = (
        pl.scan_parquet(cpr / '[model] 전체 사용량_1일 주중.parquet', glob=False)
        .filter(pl.col('r2') >= min_r2)
        .join(buildings, on='id')
        .sort(
            '기관대분류', pl.col('names').replace({'CDD': 0, 'HDD': 1, 'Intercept': 2})
        )
        .with_columns(pl.col('coef') * 1000)  # kWh/m² -> Wh/m²
        .collect()
    )
    count = (
        models.select('id', '기관대분류')
        .unique()
        .group_by('기관대분류')
        .len()
        .sort('기관대분류')
    )
    models = models.join(
        count,
        on='기관대분류',
        how='left',
    )

    if max_beta:
        outlier = models.filter(
            pl.col('names').is_in(['CDD', 'HDD']) & (pl.col('coef') > max_beta)
        )
        rich.print('outlier', outlier.drop('id', 'name'))
        models = models.filter(
            pl.col('id').is_in(outlier.select('id').to_series()).not_()
        )

    rich.print('개수', count)
    rich.print(
        '계수',
        models.group_by('기관대분류', 'names')
        .agg(pl.median('coef'))
        .pivot('names', index='기관대분류', values='coef')
        .sort('기관대분류')
        .with_columns(pl.col('CDD').truediv('HDD').alias('CDD/HDD')),
    )
    rich.print(
        '균형점 온도',
        models.group_by('기관대분류', 'names')
        .agg(pl.median('change_point'))
        .pivot('names', index='기관대분류', values='change_point')
        .sort('기관대분류'),
    )

    utils.MplTheme(context=0.9).grid().apply()

    # r2
    fig, ax = plt.subplots()
    sns.barplot(
        models.filter(pl.col('names') == 'Intercept'),
        x='r2',
        y='기관대분류',
        ax=ax,
        estimator=estimator,
    )
    ax.set_xlabel('결정계수')
    ax.set_ylabel('')
    fig.savefig(cpr / f'report-energy-r2-{estimator}.png')
    plt.close(fig)

    # coef
    for var, name in zip(
        ['Intercept', 'HDD', 'CDD'],
        ['기저부하 [Wh/m²]', '난방 민감도 [Wh/m²°C]', '냉방 민감도 [Wh/m²°C]'],
        strict=True,
    ):
        fig, ax = plt.subplots()
        sns.barplot(
            models.filter(pl.col('names') == var),
            x='coef',
            y='기관대분류',
            ax=ax,
            estimator=estimator,
        )
        ax.set_xlabel(name)
        ax.set_ylabel('')
        fig.savefig(cpr / f'report-energy-{var}-{estimator}.png')
        plt.close(fig)

    fig, ax = plt.subplots()
    sns.barplot(
        models.filter(pl.col('names') != 'Intercept').with_columns(
            pl.col('names').replace({'HDD': '난방', 'CDD': '냉방'})
        ),
        x='coef',
        y='기관대분류',
        hue='names',
        estimator=estimator,
        ax=ax,
    )
    ax.set_xlabel('냉·난방 민감도 [Wh/m²°C]')
    ax.set_ylabel('')
    ax.get_legend().set_title('')
    fig.savefig(cpr / f'report-energy-{estimator}.png')
    plt.close(fig)

    # change point
    fig, ax = plt.subplots()
    sns.pointplot(
        models.drop_nulls('change_point')
        .with_columns(pl.col('names').replace({'HDD': '난방', 'CDD': '냉방'}))
        .with_columns(),
        x='change_point',
        y='기관대분류',
        hue='names',
        estimator=estimator,
        linestyles='none',
        dodge=False,
        ax=ax,
        marker='D',
    )

    ax.set_xlabel('냉·난방 균형점 온도 [°C]')
    ax.set_ylabel('')
    ax.get_legend().set_title('')
    fig.savefig(cpr / f'report-change_point-{estimator}.png')
    plt.close(fig)


def _region_usage_data(conf: _Config, year: int | None = None):
    buildings = pl.scan_parquet(conf.data / '1.기관-주소변환.parquet').select(
        pl.col('기관ID').alias('id'),
        pl.col('asos_code').alias('region'),
        pl.col('연면적'),
        pl.when(pl.col('기관대분류').str.starts_with('국립대학병원'))
        .then(pl.lit('국립대학병원 등'))
        .otherwise(pl.col('기관대분류'))
        .alias('기관대분류'),
    )

    # 평일 일간 사용량
    energy = (
        pl.scan_parquet(conf.root / '02data/PublicAMI*.parquet')
        .filter(
            pl.col('datetime').dt.weekday().is_in([6, 7]).not_(),
            pl.lit(1).cast(pl.Boolean)
            if year is None
            else (pl.col('datetime').dt.year() == year),
        )
        .collect()
        .rename({'기관ID': 'id'})
        .group_by('id', pl.col('datetime').dt.date())
        .agg(pl.sum('사용량').alias('energy'))
    )

    cpr = (
        pl.scan_parquet(conf.cpr / '[model] 전체 사용량_1일 주중.parquet', glob=False)
        .filter(pl.col('names') == 'Intercept')
        .select('id', pl.col('coef').alias('baseline'))
    )

    return (
        energy.lazy()
        .join(cpr, on='id', how='inner')
        .join(buildings, on='id', how='left')
        .with_columns(pl.col('energy').truediv('연면적').alias('energy'))
        .collect()
    )


@app['report'].command
def report_region_usage(year: int | None = None):
    conf = _Config()

    if (path := conf.analysis / f'region-usage-{year}.parquet').exists():
        data = pl.read_parquet(path)
    else:
        data = _region_usage_data(conf, year=year)
        data.write_parquet(path)

    rich.print(data.head())
    rich.print(
        pl.from_pandas(
            pg.anova(
                data.group_by('id', 'region', '기관대분류')
                .agg(pl.sum('energy'))
                .to_pandas(),
                dv='energy',
                between=['region', '기관대분류'],
                ss_type=1,
                detailed=True,
            )
        )
    )

    unpivot = data.unpivot(['energy', 'baseline'], index=['id', 'region', '기관대분류'])
    rich.print(unpivot)

    utils.MplTheme('paper').grid(show=False).apply()
    fig, ax = plt.subplots()
    ax.set_facecolor('#EEE')
    sns.heatmap(
        data.pivot(
            'region',
            index='기관대분류',
            values='energy',
            aggregate_function='median',
            sort_columns=True,
        )
        .sort('기관대분류')
        .to_pandas()
        .set_index('기관대분류'),
        cmap=Colormap('seaborn:flare_r').to_mpl(),
        robust=True,
        annot=True,
        fmt='.2f',
        ax=ax,
        cbar_kws={'label': '일간 사용량 대표값 [kWh/m²]'},
    )
    ax.set_ylabel('')
    fig.savefig(conf.analysis / f'지역-용도별 일사용량-{year}.png')


@dc.dataclass
class PublicAmiHampel:
    ami: PublicAMI
    hf: HampelFilter

    def _data(self, building: Building):
        return (
            self.ami.data(building=building, temperature=False)
            .sort('datetime')
            .upsample('datetime', every='1h')
        )

    def hampel_filter(self, building: Building, data: pl.DataFrame | None = None):
        if data is None:
            data = self._data(building)

        return self.hf(data, value='energy')

    def _iter_hf(self):
        for bldg in utils.Progress.trace(
            self.ami.iter_building(),
            total=self.ami.building_count(),
        ):
            logger.debug(bldg)
            try:
                yield (
                    self.hampel_filter(building=bldg)
                    .filter(pl.col('is_outlier').is_not_null())
                    .select(
                        'datetime',
                        'is_outlier',
                        pl.lit(bldg.region).alias('region'),
                        pl.lit(bldg.id_).alias('id'),
                    )
                )
            except pl.exceptions.ComputeError as e:
                logger.warning('{}: {} ({})', e.__class__.__name__, e, bldg.name)
                continue

    def batch_hampel_filter(self):
        return pl.concat(self._iter_hf())


@app['report'].command
def report_hampel(
    energy: Literal['사용량', '보정사용량'] = '사용량',
    window_size: int = 4,
    t: float = 1,
):
    conf = _Config()

    if (path := conf.analysis / 'HampelFilter.parquet').exists():
        df = pl.read_parquet(path).filter(pl.col('is_outlier').is_not_null())
    else:
        ami = PublicAMI(energy=energy, institution='중앙행정기관')
        hf = HampelFilter(window_size=window_size, t=t)
        pahf = PublicAmiHampel(ami=ami, hf=hf)

        df = pahf.batch_hampel_filter()
        df.write_parquet(conf.analysis / 'HampelFilter.parquet')

    rich.print(df)

    year = 2022
    month = 7
    avg = (
        df.filter(
            pl.col('datetime').dt.year() == year,
            pl.col('datetime').dt.month() == month,
        )
        .with_columns(
            pl.col('is_outlier').cast(pl.Float64),
            pl.col('datetime').dt.hour().alias('hour'),
        )
        .filter(pl.col('hour').is_between(15, 17))
        .group_by('hour', 'region', 'id')
        .agg(pl.mean('is_outlier').alias('outlier'))
        .group_by('hour', 'region')
        .agg(pl.median('outlier'))
        .sort('hour', 'region')
    )

    rich.print(avg)

    fig, ax = plt.subplots()
    sns.heatmap(
        avg.with_columns(pl.format('{}:00', 'hour'))
        .pivot('hour', index='region', values='outlier')
        .to_pandas()
        .set_index('region'),
        ax=ax,
        cmap=Colormap('cmasher:ocean').to_mpl(),
        annot=True,
        fmt='.1%',
        vmax=0.3,
    )

    ax.set_ylabel('')
    ax.tick_params('y', rotation=0)

    fig.savefig(conf.analysis / 'HampelFilter.png')


if __name__ == '__main__':
    utils.LogHandler.set()
    utils.MplTheme('paper').grid().apply()

    app()

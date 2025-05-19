"""공공기관 건물·AMI 정보 전처리 및 분석."""

from __future__ import annotations

from io import StringIO
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import cyclopts
import matplotlib.pyplot as plt
import pint
import polars as pl
import rich
import seaborn as sns
from loguru import logger

from greenbutton import utils
from greenbutton.utils.cli import App
from greenbutton.utils.terminal import Progress
from scripts.ami.public_institution.config import Config  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Iterable


app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml',
        root_keys='public_institution',
        use_commands_as_keys=False,
    )
)


# ================================= preprocess ================================


app.command(App('prep', help='txt 데이터 변환, 전처리'))


@app['prep'].command
def prep_institution(
    *,
    conf: Config,
    src: Path | None = None,
    dst: Path | None = None,
):
    """기관/건물 정보 변환."""
    src = src or conf.dirs.raw
    dst = dst or conf.dirs.data
    dst.mkdir(exist_ok=True)
    console = rich.get_console()

    for p in src.glob('*.txt'):
        if p.name.startswith('kcl_'):
            # AMI 사용량 데이터
            continue

        console.print(p.name)

        text = p.read_text('korean', errors='ignore')

        try:
            data = pl.read_csv(StringIO(text), separator='|')
        except pl.exceptions.ComputeError:
            data = pl.read_csv(
                StringIO(text), separator='|', truncate_ragged_lines=True
            )

        console.print(data)

        data.write_parquet(dst / f'{p.stem}.parquet')
        data.write_excel(
            dst / f'{p.stem}.xlsx', column_widths=max(50, int(1600 / data.width))
        )


@app['prep'].command
def prep_ami(*, conf: Config):
    """AMI 사용량 데이터 변환."""
    src = conf.dirs.raw
    dst = conf.dirs.data
    dst.mkdir(exist_ok=True)
    files = list(src.glob('kcl_*.txt'))

    console = rich.get_console()
    console.print(files)

    for p in src.glob('kcl_*.txt'):
        data = pl.scan_csv(p, separator='|')

        n = data.select(pl.col('기관ID').n_unique()).collect().item()
        console.print(f'file={p.name}, n_building={n}')

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

        console.print(converted.head().collect())

        year = p.stem.removeprefix('kcl_')
        converted.sink_parquet(dst / f'AMI{year}.parquet')


@app['prep'].command
def prep_address(*, conf: Config):
    """
    주소 표준화 자료 검토, 저장.

    https://www.juso.go.kr/CommonPageLink.do?link=/support/AddressTransformThousand
    """
    src = conf.dirs.raw / '1.기관-주소변환.xlsx'

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
    data.write_excel(
        conf.dirs.data / f'{src.stem}-코드.xlsx',
        column_widths=max(50, int(1600 / data.width)),
    )
    data.write_parquet(conf.dirs.data / src.with_suffix('.parquet').name)

    region = (
        data.select(pl.col('주소-도로명').str.split(' ').list[0].unique().sort())
        .to_series()
        .to_list()
    )

    rich.print('지역=', region, sep='')


def _capacity_unit_conversion(units: Iterable[str]) -> dict[str, float | None]:
    ur = pint.UnitRegistry()

    ur.define('HP = 0.75 kW')
    ur.define('RT = 3320 kcal/h')
    ur.define('USRT = 3024 kcal/h')

    # 에너지법 시행규칙 에너지열량 환산기준
    # NOTE 계산오류 - 증기 보일러 용량 환산으로 대신
    ur.define('LNGkg = 54.7 MJ')
    ur.define('LNGton = 1000 LNGkg')

    # 증기 보일러 (효율등급 규정)
    ur.define('steam_boiler_kg = 626.611 kcal')
    ur.define('steam_boiler_ton = 1000 steam_boiler_kg')

    def conversion(unit: str):
        try:
            return float(ur.Quantity(1, unit).to('kW').magnitude)
        except (
            pint.UndefinedUnitError,
            pint.DimensionalityError,
            AssertionError,
            ValueError,
        ) as e:
            logger.debug('{}({}): {}', e.__class__.__name__, unit, e)
            return None

    return {x: conversion(x) for x in units}


@app['prep'].command
def prep_equipment(*, conf: Config):
    """'냉난방설비현황' 설비 종류, 단위 전처리."""
    src = conf.dirs.data / '2.냉난방설비현황.parquet'
    dst = src.with_stem(f'{src.stem}-전처리')

    area = (
        pl.scan_parquet(conf.dirs.data / '1.기관-주소변환.parquet')
        .select('기관ID', '연면적')
        .collect()
    )

    data = (
        pl.scan_parquet(src)
        .with_columns(
            pl.col('설비명')
            .str.replace_many([' ', ',', 'ㆍ'], '')
            .str.replace(r'\(.*\)', '')
            .str.replace(r'^(.*?)#?\d*(호?)$', '$1')
            .alias('equipment'),
            pl.col('설비 단위')
            .str.strip_chars()
            .replace({'w': 'W'})
            .str.replace('(?i)kw', 'kW')
            .str.replace('(?i)USRT', 'USRT')
            .str.replace_many(
                ['t/h', 'kacl', 'Kcal', '/hr', '/y'],
                ['ton/h', 'kcal', 'kcal', '/h', ''],
                ascii_case_insensitive=True,
            )
            .alias('unit'),
        )
        .with_columns(
            pl.when(pl.col('설비명') == '증기보일러')
            .then(
                pl.col('unit').str.replace_many(
                    ['ton', 'kg'], ['steam_boiler_ton', 'steam_boiler_kg']
                )
            )
            .otherwise(pl.col('unit'))
            .alias('unit')
        )
        .collect()
    )

    conversion_map = _capacity_unit_conversion(data['unit'].unique().sort())
    data = (
        data.with_columns(
            pl.col('unit')
            .replace_strict(conversion_map, default=None, return_dtype=pl.Float64)
            .alias('conversion')
        )
        .with_columns(
            (pl.col('설비 용량') * pl.col('conversion')).alias('capacity[kW]')
        )
        .join(area, on='기관ID', how='left')
        .with_columns(
            (pl.col('capacity[kW]') / pl.col('연면적')).alias('capacity[kW/m²]')
        )
    )

    data.write_parquet(dst)
    data.write_excel(
        dst.with_suffix('.xlsx'), column_widths=max(80, round(1600 / data.width))
    )


# ================================= plot ================================

app.command(App('plot'))


@app['plot'].command
def plot_equipment(*, min_count: int = 10, conf: Config):
    """설비 전처리 검토."""
    src = conf.dirs.data / '2.냉난방설비현황-전처리.parquet'
    lf = pl.scan_parquet(src).drop_nulls(['equipment', 'capacity[kW/m²]'])

    count = (
        lf.group_by('equipment').len().sort('len', descending=True).head(10).collect()
    )
    rich.print(count)

    by_equipment = (
        lf.filter(pl.len().over('equipment') >= min_count).sort('equipment').collect()
    )
    for unit in ['kW', 'kW/m²']:
        grid = (
            sns.FacetGrid(
                by_equipment,
                col='equipment',
                col_wrap=utils.mpl.ColWrap(by_equipment['equipment'].n_unique()).ncols,
                sharey=False,
                height=2,
                aspect=4 / 3,
            )
            .map_dataframe(
                sns.histplot,
                x=f'capacity[{unit}]',
                kde=True,
                log_scale=True,
            )
            .set_axis_labels(f'Capacity [{unit}]', clear_inner=False)
            .set_titles('{col_name}')
        )

        grid.savefig(
            conf.dirs.data / f'equipment-capacity-{unit.replace("/", " per ")}.png'
        )
        plt.close(grid.figure)


def _plot_institution(lf: pl.LazyFrame):
    df = lf.unpivot(['사용량', '보정사용량'], index='datetime').collect()

    fig, ax = plt.subplots()
    sns.lineplot(df, x='datetime', y='value', hue='variable', ax=ax, alpha=0.75)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_legend().set_title('')

    return fig


@app['plot'].command
def plot_each(*, conf: Config):
    """각 기관 시간별 사용량 그래프."""
    src = conf.dirs.data
    dst = conf.dirs.analysis / 'plot-institution'
    dst.mkdir(exist_ok=True)

    institution_id = '기관ID'

    ami = pl.scan_parquet(src / 'AMI*.parquet')
    institution = (
        pl.scan_parquet(src / '1.기관.parquet')
        .select(institution_id, '기관명')
        .collect()
    )

    rich.print(ami.head().collect())
    iid_series = (
        ami.select(pl.col(institution_id).unique().sort()).collect().to_series()
    )

    utils.mpl.MplTheme().grid().apply()
    utils.mpl.MplConciseDate().apply()

    for iid in Progress.iter(iid_series):
        name = (
            institution.filter(pl.col(institution_id) == iid)
            .head(1)
            .join(institution, on=institution_id)
            .select('기관명')
            .item()
        )

        fig = _plot_institution(ami.filter(pl.col(institution_id) == iid))
        fig.savefig(dst / f'{iid}_{name}.png')
        plt.close(fig)


# ================================= analyse ================================


app.command(App('analyse'))


@app['analyse'].command
def analyse_area_dist(*, max_area: float = 300000, conf: Config):
    """연면적 분포."""
    lf = pl.scan_parquet(conf.dirs.data / conf.files.institution)

    rich.print(
        lf.group_by(pl.col('연면적').cut([3000, 10000, 30000, 100000, 300000, 1000000]))
        .agg(pl.len())
        .sort('연면적')
        .collect()
    )
    # -> 만, 3만 m² 구분

    area = lf.select('연면적').collect().to_series().to_numpy()

    fig, ax = plt.subplots()
    sns.histplot(area, ax=ax)
    ax.set_xlabel('연면적 [m²]')
    fig.savefig(conf.dirs.analysis / '연면적 분포.png')

    ax.clear()
    sns.histplot(area, ax=ax, log_scale=True)
    ax.set_xlabel('연면적 [m²]')
    fig.savefig(conf.dirs.analysis / '연면적 분포-log.png')

    ax.clear()
    sns.histplot(area[area <= max_area], ax=ax)
    ax.set_xlabel('연면적 [m²]')
    fig.savefig(conf.dirs.analysis / f'연면적 분포-{max_area}m².png')
    plt.close(fig)

    grid = sns.displot(
        lf.select('연면적', '건물용도').sort('건물용도').collect(),
        x='연면적',
        col='건물용도',
        col_wrap=4,
        kind='hist',
        log_scale=True,
        facet_kws={'sharey': False},
        height=3,
    ).set_xlabels('연면적 [m²]')
    grid.savefig(conf.dirs.analysis / '연면적 분포-용도별.png')
    plt.close(grid.figure)

    grid = sns.displot(
        lf.select('연면적', '기관대분류').sort('기관대분류').collect(),
        x='연면적',
        col='기관대분류',
        col_wrap=4,
        kind='hist',
        log_scale=True,
        facet_kws={'sharey': False},
        height=3,
    ).set_xlabels('연면적 [m²]')
    grid.savefig(conf.dirs.analysis / '연면적 분포-기관대분류.png')


@app['analyse'].command
def analyse_extract(
    *,
    conf: Config,
    iid: str = 'DB_B7AE8782-40AF-8EED-E050-007F01001D51',
    xlsx: bool = True,
):
    """특정 건물 AMI 데이터 추출."""
    output = conf.dirs.extract
    output.mkdir(exist_ok=True)

    # AMI
    data = (
        pl.scan_parquet(list(conf.dirs.data.glob('AMI*.parquet')))
        .filter(pl.col('기관ID') == iid)
        .sort('datetime')
        .collect()
    )
    data.write_parquet(output / f'AMI_{iid}.parquet')
    if xlsx:
        data.write_excel(output / f'AMI_{iid}.xlsx', column_widths=200)


@app['analyse'].command
def analyse_elec_equipment(*, conf: Config):
    """전기식 설비 통계."""
    bldg = (
        pl.scan_parquet(conf.dirs.data / '1.기관-주소변환.parquet')
        .select('기관ID', '기관명', '기관대분류', '건물용도', 'asos_code')
        .collect()
    )
    equipment = pl.read_csv(
        conf.dirs.raw / '3.냉난방방식.txt',
        separator='|',
        encoding='korean',
    )

    # 기관ID, 전기식 용량 합계, 비전기식 용량 합계, 비전기식 용량 비율
    # FIXME "2.냉난방설비현황"과 "3.냉난방방식"의 전기/비전기식 용량 값이 다름
    # "2.냉난방설비현황" 기준으로 다시 계산 필요
    cols = equipment.columns
    elec = pl.col(cols[1])
    non_elec = pl.col(cols[2])
    equipment = (
        equipment.drop_nulls()
        .drop(cols[3])  # 계산 오류
        .filter(elec != 0)
        .with_columns((elec / (elec + non_elec)).alias('전기식용량비율'))
        .sort('전기식용량비율', descending=True)
        .join(bldg, on='기관ID', how='left')
    )

    equipment.write_parquet(conf.dirs.data / '냉난방방식-전기식용량비율.parquet')
    equipment.write_excel(
        conf.dirs.data / '냉난방방식-전기식용량비율.xlsx',
        column_widths=max(50, int(1600 / equipment.width)),
    )
    rich.print(equipment.filter(pl.col('전기식용량비율') == 1))


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    utils.terminal.LogHandler.set(10)

    app()

from __future__ import annotations

from io import StringIO
from pathlib import Path  # noqa: TC003

import cyclopts
import matplotlib.pyplot as plt
import polars as pl
import rich
import seaborn as sns

from greenbutton import utils
from greenbutton.utils import App, Progress
from scripts.ami.public_institution.config import Config  # noqa: TC001

app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml',
        root_keys='public_institution',
        use_commands_as_keys=False,
    )
)


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


def _plot_institution(lf: pl.LazyFrame):
    df = lf.unpivot(['사용량', '보정사용량'], index='datetime').collect()

    fig, ax = plt.subplots()
    sns.lineplot(df, x='datetime', y='value', hue='variable', ax=ax, alpha=0.75)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_legend().set_title('')

    return fig


@app.command
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

    utils.MplTheme().grid().apply()
    utils.MplConciseDate().apply()

    for iid in Progress.trace(iid_series):
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


app.command(App('analyse'))


@app['analyse'].command
def elec_equipment(*, conf: Config):
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
    cols = equipment.columns
    equipment = (
        equipment.drop_nulls()
        .drop(cols[3])  # 계산 오류
        .filter(pl.col(cols[1]) != 0)
        .with_columns(
            (
                pl.col(cols[1])  ##
                / (pl.col(cols[1]) + pl.col(cols[2]))
            ).alias('전기식용량비율')
        )
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
    app()

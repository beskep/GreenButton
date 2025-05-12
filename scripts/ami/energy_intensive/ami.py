from __future__ import annotations

import dataclasses as dc
import itertools
import re
from functools import lru_cache
from typing import TYPE_CHECKING

import cyclopts
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import pyarrow.csv
import rich
import seaborn as sns
from loguru import logger

from greenbutton import utils
from greenbutton.utils.cli import App
from greenbutton.utils.terminal import Progress
from scripts.ami.energy_intensive.common import KEMC_CODE, Buildings
from scripts.ami.energy_intensive.config import Config  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path

app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml',
        root_keys='energy_intensive',
        use_commands_as_keys=False,
    )
)

# ================================= prep ================================
app.command(App('prep', help='AMI 전처리'))


@app['prep'].command
def prep_sample(*, conf: Config, n: int = 100, suffix: str = '-sample'):
    for src in conf.dirs.raw.glob('*.csv'):
        stem = src.stem

        if stem.endswith(suffix):
            continue

        logger.info(src)

        with src.open('r', encoding='korean') as f:
            text = ''.join(itertools.islice(f, n))

        src.with_stem(f'{stem}{suffix}').write_text(text)


@dc.dataclass
class _Preprocess:
    conf: Config

    @staticmethod
    def _read_csv(source, encoding: str = 'korean'):
        arrow = pyarrow.csv.read_csv(
            source, read_options=pyarrow.csv.ReadOptions(encoding=encoding)
        )
        data = pl.from_arrow(arrow)
        assert isinstance(data, pl.DataFrame)
        return data

    @classmethod
    def prep(cls, source: Path, day: int | None, code: int):
        remove_prefix = '' if day is None else f'tb_day_lp_{day}day_bfor_data.'
        value_prefix = (
            'elcp_use_' if (day is None and '2023' not in source.name) else 'pwr_qty'
        )

        data = (
            cls._read_csv(source)
            .rename(lambda x: x.removeprefix(remove_prefix))
            .rename(
                {
                    'kemc_oldx_code': 'KEMC_OLDX_CODE',
                    'meter_dd': 'mr_ymd',
                    'ente_code': 'ente',
                },
                strict=False,
            )
            .rename({'mr_ymd': 'date'})
            .filter(pl.col('KEMC_OLDX_CODE') == code)
            .drop(
                'column_0',
                'season_code',
                'season_name',
                'weekd_weekend_code',
                'weekd_weekend_name',
                cs.starts_with('Unnamed:'),
                cs.starts_with('tb_day_lp'),
                strict=False,
            )
        )

        columns = data.columns
        values = data.select(cs.starts_with(value_prefix)).columns

        return (
            data.unpivot(
                values,
                index=[x for x in columns if x not in values],
                variable_name='time',
            )
            .with_columns(
                pl.col('KEMC_OLDX_CODE').cast(pl.UInt16),
                pl.col('ente').cast(pl.UInt32),
                # 날짜 데이터를 date 형식으로 변환
                pl.col('date')
                .cast(pl.String)
                .str.strip_suffix('.0')
                .str.to_date('%Y%m%d'),
                # 시간 데이터 중 숫자 4자리 추출 "pwr_qty0015" -> "0015"
                pl.col('time').str.extract(r'.*(\d{4})$'),
            )
            .with_columns(
                # time, datetime 형식은 `24:00:00` 데이터를 허용하지 않음
                # 시간이 "2400"인 열의 날짜를 하루 더하고, 시간은 "0000"으로 변환
                date=pl.when(pl.col('time') == '2400')
                .then(pl.col('date') + pl.duration(days=1))
                .otherwise(pl.col('date')),
                time=pl.when(pl.col('time') == '2400')
                .then(pl.lit('0000'))
                .otherwise(pl.col('time')),
            )
            .with_columns(pl.col('time').str.to_time('%H%M'))
        )

    @staticmethod
    @lru_cache(32)
    def _day(text: str):
        m = re.search(r'D\+(\d+)', text)
        return int(m.group(1)) if m else None

    @staticmethod
    def iter_files(path: Path):
        for file in path.glob('*.csv'):
            if file.stem.endswith('-sample'):
                continue

            m = re.search(r'D\+(\d+)', file.name)
            day = int(m.group(1)) if m else None

            yield day, file

    def __call__(self, sample_code: int = 501):
        dst = self.conf.dirs.data
        dst.mkdir(exist_ok=True)

        paths = [
            x
            for x in self.conf.dirs.raw.glob('*.csv')
            if not x.stem.endswith('-sample')
        ]

        for src, code in Progress.iter(
            itertools.product(paths, KEMC_CODE),
            total=len(paths) * len(KEMC_CODE),
        ):
            day = self._day(src.name)
            name = KEMC_CODE[code]

            logger.info('Day {} | {} ({}) | {}', day, code, name, src.name)

            n = f'{code}{name}_{src.stem}'
            data = self.prep(src, day=day, code=code)
            data.write_parquet(dst / f'{n}.parquet')

            if code == sample_code:
                data.head(100).write_excel(dst / f'{n}-sample.xlsx', column_widths=100)


@app['prep'].command
def prep(*, conf: Config):
    _Preprocess(conf)()


# ================================= bldg ================================
app.command(App('bldg', help='건물 정보 가공'))


@app['bldg'].command
def bldg_convert(*, conf: Config):
    # 건물 정보 엑셀 첫번째 시트 읽기
    files = [
        x
        for x in conf.dirs.raw.glob('*')
        if x.suffix == '.xlsx' and '다소비사업장data' in x.name and '가공' not in x.name
    ]
    data = pl.concat([
        pl.read_excel(x).with_columns(
            pl.col('실적연도').cast(pl.Int16), pl.col('업체코드').cast(pl.String)
        )
        for x in files
    ]).sort('실적연도', '업체코드')

    rich.print(data)

    data.write_parquet(conf.root / 'buildings.parquet')
    data.sample(1000, shuffle=True, seed=42).write_excel(
        conf.root / 'buildings-sample.xlsx'
    )


@app['bldg'].command
def bldg_elec(*, conf: Config):
    """전전화 건물(?) 목록."""
    data = (
        pl.scan_parquet(conf.root / 'buildings.parquet')
        .with_columns(cs.ends_with('(toe)').fill_null(0))
        .with_columns()
    )

    zero = data.filter(pl.all_horizontal(cs.ends_with('(toe)') == 0)).collect()
    zero.write_parquet(conf.root / 'buildings-electric.parquet')
    zero.write_excel(conf.root / 'buildings-electric.xlsx', column_widths=100)
    rich.print(zero)

    def _ente(p: Path):
        if m := re.match(r'^\d+.*?_(\d+)_.*$', p.name):
            return int(m.group(1))

        return None

    if (path := conf.dirs.analysis / '전전화 건물 사용량').exists():
        ente = {_ente(x) for x in path.glob('*')}
        ente = {x for x in ente if x is not None}

        zero = zero.filter(pl.col('업체코드').cast(pl.Int64).is_in(ente)).sort(
            '업체코드', '실적연도'
        )
        zero.write_excel(
            conf.root / 'buildings-electric-with-ami.xlsx', column_widths=100
        )


# ================================= EDA ================================
app.command(App('eda'))


@app['eda'].command
def eda_plot_elec_line(*, conf: Config):
    dst = conf.dirs.analysis / '전전화 건물 사용량'
    dst.mkdir(parents=True, exist_ok=True)

    # 실적 연도 중 하나라도 toe 0이면 포함

    buildings = Buildings(conf=conf, electric=True)
    rich.print(buildings.buildings)

    utils.mpl.MplTheme('paper').grid().apply()
    utils.mpl.MplConciseDate().apply()

    for ente, kemc, name in buildings.iter_rows('ente', 'KEMC_CODE', '업체명'):
        data = (
            buildings.ami(ente=ente, kemc=kemc)
            .group_by('date')
            .agg(pl.sum('value'))
            .sort('date')
            .collect()
        )

        if not data.height:
            continue

        fig, ax = plt.subplots()
        sns.lineplot(data, x='date', y='value', ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('일간 AMI 전력 사용량 [kWh]')
        ax.update_datalim([[0, 0]], updatex=False)
        ax.autoscale_view()

        fig.savefig(dst / f'{kemc}{KEMC_CODE[kemc]}_{ente}_{name}.png')
        plt.close(fig)


if __name__ == '__main__':
    utils.terminal.LogHandler.set()

    app()

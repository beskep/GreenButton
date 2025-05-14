from __future__ import annotations

import dataclasses as dc
import itertools
import re
from functools import lru_cache
from typing import TYPE_CHECKING

import cyclopts
import fastexcel
import polars as pl
import polars.selectors as cs
import rich
from loguru import logger
from matplotlib.figure import Figure

from greenbutton import utils
from greenbutton.utils.cli import App
from greenbutton.utils.terminal import Progress
from scripts.ami.energy_intensive.common import KEMC_CODE, Buildings, Vars
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
def prep_sample(*, conf: Config, n: int = 100):
    dst = conf.dirs.raw / 'sample'
    dst.mkdir(exist_ok=True)

    for src in conf.dirs.raw.glob('*.csv'):
        logger.info(src)

        with src.open('r', encoding='korean') as f:
            text = ''.join(itertools.islice(f, n))

        (dst / src.name).write_text(text)


@dc.dataclass
class _Preprocess:
    conf: Config

    test: bool = False
    test_rows: int = 10000

    def prep(self, source: Path, day: int | None, code: int):
        def remove_prefix(s: str):
            if day is None:
                return s

            return s.removeprefix(f'tb_day_lp_{day}day_bfor_data.')

        data = (
            pl.read_csv(
                source,
                encoding='korean',
                n_rows=self.test_rows if self.test else None,
            )
            .rename(remove_prefix)
            .rename(
                {
                    'kemc_oldx_code': Vars.KEMC_CODE,
                    'KEMC_OLDX_CODE': Vars.KEMC_CODE,
                    'ente_code': Vars.ENTE,
                    'ente': Vars.ENTE,
                    Vars.CNTR_TP_CODE.lower(): Vars.CNTR_TP_CODE,
                    Vars.CNTR_TP_NAME.lower(): Vars.CNTR_TP_NAME,
                    'mr_ymd': 'date',
                    'meter_dd': 'date',
                },
                strict=False,
            )
            .filter(pl.col(Vars.KEMC_CODE) == code)
            .drop(
                '',
                'season_code',
                'season_name',
                'weekd_weekend_code',
                'weekd_weekend_name',
                cs.starts_with('Unnamed:'),
                cs.starts_with('tb_day_lp'),
                strict=False,
            )
        )

        value_prefix = (
            'elcp_use_' if (day is None and '2023' not in source.name) else 'pwr_qty'
        )
        values = data.select(cs.starts_with(value_prefix)).columns

        return (
            data.unpivot(
                values,
                index=[x for x in data.columns if x not in values],
                variable_name='time',
            )
            .with_columns(
                pl.col(Vars.KEMC_CODE).cast(pl.UInt16),
                pl.col(Vars.ENTE).cast(pl.UInt32),
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
            m = re.search(r'D\+(\d+)', file.name)
            day = int(m.group(1)) if m else None

            yield day, file

    def __call__(self, sample_code: int = 501):
        dst = self.conf.dirs.data
        dst.mkdir(exist_ok=True)

        for code, name in KEMC_CODE.items():
            (dst / f'{code}{name}').mkdir(exist_ok=True)

        paths = list(self.conf.dirs.raw.glob('*.csv'))

        for src, code in Progress.iter(
            itertools.product(paths, KEMC_CODE),
            total=len(paths) * len(KEMC_CODE),
        ):
            if self.test and code != sample_code:
                continue

            day = self._day(src.name)
            name = KEMC_CODE[code]

            logger.info('Day {} | {} ({}) | {}', day, code, name, src.name)

            n = f'{code}{name}_{src.stem}'
            data = self.prep(src, day=day, code=code)
            data.write_parquet(dst / f'{code}{name}' / f'{n}.parquet')

            if code == sample_code:
                data.head(1000).write_excel(
                    dst / f'(sample){n}.xlsx', column_widths=120
                )


@app['prep'].command
def prep(*, conf: Config, test: bool = False):
    _Preprocess(conf=conf, test=test)()


# ================================= bldg ================================
app.command(App('bldg', help='건물 정보 가공'))


@app['bldg'].command
def bldg_convert(*, conf: Config):
    # 건물 정보 엑셀 첫번째 시트 읽기
    console = rich.get_console()

    files = [
        x
        for x in conf.dirs.raw.glob('*')
        if x.suffix == '.xlsx' and '다소비사업장data' in x.name and '가공' not in x.name
    ]
    console.print('files', files)

    cast = [
        pl.col(Vars.PERF_YEAR).cast(pl.Int16),
        pl.col('업체코드').cast(pl.String),
    ]

    for sheet in fastexcel.read_excel(files[0]).sheet_names:
        logger.info('sheet={}', sheet)

        data = (
            pl.concat(pl.read_excel(x).with_columns(cast) for x in files)
            .rename({'업체코드': Vars.ENTE})
            .sort(Vars.PERF_YEAR, Vars.ENTE)
        )

        console.print(data)

        suffix = '' if sheet == '건물' else f'-{sheet}'
        path = conf.dirs.data / f'building{suffix}.parquet'
        data.write_parquet(path)
        data.write_excel(path.with_suffix('.xlsx'), column_widths=120)


@app['bldg'].command
def bldg_elec(*, conf: Config):
    """전전화 건물(?) 목록."""
    data = (
        pl.scan_parquet(conf.dirs.data / 'building.parquet')
        .with_columns(cs.ends_with('(toe)').fill_null(0))
        .with_columns()
    )

    zero = data.filter(pl.all_horizontal(cs.ends_with('(toe)') == 0)).collect()
    zero.write_parquet(conf.dirs.data / 'building-electric.parquet')
    zero.write_excel(conf.dirs.data / 'building-electric.xlsx', column_widths=120)
    rich.print(zero)

    def _ente(p: Path):
        if m := re.match(r'^\d+.*?_(\d+)_.*$', p.name):
            return int(m.group(1))

        return None

    if (path := conf.dirs.analysis / '전전화 건물 사용량').exists():
        ente = {_ente(x) for x in path.glob('*')}
        ente = {x for x in ente if x is not None}

        (
            zero.filter(pl.col(Vars.ENTE).cast(pl.Int64).is_in(ente))
            .sort(Vars.ENTE, Vars.PERF_YEAR)
            .write_excel(
                conf.dirs.analysis / 'buildings-electric-with-ami.xlsx',
                column_widths=120,
            )
        )


# ================================= EDA ================================
app.command(App('eda'))


@app['eda'].command
def eda_plot_elec(*, conf: Config):
    """전전화 건물 사용량 선그래프."""
    dst = conf.dirs.analysis / '전전화 건물 사용량'
    dst.mkdir(parents=True, exist_ok=True)

    # 실적 연도 중 하나라도 toe 0이면 포함
    buildings = Buildings(conf=conf, electric=True)
    rich.print(buildings.buildings)

    utils.mpl.MplTheme('paper').grid().apply()
    utils.mpl.MplConciseDate().apply()

    for ente, kemc, name in buildings.iter_rows(Vars.ENTE, Vars.KEMC_CODE, Vars.NAME):
        data = (
            buildings.ami(ente=ente, kemc=kemc)
            .group_by('date')
            .agg(pl.sum('value'))
            .sort('date')
            .collect()
        )

        if not data.height:
            continue

        fig = Figure()
        ax = fig.add_subplot()
        utils.mpl.lineplot_break_nans(
            data.upsample('date', every='1d'), x='date', y='value', ax=ax
        )
        ax.set_xlabel('')
        ax.set_ylabel('일간 AMI 전력 사용량 [kWh]')
        ax.update_datalim([[0, 0]], updatex=False)
        ax.autoscale_view()

        fig.savefig(dst / f'{kemc}{KEMC_CODE[kemc]}_{ente}_{name}.png')


if __name__ == '__main__':
    utils.terminal.LogHandler.set()

    app()

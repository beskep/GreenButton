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
    from collections.abc import Iterable, Sequence
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


@dc.dataclass
class _BuildingConverter:
    cast_float: Sequence[str] = (
        '사용량(toe)',
        '사용량(toe)1',
        '사용량(toe)2',
        '사용량(toe)3',
        '용량',
        '용량(kW)',
        '총발전량(MWh)',
    )
    cast_uint: Sequence[str] = ('구입년도', '대수', '설치년도')

    @staticmethod
    def _cast(data: pl.DataFrame, column: str, dtype: type[pl.DataType]):
        if column not in data.columns or data[column].dtype != pl.String:
            return data

        return data.with_columns(
            pl.col(column).str.replace(',', '').replace('', None).cast(dtype)
        )

    def _prep(self, data: pl.DataFrame):
        data = data.with_columns(cs.string().str.strip_chars().replace('', None))

        for c in self.cast_float:
            data = self._cast(data, c, pl.Float64)
        for c in self.cast_uint:
            data = self._cast(data, c, pl.UInt16)

        return data.with_columns(
            pl.col(Vars.PERF_YEAR).cast(pl.Int16),
            pl.col('업체코드').cast(pl.String),
        )

    @staticmethod
    def _prep_fixed_equipment(data: pl.DataFrame):
        data = data.with_row_index()

        cols = [
            f'{c}{i + 1}'
            for c in ['에너지원', '사용량', '사용량(toe)', '단위']
            for i in range(3)
        ]
        equipment = (
            data.select('index', *cols)
            .unpivot(index='index')
            .with_columns(
                pl.col('variable').str.extract_groups(
                    r'^(?<variable>.*?)(?<energy_index>\d)$'
                )
            )
            .unnest('variable')
            .pivot('variable', index=['index', 'energy_index'], values='value')
            .drop_nulls('에너지원')
            .with_columns(
                cs.starts_with('사용량').cast(pl.Float64),
                pl.col('energy_index').cast(pl.UInt8),
            )
            .rename({'단위': '(에너지원)단위'})
        )

        return (
            data.drop(cols)
            .join(equipment, on='index')
            .sort('index', 'energy_index')
            .drop('index')
        )

    def __call__(self, paths: Iterable[str | Path], sheet: str):
        data = (
            pl.concat(
                (self._prep(pl.read_excel(p, sheet_name=sheet)) for p in paths),
                how='diagonal_relaxed',
            )
            .rename({'업체코드': Vars.ENTE})
            .sort(Vars.PERF_YEAR, Vars.ENTE)
        )

        if sheet == '고정설비':
            # 설비 1~3번 unpivot
            # 이동 설비도 같은 작업 필요하나, 데이터 사용이 필요 없어서 변환 X
            data = self._prep_fixed_equipment(data)

        return data


@app['bldg'].command
def bldg_convert(*, conf: Config):
    console = rich.get_console()

    paths = list(conf.dirs.raw.glob('다소비사업장data*.xlsx'))
    console.print('files', paths)

    converter = _BuildingConverter()

    for sheet in fastexcel.read_excel(paths[0]).sheet_names:
        logger.info('sheet={}', sheet)

        data = converter(paths, sheet)
        console.print(data.glimpse(max_items_per_column=5, return_as_string=True))

        suffix = '' if sheet == '건물' else f'-{sheet}'
        path = conf.dirs.data / f'building{suffix}.parquet'
        data.write_parquet(path)
        data.write_excel(path.with_suffix('.xlsx'), column_widths=100)


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

from __future__ import annotations

import dataclasses as dc
import itertools
import re
from functools import lru_cache
from typing import TYPE_CHECKING

import cyclopts
import fastexcel
import pint
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
    from collections.abc import Iterable, Mapping, Sequence
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

    kemc_code: dict[str, int] = dc.field(init=False)

    def __post_init__(self):
        self.kemc_code = {n: c for c, n in KEMC_CODE.items()}

    @staticmethod
    def _cast(data: pl.DataFrame, column: str, dtype: type[pl.DataType]):
        if column not in data.columns or data[column].dtype != pl.String:
            return data

        return data.with_columns(
            pl.col(column).str.replace(',', '').replace('', None).cast(dtype)
        )

    def _prep(self, data: pl.DataFrame):
        data = (
            data.rename(lambda x: x.replace('㎡', 'm²'))
            .with_columns(cs.string().str.strip_chars().replace('', None))
            .with_columns()
        )

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
                    r'^(?<variable>.*?)(?<index_energy>\d)$'
                )
            )
            .unnest('variable')
            .pivot('variable', index=['index', 'index_energy'], values='value')
            .drop_nulls('에너지원')
            .with_columns(cs.starts_with('사용량').cast(pl.Float64))
            .rename({'단위': '(에너지원)단위'})
        )

        return (
            data.drop(cols)
            .join(equipment, on='index')
            .with_columns(
                # 사용량 순서대로 index_energy 다시 계산
                pl.col('사용량(toe)')
                .rank(descending=True)
                .over('index')
                .cast(pl.UInt8)
                .alias('index_energy')
            )
            .sort('index', 'index_energy')
            .drop('index')
        )

    def __call__(self, paths: Iterable[str | Path], sheet: str):
        data = (
            pl.concat(
                (self._prep(pl.read_excel(p, sheet_name=sheet)) for p in paths),
                how='diagonal_relaxed',
            )
            .rename({'업체코드': Vars.ENTE, '업종': Vars.KEMC_KOR})
            .with_columns(pl.col(Vars.KEMC_KOR).replace('IDC(전화국)', 'IDC'))
            .sort(Vars.PERF_YEAR, Vars.ENTE)
        )

        data.insert_column(
            data.columns.index(Vars.KEMC_KOR),
            data[Vars.KEMC_KOR].replace_strict(self.kemc_code).alias(Vars.KEMC_CODE),
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

        if sheet == '건물':
            utils.pl.PolarsSummary(data).write_excel(
                conf.dirs.analysis / '0001.건물 summary.xlsx'
            )
            utils.pl.PolarsSummary(
                data, group=[Vars.KEMC_CODE, Vars.KEMC_KOR]
            ).write_excel(conf.dirs.analysis / '0001.건물 summary-KEMC.xlsx')


_DEFAULT_EQUIPMENT_MAPPING = {
    '관류': '보일러',
    '노통연관': '보일러',
    '빙축열 냉동기': '빙축열 냉방',
    '스크롤형 냉동기': '압축식 냉방',
    '스크류형 냉동기': '압축식 냉방',
    '왕복동형 냉동기': '압축식 냉방',
    '터어보형 냉동기': '압축식 냉방',
    '가스-터빈 열병합': '열병합 난방',
    '열교환기': '지역난방',
    '냉온수기': '흡수식 냉방',
    '흡수식1종 냉동기': '흡수식 냉방',
    '흡수식2종 냉동기': '흡수식 냉방',
    '흡수식 냉온수기': '흡수식 냉난방',
    '흡수식냉온수기': '흡수식 냉난방',
}


@dc.dataclass
class _BuildingEquipment:
    conf: Config

    value: str = '사용량(toe)'
    """설비 특성 추출/분석에 사용할 변수"""

    source: Sequence[str] = ('전기', 'LNG')  # 전기, LNG, 기타(온수 포함)로 분류
    """에너지원 분류 기준"""

    equipment_mapping: Mapping[str, str] = dc.field(
        default_factory=lambda: _DEFAULT_EQUIPMENT_MAPPING
    )
    """설비 종류 ('형식'열) 재분류 기준"""

    energy_unit: str = 'MJ'

    index: Sequence[str] = (Vars.ENTE, Vars.PERF_YEAR)
    index_full: Sequence[str] = (
        Vars.ENTE,
        Vars.PERF_YEAR,
        Vars.KEMC_CODE,
        Vars.KEMC_KOR,
        Vars.NAME,
    )
    index_equipment: Sequence[str] = (
        Vars.ENTE,
        Vars.PERF_YEAR,
        Vars.KEMC_CODE,
        Vars.KEMC_KOR,
        Vars.NAME,
        '설비명',
        '형식',
    )

    data: pl.DataFrame = dc.field(init=False)

    def __post_init__(self):
        data = (
            pl.scan_parquet(self.conf.dirs.data / 'building-고정설비.parquet')
            .with_columns(
                # 설비명 => 'type': 전기, 보일러, 열사용
                pl.col('설비명').alias('type'),
            )
            .with_columns(
                # 에너지원 => 'source': 지정한 에너지원 외 기타로 분류
                pl.col('에너지원')
                .replace('도시가스(LNG)', 'LNG')
                .replace_strict(self.source, self.source, default='기타')
                .alias('source'),
            )
            .with_columns(
                # 용도 => 'use': 생산용, 생산난방용을 기타로 분류
                pl.col('용도')
                .replace_strict(
                    {
                        '냉방용': '냉방용',
                        '난방용': '난방용',
                        '열병합용': '난방용',
                        '냉난방용': '냉난방용',
                    },
                    default='기타',
                )
                .fill_null('기타')
                .alias('use'),
            )
        )

        # '형식', '에너지원' (여러개인 경우 제일 사용량이 많은 에너지원)
        # 기준으로 설비 유형 재분류
        form = '형식'
        source = '에너지원'
        equipment = (
            data.filter(pl.col('index_energy') == 1)
            .select(form, source)
            .unique()
            .with_columns(
                pl.col(form)
                .replace_strict(self.equipment_mapping, default=None)
                .alias('equipment'),
            )
            .with_columns(
                # '냉동기' 에너지원에 따라 재분류
                pl.when(
                    pl.col(form) == '냉동기',
                    pl.col(source).is_in(['도시가스(LNG)', '온수']),
                )
                .then(pl.lit('흡수식 냉방'))
                .when(
                    pl.col(form) == '냉동기',
                    pl.col(source) == '전기',
                )
                .then(pl.lit('압축식 냉방'))
                .otherwise(pl.col('equipment'))
                .alias('equipment')
            )
        )

        self.data = data.join(
            equipment, on=[form, source], how='left', nulls_equal=True
        ).collect()

    @property
    def eui_unit(self):
        return f'EUI({self.energy_unit}/m²)'

    def source_dist(self):
        return (
            self.data.group_by('에너지원')
            .agg(pl.sum(self.value))
            .with_columns(ratio=pl.col(self.value) / pl.sum(self.value))
            .sort('ratio', descending=True)
        )

    def use_dist(self):
        return (
            self.data.group_by('용도')
            .agg(pl.len().alias('len'), pl.sum(self.value))
            .with_columns(
                ratio=pl.col('len') / pl.sum('len'),
                consumption_ratio=pl.col(self.value) / pl.sum(self.value),
            )
            .sort('len', descending=True)
        )

    def main_equipment(self):
        """사업장별 사용량이 가장 많은 냉,난,냉난방 설비 선정."""
        v = pl.col(self.value)
        group = [*self.index, 'use']
        return (
            self.data.filter(v != 0)
            # 같은 종류 설비 사용량 합산
            .group_by([*self.index_equipment, 'use', 'equipment'])
            .agg(v.sum())
            # 설비 사용량 비율, 순위 계산
            .with_columns(
                (v / v.sum().over(group)).alias('ratio'),
                v.rank('min', descending=True).over(group).alias('rank'),
                pl.col('use').str.strip_suffix('용'),  # e.g. 냉난방용 -> 냉난방
            )
            # 건물/용도 중 사용량 가장 큰 설비 필터링
            .filter(pl.col('rank') == 1)
            # 1위 설비가 2개 이상인 경우 (사용량이 같은 경우) 동시 표기
            .group_by([*self.index_full, 'use'])
            .agg('equipment', pl.sum('ratio'))
            .with_columns(
                pl.col('equipment')
                .list.sort()
                .list.join(separator='+')
                .replace('', None)
            )
            .with_columns(
                pl.when(pl.col('ratio').is_not_null(), pl.col('equipment').is_null())
                .then(pl.lit('기타'))
                .otherwise(pl.col('equipment'))
                .alias('equipment')
            )
            .sort(self.index)
        )

    def feature(self):
        """군집화 분석 등에 사용할 수치 feature 추출."""
        # NOTE 사업장당 실적연도 다수 존재
        # 연도별로 사용량(+일부 연면적 등 건축 정보)가 다르기 때문에,
        # 각 연도 데이터 별도로 취급

        # NOTE '대수' 변수는 무시 ('사용량'에 이미 반영되었다고 가정)
        # 한국에너지사용량 에너지사용량 신고 시스템 상
        # 고정설비 정보와 사용량을 별도로 입력함
        # => 사용량은 설비 수를 고려하지 않고 총합을 입력한다고 가정
        # (https://min24.energy.or.kr/EngyRpt/CST/02/02_01_010.do 참조)

        area = (
            pl.scan_parquet(self.conf.dirs.data / 'building.parquet')
            .select(*self.index, Vars.AREA)
            .collect()
        )

        # ENTE번호, 실적연도 조합이 유일한지 확인
        assert area.select(
            pl.concat_str(Vars.ENTE, Vars.PERF_YEAR, separator='-').is_unique().all()
        ).item()

        conversion_factor = float(
            pint.UnitRegistry()
            .Quantity(1, re.sub(r'.*?\((.*)\)$', r'\1', self.value))
            .to(self.energy_unit)
            .magnitude
        )
        convert = pl.col(self.value) * conversion_factor / pl.col(Vars.AREA)

        return (
            self.data
            # NOTE '로': 전체 데이터의 0.12%, 산업 목적 추정 -> 생략
            .filter(pl.col('type') != '로', pl.col(self.value) != 0)
            .group_by([*self.index_full, 'use', 'type', 'source', 'equipment'])
            .agg(pl.sum(self.value))
            .join(area, on=self.index, how='left')
            .with_columns(convert.alias(self.eui_unit))
            .sort(self.index)
        )

    def pivot_feature(self, feature: pl.DataFrame, variables: Sequence[str]):
        eui = self.eui_unit
        return (
            feature.select(
                *self.index_full,
                pl.concat_str(variables, separator='-').alias('feature'),
                eui,
            )
            .group_by([*self.index_full, 'feature'])
            .agg(pl.sum(eui))
            .pivot('feature', index=self.index_full, values=eui, sort_columns=True)
            .fill_null(0)
        )

    def __call__(self):
        console = rich.get_console()

        console.print('에너지원', self.source_dist())
        # 에너지원별 비중: 전기 76.3%, LNG 13.3%, 온수 9.45%, ...
        # => 전기, LNG, 기타로 분류

        console.print('용도', self.use_dist())
        # => 열병합용을 난방용으로, 생산/생산난방용을 기타로 분류

        # summary
        cols = ['설비명', '에너지원', '용도', 'type', 'use', 'source']
        utils.pl.PolarsSummary(self.data.select(cols)).write_excel(
            self.conf.dirs.analysis / '0002.설비 summary.xlsx'
        )
        group = [Vars.KEMC_CODE, Vars.KEMC_KOR]
        utils.pl.PolarsSummary(
            self.data.select([*group, *cols]), group=group
        ).write_excel(self.conf.dirs.analysis / '0002.설비 summary-KEMC.xlsx')

        d = self.conf.dirs.data

        # 주설비 판단
        main_equipment = self.main_equipment()
        main_equipment.write_parquet(d / 'equipment-main-equipment.parquet')
        main_equipment.write_excel(
            d / 'equipment-main-equipment.xlsx', column_widths=120
        )
        (
            main_equipment.rename({'equipment': '주설비', 'ratio': '주설비비율'})
            .pivot(
                'use',
                index=self.index_full,
                values=['주설비', '주설비비율'],
                sort_columns=True,
            )
            .write_excel(d / 'equipment-main-equipment-pivot.xlsx', column_widths=120)
        )

        # 수치 feature 계산
        feature = self.feature()
        console.print('features', feature)

        name = 'equipment-feature'
        feature.write_parquet(self.conf.dirs.data / f'{name}.parquet')
        feature.write_excel(self.conf.dirs.data / f'{name}.xlsx', column_widths=120)

        self.pivot_feature(feature, ['use', 'source']).write_excel(
            self.conf.dirs.data / f'{name}-(use-source).xlsx'
        )
        self.pivot_feature(feature, ['use', 'type', 'source']).write_excel(
            self.conf.dirs.data / f'{name}-(use-type-source).xlsx'
        )


@app['bldg'].command
def bldg_equipment(*, conf: Config):
    """설비 특성 추출 (설비 종류, 에너지원별 사용량)."""
    _BuildingEquipment(conf)()


@app['bldg'].command
def bldg_elec(*, conf: Config):
    """전전화 건물(?) 목록."""
    # TODO 방법 체크
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

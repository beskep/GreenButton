from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING, Literal, overload

import polars as pl
from loguru import logger

from greenbutton import misc
from greenbutton.utils import Progress
from scripts.utils import MetropolitanGov

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from scripts.ami.energy_intensive.config import Config

type InterpDay = Literal[4, 30] | None  # 한전 보간일 기준 데이터 구분

KEMC_CODE: dict[int, str] = {
    501: '상용',
    502: '공공',
    503: '아파트',
    504: '호텔',
    505: '병원',
    506: '학교',
    507: 'IDC',
    508: '연구소',
    509: '백화점',
    599: '건물기타',
}


class EmptyDataError(ValueError):
    pass


def _iter_ami(root: Path, code: int, interp_day: InterpDay = None):
    c = f'{code}{KEMC_CODE[code]}'
    p = 'post' if interp_day is None else f'D+{interp_day}'

    # 2020~2022년
    for path in root.glob(f'{c}_AMI*_{p}*.parquet'):
        yield pl.scan_parquet(path)

    # 2023년
    for path in root.glob(f'{c}_AMI2023.parquet'):
        yield pl.scan_parquet(path)


@dc.dataclass
class BuildingInfo:
    ente: int  # 업체 코드
    kemc: int  # 용도 코드
    name: str
    area: float  # 연면적
    address: str | None
    region: str | None

    def file_name(self):
        kemc_str = KEMC_CODE[self.kemc]
        return f'{self.kemc}({kemc_str})_{self.region}_{self.name}'


@dc.dataclass
class Buildings:
    conf: Config

    electric: bool = True
    """전전화 건물만 대상 여부"""

    buildings: pl.DataFrame = dc.field(init=False)

    def __post_init__(self):
        file = 'buildings-electric' if self.electric else 'buildings'
        self.buildings = (
            pl.scan_parquet(self.conf.root / f'{file}.parquet')
            .rename({'업체코드': 'ente'})
            .with_columns(
                pl.col('ente').cast(pl.UInt32),
                pl.col('업종')
                .replace({'IDC(전화국)': 'IDC'})
                .replace_strict({v: k for k, v in KEMC_CODE.items()})
                .alias('KEMC_CODE'),
            )
            .filter(pl.col('ente').is_first_distinct())
            .collect()
        )

    @overload
    def iter_rows(
        self,
        *cols,
        named: Literal[False] = ...,
        track: bool = ...,
    ) -> Iterable[tuple]: ...

    @overload
    def iter_rows(
        self,
        *cols,
        named: Literal[True],
        track: bool = ...,
    ) -> Iterable[dict[str, object]]: ...

    def iter_rows(self, *cols, named: bool = False, track: bool = True):
        it: Iterable[tuple] | Iterable[dict[str, object]]
        it = self.buildings.select(cols or pl.all()).iter_rows(named=named)  # type: ignore[call-overload]

        if track:
            it = Progress.trace(it, total=self.buildings.height)  # type: ignore[assignment]

        return it

    def iter_buildings(self, *, track: bool = True):
        for ente, kemc, name, area, address in self.iter_rows(
            'ente', 'KEMC_CODE', '업체명', '연면적(㎡)', '주소', track=track
        ):
            region = MetropolitanGov.search(address)

            if region is not None:
                try:
                    region = MetropolitanGov.asos_region(region)
                except ValueError as e:
                    logger.warning(repr(e))

            yield BuildingInfo(
                ente=ente,
                kemc=kemc,
                name=name,
                area=area,
                address=address,
                region=region,
            )

    def ami(
        self,
        ente: int,
        kemc: int,
        interp_day: InterpDay = None,
    ) -> pl.LazyFrame:
        return (
            pl.concat(
                _iter_ami(self.conf.dirs.data, code=kemc, interp_day=interp_day),
                how='diagonal',
            )
            .filter(pl.col('ente') == ente)
            .drop_nulls('value')
        )

    def temperature(self, region: str):
        path = self.conf.root.parents[1] / 'weather/temperature.parquet'
        return (
            pl.scan_parquet(path)
            .filter(pl.col('region2') == region)
            .group_by(pl.col('datetime').dt.date().alias('date'))
            .agg(pl.mean('ta').alias('temperature'))
        )

    def data(
        self,
        bldg: BuildingInfo,
        interp_day: InterpDay = None,
    ):
        if bldg.region is None:
            msg = 'building.region is None'
            raise ValueError(msg, bldg)

        ami = (
            self.ami(ente=bldg.ente, kemc=bldg.kemc, interp_day=interp_day)
            .group_by('date')
            .agg(pl.sum('value').truediv(bldg.area).alias('eui'))
            .sort(pl.col('date'))
            .collect()
        )

        if not ami.height:
            raise EmptyDataError(bldg)

        temperature = self.temperature(region=bldg.region).collect()
        data = ami.join(temperature, on='date', validate='1:1').sort('date')

        years = data['date'].dt.year().unique()
        return data.with_columns(
            is_holiday=misc.is_holiday(pl.col('date'), years=years)
        )

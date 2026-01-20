"""
기상청 ASOS 자료 다운로드.

Notes
-----
https://www.data.go.kr/data/15057210/openapi.do
"""

from __future__ import annotations

import dataclasses as dc
import functools
import math
import tomllib
import urllib.parse
from itertools import repeat
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, ClassVar, Self

import cyclopts
import msgspec
import polars as pl
import polars.selectors as cs
import requests
import rich
from loguru import logger
from whenever import Instant, PlainDateTime

from greenbutton.utils.cli import App
from greenbutton.utils.terminal import Progress

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from _typeshed import StrPath


class OpenApiKey(msgspec.Struct):
    encoding: str
    decoding: str


class OpenApiParameter(msgspec.Struct, rename='camel'):
    page_no: int = 1
    num_of_rows: int = 10
    data_type: str = 'JSON'
    data_cd: str = 'ASOS'
    date_cd: str = 'HR'
    start_dt: str = '20190101'
    start_hh: str = '00'
    end_dt: str = '20190102'
    end_hh: str = '01'
    stn_ids: str = '108'


class AsosOpenAPI(msgspec.Struct):
    key: OpenApiKey
    prefix: str
    parameter: OpenApiParameter

    @property
    def param(self):
        return self.parameter

    @classmethod
    def read(cls, path: str | Path = 'config/.asos_api.toml'):
        data = tomllib.loads(Path(path).read_text('UTF-8'))
        return msgspec.convert(data, type=cls)

    def url(self):
        param = urllib.parse.urlencode(msgspec.to_builtins(self.parameter))
        return f'{self.prefix}?serviceKey={self.key.encoding}&{param}'

    def request(self, timeout=1.0) -> requests.Response:
        return requests.get(self.url(), timeout=timeout)


class AsosConfig(msgspec.Struct):
    root: Path
    api: AsosOpenAPI

    @staticmethod
    def _dec_hook(t: type, obj):
        if t is Path:
            return Path(obj)
        return obj

    @classmethod
    def read(cls, path: str | Path = 'config/.asos.toml') -> Self:
        data = tomllib.loads(Path(path).read_text('UTF-8'))
        return msgspec.convert(data, type=cls, dec_hook=cls._dec_hook)


class ResponseItems(msgspec.Struct):
    item: list[dict[str, str]]


class ResponseBody(msgspec.Struct, rename='camel'):
    data_type: str
    items: ResponseItems
    page_no: int
    num_of_rows: int
    total_count: int


class ResponseHeader(msgspec.Struct, rename='camel'):
    result_code: str
    result_msg: str


class Response(msgspec.Struct):
    header: ResponseHeader
    body: ResponseBody


class AsosResponse(msgspec.Struct):
    response: Response

    @classmethod
    def is_valid(cls, path: StrPath) -> bool:
        path = Path(path)
        if not path.exists():
            return False

        try:
            cls.read_items(path)
        except msgspec.DecodeError:
            return False

        return True

    @classmethod
    def items(cls, buf: bytes | str):
        r = msgspec.json.decode(buf, type=cls)
        return r.response.body.items.item

    @classmethod
    def read_items(cls, path: StrPath):
        try:
            return cls.items(Path(path).read_text('UTF-8'))
        except msgspec.DecodeError as e:
            msg = f'DecodeError on "{path}"'
            raise msgspec.DecodeError(msg) from e

    @classmethod
    def read_dataframe(cls, path: StrPath):
        return pl.DataFrame(cls.read_items(path), orient='row')


app = App()


@app.command
def asos_station(path: StrPath = 'config/asos_station.toml'):
    """ASOS 지역, 지점 정보 dataframe/json 변환."""
    path = Path(path)
    data = tomllib.loads(path.read_text('UTF-8'))

    def _iter():
        for region1, d in data.items():
            for region2, stations in d.items():
                for station in stations:
                    yield region1, region2, station

    station = (
        pl
        .DataFrame(
            list(_iter()), schema=['region1', 'region2', 'station'], orient='row'
        )
        .with_columns(
            pl.col('station').str.extract_groups(r'^(?<station>\w+)\((?<code>\d+)\)')
        )
        .unnest('station')
        .with_columns(pl.col('code').cast(pl.Int16))
    )

    station.write_json(path.with_suffix('.json'))
    rich.print(station)


@cyclopts.Parameter(name='*')
@dc.dataclass(frozen=True)
class DownloadRange:
    start: str
    end: str | None

    FORMAT: ClassVar[dict[int, str]] = {4: '%Y', 6: '%Y%m'}

    @functools.cached_property
    def t0(self):
        return self._parse_time(self.start)

    @functools.cached_property
    def t1(self):
        return (
            self._end_time(self.start, self.t0)
            if self.end is None
            else self._parse_time(self.end)
        )

    @classmethod
    def _parse_time(cls, s: str) -> Instant:
        return PlainDateTime.parse_strptime(s, format=cls.FORMAT[len(s)]).assume_utc()

    @staticmethod
    def _end_time(start: str, t0: Instant) -> Instant:
        by_month = len(start) == 6  # noqa: PLR2004
        end = (
            t0
            .to_fixed_offset()
            .to_plain()
            .add(years=0 if by_month else 1, months=1 if by_month else 0)
            .assume_utc()
        )

        # 11시 이후 전날 자료 조회 가능 -> 최대 2일 전 자료까지 조회 설정
        available = Instant.now().subtract(hours=48)

        return min(end, available)

    def max_page(self, rows: int) -> int:
        return math.ceil((self.t1 - self.t0).in_hours() / rows)

    def months(self):
        delta = (self.t1 - self.t0).in_days_of_24h() / 30
        t0 = self.t0.to_system_tz()
        months = [t0.add(months=x) for x in range(math.ceil(delta))]
        return [x for x in months if x < Instant.now()][:-1]


class AsosDownloader(msgspec.Struct):
    duration: DownloadRange
    rows: int = 800

    conf: AsosConfig = msgspec.field(default_factory=AsosConfig.read)

    @property
    def api(self):
        return self.conf.api

    def __post_init__(self):
        self.update()

    def update(self):
        self.api.param.start_dt = self.duration.t0.py_datetime().strftime('%Y%m%d')
        self.api.param.end_dt = self.duration.t1.py_datetime().strftime('%Y%m%d')
        self.api.param.num_of_rows = self.rows
        return self

    def max_page(self):
        return self.duration.max_page(rows=self.rows)

    def iter_api(
        self, station_name: str | None = None
    ) -> Iterable[tuple[AsosOpenAPI, str]]:
        self.update()

        max_page = self.max_page()
        name = f'({station_name})' if station_name else ''
        prefix = (
            f'ASOS-{self.api.param.stn_ids}{name}-'
            f'{self.api.param.start_dt}-{self.api.param.end_dt}-'
            f'rows{self.rows}'
        )

        for page in range(1, max_page + 1):
            self.api.param.page_no = page
            case = f'{prefix}-page{page}of{max_page}'
            yield self.api, case

    def track_api(
        self,
        ids: Sequence[int],
        names: Iterable[str | None] | None = None,
    ) -> Iterable[tuple[AsosOpenAPI, str]]:
        if names is None:
            names = repeat(None, len(ids))

        def _iter():
            for id_, name in zip(ids, names, strict=True):
                self.api.param.stn_ids = str(id_)

                yield from self.iter_api(name)

        yield from Progress.iter(_iter(), total=len(ids) * self.max_page())


@app.command
def download(  # noqa: PLR0913
    output: Path | None = None,
    *,
    dry_run: bool = False,
    station: int | None = None,
    start: str = '202001',
    end: str | None = None,
    rows: int = 800,
    timeout: float = 120,
    sleep_seconds: float = 0.1,
):
    """OpenAPI 다운로드 (기상청_지상(종관, ASOS) 시간자료 조회서비스)."""
    asos = pl.read_json('config/asos_station.json')
    if station is not None:
        asos = asos.filter(pl.col('code') == station)

    duration = DownloadRange(start=start, end=end)
    downloader = AsosDownloader(duration=duration, rows=rows)

    output = output or downloader.conf.root / 'json'

    logger.info('#station={}', asos.height)
    logger.info('#page={}', duration.max_page(rows=rows))

    session = requests.Session()
    for api, case in downloader.track_api(
        ids=asos['code'].to_list(), names=asos['station'].to_list()
    ):
        if AsosResponse.is_valid(path := output / f'{case}.json'):
            continue

        logger.info(case)

        if dry_run:
            continue

        response = session.get(api.url(), timeout=timeout)
        response.raise_for_status()

        path.write_text(response.text)
        sleep(sleep_seconds)


@app.command
def batch_download(
    start: int = 2020,
    end: int = 2025,
    *,
    output: Path | None = None,
    dry_run: bool = False,
):
    t0 = PlainDateTime(start, 1, 1)
    t1 = PlainDateTime(end, 1, 1)
    delta = (t1.assume_utc() - t0.assume_utc()).in_days_of_24h() / 30  # 대략
    months = [t0.add(months=x).assume_utc() for x in range(math.ceil(delta))]
    months = [x for x in months if x < Instant.now()][:-1]

    output = output or AsosConfig.read().root / 'json'
    output.mkdir(exist_ok=True)

    for month in months:
        m = month.py_datetime().strftime('%Y%m')
        logger.info('month={}', m)
        download(output=output, start=m, dry_run=dry_run)


_DEFAULT_RANGE = DownloadRange('202001', '202501')


@app.command
def download_matched(
    r: DownloadRange = _DEFAULT_RANGE,
    *,
    output: Path | None = None,
    dry_run: bool = False,
    sleep_seconds: float = 0.05,
):
    """2025-07-14. SQI에서 제공한 건물과 매칭된 ASOS 지점 자료 다운로드."""
    root = AsosConfig.read().root
    output = output or root / 'json-matched'
    output.mkdir(exist_ok=True)

    stations = (
        pl
        .scan_parquet(root.parent / 'WeatherStation.parquet')
        .select(
            pl.col('기상관측지점').alias('station'),
            pl.col('기상청 지점명').alias('station-name'),
        )
        .unique()
        .sort('station')
        .collect()
    )

    for month in r.months():
        logger.info(month)
        m = f'{month.year}{month.month:02d}'
        downloader = AsosDownloader(DownloadRange(m, end=None))

        session = requests.Session()
        for api, case in downloader.track_api(
            ids=stations['station'].to_list(),
            names=stations['station-name'].to_list(),
        ):
            if AsosResponse.is_valid(path := output / f'{case}.json'):
                continue

            logger.info(case)

            if dry_run:
                continue

            response = session.get(api.url(), timeout=120)
            response.raise_for_status()

            path.write_text(response.text)
            sleep(sleep_seconds)


@app.command
def parse_response(src: str = 'json-matched'):
    root = AsosConfig.read().root

    floats = [
        'ts',
        'm01Te',
        'm02Te',
        'm03Te',
        'hr3Fhsc',
        'dsnw',
        'icsr',
        'ss',
        'ps',
        'pa',
        'td',
        'pv',
        'hm',
        'wd',
        'ws',
        'rn',
        'ta',
    ]

    stations = (
        pl
        .Series('files', [x.name for x in root.glob(f'{src}/*.json')])
        .str.extract_groups(r'^ASOS\-(?<station>\d+)\((?<station_name>.*)\)\-')
        .struct.unnest()
        .with_columns(pl.col('station').cast(pl.UInt16))
        .unique('station')
        .sort('station')
    )

    def it():
        for station, name in Progress.iter(stations.iter_rows(), total=stations.height):
            logger.info('station={}({})', station, name)

            df = (
                pl
                .concat(
                    AsosResponse.read_dataframe(x)
                    for x in root.glob(f'{src}/ASOS-{station}*.json')
                )
                .drop('rnum')
                .with_columns(pl.all().replace({'': None}))
                .with_columns(
                    pl.col('tm').str.to_datetime('%Y-%m-%d %H:%M'),
                    pl.col('stnId').cast(pl.UInt16),
                    pl.col(floats).cast(pl.Float64),
                    cs.ends_with('Qcflg').cast(pl.UInt8),  # null 정상, 1 오류, 9 결측
                    cs.starts_with('dc10').cast(pl.UInt8),  # 운량
                )
                .sort('tm')
            )

            yield df

    data = pl.concat(it())
    data.write_parquet(root / 'weather.parquet')
    print(data)


@app.command
def regional_average():
    """ASOS 기상자료 지역별 평균."""
    stations = (
        pl
        .read_json('config/asos_station.json')
        .select(
            'region1',
            # 주소로부터 구분하기 어려움
            pl.col('region2').replace({'강원영동': '강원', '강원영서': '강원'}),
            pl.col('code').cast(pl.UInt16).alias('stnId'),
        )
        .sort(pl.all())
    )

    root = AsosConfig.read().root

    temperature = (
        pl
        .scan_parquet(list(root.glob('binary/*.parquet')))
        .select(pl.col('tm').alias('datetime'), pl.col('stnId').cast(pl.UInt16), 'ta')
        .collect()
        .join(stations, on='stnId', how='left')
        .group_by('datetime', 'region1', 'region2')
        .agg(pl.mean('ta'))
        .sort(pl.all())
    )

    rich.print(temperature)

    temperature.write_parquet(root / 'temperature.parquet')
    temperature.head(1000).write_excel(
        root / 'temperature-sample.xlsx', column_widths=200
    )


if __name__ == '__main__':
    from greenbutton import utils

    utils.terminal.LogHandler.set()
    app()

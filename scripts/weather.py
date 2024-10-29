"""
기상청 ASOS 자료 다운로드.

Notes
-----
https://www.data.go.kr/data/15057210/openapi.do
"""

from __future__ import annotations

import math
import tomllib
import urllib.parse
from itertools import repeat
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, ClassVar

import cyclopts
import msgspec
import polars as pl
import requests
import rich
from loguru import logger
from whenever import Instant, LocalDateTime

from greenbutton.utils import Progress
from scripts.config import Config

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

    def request(self, timeout=1.0):
        return requests.get(self.url(), timeout=timeout)


class Items(msgspec.Struct):
    item: list[dict[str, str]]


class Body(msgspec.Struct, rename='camel'):
    data_type: str
    items: Items
    page_no: int
    num_of_rows: int
    total_count: int


class Header(msgspec.Struct, rename='camel'):
    result_code: str
    result_msg: str


class Response(msgspec.Struct):
    header: Header
    body: Body


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


app = cyclopts.App()


@app.command
def asos_station(path: Path = Path('config/asos_station.toml')):
    """ASOS 지역, 지점 정보 dataframe/json 변환."""
    data = tomllib.loads(path.read_text('UTF-8'))

    def _iter():
        for region1, d in data.items():
            for region2, stations in d.items():
                for station in stations:
                    yield region1, region2, station

    station = (
        pl.DataFrame(
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


class DownloadDuration(msgspec.Struct):
    start: str
    end: str | None

    t0: Instant = Instant.from_timestamp(0)
    t1: Instant = Instant.from_timestamp(0)

    FORMAT: ClassVar[dict[int, str]] = {4: '%Y', 6: '%Y%m'}

    def __post_init__(self):
        self.t0 = self._parse_time(self.start)
        self.t1 = (
            self._end_time(self.start, self.t0)
            if self.end is None
            else self._parse_time(self.end)
        )

    @classmethod
    def _parse_time(cls, s: str) -> Instant:
        return LocalDateTime.strptime(s, cls.FORMAT[len(s)]).assume_utc()

    @staticmethod
    def _end_time(start: str, t0: Instant) -> Instant:
        by_month = len(start) == 6  # noqa: PLR2004
        end = (
            t0.to_fixed_offset()
            .local()
            .add(years=0 if by_month else 1, months=1 if by_month else 0)
            .assume_utc()
        )

        # 11시 이후 전날 자료 조회 가능 -> 최대 2일 전 자료까지 조회 설정
        available = Instant.now().subtract(hours=48)

        return min(end, available)

    def max_page(self, rows: int):
        return math.ceil((self.t1 - self.t0).in_hours() / rows)


class AsosDownloader(msgspec.Struct):
    duration: DownloadDuration
    rows: int = 800

    api: AsosOpenAPI = msgspec.field(default_factory=AsosOpenAPI.read)

    def __post_init__(self):
        self.update()

    def update(self):
        self.api.param.start_dt = self.duration.t0.py_datetime().strftime('%Y%m%d')
        self.api.param.end_dt = self.duration.t1.py_datetime().strftime('%Y%m%d')
        self.api.param.num_of_rows = self.rows
        return self

    def max_page(self):
        return self.duration.max_page(rows=self.rows)

    def iter_api(self, station_name: str | None = None):
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

    def track_api(self, ids: Sequence[int], names: Iterable[str | None] | None = None):
        if names is None:
            names = repeat(None, len(ids))

        def _iter():
            for id_, name in zip(ids, names, strict=True):
                self.api.param.stn_ids = str(id_)

                yield from self.iter_api(name)

        yield from Progress.with_track(_iter(), total=len(ids) * self.max_page())


@app.command
def download(  # noqa: PLR0913
    output: Path | None = None,
    *,
    dry_run: bool = False,
    station: int | None = None,
    start: str = '202201',
    end: str | None = None,
    rows: int = 800,
    timeout: float = 120,
    sleep_seconds: float = 0.1,
):
    """OpenAPI 다운로드 (기상청_지상(종관, ASOS) 시간자료 조회서비스)."""
    output = output or Config.read().ami.root / 'Public/99weather/json'
    output.mkdir(exist_ok=True)

    asos = pl.read_json('config/asos_station.json')
    if station is not None:
        asos = asos.filter(pl.col('code') == station)

    duration = DownloadDuration(start=start, end=end)
    downloader = AsosDownloader(duration=duration, rows=rows)

    logger.info('#station={}', asos.height)
    logger.info('#page={}', duration.max_page(rows=rows))

    session = requests.Session()
    for api, case in downloader.track_api(
        ids=asos.select('code').to_series().to_list(),
        names=asos.select('station').to_series().to_list(),
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
    start: int = 2022,
    end: int = 2025,
    *,
    dry_run: bool = False,
):
    t0 = LocalDateTime(start, 1, 1)
    t1 = LocalDateTime(end, 1, 1)
    delta = (t1.assume_utc() - t0.assume_utc()).in_days_of_24h() / 30  # 대략
    months = [t0.add(months=x).assume_utc() for x in range(math.ceil(delta))]
    months = [x for x in months if x < Instant.now()][:-1]

    for month in months:
        m = month.py_datetime().strftime('%Y%m')
        logger.info('month={}', m)
        download(start=m, dry_run=dry_run)


@app.command
def parse_response():
    root = Config.read().ami.root / 'Public/99weather'

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

    regions = (
        pl.Series('files', [x.name for x in root.glob('json/*.json')])
        .str.extract('^ASOS-(.*?)-')
        .unique()
        .sort()
    )

    for region in Progress.with_track(regions):
        logger.info('region={}', region)

        df = (
            pl.concat(
                AsosResponse.read_dataframe(x)
                for x in root.glob(f'json/ASOS-{region}*.json')
            )
            .drop('rnum')
            .with_columns(
                pl.col('tm').str.to_datetime('%Y-%m-%d %H:%M'),
                pl.col('stnId').cast(pl.UInt16),
                pl.col(floats).replace('', None).cast(pl.Float64),
            )
            .sort('tm')
        )

        df.write_parquet(root / f'ASOS-{region}.parquet')


if __name__ == '__main__':
    from greenbutton import utils

    utils.set_logger()
    app()

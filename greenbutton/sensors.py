from __future__ import annotations

import dataclasses as dc
import io
import re
import tomllib
from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from typing import IO, TYPE_CHECKING, ClassVar, Literal, overload
from warnings import warn

import more_itertools as mi
import polars as pl
import polars.selectors as cs

if TYPE_CHECKING:
    from collections.abc import Collection

    import numpy as np
    from numpy.typing import NDArray
    from polars._typing import FrameType


type Source = str | Path | IO[str] | IO[bytes]


class DataFormatError(ValueError):
    pass


def _iter_line(source: Source, encoding: str = 'UTF-8') -> Iterator[str]:
    match source:
        case io.StringIO() | io.BytesIO():
            source.seek(0)

    match source:
        case io.StringIO():
            yield from source
        case io.BytesIO():
            yield from (b.decode(encoding) for b in source)
        case str() | Path():
            with Path(source).open('r', encoding=encoding) as f:
                yield from f
        case _:
            raise TypeError(source)


def read_tr7(source: Source | bytes, *, unpivot=True):
    data = (
        pl.read_csv(
            source,
            new_columns=['datetime', 'datetime2', 'T', 'RH'],
            skip_rows=2,
            schema={
                'datetime': pl.Datetime,
                'datetime2': pl.String,
                'T': pl.Float64,
                'RH': pl.Float64,
            },
        )
        .with_columns(pl.col('RH') / 100.0)
        .drop('datetime2')
        .sort('datetime')
    )

    if unpivot:
        unit = pl.col('variable').replace_strict('T', '°C', default=None).alias('unit')
        data = (
            data.unpivot(['T', 'RH'], index='datetime')
            .select('datetime', 'variable', 'value', unit)
            .with_columns()
        )

    return data


@dc.dataclass(frozen=True)
class PMVReader:
    source: Source
    _: dc.KW_ONLY
    rh_percentage: bool = False

    # TODO read clo, met

    @cached_property
    def dataframe(self) -> pl.DataFrame:
        raise NotImplementedError


@dc.dataclass(frozen=True)
class TestoPMV(PMVReader):
    """Testo400 reader."""

    _: dc.KW_ONLY
    probe_config: Mapping[int, str] | str | Path = 'config/sensor.toml'

    encoding: str = 'UTF-8'
    datetime: str = '날짜/시간'

    exclude: Collection[str] | None = (
        '압력-Turbulence',
        '압력-ComfortKit',
        '온도-Turbulence',
        '차압-ComfortKit',
        '이슬점-AirQuality',
    )
    unit_var: Collection[tuple[str, str]] | Mapping[str, str] = (
        ('℃', '온도'),
        ('°C', '온도'),
        ('%RH', '상대습도'),
        ('m/s', '기류'),
        ('bar', '압력'),
        ('hPa', '차압'),
        ('ppm', 'CO2'),
        ('g/m³', '수증기'),
    )

    @cached_property
    def probes(self) -> dict[int, str]:
        if isinstance(self.probe_config, Mapping):
            return dict(self.probe_config)

        conf = tomllib.loads(Path(self.probe_config).read_text('UTF-8'))

        def id_probe():
            for k, ids in conf['testo400'].items():
                for i in ids:
                    yield i, k

        return dict(id_probe())

    @classmethod
    def _iter_csv(cls, source: str | Path | IO[str] | IO[bytes], encoding: str):
        for line in _iter_line(source, encoding=encoding):
            if not line.removesuffix('\n'):
                continue

            if line.startswith(('전체 평균', '<')):
                break

            yield line

    def _read_csv(self):
        text = ''.join(self._iter_csv(source=self.source, encoding=self.encoding))

        if self.datetime not in text:
            raise DataFormatError(text)

        data = pl.read_csv(io.StringIO(text), null_values=['-', 'xxx']).drop('')

        fields = ('y', 'm', 'd', 't', 'p')
        datetime = (
            data.select(
                pl.col(self.datetime)
                .str.replace_many(['오전', '오후'], ['AM', 'PM'])
                .str.split(' ')
                .list.to_struct(fields=fields)
            )
            .unnest(self.datetime)
            .with_columns(
                pl.col('y', 'm', 'd').str.strip_chars(' .').str.pad_start(2, '0')
            )
            .select(
                pl.format('{}-{}-{} {} {}', *fields)
                .str.to_datetime('%y-%m-%d %r')
                .alias(self.datetime)
            )
        )

        return data.with_columns(datetime.to_series())

    def _unpivot(self, frame: FrameType) -> FrameType:
        return (
            frame.unpivot(index=self.datetime, variable_name='_variable')
            .with_columns(
                pl.col('_variable')
                .str.extract_groups(
                    r'^(?<id>\d+)?\s?(?<variable>.*?)?\s?(\[(?<unit>.*)\])?$'
                )
                .alias('_var')
            )
            .unnest('_var')
            .with_columns(
                pl.col('id')
                .cast(int)
                .replace_strict(self.probes, default='ComfortKit')
                .alias('probe'),
                pl.col('variable').replace({
                    '': None,
                    'TC1': '흑구온도',
                    'TC2': '흑구온도',
                }),
            )
            .with_columns(
                pl.when(pl.col('variable').is_null())
                .then(pl.col('unit').replace_strict(dict(self.unit_var), default=None))
                .otherwise(pl.col('variable'))
                .alias('variable')
            )
            .select(
                pl.col(self.datetime).alias('datetime'),
                'variable',
                pl.col('id').alias('probe_id'),
                'probe',
                pl.col('value').cast(pl.Float64),
                'unit',
            )
        )

    @cached_property
    def dataframe(self):
        data = self._unpivot(self._read_csv())

        if self.exclude:
            data = data.filter(
                pl.format('{}-{}', 'variable', 'probe').is_in(self.exclude).not_()
            )

        if not self.rh_percentage:
            data = data.with_columns(
                pl.when(pl.col('variable') == '상대습도')
                .then(pl.col('value') / 100.0)
                .otherwise(pl.col('value'))
                .alias('value'),
                pl.col('unit').replace({'%RH': None}),
            )

        return data


@dc.dataclass
class DeltaOhmConfig:
    pmv_only: bool = True

    separator: str = ';'
    null: str = 'ERR.'

    interval_pattern: str = r'Sample interval= ([\d.]+)sec.*'
    header_prefix: str = 'Sample interval='
    data_prefix: str = 'Date='


@dc.dataclass(frozen=True)
class DeltaOhmPMV(PMVReader):
    """DeltaOhm HD32.2 reader."""

    _: dc.KW_ONLY
    conf: DeltaOhmConfig = dc.field(default_factory=DeltaOhmConfig)

    VARIABLES: ClassVar[dict[str, str]] = {
        'Ta': '온도',
        'Tw': '습구온도',  # NOTE 불확실 - 공식 문서에 없음
        'Tg': '흑구온도',
        'RH': '상대습도',
        'Va': '기류',
    }
    MISC_INDEX: ClassVar[tuple[str, ...]] = (
        'Tr[C]',  # medium radiant temperature
        'WBGT(i)[C]',  # Wet Bulb Globe Temperature (interior)
        'WBGT(o)[C]',  # Wet Bulb Globe Temperature (exterior)
        'HI[C]',  # Heat Index
        'UTCI[C]',  # Universal Thermal Climate Index
        'PET[C]',  # Perceived Equivalent Temperature
    )

    @cached_property
    def interval(self):
        # FIXME dataframe과 동시 사용 시 오류
        for line in _iter_line(self.source):
            if m := re.match(self.conf.interval_pattern, line):
                return float(m.group(1))

        msg = f'Interval not found (pattern="{self.conf.interval_pattern}")'
        raise ValueError(msg)

    @staticmethod
    def _iter_header(data: str):
        item: tuple[str, str]
        items = mi.batched(['', *data.split(';')], 2)

        for first, last, item in mi.mark_ends(items):  # type: ignore[assignment]
            if first:
                yield item[1]
            elif last:
                yield item[0]
            else:
                yield f'{item[0]}[{item[1]}]' if item[1] else item[0]

    @classmethod
    def _header(cls, data: str):
        return ';'.join(cls._iter_header(data)).replace(';\r\n', '\r\n')

    def _iter_row(self, source: Source):
        check_header = True
        for line in _iter_line(source):
            if check_header and line.startswith(self.conf.header_prefix):
                yield self._header(line)
                check_header = False
            elif line.startswith(self.conf.data_prefix):
                yield line

        if check_header:
            raise DataFormatError(source)

    def _read_csv(self, source: Source):
        return (
            pl.read_csv(
                io.StringIO(''.join(self._iter_row(source))),
                separator=self.conf.separator,
                null_values=self.conf.null,
            )
            .with_columns(
                pl.first()
                .str.strip_prefix(self.conf.data_prefix)
                .str.to_datetime()
                .alias('datetime')
            )
            .drop(pl.first())  # type: ignore[arg-type]
            .with_columns(cs.string().str.strip_chars().cast(pl.Float64))
        )

    @cached_property
    def dataframe(self):
        data = self._read_csv(self.source).drop(
            # SARS-CoV-2 virus natural decay estimation
            cs.starts_with('COV2H', 'COV2D')
        )

        if self.conf.pmv_only:
            data = data.drop(self.MISC_INDEX)

        data = (
            data.unpivot(index='datetime')
            .drop_nulls('value')
            .with_columns(
                pl.col('variable')
                .str.extract_groups(r'^(?<variable>\w+)(\[(?<unit>.*)\])?$')
                .alias('group')
            )
            .select(
                'datetime',
                pl.col('group')
                .struct['variable']
                .replace(self.VARIABLES)
                .alias('variable'),
                pl.col('group').struct['unit'].replace({'C': '°C'}).alias('unit'),
                'value',
            )
        )

        if not self.rh_percentage:
            percent = (pl.col('variable') == '상대습도') & (pl.col('unit') == '%')
            data = data.with_columns(
                pl.when(percent)
                .then(pl.col('value') / 100.0)
                .otherwise(pl.col('value'))
                .alias('value'),
                pl.when(percent)
                .then(pl.lit(None))
                .otherwise(pl.col('unit'))
                .alias('unit'),
            )

        return data


@dc.dataclass
class DataFramePMV:
    """
    DataFrame으로부터 PMV 계산.

    `pythermalcomfort.models.pmv_ppd_ashrae` 참조.

    Parameters
    ----------
    tdb : str
        Dry bulb air temperature 열 이름.
    tr : str
        Mean radiant temperature 열 이름.
    vel : str
        Relative air speed 열 이름.
    rh : str
        Relative humidity 열 이름 ([0, 100] 범위).
    met : float | str
        Metabolic rate (고정값 또는 해당 열 이름).
    clo : float | str
        Clothing insulation (고정값 또는 해당 열 이름).
    wme : float | str
        External work (고정값 또는 해당 열 이름).
    """

    tdb: str
    tr: str
    vel: str
    rh: str

    met: float | str = 0.9
    clo: float | str = 1.0
    wme: float | str = 0.0

    units: Literal['SI', 'IP'] = 'SI'
    limit_inputs: bool = False
    airspeed_control: bool = True

    @overload
    def __call__(
        self,
        data: pl.DataFrame,
        *,
        as_dict: Literal[False] = ...,
    ) -> pl.DataFrame: ...

    @overload
    def __call__(
        self,
        data: pl.DataFrame,
        *,
        as_dict: Literal[True],
    ) -> dict[str, NDArray[np.float16]]: ...

    def __call__(self, data: pl.DataFrame, *, as_dict: bool = False):
        from pythermalcomfort.models import pmv_ppd_ashrae  # noqa: PLC0415
        from pythermalcomfort.utilities import v_relative  # noqa: PLC0415

        def value(v: float | str):
            if isinstance(v, float | int):
                return v

            return data[v].to_numpy()

        rh = data[self.rh]
        if (rh <= 1).all():
            warn('RH는 [0, 100] 범위로 입력해야 합니다.', stacklevel=2)

        met = value(self.met)
        vr = v_relative(v=value(self.vel), met=met)  # pyright: ignore[reportArgumentType]
        kwargs = {
            'tdb': data[self.tdb].to_numpy(),
            'tr': data[self.tr].to_numpy(),
            'vr': vr,
            'rh': rh.to_numpy(),
            'met': met,
            'clo': value(self.clo),
            'wme': value(self.wme),
            'units': self.units,
            'limit_inputs': self.limit_inputs,
            'airspeed_control': self.airspeed_control,
        }

        d = pmv_ppd_ashrae(**kwargs)

        if as_dict:
            return dc.asdict(d) | {'vr': vr}

        return pl.DataFrame([
            pl.Series('vr', vr),
            pl.Series('PMV', d.pmv),
            pl.Series('PPD', d.ppd),
            pl.Series('TSV', d.tsv),
        ])

    calculate = __call__  # noqa: RUF045

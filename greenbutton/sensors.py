from __future__ import annotations

import io
import re
import tomllib
from dataclasses import KW_ONLY, dataclass
from functools import cached_property
from pathlib import Path
from typing import IO, TYPE_CHECKING, Literal, overload
from warnings import warn

import more_itertools as mi
import numpy as np
import polars as pl
import polars.selectors as cs

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from numpy.typing import NDArray
    from polars._typing import FrameType


def read_tr7(path: str | Path | IO[str] | IO[bytes] | bytes, *, melt=True):
    df = (
        pl.read_csv(
            path,
            new_columns=['datetime', 'datetime2', 'T', 'RH'],
            skip_rows=2,
            schema={
                'datetime': pl.Datetime,
                'datetime2': pl.String,
                'T': pl.Float64,
                'RH': pl.Float64,
            },
        )
        .with_columns(pl.col('RH') / 100)
        .drop('datetime2')
        .sort('datetime')
    )

    if melt:
        df = df.melt(id_vars='datetime', value_vars=['T', 'RH']).select(
            'datetime',
            'variable',
            'value',
            pl.col('variable').replace('T', '℃', default=None).alias('unit'),
        )

    return df


@dataclass(frozen=True)
class TestoPMV:
    """Testo400 csv reader."""

    path: str | Path
    _: KW_ONLY
    probes: Mapping[int, str] | str | Path = 'config/sensor.toml'
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
    rh_percentage: bool = False

    def probe_dict(self) -> dict[int, str]:
        if isinstance(self.probes, str | Path):
            conf = tomllib.loads(Path(self.probes).read_text('UTF-8'))

            def _id_probe():
                for k, ids in conf['testo400'].items():
                    for i in ids:
                        yield i, k

            return dict(_id_probe())

        return dict(self.probes)

    def _iter_csv(self):
        with Path(self.path).open('r', encoding=self.encoding) as f:
            for line in f:
                if not line.removesuffix('\n'):
                    continue

                if line.startswith(('전체 평균', '<')):
                    break

                yield line

    @cached_property
    def wide_dataframe(self):
        text = ''.join(self._iter_csv())
        fields = ('y', 'm', 'd', 't', 'p')

        df = pl.read_csv(io.StringIO(text), null_values='-').drop('')

        dt = (
            df.select(
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

        return df.with_columns(dt.to_series())

    def unpivot(self, df: FrameType) -> FrameType:
        return (
            df.unpivot(index=self.datetime, variable_name='_variable')
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
                .replace_strict(self.probe_dict(), default='ComfortKit')
                .alias('probe'),
                pl.col('variable').replace({'': None, 'TC1': '흑구온도'}),
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
                'value',
                'unit',
            )
        )

    @cached_property
    def dataframe(self):
        df = self.unpivot(self.wide_dataframe)

        if self.exclude:
            df = df.filter(
                pl.format('{}-{}', 'variable', 'probe').is_in(self.exclude).not_()
            )

        if not self.rh_percentage:
            df = df.with_columns(
                pl.when(pl.col('variable') == '상대습도')
                .then(pl.col('value') / 100.0)
                .otherwise(pl.col('value'))
                .alias('value'),
                pl.col('unit').replace({'%RH': None}),
            )

        return df


@dataclass(frozen=True)
class DeltaOhmPMV:
    """DeltaOhm HD32.2 reader."""

    path: str | Path
    _: KW_ONLY
    header_prefix: bytes = b'Sample interval='
    data_prefix: bytes = b'Date='
    interval_pattern: str = r'Sample interval= ([\d.]+)sec.*'
    rh_percentage: bool = False

    @cached_property
    def interval(self):
        with Path(self.path).open('r', encoding='UTF-8') as f:
            for line in f:
                if m := re.match(self.interval_pattern, line):
                    return float(m.group(1))

        msg = f'Interval not found (pattern="{self.interval_pattern}")'
        raise ValueError(msg)

    @staticmethod
    def _iter_header(data: bytes):
        item: tuple[bytes, bytes]
        items = mi.batched([b'', *data.split(b';')], 2)

        for first, last, item in mi.mark_ends(items):  # type: ignore[assignment]
            if first:
                yield item[1]
            elif last:
                yield item[0]
            else:
                yield b'%b[%b]' % item if item[1] else item[0]

    @classmethod
    def _header(cls, data: bytes):
        return b';'.join(cls._iter_header(data)).replace(b';\r\n', b'\r\n')

    def _iter_row(self):
        with Path(self.path).open('rb') as f:
            for line in f:
                if line.startswith(self.header_prefix):
                    yield self._header(line)
                    break

            for line in f:
                if line.startswith(self.data_prefix):
                    yield line

    @cached_property
    def wide_dataframe(self):
        df = pl.read_csv(
            io.BytesIO(b''.join(self._iter_row())), separator=';', null_values='ERR.'
        )
        return (
            df.rename({df.columns[0]: 'datetime'})
            .with_columns(
                pl.col('datetime')
                .str.strip_prefix(self.data_prefix.decode())
                .str.to_datetime()
            )
            .with_columns(cs.string().str.strip_chars().cast(pl.Float64))
        )

    @cached_property
    def dataframe(self):
        df = (
            self.wide_dataframe.unpivot(index='datetime')
            .with_columns(
                pl.col('variable').str.extract_groups(
                    r'^(?<variable>\w+)(\[(?<unit>.*)\])?$'
                )
            )
            .select(
                'datetime',
                pl.col('variable').struct['variable'].alias('variable'),
                pl.col('variable').struct['unit'].replace({'C': '℃'}).alias('unit'),
                'value',
            )
        )

        if not self.rh_percentage:
            condition = (pl.col('variable') == 'RH') & (pl.col('unit') == '%')
            df = df.with_columns(
                pl.when(condition)
                .then(pl.col('value') / 100.0)
                .otherwise(pl.col('value'))
                .alias('value'),
                pl.when(condition)
                .then(pl.lit(None))
                .otherwise(pl.col('unit'))
                .alias('unit'),
            )

        return df


@dataclass
class DataFramePMV:
    tdb: str
    tr: str
    vel: str
    rh: str

    met: float | str = 0.9
    clo: float | str = 1.0
    wme: float | str = 0.0

    standard: Literal['ISO', 'ASHRAE'] = 'ISO'
    units: Literal['SI', 'IP'] = 'SI'
    limit_inputs: bool = False
    airspeed_control: bool = True

    @staticmethod
    def _value(df: pl.DataFrame, v: float | str) -> float | NDArray:
        if isinstance(v, float | int):
            return v

        return df.select(v).to_numpy().ravel()

    @overload
    def calculate(
        self, df: pl.DataFrame, *, as_dict: Literal[False] = ...
    ) -> pl.DataFrame: ...

    @overload
    def calculate(
        self, df: pl.DataFrame, *, as_dict: Literal[True]
    ) -> dict[str, NDArray[np.float16]]: ...

    def calculate(self, df: pl.DataFrame, *, as_dict: bool = False):
        from pythermalcomfort.models import pmv_ppd  # noqa: PLC0415
        from pythermalcomfort.utilities import v_relative  # noqa: PLC0415

        tdb, tr, rh = df.select(self.tdb, self.tr, self.rh).to_numpy().T

        if np.all(rh <= 1):
            warn('RH는 [0, 100] 범위로 입력.', stacklevel=1)

        met = self._value(df, self.met)
        clo = self._value(df, self.clo)
        wme = self._value(df, self.wme)
        vr = v_relative(v=self._value(df, self.vel), met=met)

        d = pmv_ppd(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            standard=self.standard,
            units=self.units,
            limit_inputs=self.limit_inputs,
            airspeed_control=self.airspeed_control,
        )

        if as_dict:
            return d | {'vr': vr}

        return pl.DataFrame([
            pl.Series('vr', vr),
            pl.Series('PMV', d['pmv']),
            pl.Series('PPD', d['ppd']),
        ])

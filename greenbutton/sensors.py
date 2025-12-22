from __future__ import annotations

import dataclasses as dc
import io
import re
import tomllib
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import IO, TYPE_CHECKING, ClassVar, Literal, overload
from warnings import warn

import more_itertools as mi
import polars as pl
import polars.selectors as cs

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator

    import numpy as np
    from numpy.typing import NDArray
    from polars._typing import FrameType


type Source = str | Path | IO[str] | IO[bytes]


class DataFormatError(ValueError):
    pass


_VALUE = r'\d+(?:\.\d+)?'


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


def _parse_datetime(data: pl.Series, formats: Iterable[str]):
    for fmt in formats:
        try:
            return data.str.to_datetime(fmt)
        except pl.exceptions.InvalidOperationError:
            continue

    msg = 'Cannot parse datetime'
    raise DataFormatError(msg, data)


def _match_group(pattern: re.Pattern, text: str):
    for m in pattern.finditer(text):
        for key, value in m.groupdict().items():
            if value:
                yield key, value


def read_tr7(source: Source | bytes, *, unpivot=True):
    data = (
        pl
        .read_csv(
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
            data
            .unpivot(['T', 'RH'], index='datetime')
            .select('datetime', 'variable', 'value', unit)
            .with_columns()
        )

    return data


@dc.dataclass(frozen=True)
class PMVReader:
    source: Source
    _: dc.KW_ONLY
    encoding: str = 'UTF-8'
    rh_percentage: bool = False

    @property
    def meta(self) -> dict[str, float]:
        # met, clo, interval, ...
        raise NotImplementedError

    @property
    def data(self) -> pl.DataFrame:
        raise NotImplementedError


@dc.dataclass(frozen=True)
class TestoPMV(PMVReader):
    """Testo400, Testo480 reader."""

    _: dc.KW_ONLY
    datetime: str = '날짜/시간'
    exclude: Collection[str] | None = (
        '압력-Turbulence',
        '압력-ComfortKit',
        '압력-Unknown',
        '온도-Turbulence',
        '차압-ComfortKit',
        '차압-Unknown',
        '이슬점-AirQuality',
    )

    probe_config: Mapping[int, str] | str | Path = 'config/sensor.toml'

    UNIT_VAR: ClassVar[Collection[tuple[str, str]]] = (
        ('℃', '온도'),
        ('°C', '온도'),
        ('%RH', '상대습도'),
        ('m/s', '기류'),
        ('bar', '압력'),
        ('hPa', '압력'),
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

    def _read(self):
        csv = []
        meta = []
        csv_section = True

        for line in _iter_line(self.source, encoding=self.encoding):
            if not line.removeprefix('\n'):
                continue

            if line.startswith(('전체 평균', '<')):
                csv_section = False

            if csv_section:
                csv.append(line)
            else:
                meta.append(line)

        return ''.join(csv), ''.join(meta)

    @cached_property
    def _data(self):
        csv, meta = self._read()

        # csv
        if self.datetime not in csv:
            raise DataFormatError(self.source)

        data = (
            pl
            .read_csv(io.StringIO(csv), null_values=['-', 'xxx'])
            .drop('', strict=False)
            .with_columns(
                pl.col(self.datetime).str.replace_many(['오전', '오후'], ['AM', 'PM'])
            )
        )
        data = data.with_columns(
            _parse_datetime(
                data[self.datetime],
                formats=['%y. %m. %d. %I:%M:%S %p', '%F %p %I:%M:%S'],
            )
        )

        # meta
        pattern = re.compile(
            rf"""
            (clo\s?=(?P<clo>{_VALUE}))
            | (met\s?=\s?(?P<met>{_VALUE}))
            | ((?P<clo_unit>{_VALUE})\s?m²K/W)
            | ((?P<met_unit>{_VALUE})\s?W/m²)
            """,
            re.VERBOSE | re.IGNORECASE,
        )
        meta_dict = {k: float(v) for k, v in _match_group(pattern, meta)}

        return data, meta_dict

    def _unpivot_testo400(self, frame: FrameType) -> FrameType:
        return (
            frame
            .unpivot(index=self.datetime, variable_name='_variable')
            .with_columns(
                pl
                .col('_variable')
                .str.extract_groups(
                    r'^(?<id>\d+)?\s?(?<variable>.*?)?\s?(\[(?<unit>.*)\])?$'
                )
                .alias('_var')
            )
            .unnest('_var')
            .with_columns(
                pl
                .col('id')
                .cast(int)
                .replace_strict(self.probes, default='Unknown')
                .alias('probe'),
                pl.col('variable').replace({
                    '': None,
                    'TC1': '흑구온도',
                    'TC2': '흑구온도',
                }),
            )
            .with_columns(
                pl
                .when(pl.col('variable').is_null())
                .then(pl.col('unit').replace_strict(dict(self.UNIT_VAR), default=None))
                .otherwise(pl.col('variable'))
                .alias('variable')
            )
            .select(
                self.datetime,
                'variable',
                pl.col('id').alias('probe_id'),
                'probe',
                pl.col('value').cast(pl.Float64),
                'unit',
            )
        )

    def _unpivot_testo480(self, frame: FrameType) -> FrameType:
        p = re.compile(r'^(.*?) \-\d+$')
        d = {
            '°C INT': '온도',
            '°C': '흑구온도',
            '%RH': '상대습도',
            'M/S': '기류',
            'PMV CALC': 'PMV',
            '% PPD CALC': 'PPD',
            'PPM': 'CO2',
        }

        def rename(text: str):
            if m := p.match(text):
                text = m.group(1)

            return d.get(text.upper(), text)

        var_unit = {v: u for u, v in self.UNIT_VAR}
        return (
            frame
            .drop('SecRuntime', strict=False)
            .rename(rename)
            .unpivot(index=self.datetime)
            .with_columns(
                pl.col('variable').replace_strict(var_unit, default=None).alias('unit')
            )
        )

    @property
    def meta(self):
        return self._data[1]

    @cached_property
    def data(self):
        wide, meta = self._data

        if 'PPD [%]' in wide.columns:
            data = self._unpivot_testo400(wide)
        else:
            data = self._unpivot_testo480(wide)

        data = data.rename({self.datetime: 'datetime'}).drop_nulls('value')

        if self.exclude and 'probe' in data.columns:
            data = data.filter(
                pl.format('{}-{}', 'variable', 'probe').is_in(self.exclude).not_()
            )

        if not self.rh_percentage:
            data = data.with_columns(
                pl
                .when(pl.col('variable') == '상대습도')
                .then(pl.col('value') / 100.0)
                .otherwise(pl.col('value'))
                .alias('value'),
                pl.col('unit').replace({'%RH': None}),
            )

        if meta:
            rename = {'clo_unit': 'clo [m²K/W]', 'met_unit': 'met [W/m²]'}
            data = data.with_columns([
                pl.lit(v).alias(rename.get(k, k)) for k, v in meta.items()
            ])

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

    def _read(self):
        csv = []
        meta = []
        check_header = True

        for line in _iter_line(self.source, encoding=self.encoding):
            if check_header and line.startswith(self.conf.header_prefix):
                check_header = False
                meta.append(line)
                csv.append(self._header(line))
            elif check_header:
                meta.append(line)
            elif line.startswith(self.conf.data_prefix):
                csv.append(line)
            else:
                break

        if check_header:
            raise DataFormatError(self.source)

        return ''.join(csv), ''.join(meta)

    @cached_property
    def _data(self):
        csv, meta = self._read()

        # csv
        data = (
            pl
            .read_csv(
                io.StringIO(csv),
                separator=self.conf.separator,
                null_values=self.conf.null,
            )
            .with_columns(
                pl
                .first()
                .str.strip_prefix(self.conf.data_prefix)
                .str.to_datetime()
                .alias('datetime')
            )
            .drop(pl.first())  # type: ignore[arg-type]
            .with_columns(cs.string().str.strip_chars().cast(pl.Float64))
        )

        # meta
        pattern = re.compile(
            rf"""
            (clo\s?=\s?(?P<clo>{_VALUE}))
            | (met\s?=\s?(?P<met>{_VALUE}))
            | (sample\sinterval\s?=\s?(?P<interval>{_VALUE})sec)
            """,
            re.VERBOSE | re.IGNORECASE,
        )
        meta_dict = {k: float(v) for k, v in _match_group(pattern, meta)}

        return data, meta_dict

    @property
    def meta(self):
        return self._data[1]

    @cached_property
    def data(self):
        data, meta = self._data

        data = data.drop(
            # SARS-CoV-2 virus natural decay estimation
            cs.starts_with('COV2H', 'COV2D')
        )

        if self.conf.pmv_only:
            data = data.drop(self.MISC_INDEX)

        data = (
            data
            .unpivot(index='datetime')
            .drop_nulls('value')
            .with_columns(
                pl
                .col('variable')
                .str.extract_groups(r'^(?<variable>\w+)(\[(?<unit>.*)\])?$')
                .alias('group')
            )
            .select(
                'datetime',
                pl
                .col('group')
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
                pl
                .when(percent)
                .then(pl.col('value') / 100.0)
                .otherwise(pl.col('value'))
                .alias('value'),
                pl
                .when(percent)
                .then(pl.lit(None))
                .otherwise(pl.col('unit'))
                .alias('unit'),
            )

        if meta:
            data = data.with_columns(pl.lit(v).alias(k) for k, v in meta.items())

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

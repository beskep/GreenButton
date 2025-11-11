import dataclasses as dc
from pathlib import Path
from typing import ClassVar

import cyclopts
import polars as pl

from greenbutton.utils.cli import App


@dc.dataclass
class _Path:
    root: Path
    raw: Path = Path('00.raw')
    data: Path = Path('01.data')

    def __post_init__(self):
        for field in (f.name for f in dc.fields(self)):
            if field == 'root':
                continue

            setattr(self, field, self.root / getattr(self, field))


@dc.dataclass
class _Ean:
    threshold: dict[str, float]
    weather_station: dict[str, str]


@dc.dataclass
class Config:
    path: _Path
    ean: _Ean


_config = Path(__file__).parent / '.validation.toml'
app = App(
    config=cyclopts.config.Toml(_config, use_commands_as_keys=False),
    result_action=['call_if_callable', 'print_non_int_sys_exit'],
)


@dc.dataclass
class BasePrep:
    conf: Config

    NAME: ClassVar[str] = 'BUILDING'

    def write(self, data: pl.DataFrame):
        if 'building' not in data.columns:
            data = data.with_columns(pl.lit(self.NAME).alias('building'))

        fields = ['building', 'date', 'is_holiday', 'temperature', 'energy']
        fields = [*fields, *(x for x in data.columns if x not in fields)]
        data = data.select(fields)

        output = self.conf.path.data
        data.write_parquet(output / f'00.{self.NAME}.parquet')
        (
            (output)
            .joinpath(f'01.glimpse-{self.NAME}.txt')
            .write_text(data.glimpse(return_type='string'), encoding='utf-8')
        )


if __name__ == '__main__':

    @app.default
    def print_config(conf: Config):
        return conf

    app()

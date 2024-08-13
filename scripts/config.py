from collections.abc import Callable
from pathlib import Path

import msgspec
import polars as pl
from msgspec import Struct


def dec_hook(t: type, obj):
    if t is Path:
        return Path(obj)

    return obj


_dec_hook = dec_hook


class Ami(Struct):
    root: Path = msgspec.field(name='dir')


class ExpBuilding(Struct):
    yeonseo: str
    cheolsan: str
    kepco_paju: str
    yecheon_gov: str
    neulpum: str


class ExpSubDir(Struct):
    TR7: str
    PMV: str
    DB: str
    PLOT: str


class _ExpDir(Struct):
    ROOT: Path

    TR7: Path
    PMV: Path
    DB: Path
    PLOT: Path


class Experiment(Struct):
    root: Path = msgspec.field(name='dir')
    building: ExpBuilding
    subdir: ExpSubDir

    def directory(self, building: str, date: str | None = None):
        try:
            bldg = getattr(self.building, building)
        except AttributeError:
            bldg = building

        root = self.root / bldg / (date or '')
        s = msgspec.to_builtins(self.subdir)

        return _ExpDir(ROOT=root, **{k: root / v for k, v in s.items()})


class Config(Struct):
    ami: Ami
    experiment: Experiment

    @classmethod
    def read(
        cls,
        path='config/config.toml',
        *,
        strict: bool = True,
        dec_hook: Callable | None = None,
    ):
        return msgspec.toml.decode(
            Path(path).read_bytes(),
            type=cls,
            strict=strict,
            dec_hook=dec_hook or _dec_hook,
        )


def sensor_location(path: str | Path = 'config/sensor_location.json', *, xlsx=False):
    path = Path(path)
    schema_overrides = dict.fromkeys(['floor', 'point', 'PMV', 'TR', 'GT'], pl.UInt8)

    if xlsx or not path.exists():
        df = pl.read_excel(
            path.with_suffix('.xlsx'), schema_overrides=schema_overrides
        ).with_columns(pl.col('date').cast(pl.String))
        df.write_json(path)
    else:
        df = pl.read_json(path, schema_overrides=schema_overrides)

    return df


if __name__ == '__main__':
    from rich import print  # noqa: A004

    conf = Config.read()
    print(conf)
    print(conf.experiment.directory(conf.experiment.building.yeonseo))

    print(sensor_location())

from __future__ import annotations

import dataclasses as dc
from pathlib import Path
from typing import Annotated

import cyclopts


@dc.dataclass
class Dirs:
    raw: Path = Path('0000.raw')
    data: Path = Path('0001.data')
    weather: Path = Path('0002.weather')

    analysis: Path = Path('0100.analysis')
    cpr: Path = Path('0200.CPR')


@dc.dataclass
class Config:
    root: Path
    dirs: Dirs

    def __post_init__(self):
        self.update()

    def update(self):
        for field in (f.name for f in dc.fields(self.dirs)):
            p = getattr(self.dirs, field)
            setattr(self.dirs, field, self.root / p)


ConfigParam = Annotated[Config, cyclopts.Parameter(name='*')]


if __name__ == '__main__':
    import tomllib

    import msgspec
    import rich

    def dec_hook(t: type, obj):
        if t is Path:
            return Path(obj)
        return obj

    path = Path('config/.ami.toml')
    data = tomllib.loads(path.read_text('UTF-8'))['public_institution']
    rich.print(msgspec.convert(data, Config, dec_hook=dec_hook))

import dataclasses as dc
from typing import ClassVar

import cyclopts

import scripts.exp.experiment as exp
from greenbutton.utils.cli import App

app = App(
    config=[
        cyclopts.config.Toml(f'config/{x}.toml', use_commands_as_keys=False)
        for x in ['.experiment', 'experiment']
    ],
)


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'ecpm'

    def bldg_dirs(self, bldg: str):
        return exp.Dirs(root=self.root / self.buildings[bldg])

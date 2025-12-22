import dataclasses as dc
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar

import cyclopts
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from loguru import logger

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils.cli import App
from greenbutton.utils.terminal import Progress


@dc.dataclass
class DBDirs:
    log: Path = Path('0000.log')  # C드라이브 `BigData` 폴더 안 데이터
    web: Path = Path('0000.web')  # BEMS 웹 인터페이스로 다운받은 엑셀 파일

    data: Path = Path('0001.data')


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'cheolsan'

    db_dirs: DBDirs = dc.field(default_factory=DBDirs)

    def __post_init__(self):
        for field in (f.name for f in dc.fields(self.db_dirs)):
            p = getattr(self.db_dirs, field)
            setattr(self.db_dirs, field, self.dirs.database / p)


app = App(
    config=cyclopts.config.Toml('config/.experiment.toml', use_commands_as_keys=False)
)


@app.command
def init(*, conf: Config):
    conf.dirs.mkdir()


app.command(App('sensor'))


@app['sensor'].command
def sensor_parse(*, conf: Config, parquet: bool = True, xlsx: bool = True):
    exp = conf.experiment()
    exp.parse_sensors(write_parquet=parquet, write_xlsx=xlsx)


@app['sensor'].command
def sensor_plot(*, conf: Config, pmv: bool = True, tr7: bool = True):
    exp = conf.experiment()
    exp.plot_sensors(pmv=pmv, tr7=tr7)


app.command(App('db'))


class LogReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @staticmethod
    def _iter_deepest_dir(path: Path):
        for p in path.rglob('*'):
            if p.is_dir() and not any(x.is_dir() for x in p.iterdir()):
                yield p

    @staticmethod
    def _read_log(path: Path, *, stem_date: bool, **kwargs):
        data = pl.read_csv(StringIO(path.read_text('utf-8')), **kwargs)

        if stem_date:
            date = pl.lit(path.stem).str.to_date('%Y-%m-%d')
            data = data.with_columns(
                datetime=date.dt.combine(pl.col('datetime').str.to_time('%H:%M:%S'))
            )

        assert data.width == 2  # noqa: PLR2004
        return data

    @classmethod
    def _read_log_dir(cls, path: Path):
        if not (files := list(path.glob('*'))):
            return None

        ext = files[0].suffix
        if not all(ext == f.suffix for f in files[1:]):
            return None

        kwargs: dict[str, Any]
        match ext:
            case '.csv':
                kwargs = {'has_header': True, 'new_columns': None}
            case '.txt':
                kwargs = {'has_header': False, 'new_columns': ['datetime', 'value']}
            case _:
                return None

        data = pl.concat(
            (cls._read_log(f, stem_date=ext == '.txt', **kwargs) for f in files),
            how='vertical_relaxed',
        )

        if '시간' in data.columns:
            data = (
                data
                .rename({'시간': 'datetime'})
                .with_columns(pl.col('datetime').str.to_datetime())
                .with_columns()
            )

        return data

    def __iter__(self):
        paths = list(self._iter_deepest_dir(self.path))

        for path in Progress.iter(paths):
            if (data := self._read_log_dir(path)) is None:
                continue

            if not data.height or (data.to_series(1) == 0).all():
                continue

            parts = path.relative_to(self.path).parts

            try:
                equipment = next(x for x in parts if x.startswith('설비'))
            except StopIteration:
                msg = f'대상 파일 경로에서 설비 정보를 인식할 수 없음: {path}'
                raise ValueError(msg) from None

            kind = parts[0]
            tag = parts[parts.index(equipment) + 1]

            unpivot = (
                data
                .unpivot(index='datetime')
                .select(
                    'datetime',
                    pl.lit(tag).alias('tag'),
                    'variable',
                    pl.col('value').cast(pl.Float64),
                )
                .with_columns()
            )

            yield f'{kind}-{equipment}', unpivot


@app['db'].command
def db_convert_log(*, conf: Config):
    """Log 폴더 (C드라이브 BigData 폴더) 정리."""
    src = conf.db_dirs.log
    dst = conf.db_dirs.data
    dst.mkdir(exist_ok=True)

    dd: defaultdict[str, list[pl.DataFrame]] = defaultdict(list)

    for kind, data in LogReader(src):
        dd[kind].append(data)

    for kind, dfs in dd.items():
        logger.info(f'{kind=}')

        pl.concat(dfs).write_parquet(dst / f'{kind}.parquet')


@app['db'].command
def db_convert_web(*, conf: Config):
    """Web 폴더 (BEMS 웹 인터페이스로 다운받은 자료) 정리."""
    src = conf.db_dirs.web
    dst = conf.db_dirs.data

    def read(path):
        return (
            pl
            .read_excel(path)
            .rename({'날짜': 'datetime'})
            .with_columns(
                pl.format('{}:00', 'datetime').str.to_datetime().alias('datetime')
            )
            .unpivot(index='datetime')
            .with_columns(pl.col('value').cast(pl.Float64))
        )

    for pattern in [
        '계측기 모니터링',
        '에너지원별 사용량*기간',
        '에너지원별 사용량*실시간',
    ]:
        data = pl.concat(read(x) for x in src.glob(f'{pattern}/*.xlsx'))

        name = pattern.replace('*', '_')
        data.write_parquet(dst / f'web-{name}.parquet')
        (
            data
            .filter(pl.col('value') != 0)
            .with_columns()
            .write_excel(dst / f'web-{name}.xlsx', column_widths=200)
        )


@app['db'].command
def db_plot(*, conf: Config):
    src = conf.db_dirs.data
    dst = conf.dirs.analysis

    utils.mpl.MplTheme('paper').grid().apply()

    for path in Progress.iter(list(src.glob('*.parquet'))):
        logger.info(path)
        data = pl.read_parquet(path)

        if 'tag' in data.columns:
            data = data.with_columns(
                pl.format('{}/{}', 'tag', 'variable').alias('variable')
            )

        if '계측기 모니터링' in path.name:
            data = (
                data
                .with_columns(pl.col('variable').str.split(' - '))
                .with_columns(
                    pl.col('variable').list[0].alias('equipment'),
                    pl.col('variable').list[1].alias('variable'),
                )
                .sort('equipment', 'variable', 'datetime')
            )

            grid = (
                sns
                .FacetGrid(
                    data,
                    col='equipment',
                    col_wrap=4,
                    sharey=False,
                    height=4,
                    despine=False,
                )
                .map_dataframe(sns.lineplot, x='datetime', y='value', hue='variable')
                .set_xlabels('')
                .set_ylabels('')
            )
            for ax in grid.axes.ravel():
                ax.legend()

            grid.savefig(dst / f'database-{path.stem}.png')
            plt.close(grid.figure)
        else:
            fig, ax = plt.subplots()
            sns.lineplot(
                data, x='datetime', y='value', hue='variable', ax=ax, alpha=0.75
            )
            sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
            ax.set_yscale('asinh')
            ax.autoscale_view()
            ax.set_xlabel('')
            ax.set_ylabel('')
            fig.savefig(dst / f'database-{path.stem}.png')
            plt.close(fig)


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplConciseDate(bold_zero_format=False).apply()
    utils.mpl.MplTheme(palette='tol:vibrant').grid().apply()

    app()

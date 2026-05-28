"""2025-09-12 EAN 데이터 검토 & 검수용 그래프."""

import dataclasses as dc
from typing import ClassVar

import cyclopts
import matplotlib.pyplot as plt
import polars as pl
import rich
import seaborn as sns
from loguru import logger

import scripts.exp.experiment as exp
from greenbutton import utils


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'eanbems'

    energy_threshold: float = 40000
    bldgs: tuple[str, ...] = (
        '개원초',
        '개포중',
        '광주기후에너지진흥원',
        '대구로봇산업진흥원',
        '한울권',
    )


app = utils.cli.App(
    config=[
        cyclopts.config.Toml('config/.experiment.toml', use_commands_as_keys=False),
        cyclopts.config.Toml('config/experiment.toml', use_commands_as_keys=False),
    ]
)


@app.command
def convert(conf: Config):
    root = conf.dirs.root
    prefix = '(raw)'

    for xlsx in root.glob('*.xlsx'):
        if xlsx.name.startswith(prefix):
            continue

        logger.info(xlsx)

        data = pl.read_excel(xlsx)
        data.write_parquet(root / f'01.{xlsx.stem}.parquet')
        xlsx.rename(root / f'{prefix}{xlsx.name}')


@app.command
def concat(conf: Config):
    root = conf.dirs.root
    pl.Config.set_tbl_cols(20)

    for t in ['에너지', '실내환경']:
        dfs = [
            pl.read_parquet(x, include_file_paths='path')
            for x in root.glob(f'01.*_{t}*.parquet')
        ]
        data = (
            pl
            .concat(dfs, how='diagonal_relaxed')
            .select(
                pl
                .col('path')
                .str.extract(rf'.*\\(.*)_{t}\.parquet')
                .str.strip_prefix('01.')
                .alias('건물'),
                '시간',
                pl.all().exclude('path', '시간'),
            )
            .with_columns(
                pl
                .col('시간')
                .str.extract(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2})(:00\.0+)?')
                .str.to_datetime('%Y-%m-%d %H:%M')
            )
        )
        rich.print(data)

        data.write_parquet(root / f'02.{t}.parquet')


@app.command
def per_bldg(conf: Config):
    # 건물별 데이터 정리 (시험성적서 신청용)
    root = conf.dirs.root
    time = '시간'

    for bldg in conf.bldgs:
        logger.info(bldg)

        energy = (
            pl
            .scan_parquet(root / f'01.{bldg}_에너지.parquet')
            .rename(lambda x: x if x == time else f'사용량:{x}')
            .with_columns(pl.col(time).str.to_datetime())
            .collect()
        )
        env = (
            pl
            .scan_parquet(root / f'01.{bldg}_실내환경.parquet')
            .unpivot(index=[time, '실이름'])
            .drop_nulls('value')
            .with_columns(
                pl.col(time).str.to_datetime(),
                pl.col('value').cast(pl.Float64),
                pl.format('환경:{}:{}', '실이름', 'variable').alias('v'),
            )
            .collect()
            .pivot('v', index=time, values='value', sort_columns=True)
        )

        data = energy.join(env, on=time, how='full', coalesce=True).sort(time)
        data.write_excel(root / f'04.{bldg}.xlsx')


@app.command
def vis_energy(conf: Config):
    utils.mpl.MplTheme().grid().apply()
    utils.mpl.MplConciseDate(matplotlib_default=True).apply()
    root = conf.dirs.root

    data = (
        pl
        .scan_parquet(root / '02.에너지.parquet')
        .unpivot(index=['건물', '시간'])
        .filter(pl.col('value').is_between(0, conf.energy_threshold))
        .collect()
    )

    grid = (
        sns
        .FacetGrid(
            data, col='건물', col_wrap=3, sharey=False, despine=False, aspect=4 / 3
        )
        .map_dataframe(sns.lineplot, x='시간', y='value', hue='variable', alpha=0.5)
        .set_axis_labels('', '에너지 사용량 [kWh]')
        .set_titles('{col_name}')
        .add_legend()
    )
    utils.mpl.move_grid_legend(grid)
    grid.savefig(root / '03.에너지.png')


@app.command
def vis_env(conf: Config):
    utils.mpl.MplTheme().grid().apply()
    utils.mpl.MplConciseDate(matplotlib_default=True).apply()
    root = conf.dirs.root

    data = pl.read_parquet(root / '02.실내환경.parquet')

    for v in ['온도', '상대습도']:
        grid = (
            sns
            .FacetGrid(
                data,
                col='건물',
                col_wrap=3,
                sharex=False,
                sharey=False,
                despine=False,
                aspect=4 / 3,
            )
            .map_dataframe(sns.lineplot, x='시간', y=v, hue='실이름', alpha=0.5)
            .set_axis_labels('', f'{v} [{"°C" if v == "온도" else "%"}]')
            .set_titles('{col_name}')
        )
        grid.savefig(root / f'03.실내환경-{v}.png')
        plt.close(grid.figure)


if __name__ == '__main__':
    app()

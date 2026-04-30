"""2026-04-30 두올 분석용 전력 AMI 데이터."""

import dataclasses as dc
import functools
from collections.abc import Sequence  # noqa: TC003
from pathlib import Path  # noqa: TC003

import cyclopts
import polars as pl
import structlog
from cyclopts.config import Toml
from matplotlib.figure import Figure

from greenbutton import cpr, misc
from greenbutton.utils import mpl as mplu
from greenbutton.utils import tqdm
from greenbutton.utils.cli import App

logger = structlog.stdlib.get_logger()


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config:
    root: Path
    temperature: Path
    samples: Sequence[str]

    @functools.cached_property
    def output(self):
        return self.root / '99.samples'


configs = [
    Toml(
        'config/.ami.toml',
        root_keys='public_institution',
        use_commands_as_keys=False,
        allow_unknown=True,
    ),
    Toml('config/ami.toml', use_commands_as_keys=False),
]
app = App(config=configs)


@app.command
def extract(conf: Config):
    data_dir = conf.root / '0001.data'
    temperature = (
        pl
        .scan_parquet(conf.temperature)
        .select(
            'datetime',
            pl.col('region2').replace({'제주도': '제주'}).alias('region'),
            pl.col('ta').alias('temperature'),
        )
        .collect()
    )
    inst = (
        pl
        .scan_parquet(data_dir / '1.기관-주소변환.parquet')
        .select(
            pl.col('기관ID').alias('id'),
            pl.col('기관대분류').alias('type'),
            pl.col('연면적').alias('GFA'),
            pl.col('asos_code').alias('region'),
        )
        .filter(pl.col('id').is_in(conf.samples))
        .collect()
    )
    ami = (
        pl
        .scan_parquet(list(data_dir.glob('AMI*.parquet')))
        .rename({
            '기관ID': 'id',
            '보정사용량': 'energy',
        })
        .select('id', 'datetime', 'energy')
        .filter(pl.col('id').is_in(conf.samples))
        .collect()
        .join(inst, on='id', how='left', validate='m:1')
        .with_columns((pl.col('energy') / pl.col('GFA')).alias('EUI'))
        .join(temperature, on=['datetime', 'region'], how='left', validate='m:1')
    )

    years = (
        ami.select(pl.col('datetime').dt.year().unique().sort()).to_series().to_list()
    )
    ami = ami.select(
        'id',
        'type',
        'GFA',
        'region',
        'datetime',
        misc.is_holiday(pl.col('datetime'), years=years).alias('holiday'),
        'temperature',
        'energy',
        'EUI',
    )

    ami.write_parquet(conf.output / '01.AMI.parquet')
    ami.write_csv(conf.output / '01.AMI.csv.zst', compression='zstd')


@app.command
def cpm_params(conf: Config):
    """전체 CPM 분석 결과 파라미터."""
    model = (
        pl
        .scan_parquet(conf.root / '0200.CPM/model.parquet')
        .filter(pl.col('validity') == 1, pl.col('holiday').not_())
        .drop('name', 'elec_ratio', 'holiday', 'validity')
        .rename({'category': 'type'})
        .collect()
    )
    model_params = (
        model
        .unpivot(['coef', 'change_points'], index=['id', 'type', 'names', 'r2'])
        .drop_nulls('value')
        .with_columns(
            pl
            .concat_str('names', 'variable', separator='-')
            .replace_strict({
                'Intercept-coef': '기저부하',
                'HDD-coef': '난방민감도',
                'CDD-coef': '냉방민감도',
                'HDD-change_points': '난방균형점온도',
                'CDD-change_points': '냉방균형점온도',
            })
            .alias('variable')
        )
        .pivot('variable', index=['id', 'type', 'r2'], values='value')
    )

    model.write_parquet(conf.output / '02.CPM-models.parquet')
    model.write_excel(conf.output / '02.CPM-models.xlsx', column_widths=100)
    model_params.write_excel(conf.output / '02.CPM-parameters.xlsx', column_widths=150)


@app.command
def cpm(conf: Config):
    output = conf.output / '03.CPM Plot'
    output.mkdir(exist_ok=True)

    ami = (
        pl
        .scan_parquet(conf.output / '01.AMI.parquet')
        .filter(pl.col('holiday').not_())
        .sort('id', 'datetime')
        .group_by_dynamic('datetime', every='1d', group_by=['id', 'GFA', 'region'])
        .agg(pl.sum('energy'), pl.mean('temperature'))
        .with_columns(energy=pl.col('energy') / pl.col('GFA'))
        .collect()
    )

    mplu.MplTheme().grid().apply()
    for by, df in tqdm(
        ami.group_by('id', maintain_order=True),
        total=ami.select('id').n_unique(),
    ):
        id_: str = by[0]
        logger.info('id=%s', id_)

        model = cpr.CprEstimator(df).fit()
        fig = Figure()
        ax = fig.subplots()
        model.plot(ax=ax, style={'scatter': {'s': 12, 'alpha': 0.25}})
        ax.set_xlabel('기온 [℃]')
        ax.set_ylabel('전력사용량 [kWh/m²]')
        ax.set_title(f'r²={model.model_dict["r2"]:.4f}', loc='left')
        fig.savefig(output / f'{id_}.png')


if __name__ == '__main__':
    app()

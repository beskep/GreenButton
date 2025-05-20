from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING

import cyclopts
import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine

from greenbutton import utils
from scripts.ami.energy_intensive.common import Vars
from scripts.ami.energy_intensive.config import Config  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Sequence

app = utils.cli.App(
    config=cyclopts.config.Toml(
        'config/.ami.toml',
        root_keys='energy_intensive',
        use_commands_as_keys=False,
    )
)


@dc.dataclass
class _Prep:
    conf: Config

    include_holiday: bool = False
    fill_building_null: bool = True

    index: Sequence[str] = (Vars.ENTE, Vars.PERF_YEAR)

    def building(self):
        return pl.read_parquet(self.conf.dirs.data / 'building.parquet')

    def equipment(self):
        eui = 'EUI(MJ/m²)'
        return (
            pl.scan_parquet(self.conf.dirs.data / 'equipment-feature.parquet')
            .with_columns(pl.col('use').str.strip_suffix('용'))
            .group_by(
                *self.index,
                pl.format(
                    '설비:{}',
                    pl.concat_str('use', 'source', separator=':'),
                ).alias('on'),
            )
            .agg(pl.sum(eui))
            .collect()
            .pivot('on', index=self.index, values=eui, sort_columns=True)
            .fill_null(0)
        )

    def elec(self):
        # NOTE 용도(냉/난/냉난방, 기타)별 사용량이 없는 경우 '전력사용량비'는 null,
        # 사용량은 있으나 전력 비중이 0일 경우 '전력사용량비' 0
        return (
            pl.scan_parquet(self.conf.dirs.data / 'equipment-main-equipment.parquet')
            .select(
                *self.index,
                pl.col(Vars.Ratio.ELEC_HVAC).fill_null(0),
                pl.col(Vars.Ratio.ELEC_BY_USE).fill_null(0),
                pl.format('전력사용량비:{}', 'use').alias('on'),
            )
            .collect()
            .pivot(
                'on',
                index=[*self.index, Vars.Ratio.ELEC_HVAC],
                values=Vars.Ratio.ELEC_BY_USE,
            )
        )

    def cpr_params(self):
        return (
            pl.scan_parquet(self.conf.dirs.cpr / 'models.parquet')
            .filter(
                pl.lit(value=True)
                if self.include_holiday
                else pl.col('is_holiday').not_()
            )
            .rename({'change_points': 'CP', 'names': 'operation'})
            .with_columns(
                pl.col('is_holiday').replace_strict(
                    {False: 'workday', True: 'holiday'}, return_dtype=pl.String
                )
            )
            .unpivot(
                ['CP', 'coef'],
                index=[Vars.ENTE, 'operation', 'is_holiday', 'r2'],
            )
            .drop_nulls('value')
            .with_columns(
                pl.concat_str(
                    'operation', 'variable', 'is_holiday', separator=':'
                ).alias('variable')
            )
            .collect()
            .pivot(
                'variable', index=[Vars.ENTE, 'r2'], values='value', sort_columns=True
            )
        )

    def prep(self):
        # TODO 실적연도 처리 문제 (현재 1번)
        # 1. 각 실적 연도 별도 처리하고 CPR 파라미터는 모든 연도 같은 데이터 사용
        # 2. 각 실적 연도 별도 처리하고 CPR 연도별 분석 (CPR 데이터 일부 소실)
        # 3. 건물/설비 정보 건물 당 하나 사용 (연도 통합)
        ente = pl.col(Vars.ENTE).cast(pl.UInt32)
        bldg = self.building().with_columns(ente)
        equipment = self.equipment().with_columns(ente)
        elec = self.elec().with_columns(ente)
        cpr_params = self.cpr_params().with_columns(ente)

        data = (
            bldg.join(elec, on=self.index, how='full', validate='1:1', coalesce=True)
            .join(equipment, on=self.index, how='full', validate='1:1', coalesce=True)
            .join(cpr_params, on=Vars.ENTE, how='full', validate='m:1', coalesce=True)
            .sort(self.index)
        )

        if self.fill_building_null:
            # NOTE '건물' 시트에 없는 실적연도 정보가 '고정설비'에
            # 존재하는 경우 일부 존재 (e.g. ENTE 56,943)
            # 건물 정보는 변하지 않았다고 가정하고 다른 연도 데이터로 보간
            cols = pl.col(bldg.drop(self.index).columns)
            data = (
                data.with_columns(cols.fill_null(strategy='forward').over(Vars.ENTE))
                .with_columns(cols.fill_null(strategy='backward').over(Vars.ENTE))
                .with_columns()
            )

        return data

    @staticmethod
    def plot_elec(data: pl.DataFrame, *, drop_zero: bool = False):
        elec = (
            data.unpivot(cs.contains('전력사용량비'), index=Vars.PERF_YEAR)
            .drop_nulls('value')
            .with_columns(
                pl.col('variable')
                .replace({Vars.Ratio.ELEC_HVAC: '냉난방 종합'})
                .str.strip_prefix('전력사용량비:')
            )
        )

        if drop_zero:
            elec = elec.filter(pl.col('value') > 0)

        grid = (
            sns.displot(
                elec,
                x='value',
                col='variable',
                col_order=['냉방', '난방', '냉난방', '냉난방 종합'],
                col_wrap=2,
                kind='hist',
                kde=True,
                facet_kws={'sharey': False},
            )
            .set_titles('{col_name}')
            .set_axis_labels('냉난방 전력 비율')
        )
        grid.figure.set_size_inches(mpl.rcParams['figure.figsize'])
        return grid

    @staticmethod
    def plot_joint(data: pl.DataFrame, *, drop_zero: bool = True):
        if drop_zero:
            data = data.filter(pl.col(Vars.Ratio.ELEC_HVAC) > 0)

        grid = sns.JointGrid(
            data, x=Vars.Ratio.ELEC_HVAC, y='r2', palette='crest', marginal_ticks=True
        )
        cax = grid.figure.add_axes((0.85, 0.85, 0.02, 0.1))
        return (
            grid.plot_joint(sns.histplot, cbar=True, cbar_ax=cax, binwidth=0.05)
            .plot_marginals(sns.histplot, kde=True, binwidth=0.05, color='gray')
            .set_axis_labels('냉난방 최소 전력 비율', 'CPR 결정계수')
        )

    def __call__(self):
        data = self.prep()

        output = self.conf.dirs.cluster
        output.mkdir(exist_ok=True)

        data.write_parquet(output / '0000.data.parquet')
        data.write_excel(output / '0000.data.xlsx', column_widths=80)

        layout = ConstrainedLayoutEngine()

        # 전력 비율
        grid = self.plot_elec(data, drop_zero=False)
        layout.execute(grid.figure)
        grid.savefig(output / '0001.냉난방 전력 비율.png')
        plt.close(grid.figure)

        grid = self.plot_elec(data, drop_zero=True)
        layout.execute(grid.figure)
        grid.savefig(output / '0001.냉난방 전력 비율-positive.png')
        plt.close(grid.figure)

        # CPR r2
        fig = Figure()
        ax = fig.add_subplot()
        sns.histplot(data.drop_nulls('r2'), x='r2', ax=ax, kde=True)
        layout.execute(grid.figure)
        fig.savefig(output / '0001.CPR r2.png')

        # joint
        grid = self.plot_joint(data)
        grid.savefig(output / '0001.joint.png')
        plt.close(grid.figure)

        grid = self.plot_joint(data, drop_zero=False)
        grid.savefig(output / '0001.joint-비전력 냉난방 포함.png')
        plt.close(grid.figure)


@app.command
def prep(*, conf: Config):
    """클러스터링 대상 데이터셋 전처리."""
    _Prep(conf)()


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    app()

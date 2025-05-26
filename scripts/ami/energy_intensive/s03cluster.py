from __future__ import annotations

import dataclasses as dc
import itertools
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Self

import cyclopts
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import rich
import seaborn as sns
import sklearn.preprocessing as skp
from loguru import logger
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from scipy.cluster import hierarchy as hrc
from sklearn import metrics
from sklearn.neighbors import LocalOutlierFactor

from greenbutton import utils
from greenbutton.utils.terminal import Progress
from scripts.ami.energy_intensive.common import Vars
from scripts.ami.energy_intensive.config import Config  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.text import Text

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
    index_full: Sequence[str] = (
        Vars.ENTE,
        Vars.PERF_YEAR,
        Vars.KEMC_CODE,
        Vars.KEMC_KOR,
        Vars.NAME,
    )

    def building(self):
        path = self.conf.dirs.data / 'building.parquet'
        data = pl.read_parquet(path).drop('주소', '부문')
        columns = data.drop(self.index_full).columns
        renamed = [
            f'면적:{x}' if '면적' in x else f'연간사용량:{x}'.replace(')', '/m²)')
            for x in columns
        ]
        return (
            data.rename(dict(zip(columns, renamed, strict=True)))
            .with_columns(cs.starts_with('연간사용량:') / pl.col('면적:연면적(m²)'))
            .with_columns()
        )

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
        elec = self.elec().with_columns(ente)
        equipment = self.equipment().with_columns(ente)
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
            .set_titles('{col_name}', weight=500)
            .set_axis_labels('냉난방 전력 사용량 비중')
        )
        grid.figure.set_size_inches(*(1.8 * x for x in mpl.rcParams['figure.figsize']))
        return grid

    @staticmethod
    def plot_r2_ecdf(data: pl.DataFrame, threshold: float = 0.6, alpha: float = 0.4):
        data = data.drop_nulls('r2')
        r2 = pl.col('r2')

        rank, percentile = (
            data.with_columns(r2.rank(descending=True).alias('rank'))
            .with_columns((pl.col('rank') / pl.len()).alias('percentile'))
            .filter(r2 >= threshold)
            .filter(r2 == r2.min())
            .select('rank', 'percentile')
            .row(0)
        )

        fig = Figure()
        ax = fig.add_subplot()
        sns.ecdfplot(data, x='r2', stat='count', complementary=True, ax=ax)
        ax.set_xlim(0, 1)
        ax.set_xlabel('r²')

        ax.axvline(threshold, alpha=alpha, ls='--')
        ax.axhline(rank, alpha=alpha, ls='--')
        ax.annotate(
            f'r² ≥ {threshold} 모델 {int(rank):,}개 ({percentile:.1%})',
            (0, rank),
            xytext=(5, -2.5),
            textcoords='offset points',
            va='top',
            fontsize='small',
            weight=500,
        )

        return fig

    @staticmethod
    def plot_joint(data: pl.DataFrame, *, drop_zero: bool = True):
        if drop_zero:
            data = data.filter(pl.col(Vars.Ratio.ELEC_HVAC) > 0, pl.col('r2') > 0)

        grid = sns.JointGrid(
            data, x=Vars.Ratio.ELEC_HVAC, y='r2', palette='crest', marginal_ticks=True
        )
        cax = grid.figure.add_axes((0.85, 0.85, 0.02, 0.1))
        return (
            grid.plot_joint(sns.histplot, cbar=True, cbar_ax=cax, binwidth=0.05)
            .plot_marginals(sns.histplot, kde=True, binwidth=0.05, color='gray')
            .set_axis_labels('냉난방 전력 사용량 비중', 'CPR 결정계수')
        )

    def __call__(self, *, plot: bool = True):
        data = self.prep()

        output = self.conf.dirs.cluster
        output.mkdir(exist_ok=True)

        data.write_parquet(output / '0000.data.parquet')
        data.write_excel(output / '0000.data.xlsx', column_widths=80)

        if not plot:
            return

        layout = ConstrainedLayoutEngine()

        # 전력 비중
        grid = self.plot_elec(data, drop_zero=False)
        layout.execute(grid.figure)
        grid.savefig(output / f'0001.{Vars.Ratio.ELEC_HVAC}.png')
        plt.close(grid.figure)

        grid = self.plot_elec(data, drop_zero=True)
        layout.execute(grid.figure)
        grid.savefig(output / f'0001.{Vars.Ratio.ELEC_HVAC}-positive.png')
        plt.close(grid.figure)

        # CPR r2
        r2 = data.select(Vars.ENTE, 'r2').unique()
        fig = Figure()
        ax = fig.add_subplot()
        sns.histplot(r2.drop_nulls('r2'), x='r2', ax=ax, kde=True)
        ax.set_xlabel('r²')
        layout.execute(grid.figure)
        fig.savefig(output / '0001.CPR r2 hist.png')

        # CPR r2 ecdf
        fig = self.plot_r2_ecdf(r2)
        layout.execute(fig)
        fig.savefig(output / '0001.CPR r2 ecdf.png')

        # joint
        grid = self.plot_joint(data)
        grid.savefig(output / '0001.joint.png')
        plt.close(grid.figure)

        grid = self.plot_joint(data, drop_zero=False)
        grid.savefig(output / '0001.joint-비전력 냉난방 포함.png')
        plt.close(grid.figure)


@app.command
def prep(*, plot: bool = True, conf: Config):
    """클러스터링 대상 데이터셋 전처리."""
    utils.mpl.MplTheme(constrained=False).grid().apply()
    _Prep(conf)(plot=plot)


@dc.dataclass
class _ClusterFeature:
    area: bool = True
    consumption: bool = True
    elec: bool = False
    equipment: bool = True
    cpr: bool = True

    def __str__(self):
        return (
            super()
            .__str__()
            .removeprefix(f'{self.__class__.__name__}(')
            .removesuffix(')')
            .replace(',', '')
            .replace('True', '1')
            .replace('False', '0')
        )

    @classmethod
    def iter(cls):
        for area, consumption in itertools.product([False, True], [False, True]):
            yield cls(area=area, consumption=consumption)


@dc.dataclass
class _ClusterParam:
    cpr_min_r2: float = 0.6
    """CPR 모델 최소 r2 (평일 모델만 이용)"""

    equipment_min_elec: float = 0.0
    """냉난방설비 전력 사용량 최소 비중."""

    contamination: float | Literal['auto'] | None = 'auto'
    """LOF 이상치 제거 contamination"""

    center: Literal['mean', 'median'] = 'median'
    scaler: Literal['standard', 'robust'] = 'standard'
    scale_data: Literal['all', 'center'] = 'all'

    feature: _ClusterFeature = dc.field(default_factory=_ClusterFeature)

    def asdict(self):
        d = dc.asdict(self)
        d.pop('feature')
        return d | dc.asdict(self.feature)

    def str(self, *, feature: bool = False, cluster: bool = False):
        s = (
            f'r2={self.cpr_min_r2} '
            f'elec={self.equipment_min_elec} '
            f'cntm={self.contamination}'
        )
        if feature:
            s = f'{s} {self.feature}'
        if cluster:
            s = f'{s} {self.center} {self.scaler} {self.scale_data}'

        return s

    def __str__(self):
        return self.str()

    @classmethod
    def iter(
        cls,
        param: _ClusterParam | None = None,
        *,
        track: bool = True,
    ) -> Iterable[Self]:
        if param is None:
            param = cls()

        def fn():
            for center, scale_data, feature in itertools.product(
                ('mean', 'median'),
                ('all', 'center'),
                _ClusterFeature.iter(),
            ):
                yield cls(
                    cpr_min_r2=param.cpr_min_r2,
                    equipment_min_elec=param.equipment_min_elec,
                    contamination=param.contamination,
                    center=center,  # type: ignore[arg-type]
                    scaler=param.scaler,
                    scale_data=scale_data,  # type: ignore[arg-type]
                    feature=feature,
                )

        it: Iterable = list(fn())
        if track:
            it = Progress.iter(it)

        return it


@dc.dataclass
class _HierarchicalCluster:
    conf: Config

    index: Sequence[str] = (Vars.ENTE, Vars.PERF_YEAR)
    index_full: Sequence[str] = (
        Vars.ENTE,
        Vars.PERF_YEAR,
        Vars.KEMC_CODE,
        Vars.KEMC_KOR,
        Vars.NAME,
    )

    _cache: Path = dc.field(init=False)

    class Cluster(NamedTuple):
        data: pl.DataFrame
        score: dict[str, float]
        fig: Figure
        ax: Axes

    def __post_init__(self):
        self._cache = self.conf.dirs.cluster / '.cache'
        self._cache.mkdir(exist_ok=True)

    @staticmethod
    def evaluate(array: np.ndarray, labels: Sequence[str]):
        return {
            'n_point': array.shape[0],
            'n_feature': array.shape[1],
            'silhouette': metrics.silhouette_score(array, labels),
            'calinski_harabasz': metrics.calinski_harabasz_score(array, labels),
            'davies_bouldin': metrics.davies_bouldin_score(array, labels),
        }

    @staticmethod
    def annotated_dendrogram(
        z,
        labels: Sequence[str] | None = None,
        ax: Axes | None = None,
        *args,
        **kwarg,
    ):
        # https://stackoverflow.com/questions/11917779/how-to-plot-and-annotate-hierarchical-clustering-dendrograms-in-scipy-matplotlib/12311782#12311782
        if ax is None:
            ax = plt.gca()

        r = hrc.dendrogram(z, *args, labels=labels, orientation='right', **kwarg)
        ax.yaxis.set_tick_params(labelsize='small')
        ax.set_xlabel('Distance')
        ax.grid(visible=False, which='both', axis='y')

        if kwarg.get('no_plot'):
            return r

        for i, d in zip(r['icoord'], r['dcoord'], strict=True):
            x = d[1]
            y = sum(i[1:3]) / 2.0
            ax.annotate(
                f'{x:.3g}',
                (x, y),
                xytext=(-2, 0),
                textcoords='offset points',
                va='center',
                ha='right',
                fontsize='small',
                alpha=0.8,
            )

        return r

    def _data(self, param: _ClusterParam):
        data = (
            pl.scan_parquet(self.conf.dirs.cluster / '0000.data.parquet')
            .with_columns(pl.col('r2', Vars.Ratio.ELEC_HVAC).fill_null(0))
            .filter(
                pl.col('r2') >= param.cpr_min_r2,
                pl.col(Vars.Ratio.ELEC_HVAC) >= param.equipment_min_elec,
            )
            .collect()
        )

        # NOTE 사용량, CPR 모델 결측치 처리
        # (해당 설비가 없거나 냉난방을 하지 않는 케이스)
        cols = pl.col(data.select(cs.contains('전력사용량비', 'coef', 'CP')).columns)
        data = data.with_columns(cols.fill_null(cols.median()))

        if not param.feature.area:
            data = data.drop(cs.starts_with('면적:'))
        if not param.feature.consumption:
            data = data.drop(cs.starts_with('연간사용량:'))
        if not param.feature.elec:
            data = data.drop(cs.contains('전력사용량비'))
        if not param.feature.equipment:
            data = data.drop(cs.contains('설비:'))
        if not param.feature.cpr:
            data = data.drop(cs.contains(':CP:', ':coef:'))

        data = data.filter(pl.all_horizontal(cs.numeric().is_finite()))

        # LOF
        if not param.contamination:
            data = data.with_columns(pl.lit(None).alias('lof'))
        else:
            array = data.drop(self.index_full).fill_null(strategy='mean').to_numpy()
            lof = LocalOutlierFactor(contamination=param.contamination)
            labels = lof.fit_predict(array)
            data = data.with_columns(pl.Series('lof', labels))

        return data

    def data(self, param: _ClusterParam):
        name = param.str(feature=True, cluster=False)
        path = self._cache / f'{name}.parquet'
        fn = utils.pl.frame_cache(path, timeout='1H')(self._data)
        return fn(param).lazy().collect()

    def cluster(self, param: _ClusterParam, group: str = Vars.KEMC_KOR):
        data = (
            self.data(param)
            .filter((pl.col('lof') == 1) | pl.col('lof').is_null())
            .drop('lof', 'r2')
            .drop_nulls()
        )

        # 각 업종 대표 데이터
        index = [x for x in self.index_full if x != group]
        center = (
            data.drop(index)
            .group_by(group)
            .agg(pl.all().mean() if param.center == 'mean' else pl.all().median())
            .sort(group)
        )

        groups = center[group].to_list()
        labels = dict(
            # `group (count)`
            data.select(
                group,
                pl.format('{} ({})', group, pl.len().over(group)).alias('count'),
            ).iter_rows()
        )

        scaler = (
            skp.StandardScaler() if param.scaler == 'standard' else skp.RobustScaler()
        )
        scaler.fit(
            (
                data.drop(self.index_full)
                if param.scale_data == 'all'
                else center.drop(group)
            ).to_numpy()
        )
        scaled_center = scaler.transform(center.drop(group).to_numpy())

        linked = hrc.ward(scaled_center)
        fig, ax = plt.subplots()
        r = self.annotated_dendrogram(linked, labels=groups, ax=ax)
        ax.set_yticks(
            ax.get_yticks(), [labels[x.get_text()] for x in ax.get_yticklabels()]
        )

        # 클러스터 평가 (0.7 * max(dist) 기준)
        colors = [
            f'{s}-{i}' if s == 'C0' else s for i, s in enumerate(r['leaves_color_list'])
        ]
        cluster_map = dict(zip(r['ivl'], colors, strict=True))
        cluster = data.with_columns(
            pl.col(group).replace_strict(cluster_map).alias('cluster')
        )
        score = self.evaluate(
            scaler.transform(data.drop(self.index_full).to_numpy()),
            data[group].replace_strict(cluster_map).to_list(),
        )

        return self.Cluster(data=cluster, score=score, fig=fig, ax=ax)

    def batch_cluster(self, param: _ClusterParam, group: str = Vars.KEMC_KOR):
        score = []  # 군집화 점수

        output = self.conf.dirs.cluster / f'0100.HierarchyCluster {param}'
        output.mkdir(exist_ok=True)

        for idx, p in enumerate(_ClusterParam.iter(param)):
            name = f'{idx:04d}. {p.str(feature=True, cluster=True)}'
            logger.info(name)

            cluster = self.cluster(p, group=group)

            cluster.data.write_excel(output / f'{name}.xlsx', column_widths=120)
            cluster.fig.savefig(output / f'{name}.png')
            plt.close(cluster.fig)

            score.append(p.asdict() | cluster.score)

        utils.pl.PolarsSummary(self.data(param)).write_excel(
            output / f'0000 data summary {param}.xlsx'
        )

        (
            pl.from_dicts(score)
            .with_row_index()
            .sort('silhouette', descending=True)
            .write_excel(output / f'0000 score {param}.xlsx', column_widths=80)
        )


_DEFAULT_PARAM = _ClusterParam()


@app.command
def hierarchy(
    cmd: Literal['once', 'batch', 'sample'] = 'sample',
    *,
    conf: Config,
    param: _ClusterParam = _DEFAULT_PARAM,
):
    hc = _HierarchicalCluster(conf=conf)
    d = conf.dirs.cluster

    match cmd:
        case 'batch':
            hc.batch_cluster(param=param)
        case 'once':
            cluster = hc.cluster(param=param)

            cluster.data.write_excel(d / f'0100.cluster {param.str(cluster=True)}.xlsx')
            cluster.fig.savefig(d / f'0100.cluster {param.str(cluster=True)}.png')
            rich.print(cluster.score)
        case 'sample':
            feature = _ClusterFeature(
                area=True, consumption=True, elec=True, equipment=True, cpr=True
            )
            data = hc.data(dc.replace(param, feature=feature))

            name = param.str(feature=False, cluster=False)
            data.sample(1000).write_excel(d / f'[sample] data {name}.xlsx')
            utils.pl.PolarsSummary(data).write_excel(
                d / f'[sample] summary {name}.xlsx'
            )


app.command(utils.cli.App('cluster', '클러스터링 결과 분석.'))


@dc.dataclass
class _ClusterDist:
    conf: Config
    clusters: Mapping[str, str] = dc.field(
        default_factory=lambda: {
            'IDC': 'IDC',
            '아파트': '아파트',
            '병원': '병원+호텔',
            '호텔': '병원+호텔',
        }
    )
    contamination: float | Literal['auto'] = 'auto'

    data: pl.DataFrame = dc.field(init=False)
    _clusters: tuple[str, ...] = dc.field(init=False)

    @dc.dataclass
    class Case:
        group: str
        kind: Literal['hist', 'kde', 'bar', 'violin']
        log: bool
        lof: bool

        def __str__(self):
            return (
                f'{self.group}-{self.kind}'
                f'-{"log" if self.log else "linear"}'
                f'-{"lof" if self.lof else "none"}'
            )

        @classmethod
        def iter(cls, *, track: bool = True):
            it: Iterable = itertools.product(
                ['면적', '연간사용량', '설비', 'CPR'],
                ['kde', 'bar', 'violin'],
                [False, True],
                [False, True],
            )

            if track:
                it = Progress.iter(list(it))

            yield from itertools.starmap(cls, it)

    def __post_init__(self):
        def rename(n: str):
            if any(x in n for x in [':coef:', ':CP:']):
                return f'CPR:{n}'
            return n

        self.data = (
            pl.scan_parquet(self.conf.dirs.cluster / '0000.data.parquet')
            .with_columns(
                pl.col(Vars.KEMC_KOR)
                .replace_strict(self.clusters, default='기타')
                .alias('cluster')
            )
            .rename(rename)
            .collect()
        )
        self._clusters = (*sorted(set(self.clusters.values())), '기타')

    def _data(self, case: Case):
        index = [] if case.group in {'면적', 'CPR'} else [Vars.PERF_YEAR]
        index = ['cluster', Vars.ENTE, *index]

        data = (
            self.data.select(cs.starts_with(f'{case.group}:'), *index)
            .filter(pl.all_horizontal(cs.numeric().is_finite()))
            .with_columns()
        )

        if case.lof:
            arr = data.drop(index).fill_null(strategy='mean').to_numpy()
            lof = LocalOutlierFactor(contamination=self.contamination).fit_predict(
                skp.StandardScaler().fit_transform(arr)
            )
            logger.debug('이상치 비율: {:.2%}', np.sum(lof == -1) / lof.size)
            data = data.filter(pl.Series(values=lof) == 1)

        unpivot = (
            data.unpivot(index=index)
            .with_columns(
                pl.col('variable')
                .str.strip_prefix(f'{case.group}:')
                .str.replace('(m²)', ' [m²]', literal=True)
            )
            .drop_nulls()
        )

        if case.group == 'CPR':
            unpivot = unpivot.with_columns(
                pl.col('variable')
                .str.strip_suffix(':workday')
                .str.replace_many(
                    ['Intercept:coef', 'CDD', 'HDD', ':CP', ':coef'],
                    [
                        '기저부하',
                        '냉방',
                        '난방',
                        ' 균형점 온도 [°C]',
                        ' 민감도 [kWh/m²]',
                    ],
                )
            )

        return unpivot

    @staticmethod
    def _plot(case: Case, order: list[str]):
        match case.kind:
            case 'hist':
                plot = {
                    'func': sns.histplot,
                    'stat': 'probability',
                    'element': 'step',
                    'alpha': 0.25,
                }
            case 'kde':
                plot = {
                    'func': sns.kdeplot,
                    'alpha': 0.25,
                    'fill': True,
                    'common_norm': False,
                    'cut': 2,
                    'warn_singular': False,
                }
            case 'bar':
                plot = {'func': sns.barplot, 'y': 'cluster', 'order': order}
            case 'violin':
                plot = {'func': sns.violinplot, 'y': 'cluster', 'order': order}

        return {'x': 'value', 'log_scale': case.log} | plot

    def plot(self, case: Case):
        data = self._data(case)

        if case.log:
            data = data.filter(pl.col('value') > 0)

        variables = data['variable'].unique().sort().to_list()
        order = sorted(
            data['cluster'].unique().to_list(),
            key=lambda x: (1 if x == '기타' else 0, x),
        )
        col_wrap = 4 if len(variables) == 12 else int(utils.mpl.ColWrap(len(variables)))  # noqa: PLR2004
        plot = self._plot(case, order)
        hue = 'cluster' if case.kind in {'hist', 'kde'} else None

        grid = (
            sns.FacetGrid(
                data,
                col='variable',
                col_wrap=col_wrap,
                col_order=variables,
                hue=hue,
                hue_order=order,
                sharex=False,
                sharey=case.kind in {'bar', 'violin'},
                aspect=4 / 3,
            )
            .map_dataframe(**plot)
            .set_titles('')
            .set_titles('{col_name}', loc='left', weight=500)
            .set_xlabels('연간 사용량 [MJ/m²]' if case.group == '설비' else '')
            .add_legend()
        )

        if hue:
            if legend := grid.legend:
                legend.set_title('')

            utils.mpl.move_grid_legend(grid)
        else:
            grid.set_ylabels('')

        return grid

    def __call__(self):
        output = self.conf.dirs.cluster / '0200.cluster-dist'
        output.mkdir(exist_ok=True)

        for case in self.Case.iter():
            if case.kind == 'bar' and case.log:
                continue

            logger.info(repr(case))

            grid = self.plot(case)
            grid.savefig(output / f'0200.cluster-{case}.png')
            plt.close(grid.figure)
            return


@app['cluster'].command
def cluster_dist(*, conf: Config):
    utils.mpl.MplTheme(
        constrained=False, rc={'axes.formatter.limits': (-4, 5)}
    ).grid().apply()

    _ClusterDist(conf)()


@dc.dataclass
class _ClusterMainEquipment:
    conf: Config
    label_count: bool = False
    label_threshold: int = 4
    ellipsis: str = '(...)'

    clusters: Mapping[str, str] = dc.field(
        default_factory=lambda: {
            'IDC': 'IDC',
            '아파트': '아파트',
            '병원': '병원+호텔',
            '호텔': '병원+호텔',
        }
    )

    data: pl.DataFrame = dc.field(init=False)
    _clusters: tuple[str, ...] = dc.field(init=False)

    def __post_init__(self):
        self.data = (
            pl.scan_parquet(self.conf.dirs.data / 'equipment-main-equipment.parquet')
            .with_columns(
                pl.col(Vars.KEMC_KOR)
                .replace_strict(self.clusters, default='기타')
                .alias('cluster')
            )
            .collect()
        )
        self._clusters = (*sorted(set(self.clusters.values())), '기타')

    def _data(self, *, source):
        data = self.data

        if source:
            data = data.with_columns(
                pl.format('{} ({})', 'equipment', 'source').alias('equipment')
            )

        return (
            data.group_by('cluster', 'use', 'equipment')
            .agg(pl.len().alias('count'))
            .with_columns(
                (
                    pl.col('count')  # fmt
                    / pl.sum('count').over('cluster', 'use')
                ).alias('ratio')
            )
            .sort('use', 'count', descending=[False, True])
        )

    def _plot_data(self, data: pl.DataFrame):
        return (
            data.with_columns(
                pl.col('count')
                .rank(descending=True)
                .over('cluster', 'use')
                .alias('rank')
            )
            .with_columns(
                pl.when(pl.col('rank') > self.label_threshold)
                .then(pl.lit(self.ellipsis))
                .otherwise(pl.col('equipment'))
                .alias('equipment')
            )
            .group_by('cluster', 'use', 'equipment')
            .agg(pl.sum('count', 'ratio'))
            .with_columns(percent=pl.col('ratio').mul(100).round(1))
            .with_columns(
                label=pl.format('{} ({}%)', 'count', 'percent')
                if self.label_count
                else pl.format('{}%', 'percent'),
                ellipsis=pl.col('equipment') == self.ellipsis,
            )
            .sort(
                'use',
                'ellipsis',
                'count',
                'equipment',
                descending=[False, False, True, False],
            )
        )

    def plot(self, data: pl.DataFrame, threshold: float = 0.25):
        data = self._plot_data(data)
        palette = [sns.color_palette(n_colors=1)[0], 'darkgray']
        grid = (
            sns.FacetGrid(
                data,
                col='use',
                row='cluster',
                col_order=['난방', '냉방', '냉난방', '기타'],
                row_order=self._clusters,
                hue='ellipsis',
                palette=palette,
                sharex=False,
                sharey=False,
                margin_titles=True,
                height=2.5,
                aspect=16 / 9,
            )
            .map_dataframe(sns.barplot, x='count', y='equipment')
            .set_axis_labels('건수', '')
            .set_titles(
                row_template='{row_name}',
                col_template='{col_name}',
                weight=500,
                size='large',
            )
        )

        center_color = utils.mpl.text_color(palette[0])

        container: Any
        for (row, col), ax in grid.axes_dict.items():
            filtered = data.filter(pl.col('cluster') == row, pl.col('use') == col)
            indices = list(
                itertools.pairwise(
                    itertools.accumulate([0, *(len(x) for x in ax.containers)])
                )
            )
            labels = [filtered['label'][i[0] : i[1]] for i in indices]
            values = [filtered['count'][i[0] : i[1]] for i in indices]
            max_x = ax.get_xlim()[1]

            for container, label, value in zip(
                ax.containers, labels, values, strict=True
            ):
                centers = ax.bar_label(
                    container,
                    label,
                    label_type='center',
                    color=center_color,
                    weight=500,
                )
                edges = ax.bar_label(container, label, label_type='edge', padding=5)

                for center, edge, v in zip(centers, edges, value, strict=True):
                    if v / max_x < threshold:
                        center.remove()
                    else:
                        edge.remove()

        text: Text
        for text in grid._margin_titles_texts:  # noqa: SLF001
            text.set_rotation(0)

        ConstrainedLayoutEngine().execute(grid.figure)

        return grid

    def __call__(self, *, source: bool):
        suffix = '-에너지원' if source else ''
        path = self.conf.dirs.cluster / f'0210.주요 설비{suffix}.xlsx'

        data = self._data(source=source)
        data.write_excel(path, column_widths=120)

        grid = self.plot(data)
        grid.savefig(path.with_suffix('.png'))
        plt.close(grid.figure)


@app['cluster'].command
def cluster_main_equipment(*, scale: float = 1.1, conf: Config):
    utils.mpl.MplTheme(scale).grid().apply()
    main_equipment = _ClusterMainEquipment(conf=conf)
    main_equipment(source=False)
    main_equipment(source=True)


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplTheme().grid().apply()

    app()

"""2025-04 공공기관 건물 특성, CPR 파라미터 군집화 - 벤치마킹 비교 대상 집단 선정."""
# ruff: noqa: PLC0415

from __future__ import annotations

import dataclasses as dc
import enum
import functools
import itertools
import warnings
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import rich
import seaborn as sns
from loguru import logger
from matplotlib import patheffects
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D
from scipy.cluster import hierarchy as hrc
from sklearn import metrics
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from statannotations.Annotator import Annotator

from greenbutton import cpr, utils
from greenbutton.utils.cli import App
from greenbutton.utils.terminal import Progress
from scripts.ami.public_institution.config import Config  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


class VAR(enum.StrEnum):
    IID = '기관ID'
    GROUP = '기관대분류'
    OWNERSHIP = '소유여부'
    GFA = '연면적'
    REGION = '지역'
    USE = '건물용도'

    EQUIPMENT = '냉난방설비'
    CAPACITY = '용량[kW/m²]'

    @classmethod
    def indices(cls):
        """
        건물당 하나씩 존재하는 인덱스 변수.

        Returns
        -------
        tuple[str, str, str, str, str, str]
        """
        return (cls.IID, cls.GROUP, cls.USE, cls.OWNERSHIP, cls.GFA, cls.REGION)


@dc.dataclass
class _DatasetParams:
    equip_min_count: int = 10

    equip_by_use: bool = False
    """설비 용도에 따라 재분류 (종류 축소)"""

    cpr_min_r2: float = 0.6

    contamination: float | Literal['auto'] = 'auto'
    """LOF 이상치 제거 contamination"""

    day: Literal['workday', 'both'] | None = 'workday'
    """평일/휴일/전체"""

    def __post_init__(self):
        if self.day == 'both':
            self.day = None

    def equip_param(self):
        eq = 'Use' if self.equip_by_use else 'Original'
        return f'MinEq{self.equip_min_count}_Eq{eq}'

    def __str__(self):
        day = self.day or 'work&holiday'
        cntm = str(self.contamination).title()
        return f'{self.equip_param()}_MinR²{self.cpr_min_r2}_{day}_Lof{cntm}'


_DEFAULT_DATASET_PARAMS = _DatasetParams()


@dc.dataclass
class _Dataset:
    """
    차원축소/군집화 데이터 전처리.

    - 1.기관 파일
        - 연면적
        - 기관대분류
        - 지역
        - 소유여부
    - 2.냉난방설비현황(전처리)
        - equipment
        - capacity[kW]
        - (사용 빈도 많은 일부 설비만 이용)
    - CPR 파라미터
        - 근무일/휴일 CPR 파라미터 5개 또는 균형점 온도 제외한 3개
    """

    conf: Config
    param: _DatasetParams = dc.field(default_factory=_DatasetParams)

    EQUIP_USE_MAP: ClassVar[dict[str, str]] = {
        'EHP': 'HP',
        'GHP': 'HP',
        '지열히트펌프': 'HP',
        '온수보일러': '중앙식 보일러',
        '증기보일러': '중앙식 보일러',
    }

    EQUIP_PREFIX: ClassVar[str] = '설비:'

    def __hash__(self):
        return hash(repr(self.conf))

    def building(self):
        group = pl.col(VAR.GROUP)
        return (
            pl
            .scan_parquet(self.conf.dirs.data / '1.기관-주소변환.parquet')
            .select(
                VAR.IID,
                VAR.GROUP,
                VAR.USE,
                VAR.OWNERSHIP,
                VAR.GFA,
                pl.col('asos_code').alias(VAR.REGION),
            )
            .with_columns(
                pl
                .when(group.str.starts_with('국립대학병원'))
                .then(pl.lit('국립대학병원 외'))
                .otherwise(group)
                .alias(VAR.GROUP)
            )
        )

    def equipment_params(self):
        # NOTE 전기식 용량 비율도 계산할 수 있지만,
        # KEIT 사례를 고려할 때 정확도는 낮아보임
        p = self.EQUIP_PREFIX
        lf = (
            pl
            .scan_parquet(self.conf.dirs.data / '2.냉난방설비현황-전처리.parquet')
            .select(
                VAR.IID,
                pl
                .col('equipment')
                .replace({'흡수식냉온수기': '냉온수기'})
                .alias(VAR.EQUIPMENT),
                pl.col('capacity[kW/m²]').alias(VAR.CAPACITY),
            )
            .drop_nulls(VAR.CAPACITY)
            .filter(pl.len().over(VAR.EQUIPMENT) >= self.param.equip_min_count)
            .with_columns(pl.format(f'{p}{{}}', VAR.EQUIPMENT).alias(VAR.EQUIPMENT))
        )

        if self.param.equip_by_use:
            m = {f'{p}{k}': f'{p}{v}' for k, v in self.EQUIP_USE_MAP.items()}
            lf = lf.with_columns(pl.col(VAR.EQUIPMENT).replace(m))

        return (
            lf
            .drop_nulls(VAR.CAPACITY)
            .group_by([VAR.IID, VAR.EQUIPMENT])
            .agg(pl.col(VAR.CAPACITY).sum())
            .rename({
                VAR.EQUIPMENT: 'variable',
                VAR.CAPACITY: 'value',
            })
        )

    def cpr_params(self):
        return (
            pl
            .scan_parquet(self.conf.dirs.cpm / 'model.parquet')
            .rename({
                'id': VAR.IID,
                'change_points': 'CP',
                'names': 'cpr-param',
            })
            .with_columns(
                pl.col('holiday').replace_strict(
                    {False: 'workday', True: 'holiday'}, return_dtype=pl.String
                )
            )
            .filter(pl.col('r2') >= self.param.cpr_min_r2)
            .unpivot(
                ['CP', 'coef'],
                index=[VAR.IID, 'cpr-param', 'holiday', 'r2'],
            )
            .drop_nulls('value')
            .with_columns(
                pl.concat_str('cpr-param', 'variable', 'holiday', separator=':').alias(
                    'variable'
                )
            )
        )

    def prep(self):
        params = pl.concat([self.equipment_params(), self.cpr_params()], how='diagonal')
        return (
            self
            .building()
            .join(params, on=VAR.IID, how='full', coalesce=True)
            .drop_nulls('variable')
            .collect()
        )

    def cached(self):
        path = self.conf.dirs.cluster / f'0000.[dataset]{self.param}.parquet'
        mtime = path.stat().st_mtime if path.exists() else 0

        data = utils.pl.frame_cache(path, timeout='1H')(self.prep)().lazy().collect()

        if (
            not (xlsx := path.with_suffix('.xlsx')).exists()
            or mtime != path.stat().st_mtime
        ):
            data.write_excel(xlsx, column_widths=120)

        return data

    def lof(self, data: pl.DataFrame, **kwargs):
        pivot = (
            data
            .pivot('variable', index=[VAR.IID, VAR.GROUP], values='value')
            .with_columns(cs.starts_with(self.EQUIP_PREFIX).fill_null(0))
            .with_columns(cs.numeric().fill_null(cs.numeric().median().over(VAR.GROUP)))
        )
        array = pivot.select(cs.numeric()).to_numpy()

        lof = LocalOutlierFactor(contamination=self.param.contamination, **kwargs)
        labels = lof.fit_predict(array)
        labels_frame = pl.DataFrame([pivot[VAR.IID], pl.Series('lof', labels)])

        return data.join(labels_frame, on=VAR.IID, how='left')

    def __call__(
        self,
        *,
        cache: bool = True,
        lof: bool | dict = False,
        pivot: bool = False,
    ):
        data = self.cached() if cache else self.prep()

        if self.param.cpr_min_r2:
            r2 = pl.col('r2')
            data = data.filter(r2.is_null() | (r2 >= self.param.cpr_min_r2))

        if self.param.day:
            d = pl.col('holiday')
            data = data.filter(d.is_null() | (d == self.param.day))

        if lof and self.param.contamination:
            kwargs = lof if isinstance(lof, dict) else {}
            data = self.lof(data, **kwargs)
        else:
            data = data.with_columns(pl.lit(None).alias('lof'))

        if pivot:
            prefix = 'Intercept:coef:'
            baseline = (
                (f'{prefix}workday',)
                if self.param.day == 'workday'
                else (f'{prefix}workday', f'{prefix}holiday')
            )
            data = (
                data
                .pivot(
                    'variable',
                    index=[*VAR.indices(), 'lof'],
                    values='value',
                    sort_columns=True,
                )
                .drop_nulls(baseline)  # CPR 분석 정보가 있는 데이터
                .filter(
                    # 설비 정보가 하나 이상 있는 데이터
                    pl.any_horizontal(cs.starts_with(self.EQUIP_PREFIX).is_not_null())
                )
                # 설비 빈 항목 -> 0
                .with_columns(cs.starts_with(self.EQUIP_PREFIX).fill_null(0))
            )

        return data


# ===========================================================================


app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml',
        root_keys='public_institution',
        allow_unknown=True,
        use_commands_as_keys=False,
    )
)

app.command(App('eda', help='EDA'))


@app['eda'].command
def eda_count_categories(*, conf: Config, threshold: int = 5):
    """공공기관 카테고리별 개수 파악."""
    src = conf.dirs.data / '1.기관-주소변환.parquet'
    group = pl.col(VAR.GROUP)
    use = pl.col(VAR.USE)

    institutions = (
        pl
        .scan_parquet(src)
        .rename({'asos_code': VAR.REGION})
        .with_columns(
            pl
            .when(pl.len().over(VAR.USE) <= threshold)
            .then(pl.lit('(기타)'))
            .otherwise(use)
            .alias(VAR.USE)
        )
        .with_columns(
            pl
            .when(group.str.starts_with('국립대학병원'))
            .then(pl.lit('국립대학병원 외'))
            .otherwise(group)
            .alias(VAR.GROUP),
            pl
            .when(use.str.starts_with('그 밖에'))
            .then(pl.lit('그 밖에 (...)'))
            .otherwise(use)
            .str.replace(r'\(矯正\)', '')
            .alias(VAR.USE),
        )
    )

    console = rich.get_console()
    utils.mpl.MplTheme(0.85, fig_size=(16, None, 3 / 4)).grid().apply()
    fig, axes = plt.subplots(2, 2)

    ax: Axes
    for ax, var in zip(
        axes.ravel(),
        [VAR.GROUP, VAR.USE, VAR.REGION, VAR.OWNERSHIP],
        strict=True,
    ):
        count = (
            institutions
            .group_by(pl.col(var).fill_null('N/A'))
            .len('count')
            .sort('count', descending=True)
            .collect()
        )
        console.print(count)

        sns.barplot(count, x='count', y=var, ax=ax)
        ax.autoscale_view()
        ax.margins(x=0.2)
        ax.bar_label(ax.containers[0], padding=2, fontsize='small')  # type: ignore[arg-type]
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'{var}별 자료 수', loc='left', weight=500)

    fig.savefig(conf.dirs.cluster / '0000.count.png')


@dc.dataclass
class _CprEda:
    conf: Config
    r2_threshold: float = 0.6

    _data: pl.LazyFrame = dc.field(init=False)

    def __post_init__(self):
        h = pl.col('holiday').replace_strict(
            {'workday': '근무일', 'holiday': '휴일'}, return_dtype=pl.String
        )
        self._data = (
            _Dataset(conf=self.conf, param=_DatasetParams(cpr_min_r2=0))
            .cpr_params()
            .with_columns(h)
        )

    def r2_dist(
        self,
        kind: Literal['hist', 'ecdf'],
        **kwargs: Any,
    ):
        data = (
            self._data
            .filter(pl.col('variable').str.starts_with('Intercept:coef'))
            .select('holiday', 'r2')
            .collect()
        )

        match kind:
            case 'hist':
                fn = sns.histplot
                kwargs.setdefault('bins', np.arange(0, 1.01, 0.05))
            case 'ecdf':
                fn = sns.ecdfplot
                kwargs = {'stat': 'count', 'complementary': True} | kwargs

        fig, ax = plt.subplots()
        fn(
            data.with_columns(pl.format('{} CPR 모델', 'holiday').alias('holiday')),
            x='r2',
            hue='holiday',
            hue_order=['근무일 CPR 모델', '휴일 CPR 모델'],
            ax=ax,
            **kwargs,
        )
        ax.set_xlabel('결정계수')
        if legend := ax.get_legend():
            legend.set_title('')

        if kind == 'ecdf':
            r2 = pl.col('r2')
            h = 'holiday'
            data = (
                data
                .select(h, 'r2')
                .with_columns(r2.rank(descending=True).over(h).alias('rank'))
                .with_columns((pl.col('rank') / pl.len().over(h)).alias('percentile'))
                .filter(r2 >= self.r2_threshold)
                .filter(r2.rank().over(h) == 1)
                .sort(
                    pl.col(h).replace_strict(
                        {'근무일': 0, '휴일': 1}, return_dtype=pl.Int8
                    )
                )
            )

            ax.set_xlim(0, 1)
            ax.axvline(self.r2_threshold, color='k', alpha=0.2, ls='--')
            for y, p, c in zip(
                data['rank'],
                data['percentile'],
                sns.color_palette(n_colors=2),
                strict=True,
            ):
                ax.axhline(y, color=c, alpha=0.5, ls='--')
                ax.annotate(
                    f'r² ≥ {self.r2_threshold} 모델 {int(y)}개 ({p:.1%})',
                    (0, y),
                    xytext=(5, -2),
                    textcoords='offset points',
                    va='top',
                    fontsize='small',
                    weight=500,
                    c=c,
                )

            rich.print(data)

        return fig

    def param_dist(self):
        data = self._data.filter(pl.col('r2') >= self.r2_threshold)

        for (var,), df in (
            data
            .with_columns(pl.col('variable').str.extract(r'(.+:.+):'))
            .collect()
            .group_by('variable')
        ):
            labels = LocalOutlierFactor().fit_predict(
                df.select('r2', 'value').to_numpy()
            )
            inliers = df.filter(labels == 1)
            grid = sns.jointplot(
                inliers, x='r2', y='value', hue='holiday', height=4, alpha=0.5
            )
            grid.figure.legend().set_title('')
            yield var, grid

    def __call__(self):
        d = self.conf.dirs.cluster

        v: Any
        for v in ['hist', 'ecdf']:
            fig = self.r2_dist(v)
            s = f'_r2={self.r2_threshold}' if v == 'ecdf' else ''
            fig.savefig(d / f'0001.CPR-r2-{v}{s}.png')
            plt.close(fig)

        for v, grid in self.param_dist():
            grid.savefig(d / f'0001.CPR-r2 vs {v.replace(":", "-")}.png')
            plt.close(grid.figure)


@app['eda'].command
def eda_cpr_dist(*, conf: Config):
    _CprEda(conf)()


@app['eda'].command
def eda_cpr_dist_example(*, conf: Config):
    """CPR 기저/민감도 분포 예시."""
    variables = {
        'Intercept:coef:workday': '기저부하',
        'HDD:CP:workday': '난방민감도',
        'CDD:CP:workday': '냉방민감도',
    }
    groups = ['공공기관', '중앙행정기관', '지방공사 및 지방공단']
    data = (
        _Dataset(conf=conf, param=_DatasetParams(cpr_min_r2=0.6))()
        .lazy()
        .with_columns(pl.col(VAR.GROUP).str.replace(r'공공기관.*', '공공기관'))
        .filter(
            pl.col('variable').is_in(variables),
            pl.col(VAR.GROUP).is_in(groups),
            ~(
                (pl.col('variable') == 'Intercept:coef:workday')
                & ~pl.col('value').is_between(0, 2)
            ),
            ~(
                (pl.col('variable') == 'HDD:CP:workday')
                & ~pl.col('value').is_between(0, 25)
            ),
            ~(
                (pl.col('variable') == 'CDD:CP:workday')
                & ~pl.col('value').is_between(0, 30)
            ),
        )
        .select(VAR.GROUP, pl.col('variable').replace_strict(variables), 'value')
        .collect()
    )

    rich.print(data)

    grid = (
        sns
        .FacetGrid(
            data,
            row='variable',
            hue=VAR.GROUP,
            sharex=False,
            height=2,
            aspect=2.5 * 16 / 9,
        )
        .map_dataframe(sns.histplot, x='value', kde=True, alpha=0.3)
        .set_axis_labels('')
        .set_titles('')
        .set_titles('{row_name} 분포', weight=500)
        .add_legend()
    )

    for h in getattr(getattr(grid, 'legend', None), 'legend_handles', []):
        h.set_alpha(0.8)

    grid.savefig(conf.dirs.cluster / '0000.cpr-dist.png')


@app['eda'].command
def eda_plot_dist(
    *,
    conf: Config,
    param: _DatasetParams = _DEFAULT_DATASET_PARAMS,
):
    """카테고리별 각 변수 시각화."""
    data = (
        _Dataset(conf=conf, param=param)()
        .drop_nulls('variable')
        .select('variable', 'value', VAR.GROUP, (pl.col('lof') == 1).alias('inlier'))
    )

    output_dir = conf.dirs.cluster / '0001.dist'
    output_dir.mkdir(exist_ok=True)

    file_name = f'0001.dist_{param}'

    utils.mpl.MplTheme(0.8).grid().apply()
    for (group,), df in data.sort('variable').group_by('variable', maintain_order=True):
        logger.info(group)
        g = str(group).replace(':', '-')

        fig, ax = plt.subplots()
        sns.violinplot(
            df.filter(pl.col('inlier')),
            x='value',
            y=VAR.GROUP,
            ax=ax,
            cut=0,
            linewidth=0.5,
        )
        fig.savefig(output_dir / f'{file_name}_{g}_inlier.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.scatterplot(
            df,
            x='value',
            y=VAR.GROUP,
            hue='inlier',
            hue_order=[True, False],
            ax=ax,
            alpha=0.6,
        )
        ax.set_xscale('symlog')
        fig.savefig(output_dir / f'{file_name}_{g}_all.png')
        plt.close(fig)


# ===========================================================================


@app.command
def umap_(
    *,
    conf: Config,
    param: _DatasetParams = _DEFAULT_DATASET_PARAMS,
):
    """UMAP 차원 축소."""
    import plotly.express as px
    import plotly.graph_objects as go
    import umap
    import xlsxwriter

    data = (
        _Dataset(conf=conf, param=param)(pivot=True)
        .filter(pl.col('lof') == 1)
        .with_columns(cs.numeric().fill_null(cs.numeric().median().over(VAR.GROUP)))
    )
    numeric = data.select(cs.numeric())
    scaled = StandardScaler().fit_transform(numeric.to_numpy())

    reducer = umap.UMAP()
    arr = reducer.fit_transform(scaled)
    embedding = pl.concat([data, pl.DataFrame(arr, ['v1', 'v2'])], how='horizontal')

    # ===========================================================================
    file_name = f'0100.cluster_{param}'

    with xlsxwriter.Workbook(conf.dirs.cluster / f'{file_name}.xlsx') as wb:
        embedding.write_excel(wb, worksheet='embedding', column_widths=100)
        pl.concat(
            [
                data.select(~cs.numeric()),
                pl.DataFrame(scaled, numeric.columns),
                embedding.select('v1', 'v2'),
            ],
            how='horizontal',
        ).write_excel(wb, worksheet='scaled', column_widths=100)

    for var in [VAR.GROUP, VAR.USE, VAR.REGION, VAR.OWNERSHIP]:
        fig = px.scatter(
            embedding, x='v1', y='v2', color=var, template='plotly_white'
        ).update_traces(marker=go.scatter.Marker(size=10, opacity=0.5))

        path = conf.dirs.cluster / f'{file_name}_{var}.html'
        fig.write_html(path)


@dc.dataclass
class _HierarchicalClusterParam:
    var: Literal[VAR.GROUP, VAR.USE, VAR.REGION, VAR.OWNERSHIP]
    equip_by_use: bool

    cpr_day: Literal['workday', 'both']
    cpr_cp: bool

    center: Literal['mean', 'median']
    scale_data: Literal['all', 'center'] = 'all'

    def __str__(self):
        return (
            f'var={self.var.name} '
            f'EquipUse={self.equip_by_use} day={self.cpr_day} cp={self.cpr_cp} '
            f'scale={self.scale_data}-{self.center}'
        )

    @classmethod
    def _iter(cls):
        for x in itertools.product(
            [VAR.GROUP, VAR.USE, VAR.REGION, VAR.OWNERSHIP],
            [False, True],
            ['workday'],  # 근무일만
            [False, True],  # CP
            ['mean', 'median'],
            ['all', 'center'],
        ):
            yield cls(*x)  # type: ignore[arg-type]

    @classmethod
    def iter(cls, *, track: bool = True):
        it = cls._iter()
        if track:
            it = Progress.iter(tuple(it))
        return it


@dc.dataclass
class _HierarchicalCluster:
    conf: Config
    merge: bool = False  # 공공기관 시장형, 비시장형 통합 여부
    param: dc.InitVar[_DatasetParams] = _DEFAULT_DATASET_PARAMS
    _dataset: _Dataset = dc.field(init=False)

    class Cluster(NamedTuple):
        fig: Figure
        data: pl.DataFrame
        score: dict[str, float]

    def __post_init__(self, param):
        self._dataset = _Dataset(conf=self.conf, param=param)

    def __hash__(self):
        return hash(str(self.conf))

    @staticmethod
    @functools.lru_cache
    def _data(
        dataset: _Dataset,
        *,
        equip_by_use: bool,
        cpr_day: Literal['workday', 'both'],
        cpr_cp: bool,
    ):
        dataset.param.day = None if cpr_day == 'both' else cpr_day
        dataset.param.equip_by_use = equip_by_use

        data = (
            dataset(lof=False, pivot=True)
            .with_columns(
                cs.numeric().fill_null(cs.numeric().median()),
                pl.col(VAR.USE).str.replace(r'^그 밖에.*', '그 밖에 (...)'),
            )
            .with_columns(
                pl.col(VAR.GROUP, VAR.USE, VAR.REGION, VAR.OWNERSHIP).fill_null('N/A')
            )
        )

        if not cpr_cp:
            data = data.drop(cs.contains(':CP:'))

        # LOF
        array = data.select(cs.numeric()).to_numpy()
        labels = LocalOutlierFactor(n_neighbors=8).fit_predict(array)

        return data.with_columns(pl.Series('lof', labels))

    def data(
        self,
        *,
        equip_by_use: bool,
        cpr_day: Literal['workday', 'both'],
        cpr_cp: bool,
    ):
        data = self._data(
            dataset=self._dataset,
            equip_by_use=equip_by_use,
            cpr_day=cpr_day,
            cpr_cp=cpr_cp,
        )

        if self.merge:
            data = data.with_columns(
                pl
                .when(pl.col(VAR.GROUP).str.starts_with('공공기관'))
                .then(pl.lit('공공기관'))
                .otherwise(pl.col(VAR.GROUP))
                .alias(VAR.GROUP)
            )

        return data

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
                fontsize='x-small',
                alpha=0.8,
                path_effects=[patheffects.withStroke(linewidth=2, foreground='w')],
            )

        return r

    def cluster(self, *, param: _HierarchicalClusterParam):
        data = self.data(
            cpr_day=param.cpr_day,
            equip_by_use=param.equip_by_use,
            cpr_cp=param.cpr_cp,
        ).filter(pl.col('lof') == 1)

        counts = dict(
            data.select(
                param.var,
                pl.format('{} ({})', param.var, pl.len().over(param.var)).alias(
                    'count'
                ),
            ).iter_rows()
        )

        # 각 그룹 대표값
        center = (
            data
            .group_by(param.var)
            .agg(
                cs.numeric().mean() if param.center == 'mean' else cs.numeric().median()
            )
            .sort(param.var)
        )

        scaler = StandardScaler().fit(
            (data if param.scale_data == 'all' else center)
            .drop(param.var, cs.string())
            .to_numpy()
        )

        # 그룹 대표값 클러스터링
        scaled_center = scaler.transform(center.drop(param.var).to_numpy())
        linked = hrc.ward(scaled_center)

        fig, ax = plt.subplots()
        r = self.annotated_dendrogram(linked, labels=center[param.var].to_list(), ax=ax)

        ax.set_yticks(
            ax.get_yticks(), [counts[x.get_text()] for x in ax.get_yticklabels()]
        )

        # 클러스터 평가 (0.7 * max(dist) 기준)
        colors = [
            f'{s}-{i}' if s == 'C0' else s for i, s in enumerate(r['leaves_color_list'])
        ]
        cluster_map = dict(zip(r['ivl'], colors, strict=True))
        cluster = data.with_columns(
            pl.col(param.var).replace_strict(cluster_map).alias('cluster')
        )
        score = self.evaluate(
            scaler.transform(data.select(cs.numeric()).to_numpy()),
            data[param.var].replace_strict(cluster_map).to_list(),
        )

        return self.Cluster(fig, cluster, score)

    def batch_cluster(self):
        # 군집화 결과 저장할 데이터
        score = []  # 군집화 점수

        merge = '-merge' if self.merge else ''
        output = self.conf.dirs.cluster / f'0200.HierarchyCluster{merge}'
        output.mkdir(exist_ok=True)

        for idx, param in enumerate(_HierarchicalClusterParam.iter()):
            logger.info(param)

            cluster = self.cluster(param=param)

            name = f'{idx:04d}. {param}'
            cluster.data.write_excel(output / f'{name}.xlsx', column_widths=120)
            cluster.fig.savefig(output / f'{name}.png')
            plt.close(cluster.fig)

            score.append(dc.asdict(param) | cluster.score)

        (
            pl
            .from_dicts(score)
            .with_row_index()
            .sort('silhouette', descending=True)
            .write_excel(output / '0000 score.xlsx', column_widths=120)
        )


@app.command
def hierarchy(
    *,
    scale: float = 1.0,
    merge: bool = False,
    conf: Config,
    param: _DatasetParams = _DEFAULT_DATASET_PARAMS,
):
    """
    공공기관 기관대분류 계층적 군집.

    Parameters
    ----------
    conf : Config
    """
    utils.mpl.MplTheme(scale).grid().apply({'lines.linewidth': 2})
    _HierarchicalCluster(conf=conf, merge=merge, param=param).batch_cluster()


@dc.dataclass
class _ClusterDist:
    conf: Config
    data: pl.DataFrame = dc.field(init=False)

    # TODO anova
    cluster1: Sequence[str] = ('국립대학 및 공립대학',)
    cluster2: Sequence[str] = ('국립대학병원 외', '광역지방자치단체', '정부청사관리')

    @dc.dataclass
    class Case:
        clusters: Literal[2, 3]
        group: Literal['면적', '설비', 'CPR']
        plot: Literal['hist', 'kde']
        log: bool

        def __str__(self):
            return (
                f'clusters{self.clusters} {self.group} '
                f'{self.plot} {"log" if self.log else "linear"}'
            )

        @classmethod
        def iter(cls):
            args: Any
            for args in itertools.product(
                [2, 3], ['면적', '설비', 'CPR'], ['hist', 'kde'], [False, True]
            ):
                yield cls(*args)

    def __post_init__(self):
        variables = {
            'Intercept:coef': '기저부하',
            'CDD:CP': '냉방 균형점 온도',
            'HDD:CP': '난방 균형점 온도',
            'CDD:coef': '냉방 민감도',
            'HDD:coef': '난방 민감도',
            'HP': '히트펌프',
        }
        param = _DatasetParams(equip_by_use=True, cpr_min_r2=0.6)
        self.data = (
            _Dataset(self.conf, param)(lof=False, pivot=False)
            .with_columns(
                pl
                .col('variable')
                .str.starts_with('설비:')
                .replace_strict({True: '설비', False: 'CPR'}, return_dtype=pl.String)
                .alias('group')
            )
            .with_columns(
                pl
                .col('variable')
                .str.strip_suffix(':workday')
                .str.strip_prefix('설비:')
                .replace(variables)
            )
        )

    def __hash__(self):
        return hash(str(self))  # XXX

    @functools.lru_cache  # noqa: B019
    def _prep_data(self, clusters, group):
        replace = dict.fromkeys(self.cluster1, '국공립대학')

        if clusters == 3:  # noqa: PLR2004
            replace |= dict.fromkeys(self.cluster2, '군집 2')
            default = '군집 3'
        else:
            default = '기타'

        data = self.data.with_columns(
            pl
            .col('기관대분류')
            .replace_strict(replace, default=default)
            .alias('cluster')
        )

        if group == '면적':
            data = data.select(
                'cluster',
                pl.lit('연면적 [m²]').alias('variable'),
                pl.col('연면적').alias('value'),
            )
        else:
            data = data.filter(pl.col('group') == group).select(
                'cluster', 'variable', 'value'
            )

        return data

    @staticmethod
    def _kdeplot(*args, **kwargs):
        ax: Axes = kwargs.pop('ax', None) or plt.gca()
        kwargs['log_scale'] = (
            kwargs.get('log_scale', False)  # fmt
            and ('온도' not in ax.get_title(loc='left'))
        )
        return sns.kdeplot(*args, **kwargs, ax=ax)

    @staticmethod
    def _plot_label(grid: sns.FacetGrid, case: Case):
        if legend := grid.legend:
            legend.set_title('')

        match case.group:
            case '면적':
                utils.mpl.move_legend_fig_to_ax(grid.figure, grid.axes.ravel()[0])
                grid = grid.set_titles('', loc='left').set_xlabels('연면적 [m²]')
            case '설비':
                utils.mpl.move_grid_legend(grid)
                grid = grid.set_xlabels('용량 [kW/m²]')
            case 'CPR':
                utils.mpl.move_grid_legend(grid)
                for col, ax in grid.axes_dict.items():
                    if '민감도' in col:
                        label = '민감도 [kWh/m²°C]'
                    elif '균형점' in col:
                        label = '일평균 외기온 [°C]'
                    else:
                        label = '기저부하 [kWh/m²]'

                    ax.set_xlabel(label)

        return grid.tight_layout()

    @classmethod
    def _plot(cls, data: pl.DataFrame, case: Case):
        variables = data['variable'].unique().sort().to_list()
        if case.group == 'CPR':
            variables = sorted(variables, key=lambda x: ('온도' in x, x))

        clusters = sorted(data['cluster'].unique(), key=lambda x: (x == '기타', x))

        if case.plot == 'hist':
            plot = {
                'func': sns.histplot,
                'stat': 'probability',
                'element': 'step',
                'alpha': 0.25,
            }
        else:
            plot = {
                'func': cls._kdeplot,
                'fill': True,
                'warn_singular': False,
                'alpha': 0.25,
            }

        grid = (
            sns
            .FacetGrid(
                data,
                col='variable',
                col_wrap=int(utils.mpl.ColWrap(len(variables))),
                col_order=variables,
                hue='cluster',
                hue_order=clusters,
                sharex=case.log and case.group != 'CPR',
                sharey=False,
                aspect=4 / 3,
                height=4 if case.group == '면적' else 3,
            )
            .set_titles('')
            .set_titles('{col_name}', loc='left', weight=500)
            .map_dataframe(x='value', log_scale=case.log, **plot)
            .set_xlabels('')
            .add_legend()
        )

        return cls._plot_label(grid, case)

    def _output(self):
        return self.conf.dirs.cluster / '0201.distribution'

    def visualize(self, *, hist: bool = False):
        output = self._output()
        output.mkdir(exist_ok=True)

        for case in Progress.iter(list(self.Case.iter())):
            if not hist and case.plot == 'hist':
                continue

            logger.info(repr(case))

            data = self._prep_data(case.clusters, case.group)

            if case.log:
                data = data.filter(pl.col('value') > 0)

            grid = self._plot(data, case)
            grid.savefig(output / f'{case}.png')
            plt.close(grid.figure)

    def stat_test(self):
        """군집 간 통계적 검정."""
        clusters = ('국공립대학', '군집 2', '군집 3')
        data = (
            pl
            .concat([
                self._prep_data(3, '면적').with_columns(
                    pl.col('variable').str.strip_suffix(' [m²]')
                ),
                self._prep_data(3, '설비').with_columns(
                    pl.format('{} 용량', 'variable')
                ),
                self._prep_data(3, 'CPR'),
            ])
            .with_columns(
                pl
                .col('variable')
                .str.extract('(면적|용량|온도|기저부하|민감도)')
                .replace_strict({
                    '면적': '면적 [m²]',
                    '용량': '용량 [kW/m²]',
                    '온도': '평균 외기온 [°C]',
                    '기저부하': '면적 [kWh/m²]',
                    '민감도': '민감도 [kW/m²°C]',
                })
                .alias('label')
            )
            .with_columns(hue=pl.col('cluster') != clusters[0])
        )

        col_wrap = utils.mpl.ColWrap(data['variable'].n_unique())
        fig = Figure(figsize=(32 / 2.54, 18 / 2.54))
        axes = fig.subplots(col_wrap.nrows, col_wrap.ncols, sharex=False)
        annotator = Annotator(
            ax=None,
            pairs=list(itertools.combinations(clusters, 2)),
        )

        by: Any
        ax: Axes
        for (by, df), ax in zip(
            data.group_by('variable', maintain_order=True),
            axes.ravel(),
            strict=False,
        ):
            kwargs = {
                'data': df.to_pandas(),
                'x': 'value',
                'y': 'cluster',
                'order': clusters,
            }
            sns.boxplot(
                **kwargs,
                ax=ax,
                hue='hue',
                log_scale='온도' not in by[0],
                legend=False,
            )
            (
                annotator
                .new_plot(ax=ax, orient='h', **kwargs)
                .configure(test='t-test_ind')
                .apply_test()
                .annotate()
            )
            ax.set_title(by[0], loc='left', weight=500)
            ax.set_ylabel('')
            ax.set_xlabel(df['label'][0])

        for ax in axes.ravel():
            if not ax.has_data():
                ax.set_axis_off()

        output = self._output()
        output.mkdir(exist_ok=True)
        fig.savefig(output / 'stat-test.png')


@app.command
def cluster_dist(*, conf: Config):
    """군집별 인자 분포 시각화."""
    warnings.filterwarnings('ignore', message='The figure layout has changed to tight')
    utils.mpl.MplTheme().grid().tick('x', 'both', color='.4').apply()

    dist = _ClusterDist(conf)
    dist.visualize()
    dist.stat_test()


@dc.dataclass
class _RelativeEval:
    class Case(NamedTuple):
        var: str
        groups: Sequence[Sequence[str]]

        def elements(self):
            return itertools.chain.from_iterable(self.groups)

        def filters(self):
            expr = pl.col(self.var)
            yield from ((x, expr.is_in(x)) for x in self.groups)
            yield None, expr.is_in(list(self.elements())).not_()

        def replace(self):
            def it():
                for group in self.groups:
                    name = '+'.join(group)
                    for g in group:
                        yield g, name

            return dict(it())

    conf: Config

    target: str = 'DB_B7AE8782-40AF-8EED-E050-007F01001D51'  # KEA
    cases: Sequence[Case] = (
        Case(
            '기관대분류',
            [['국립대학 및 공립대학'], ['중앙행정기관', '공공기관(시장형·준시장형)']],
        ),
        Case('기관대분류', [['국립대학 및 공립대학']]),
        Case('건물용도', [['업무시설', '교육연구시설']]),
        Case(
            '건물용도',
            [['업무시설', '교육연구시설'], ['의료시설', '방송통신시설', '판매시설']],
        ),
    )

    dataset: _Dataset = dc.field(init=False)
    data: pl.DataFrame = dc.field(init=False)
    params: tuple[str, ...] = dc.field(init=False)
    plot_style: dict = dc.field(init=False)

    def __post_init__(self):
        self.dataset = _Dataset(
            conf=self.conf,
            param=_DatasetParams(
                cpr_min_r2=0.5,  # KEA
                day='workday',
            ),
        )
        self.data = self.dataset(lof=True, pivot=True)
        self.params = tuple(
            sorted(self.data.select(cs.numeric() & cs.contains(':')).columns)
        )
        self.plot_style = {
            'col_wrap': 4,
            'sharex': False,
            'sharey': False,
            'height': 2.5,
            'aspect': 16 / 9,
        }

    def plot_dist(self, case: Case):
        """클러스터별 파라미터 분포."""
        data = (
            self.data
            .filter(pl.col('lof') == 1)
            .with_columns(
                pl
                .col(case.var)
                .replace_strict(case.replace(), default='기타')
                .alias('group')
            )
            .filter(pl.col('lof') == 1)
            .unpivot(self.params, index=['기관ID', 'group'])
        )
        grid = (
            sns
            .FacetGrid(
                data,
                col='variable',
                col_order=self.params,
                hue='group',
                **self.plot_style,
            )
            .map_dataframe(
                sns.histplot,
                'value',
                stat='probability',
                common_norm=False,
                bins='doane',
                kde=True,
            )
            .set_titles('')
            .set_titles('{col_name}', loc='left', weight=500)
            .set_axis_labels('', '')
            .add_legend()
        )

        legend = utils.mpl.move_legend_fig_to_ax(grid.figure, ax=grid.axes.ravel()[-1])
        legend.set_title('')  # FIXME

        return grid

    def plot_model(self, data: pl.DataFrame):
        """클러스터 대표CPR 모델 vs 대상 건물."""
        params = [x for x in self.params if not x.startswith('설비:')]

        target_params = dict(
            zip(
                params,
                self.data.filter(pl.col('기관ID') == self.target).select(params).row(0),
                strict=True,
            )
        )
        repr_ = (
            data
            .filter(pl.col('lof') == 1)
            .unpivot(params)
            .group_by('variable')
            .agg(pl.median('value'))
        )
        repr_params = dict(zip(repr_['variable'], repr_['value'], strict=True))

        def model(d: dict):
            return cpr.CprModel.from_params(
                intercept=d['Intercept:coef:workday'],
                cp=(d['HDD:CP:workday'], d['CDD:CP:workday']),
                coef=(d['HDD:coef:workday'], d['CDD:coef:workday']),
            )

        fig = Figure()
        ax = fig.add_subplot()
        colors = ['dimgray', 'steelblue']

        for p, color in zip([target_params, repr_params], colors, strict=True):
            m = model(p)
            m.plot(
                data=None,
                ax=ax,
                segments=True,
                scatter=False,
                style={'line': {'c': color}},
            )

        ax.legend(
            handles=[Line2D([0], [0], color=c) for c in colors],
            labels=['대표 모델', '대상 건물'],
        )
        ax.set_xlabel('일평균 기온 [°C]')
        ax.set_ylabel('에너지 소비량 [kWh/m²]')
        ax.set_ylim(0)

        ConstrainedLayoutEngine().execute(fig)

        return fig

    def plot_percentile(self, data: pl.DataFrame):
        """클러스터 파라미터 분포에 대상 건물 수치와 백분위수 표시."""
        target_params = (
            self.data.filter(pl.col('기관ID') == self.target).select(self.params).row(0)
        )
        unpivot = data.filter(pl.col('lof') == 1).unpivot(self.params, index='기관ID')

        grid = (
            sns
            .FacetGrid(
                unpivot,
                col='variable',
                col_order=self.params,
                **self.plot_style,
            )
            .map_dataframe(sns.histplot, 'value', bins='doane', kde=True)
            .set_titles('')
            .set_titles('{col_name}', loc='left', weight=500)
            .set_axis_labels('', '')
        )

        # 대상 건물의 백분위수 계산
        percentile = (
            pl
            .concat(
                [
                    unpivot,
                    pl.DataFrame({'variable': self.params, 'value': target_params}),
                ],
                how='diagonal',
            )
            .with_columns(
                (
                    pl.col('value').rank(descending=True).over('variable')
                    / pl.len().over('variable')
                ).alias('percentile')
            )
            .filter(pl.col('기관ID').is_null())
            .sort('variable')
        )

        for row in percentile.iter_rows(named=True):
            ax = grid.axes_dict[row['variable']]
            ax.axvline(row['value'], c='darkslategray', ls='--')
            ax.text(
                0.98,
                0.98,
                f'{row["percentile"]:.1%}',
                transform=ax.transAxes,
                ha='right',
                va='top',
                weight=400,
                color='darkslategray',
            )

        return grid

    def _iter_case(self):
        yield 0, '전체', None, self.data

        for idx, case in enumerate(self.cases):
            if left := (
                set(case.elements())
                - set(self.data.select(pl.col(case.var).unique()).to_series())
            ):
                raise ValueError(left)

            for group, expr in case.filters():
                yield idx + 1, case.var, group, self.data.filter(expr)

    def __call__(
        self,
        *,
        dist: bool = True,
        model: bool = True,
        percentile: bool = True,
    ):
        output = self.conf.dirs.cluster / '0300.Evaluation'
        output.mkdir(exist_ok=True)

        if dist:
            # 클러스터별 분포
            for idx, case in enumerate(Progress.iter(self.cases)):
                logger.info('case={}', case)

                grid = self.plot_dist(case)
                grid.savefig(output / f'0001.dist-{idx:04d}-{case.var}.png')
                plt.close(grid.figure)

        if not (model or percentile):
            return

        total = (
            1 + sum(len(x.groups) + 1 for x in self.cases)
            if (model or percentile)
            else None
        )
        # 클러스터 분포 vs 대상 건물
        for idx, case, group, data in Progress.iter(self._iter_case(), total=total):
            logger.info('case={} | group={}', case, group)
            g = ''.join(x[0] for x in group) if group else 'none'
            name = f'case{idx:04d}-{case}-{g}'

            if model:
                fig = self.plot_model(data)
                fig.savefig(output / f'0002.model-{name}.png')

            if percentile:
                grid = self.plot_percentile(data)
                grid.savefig(output / f'0003.percentile-{name}.png')
                plt.close(grid.figure)


@app.command
def relative_eval(
    target: str = 'DB_B7AE8782-40AF-8EED-E050-007F01001D51',  # KEA
    *,
    dist: bool = True,
    model: bool = True,
    percentile: bool = True,
    conf: Config,
):
    """클러스터링 결과 중 테스트 건물의 위치 평가."""
    utils.mpl.MplTheme(constrained=False).grid().apply()

    r = _RelativeEval(conf=conf, target=target)
    r(dist=dist, model=model, percentile=percentile)


@app.command
def equipment_capacity(conf: Config):
    """설비 용량 시각화 (발표자료용)."""
    dataset = _Dataset(conf)
    params = (
        dataset
        .equipment_params()
        .with_columns(pl.col('variable').str.strip_prefix('설비:'))
        .collect()
    )

    color = sns.color_palette(n_colors=3)[2]

    grid = (
        sns
        .FacetGrid(
            params,
            col='variable',
            col_order=[
                'EHP',
                'GHP',
                '지열히트펌프',
                '열교환기',
                '온수보일러',
                '증기보일러',
                '냉온수기',
            ],
            col_wrap=3,
            despine=False,
            sharey=False,
            height=2,
            aspect=4 / 3,
        )
        .map_dataframe(sns.histplot, x='value', kde=True, log_scale=True, color=color)
        .set_xlabels('용량 [kWh/m²]')
        .set_titles('')
        .set_titles('{col_name}', loc='left', weight=500)
    )
    grid.savefig(r'D:\wd\greenbutton\AMI\PublicInstitution\0300.cluster\capacity.png')


if __name__ == '__main__':
    utils.mpl.MplTheme().grid().apply()
    utils.terminal.LogHandler.set()

    app()

    # NOTE
    # - 연면적 추가?
    # - ANOVA?

from __future__ import annotations

import dataclasses as dc
import shutil
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import matplotlib.pyplot as plt
import pingouin as pg
import polars as pl
import polars.selectors as cs
import seaborn as sns
from cmap import Colormap
from loguru import logger

from greenbutton import cpr, misc, utils
from greenbutton.utils import App, Progress
from scripts.ami.public_institution.config import Config  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.typing import ColorType
    from polars._typing import FrameType


class AreaError(ValueError):
    def __init__(self, area: float):
        self.area = area
        super().__init__(f'{area=}')


class VAR:
    IID = '기관ID'
    CATEGORY = '기관대분류'
    NAME = '기관명'
    AREA = '연면적'
    REGION = '지역'
    ELEC_RATIO = '전기식용량비율'


@dc.dataclass
class Institution:
    iid: str
    category: str
    name: str
    area: float
    region: str
    elec_ratio: float

    def file_name(self):
        return f'{self.category}_{self.name}_{self.iid}'


@dc.dataclass
class Dataset:
    conf: Config
    energy: Literal['사용량', '보정사용량'] = '보정사용량'
    institutions: pl.LazyFrame = dc.field(init=False)

    def __post_init__(self):
        data_dir = self.conf.dirs.data
        equipment = (
            pl.scan_parquet(data_dir / self.conf.files.equipment)
            .select(VAR.IID, VAR.ELEC_RATIO)
            .with_columns()
        )
        self.institutions = (
            pl.scan_parquet(data_dir / self.conf.files.institution)
            .with_columns(
                pl.when(pl.col('기관대분류').str.starts_with('국립대학병원'))
                .then(pl.lit('국립대학병원 등'))
                .otherwise(pl.col('기관대분류'))
                .alias('기관대분류'),
            )
            .rename({'asos_code': VAR.REGION})
            .with_row_index()
            .join(equipment, on=VAR.IID, how='left')
            .select(
                VAR.IID, VAR.CATEGORY, VAR.NAME, VAR.AREA, VAR.REGION, VAR.ELEC_RATIO
            )
        )

    def institution(self, iid: str):
        data = self.institutions.filter(pl.col(VAR.IID) == iid).collect()
        if data.height != 1:
            raise ValueError(iid, data)
        return Institution(*data.row(0))

    def iter_institutions(self, *, track: bool = True):
        it: Iterable[str] = (
            self.institutions.select(pl.col(VAR.IID).unique().sort())
            .collect()
            .to_series()
            .to_list()
        )
        if track:
            it = Progress.trace(it)

        return (self.institution(x) for x in it)

    def ami(self, iid: str):
        return (
            pl.scan_parquet(self.conf.dirs.data / 'AMI*.parquet')
            .filter(pl.col(VAR.IID) == iid)
            .with_columns()
        )

    def temperature(self, region: str):
        path = self.conf.root.parents[1] / 'weather' / self.conf.files.temperature
        return (
            pl.scan_parquet(path)
            .with_columns(pl.col('region2').replace({'제주도': '제주'}))
            .filter(pl.col('region2') == region)
            .select('datetime', pl.col('ta').alias('temperature'))
        )

    @staticmethod
    def _with_is_holiday(data: FrameType):
        years = (
            data.lazy()
            .select(pl.col('datetime').dt.year().unique())
            .collect()
            .to_series()
        )
        return data.with_columns(
            is_holiday=misc.is_holiday(pl.col('datetime'), years=years)
        )

    def data(
        self,
        institution: str | Institution,
        *,
        interval: str = '1d',
        with_temperature: bool = True,
        with_holiday: bool = True,
    ) -> tuple[Institution, pl.LazyFrame]:
        if isinstance(institution, str):
            inst = self.institution(institution)
        else:
            inst = institution

        if not inst.area or inst.area <= 0:
            raise AreaError(inst.area)

        data = (
            self.ami(inst.iid)
            .select('datetime', pl.col(self.energy).truediv(inst.area).alias('energy'))
            .sort('datetime')
            .group_by_dynamic('datetime', every=interval)
            .agg(pl.sum('energy'))
        )

        if with_temperature:
            temperature = (
                self.temperature(inst.region)
                .sort('datetime')
                .group_by_dynamic('datetime', every=interval)
                .agg(pl.mean('temperature'))
            )
            data = data.join(temperature, on='datetime', how='left').sort('datetime')

        if with_holiday:
            data = self._with_is_holiday(data)

        return inst, data


@cyclopts.Parameter(name='cpr')
@dc.dataclass
class CprConfig:
    energy: Literal['사용량', '보정사용량'] = '보정사용량'
    interval: str = '1d'
    holiday: bool | None = False
    plot: bool = True
    year: int | None = None

    min_samples: int = 4

    style: ClassVar[cpr.PlotStyle] = {'scatter': {'s': 12, 'alpha': 0.25}}

    def suffix(self):
        h = {True: '휴일', False: '평일', None: '전체'}[self.holiday]
        return f'{self.interval}_{h}_{self.energy}'

    def name(self, t: Literal['model', 'plot']):
        return f'{t}{"" if self.year is None else self.year}'


@dc.dataclass(frozen=True)
class CprCalculator:
    conf: Config
    cpr_conf: CprConfig
    dataset: Dataset

    def file_name(self, institution: Institution):
        return f'{institution.file_name()}_{self.cpr_conf.suffix()}'

    def dir(self, t: Literal['model', 'plot']):
        return self.conf.dirs.cpr / self.cpr_conf.name(t)

    def cpr(self, institution: str | Institution):
        conf = self.cpr_conf

        inst, lf = self.dataset.data(
            institution,
            interval=conf.interval,
            with_holiday=conf.holiday is not None,
        )
        if conf.holiday is not None:
            lf = lf.filter(pl.col('is_holiday') == conf.holiday)
        if self.cpr_conf.year is not None:
            lf = lf.filter(pl.col('datetime').dt.year() == self.cpr_conf.year)

        model = cpr.CprEstimator(
            lf.collect(), conf=cpr.CprConfig(min_samples=conf.min_samples)
        ).fit()

        return inst, model

    def _plot(self, model: cpr.CprAnalysis, ax: Axes | None = None):  # type: ignore[name-defined]
        if ax is None:
            ax = plt.gca()

        model.plot(ax=ax, style=self.cpr_conf.style)

        if ax.dataLim.y0 > 0:
            ax.set_ylim(bottom=0)

        ax.set_xlabel('기온 [℃]')
        ax.set_ylabel('전력사용량 [kWh/m²]')

        return ax

    def cpr_and_write(self, institution: str | Institution):
        inst, model = self.cpr(institution)
        name = self.file_name(inst)

        model_frame = model.model_frame.select(
            pl.lit(inst.iid).alias('id'),
            pl.lit(inst.category).alias('category'),
            pl.lit(inst.name).alias('name'),
            pl.lit(inst.elec_ratio).alias('elec_ratio'),
            pl.lit(self.cpr_conf.interval).alias('interval'),
            pl.lit(self.cpr_conf.holiday).alias('holiday'),
            pl.all(),
        )

        model_frame.write_parquet(self.dir('model') / f'{name}.parquet')

        if self.cpr_conf.plot:
            fig, ax = plt.subplots()
            self._plot(model, ax=ax)
            ax.set_title(
                f'{inst.name} (r²={model.model_dict["r2"]:.4g})', loc='left', weight=500
            )

            fig.savefig(self.dir('plot') / f'{name}.png')
            plt.close(fig)

    def batch_cpr(self, *, skip_existing: bool = True):
        model_dir = self.dir('model')
        model_dir.mkdir(exist_ok=True)
        self.dir('plot').mkdir(exist_ok=True)

        for institution in self.dataset.iter_institutions():
            model = model_dir / f'{self.file_name(institution)}.parquet'
            if skip_existing and model.exists():
                continue

            logger.info(institution)

            try:
                self.cpr_and_write(institution)
            except (cpr.CprError, AreaError) as e:
                logger.error(str(e))


app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml',
        root_keys='public_institution',
        use_commands_as_keys=False,
    )
)

_DEFAULT_CPR_CONF = CprConfig()


@app.command
def cpr_(
    *,
    iid: str,
    conf: Config,
    cpr_conf: CprConfig = _DEFAULT_CPR_CONF,
):
    """CPR 분석."""
    cc = CprCalculator(conf=conf, cpr_conf=cpr_conf, dataset=Dataset(conf=conf))
    cc.cpr_and_write(institution=iid)


@app.command
def cpr_parallel(
    *,
    iid: str = 'DB_B7AE8782-40AF-8EED-E050-007F01001D51',
    conf: Config,
    cpr_conf: CprConfig = _DEFAULT_CPR_CONF,
):
    """근무일, 휴일 비교 그래프."""
    cc = CprCalculator(conf=conf, cpr_conf=cpr_conf, dataset=Dataset(conf=conf))

    cc.cpr_conf.holiday = False
    _, workday_model = cc.cpr(iid)
    cc.cpr_conf.holiday = True
    _, holiday_model = cc.cpr(iid)

    fig, axes = plt.subplots(
        1, 2, sharex=True, sharey=True, figsize=(32 / 2.54, 12 / 2.54)
    )
    style: cpr.PlotStyle = {
        'datetime_hue': False,
        'scatter': {'alpha': 0.25, 'color': 'slategray'},
    }
    workday_model.plot(ax=axes[0], style=style)
    holiday_model.plot(ax=axes[1], style=style)

    for ax in axes:
        ax.set_xlabel('일평균 기온 [°C]')
        ax.set_ylabel('전력 사용량 [kWh/m²]')

    fig.savefig(
        r'D:\wd\greenbutton\AMI\PublicInstitution\0201.CPRcompensation\공단 모델.png'
    )


@app.command
def concat_cpr(*, conf: Config, cpr_conf: CprConfig = _DEFAULT_CPR_CONF):
    """기관별 CPR 분석 결과 통합."""
    d = conf.dirs.cpr
    n = cpr_conf.name('model')

    data = pl.concat(
        (
            pl.scan_parquet(x).with_columns(cs.by_dtype(pl.Null).cast(pl.Float64))
            for x in (d / n).glob('*.parquet')
        ),
        how='vertical',
    ).collect()

    data.write_parquet(d / f'{n}.parquet')
    data.write_excel(d / f'{n}.xlsx', column_widths=max(50, int(1600 / data.width)))


@app.command
def batch_cpr(*, conf: Config, cpr_conf: CprConfig = _DEFAULT_CPR_CONF):
    """전체 기관 CPR 일괄 분석."""
    cc = CprCalculator(conf=conf, cpr_conf=cpr_conf, dataset=Dataset(conf=conf))
    cc.batch_cpr()
    concat_cpr(conf=conf, cpr_conf=cpr_conf)


@app.command
def plot_elec_r2(conf: Config, cpr_conf: CprConfig = _DEFAULT_CPR_CONF):
    data = (
        pl.scan_parquet(conf.dirs.cpr / f'{cpr_conf.name("plot")}.parquet')
        .filter(pl.col('names') == 'Intercept', pl.col('elec_ratio').is_not_null())
        .with_columns(
            pl.col('holiday').replace_strict(
                {True: '휴일', False: '평일'}, return_dtype=pl.String
            )
        )
        .collect()
    )

    grid = sns.lmplot(
        data,  # pyright: ignore[reportArgumentType]
        x='elec_ratio',
        y='r2',
        hue='holiday',
        hue_order=['평일', '휴일'],
        scatter_kws={'alpha': 0.4},
        line_kws={'alpha': 0.75},
        facet_kws={'despine': False},
    ).set_axis_labels('전기식 설비 용량비', 'r²')
    if legend := grid.legend:
        legend.set_title('')
    grid.savefig(conf.dirs.cpr / '전기식 설비 용량비 vs r².png')


@dc.dataclass
class CprCoefPlotter:
    """건물 유형별 CPR 모델 파라미터 그래프."""

    conf: Config
    cpr_conf: CprConfig

    estimator: Literal['median', 'mean'] = 'median'
    min_r2: float | None = None

    # TODO max_beta [kW/m²]

    palette: Sequence[ColorType] = ()
    data: pl.LazyFrame = dc.field(init=False)

    def __post_init__(self):
        data = (
            pl.scan_parquet(
                self.conf.dirs.cpr / f'{self.cpr_conf.name("model")}.parquet'
            )
            .rename({'id': VAR.IID}, strict=False)
            .with_columns(
                pl.col('holiday').replace_strict(
                    {False: '평일', True: '휴일'}, return_dtype=pl.String
                )
            )
            .filter(pl.col('r2') > (self.min_r2 or 0))
        )

        institution = (
            pl.scan_parquet(self.conf.dirs.data / self.conf.files.institution)
            .select(VAR.IID, VAR.CATEGORY)
            .with_columns(
                pl.when(pl.col(VAR.CATEGORY).str.starts_with('국립대학병원'))
                .then(pl.lit('국립대학병원 등'))
                .otherwise(pl.col(VAR.CATEGORY))
                .alias(VAR.CATEGORY)
            )
        )
        self.data = (
            data.join(institution, on=VAR.IID, how='left')
            .sort(VAR.CATEGORY)
            .collect()
            .lazy()
        )

        if not self.palette:
            self.palette = Colormap('tol:bright-alt')([0, 1]).tolist()

    def barplot(
        self,
        variable: Literal['Intercept', 'CDD', 'HDD'],
        x: Literal['r2', 'coef', 'change_point'],
        xlabel: str | None = None,
    ):
        fig, ax = plt.subplots()
        sns.barplot(
            self.data.filter(pl.col('names') == variable).collect(),
            x=x,
            y=VAR.CATEGORY,
            hue='holiday',
            hue_order=['평일', '휴일'],
            palette=self.palette,
            ax=ax,
            estimator=self.estimator,
            alpha=0.9,
        )
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel('')
        ax.get_legend().set_title('')
        return fig, ax

    def save_barplots(self):
        for v, x, label, unit in [
            ['Intercept', 'r2', '결정계수', ''],
            ['Intercept', 'coef', '기저부하', 'kWh/m²'],
            ['CDD', 'coef', '냉방민감도', 'kWh/m²°C'],
            ['HDD', 'coef', '난방민감도', 'kWh/m²°C'],
        ]:
            xlabel = f'{label} [{unit}]' if unit else label
            fig, _ = self.barplot(variable=v, x=x, xlabel=xlabel)  # type: ignore[arg-type]
            fig.savefig(
                self.conf.dirs.cpr / f'plot-bar-{label}-MinR2={self.min_r2}.png'
            )

    def change_point(self, holiday: Literal['평일', '휴일']):
        fig, ax = plt.subplots()
        sns.pointplot(
            self.data.filter(pl.col('holiday') == holiday)
            .drop_nulls('change_points')
            .with_columns(pl.col('names').replace({'HDD': '난방', 'CDD': '냉방'}))
            .collect(),
            x='change_points',
            y=VAR.CATEGORY,
            hue='names',
            hue_order=['난방', '냉방'],
            palette=self.palette[::-1],
            ax=ax,
            estimator=self.estimator,
            linestyles='none',
            dodge=False,
            marker='D',
            alpha=0.8,
        )

        ax.set_xlabel('냉·난방 균형점 온도 [°C]')
        ax.set_ylabel('')
        ax.get_legend().set_title('')
        return fig, ax

    def save_change_points(self):
        for h in ['평일', '휴일']:
            fig, _ = self.change_point(h)  # type: ignore[arg-type]
            fig.savefig(self.conf.dirs.cpr / f'plot-CP-{h}-MinR2={self.min_r2}.png')
            plt.close(fig)


@app.command
def plot_cpr_coef(
    *,
    conf: Config,
    cpr_conf: CprConfig = _DEFAULT_CPR_CONF,
    min_r2: float | None = None,
):
    utils.MplTheme(palette='tol:bright').grid().apply()

    plotter = CprCoefPlotter(conf=conf, cpr_conf=cpr_conf, min_r2=min_r2)
    plotter.save_barplots()
    plotter.save_change_points()


@app.command
def cpr_aeb(*, conf: Config, cpr_conf: CprConfig = _DEFAULT_CPR_CONF):
    """All-electric building CPR 결과 요약."""
    r2 = (
        pl.scan_parquet(conf.dirs.cpr / f'{cpr_conf.name("model")}.parquet')
        .filter(pl.col('names') == 'Intercept')
        .with_columns(
            pl.col('holiday').replace_strict(
                {False: 'r2(평일)', True: 'r2(휴일)'}, return_dtype=pl.String
            )
        )
        .select(VAR.IID, 'holiday', 'r2')
        .collect()
        .pivot('holiday', index=VAR.IID, values='r2')
    )

    data_dir = conf.dirs.data
    data = (
        pl.scan_parquet(data_dir / conf.files.institution)
        .rename({'asos_code': '지역'})
        .join(
            pl.scan_parquet(data_dir / conf.files.equipment).select(
                VAR.IID, VAR.ELEC_RATIO
            ),
            on=VAR.IID,
            how='left',
        )
        .collect()
        .join(r2, on=VAR.IID, how='left')
        .sort(VAR.ELEC_RATIO, descending=True, nulls_last=True)
    )

    data.write_excel(
        conf.dirs.cpr / '전기식용량비율별 CPR 결정계수.xlsx',
        column_widths=max(50, int(1600 / data.width)),
    )

    # 전기식용량비가 1인 기관 CPR 결과 복사
    src = conf.dirs.cpr / cpr_conf.name('plot')
    dst = conf.dirs.cpr / '전기식용량비율 100% CPR 그래프'
    dst.mkdir(exist_ok=True)
    for iid in data.filter(pl.col(VAR.ELEC_RATIO) == 1).select(VAR.IID).to_series():
        try:
            file = next(src.glob(f'{iid}*.png'))
        except StopIteration:
            logger.info('case not found: {}', iid)
        else:
            shutil.copy2(file, dst)


@app.command
def analyse_anova(
    *,
    area_breaks: tuple[float, ...] = (10000, 30000),
    min_r2: float = 0.4,
    conf: Config,
):
    """연면적별 CPR 파라미터 ANOVA."""
    institutions = pl.scan_parquet(conf.dirs.data / conf.files.institution).select(
        VAR.IID, VAR.AREA, VAR.NAME
    )
    models = (
        pl.scan_parquet(list(conf.dirs.cpr.glob('model/*.parquet')))
        .filter(pl.col('r2') >= min_r2)
        .select(VAR.IID, 'holiday', 'names', 'change_points', 'coef', 'r2')
        .join(institutions, on=VAR.IID, how='left')
        .with_columns(pl.col(VAR.AREA).cut(area_breaks).alias('연면적 구간'))
        .unpivot(
            ['change_points', 'coef'],
            index=[
                VAR.IID,
                VAR.NAME,
                VAR.AREA,
                '연면적 구간',
                'holiday',
                'r2',
                'names',
            ],
        )
        .drop_nulls('value')
        .with_columns(
            pl.format('{}-{}', 'names', 'variable')
            .replace_strict({
                'Intercept-coef': '기저부하',
                'HDD-coef': '난방민감도',
                'HDD-change_points': '난방균형점온도',
                'CDD-coef': '냉방민감도',
                'CDD-change_points': '냉방균형점온도',
            })
            .alias('variable')
        )
        .drop('names')
        .collect()
    )

    anova_dfs: list[pl.DataFrame] = []
    for (holiday, variable), df in models.group_by('holiday', 'variable'):
        anova = pl.from_pandas(
            pg.anova(df.to_pandas(), dv='value', between='연면적 구간')
        ).select(
            pl.lit(holiday).alias('holiday'),
            pl.lit(variable).alias('variable'),
            pl.all(),
        )
        anova_dfs.append(anova)

    output = conf.dirs.cpr / '연면적별'
    output.mkdir(exist_ok=True)

    fn = f'연면적별 CPR 모델_min-r²={min_r2}'
    models.write_excel(output / f'{fn}.xlsx', column_widths=150)
    pl.concat(anova_dfs).sort(pl.all()).write_excel(
        output / f'{fn} ANOVA.xlsx', column_widths=150
    )


if __name__ == '__main__':
    utils.LogHandler.set()
    (
        utils.MplTheme(palette='tol:bright')
        .tick(which='both', color='.5', direction='in')
        .grid()
        .apply()
    )

    app()

    # 과거 `public_building.py`의 report_region_usage, report_hampel은 옮기지 않음
    # 재분석 시 제주도 포함할 것

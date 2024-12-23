from __future__ import annotations

import dataclasses as dc
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import cyclopts
import holidays
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns
from loguru import logger

from greenbutton import cpr, utils
from greenbutton.utils import App, Progress
from scripts.ami.public_institution import config

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from polars._typing import FrameType


class AreaError(ValueError):
    def __init__(self, area: float):
        self.area = area
        super().__init__(f'{area=}')


@dc.dataclass
class Files:
    institution: Path = Path('1.기관-주소변환.parquet')
    equipment: Path = Path('냉난방방식-전기식용량비율.parquet')
    temperature: Path = Path('temperature.parquet')


@dc.dataclass
class Config(config.Config):
    dirs: config.Dirs
    files: Files = dc.field(default_factory=Files)

    def update(self):
        super().update()
        self.files.institution = self.dirs.data / self.files.institution
        self.files.equipment = self.dirs.data / self.files.equipment
        self.files.temperature = (
            self.root.parents[1] / 'weather' / self.files.temperature
        )
        return self


ConfigParam = Annotated[Config, cyclopts.Parameter(name='*')]


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


@dc.dataclass
class Dataset:
    conf: Config
    energy: Literal['사용량', '보정사용량'] = '보정사용량'
    institutions: pl.LazyFrame = dc.field(init=False)

    def __post_init__(self):
        equipment = (
            pl.scan_parquet(self.conf.files.equipment)
            .select(VAR.IID, VAR.ELEC_RATIO)
            .with_columns()
        )
        self.institutions = (
            pl.scan_parquet(self.conf.files.institution)
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
        return (
            pl.scan_parquet(self.conf.files.temperature)
            .with_columns(pl.col('region2').replace({'제주도': '제주'}))
            .filter(pl.col('region2') == region)
            .select('datetime', pl.col('ta').alias('temperature'))
        )

    @staticmethod
    def _add_is_holiday(data: FrameType):
        years = (
            data.lazy()
            .select(pl.col('datetime').dt.year().unique())
            .collect()
            .to_series()
        )
        hd = set(holidays.country_holidays('KR', years=years).keys())
        return data.with_columns(
            is_holiday=pl.col('datetime').dt.date().is_in(hd)
            | pl.col('datetime').dt.weekday().is_in([6, 7])
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
            data = self._add_is_holiday(data)

        return inst, data


@dc.dataclass
class CprConfig:
    energy: Literal['사용량', '보정사용량'] = '보정사용량'
    interval: str = '1d'
    holiday: bool | None = False
    plot: bool = True

    min_samples: int = 4

    style: cpr.PlotStyle = dc.field(  # type: ignore[assignment]
        default_factory=lambda: {'scatter': {'s': 12, 'alpha': 0.25}}
    )

    def suffix(self):
        h = {True: '휴일', False: '평일', None: '전체'}[self.holiday]
        return f'{self.interval}_{h}_{self.energy}'


@dc.dataclass
class CprCalculator:
    conf: Config
    cpr_conf: CprConfig
    dataset: Dataset

    def file_name(self, institution: Institution):
        return f'{institution.iid}_{institution.name}_{self.cpr_conf.suffix()}'

    def _cpr(self, institution: str | Institution):
        conf = self.cpr_conf

        inst, lf = self.dataset.data(
            institution,
            interval=conf.interval,
            with_holiday=conf.holiday is not None,
        )
        if conf.holiday is not None:
            lf = lf.filter(pl.col('is_holiday') == conf.holiday)

        df = lf.collect()

        model = cpr.ChangePointRegression(
            df, temperature='temperature', energy='energy', min_samples=conf.min_samples
        ).optimize_multi_models()

        return inst, model

    def _plot(self, model: cpr.ChangePointModel, ax: Axes | None = None):
        if ax is None:
            ax = plt.gca()

        model.plot(ax=ax, style=self.cpr_conf.style)

        if ax.dataLim.y0 > 0:
            ax.set_ylim(bottom=0)

        ax.set_xlabel('기온 [℃]')
        ax.set_ylabel('전력사용량 [kWh/m²]')

        return ax

    def cpr(self, institution: str | Institution):
        inst, model = self._cpr(institution)
        name = self.file_name(inst)

        model_frame = model.model_frame.select(
            pl.lit(inst.iid).alias(VAR.IID),
            pl.lit(inst.elec_ratio).alias('elec_ratio'),
            pl.lit(self.cpr_conf.interval).alias('interval'),
            pl.lit(self.cpr_conf.holiday).alias('holiday'),
            pl.all(),
        )

        model_frame.write_parquet(self.conf.dirs.cpr / f'model/{name}.parquet')

        if self.cpr_conf.plot:
            fig, ax = plt.subplots()
            self._plot(model, ax=ax)
            ax.set_title(
                f'{inst.name} (r²={model.model_dict["r2"]:.4g})', loc='left', weight=500
            )

            fig.savefig(self.conf.dirs.cpr / f'plot/{name}.png')
            plt.close(fig)

    def batch_cpr(self, *, skip_existing: bool = True):
        for institution in self.dataset.iter_institutions():
            if (
                skip_existing
                and (
                    self.conf.dirs.cpr / f'model/{self.file_name(institution)}.parquet'
                ).exists()
            ):
                continue

            logger.info(institution)

            try:
                self.cpr(institution)
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
CprConfigParam = Annotated[CprConfig, cyclopts.Parameter(name='cpr')]


@app.command
def cpr_(
    *,
    iid: str,
    conf: ConfigParam,
    cpr_conf: CprConfigParam = _DEFAULT_CPR_CONF,
):
    """CPR 분석."""
    cc = CprCalculator(conf=conf, cpr_conf=cpr_conf, dataset=Dataset(conf=conf))
    cc.cpr(institution=iid)


@app.command
def batch_cpr(
    *,
    conf: ConfigParam,
    cpr_conf: CprConfigParam = _DEFAULT_CPR_CONF,
):
    """전체 기관 CPR 일괄 분석."""
    cc = CprCalculator(conf=conf, cpr_conf=cpr_conf, dataset=Dataset(conf=conf))
    cc.batch_cpr()
    concat_cpr(conf=conf)


@app.command
def concat_cpr(*, conf: ConfigParam):
    """기관별 CPR 분석 결과 통합."""
    src = conf.dirs.cpr / 'model'
    data = pl.concat(
        (
            pl.scan_parquet(x).with_columns(cs.by_dtype(pl.Null).cast(pl.Float64))
            for x in src.glob('*.parquet')
        ),
        how='vertical',
    ).collect()
    data.write_parquet(conf.dirs.cpr / 'model.parquet')
    data.write_excel(
        conf.dirs.cpr / 'model.xlsx', column_widths=max(50, int(1600 / data.width))
    )


@app.command
def plot_elec_r2(conf: ConfigParam):
    data = (
        pl.scan_parquet(conf.dirs.cpr / 'model.parquet')
        .filter(pl.col('names') == 'Intercept', pl.col('elec_ratio').is_not_null())
        .with_columns(
            pl.col('holiday').replace_strict(
                {True: '휴일', False: '평일'}, return_dtype=pl.String
            )
        )
        .collect()
    )

    grid = sns.lmplot(
        data,
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

    estimator: Literal['median', 'mean'] = 'median'
    min_r2: float | None = None

    # TODO max_beta [kW/m²]

    data: pl.LazyFrame = dc.field(init=False)

    def __post_init__(self):
        data = pl.scan_parquet(self.conf.dirs.cpr / 'model.parquet').with_columns(
            pl.col('holiday').replace_strict(
                {False: '평일', True: '휴일'}, return_dtype=pl.String
            )
        )

        if self.min_r2:
            data = data.filter(pl.col('r2') > self.min_r2)

        institution = (
            pl.scan_parquet(self.conf.files.institution)
            .select(VAR.IID, VAR.CATEGORY)
            .with_columns(
                pl.when(pl.col(VAR.CATEGORY).str.starts_with('국립대학병원'))
                .then(pl.lit('국립대학병원 등'))
                .otherwise(pl.col(VAR.CATEGORY))
                .alias(VAR.CATEGORY)
            )
        )
        data = data.join(institution, on=VAR.IID, how='left').sort(VAR.CATEGORY)

        self.data = data.collect().lazy()

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
            fig.savefig(self.conf.dirs.cpr / f'plot-bar-{label}.png')

    def change_point(self, holiday: Literal['평일', '휴일']):
        fig, ax = plt.subplots()
        sns.pointplot(
            self.data.filter(pl.col('holiday') == holiday)
            .drop_nulls('change_point')
            .with_columns(pl.col('names').replace({'HDD': '난방', 'CDD': '냉방'}))
            .collect(),
            x='change_point',
            y=VAR.CATEGORY,
            hue='names',
            ax=ax,
            estimator=self.estimator,
            linestyles='none',
            dodge=False,
            marker='D',
            alpha=0.9,
        )

        ax.set_xlabel('냉·난방 균형점 온도 [°C]')
        ax.set_ylabel('')
        ax.get_legend().set_title('')
        return fig, ax

    def save_change_points(self):
        for h in ['평일', '휴일']:
            fig, _ = self.change_point(h)  # type: ignore[arg-type]
            fig.savefig(self.conf.dirs.cpr / f'plot-CP-{h}.png')
            plt.close(fig)


@app.command
def plot_cpr_coef(*, conf: ConfigParam, min_r2: float | None = None):
    plotter = CprCoefPlotter(conf=conf, min_r2=min_r2)
    plotter.save_barplots()
    plotter.save_change_points()


@app.command
def cpr_aeb(*, conf: ConfigParam):
    """All-electric building CPR 결과 요약."""
    r2 = (
        pl.scan_parquet(conf.dirs.cpr / 'model.parquet')
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

    data = (
        pl.scan_parquet(conf.files.institution)
        .rename({'asos_code': '지역'})
        .join(
            pl.scan_parquet(conf.files.equipment).select(VAR.IID, VAR.ELEC_RATIO),
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
    src = conf.dirs.cpr / 'plot'
    dst = conf.dirs.cpr / '전기식용량비율 100% CPR 그래프'
    dst.mkdir(exist_ok=True)
    for iid in data.filter(pl.col(VAR.ELEC_RATIO) == 1).select(VAR.IID).to_series():
        try:
            file = next(src.glob(f'{iid}*.png'))
        except StopIteration:
            logger.info('case not found: {}', iid)

        shutil.copy2(file, dst)


if __name__ == '__main__':
    utils.LogHandler.set()
    utils.MplTheme(palette='tol:bright').grid().apply()

    app()

    # 과거 `public_building.py`의 report_region_usage, report_hampel은 옮기지 않음
    # 재분석 시 제주도 포함할 것

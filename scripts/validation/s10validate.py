from __future__ import annotations

import dataclasses as dc
import functools
from typing import TYPE_CHECKING, Literal

import polars as pl
import seaborn as sns
from loguru import logger
from matplotlib.figure import Figure

from greenbutton import cpr, utils
from greenbutton.utils import tqdmr
from scripts.validation.common import Config, app  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

Split = Literal['random', 'sequential']


@dc.dataclass
class _Validate:
    conf: Config
    split: Split = 'random'
    val_ratio: float = 0.3

    @property
    def param(self):
        return f'{self.split}-{self.val_ratio}'

    @functools.cached_property
    def output(self):
        return _Output.create(self)


@dc.dataclass
class _Output:
    eda: Path
    model: Path
    pred: Path

    @classmethod
    def create(cls, v: _Validate):
        d = v.conf.path.validation
        param = f'{v.param}'
        return cls(
            eda=d / f'{param}-00.eda',
            model=d / f'{param}-01.model',
            pred=d / f'{param}-02.pred',
        )

    def mkdir(self):
        for field in dc.fields(self):
            p = getattr(self, field.name)
            p.mkdir(exist_ok=True)


@dc.dataclass
class ValidateSingleBuilding(_Validate):
    _: dc.KW_ONLY
    index: int
    building: str

    train: pl.DataFrame
    val: pl.DataFrame

    def fit(self, *, holiday: bool):
        data = self.train.filter(pl.col('is_holiday') == holiday)
        estimator = cpr.CprEstimator(
            data, datetime='date', conf=cpr.CprConfig(min_r2=0.001)
        )
        return estimator.fit()

    def write_model(self, analysis: cpr.CprAnalysis, name: str):
        analysis.model_frame.write_parquet(self.output.model / f'{name}.parquet')

        fig = Figure()
        ax = fig.subplots()
        analysis.plot(ax=ax)
        ax.set_ylim(0)
        fig.savefig(self.output.model / f'{name}.png')

    def _pred(self, workday: cpr.CprAnalysis, holiday: cpr.CprAnalysis):
        f = pl.col('is_holiday')
        for analysis, is_holiday in zip([workday, holiday], [False, True], strict=True):
            yield analysis.predict(self.train.filter(f == is_holiday))
            yield analysis.predict(self.val.filter(f == is_holiday))

    def __call__(self):
        try:
            workday = self.fit(holiday=False)
        except cpr.CprError:
            logger.warning(f'분석 오류 (workday, {self.building=})')
            raise

        try:
            holiday = self.fit(holiday=True)
        except cpr.CprError:
            logger.warning(f'분석 오류 (holiday, {self.building=})')
            raise

        name = f'{self.index:04d}.{self.building}'
        self.write_model(workday, f'{name} workday')
        self.write_model(holiday, f'{name} holiday')

        pred = pl.concat(self._pred(workday, holiday))
        pred.write_parquet(self.output.pred / f'{name}.parquet')

        return pred


@app.default
@dc.dataclass
class Validate(_Validate):
    """CPM 모델 검증."""

    _: dc.KW_ONLY
    fit: bool = True
    seed: int = 42

    def read_data(self):
        d = self.conf.path.data

        sources = list(d.glob('00.*.parquet'))
        cache = d / '99.data.parquet'

        if (
            cache.exists()  # fmt
            and (max(x.stat().st_mtime for x in sources) < cache.stat().st_mtime)
        ):
            return pl.scan_parquet(cache)

        lfs = (pl.scan_parquet(x) for x in sources)
        data = pl.concat(lfs, how='diagonal').collect()
        data.write_parquet(cache)

        return data.lazy()

    def _timeline(
        self,
        index: int,
        building: str,
        data: pl.DataFrame,
        *,
        split: bool = False,
    ):
        fig = Figure(figsize=(16 / 2.54, 7 / 2.54))
        axes = fig.subplots(2 if split else 1, 1)

        if split:
            for ax, day in zip(axes, ('workday', 'holiday'), strict=True):
                utils.mpl.lineplot_break_nans(
                    (data)
                    .filter(pl.col('is_holiday') == (day == 'holiday'))
                    .upsample('date', every='1d'),
                    x='date',
                    y='energy',
                    ax=ax,
                    alpha=0.5,
                    lw=1,
                )
                ax.set_ylim(0)
                ax.set_xlabel('')
        else:
            utils.mpl.lineplot_break_nans(
                data.sort('date').upsample('date', every='1d'),
                x='date',
                y='energy',
                ax=axes,
                alpha=0.75,
            )
            axes.set_ylim(0)
            axes.set_xlabel('Date')
            axes.set_ylabel('Energy [kWh]')

        fig.savefig(self.output.eda / f'{index:04d}.{building} timeline.png')

    def _split(self, data: pl.DataFrame):
        data = data.with_row_index()
        train_count = round(data.height * (1 - self.val_ratio))

        match self.split:
            case 'random':
                index = data.select(
                    pl.col('index').sample(train_count, seed=self.seed),
                    pl.lit('train').alias('dataset'),
                )
                data = (
                    (data)
                    .join(index, on='index', how='left', validate='1:1')
                    .with_columns(pl.col('dataset').fill_null('val'))
                )
            case 'sequential':
                data = data.with_columns(
                    pl.when(pl.col('index') < train_count)
                    .then(pl.lit('train'))
                    .otherwise(pl.lit('val'))
                    .alias('dataset')
                )
            case _:
                raise ValueError(self.split)

        return data

    def _validate(self, index: int, building: str, data: pl.DataFrame):
        data = self._split(data)

        validate = ValidateSingleBuilding(
            conf=self.conf,
            split=self.split,
            val_ratio=self.val_ratio,
            index=index,
            building=building,
            train=data.filter(pl.col('dataset') == 'train'),
            val=data.filter(pl.col('dataset') == 'val'),
        )
        validate()

    def _fit(self):
        self.output.mkdir()

        data = (
            self.read_data()
            .select('building', 'date', 'is_holiday', 'temperature', 'energy')
            .collect()
        )
        buildings = data['building'].unique().sort().to_list()

        for idx, building in enumerate(tqdmr(buildings)):
            logger.info(f'{idx:04d}. {building=}')
            df = data.filter(pl.col('building') == building)
            self._timeline(index=idx, building=building, data=df)
            self._validate(index=idx, building=building, data=df)

    def _eval(self):
        if not (files := list(self.output.pred.glob('*.parquet'))):
            raise FileNotFoundError(self.output.pred / '*.parquet')

        model = (
            pl.scan_parquet(self.output.model / '*.parquet', include_file_paths='path')
            .filter(pl.col('names') == 'Intercept')
            .select(
                pl.col('path').str.extract_groups(
                    r'(?P<index>\d+)\.(?P<building>[\w\-]+) '
                    r'(?P<day>workday|holiday)'
                ),
                'r2',
            )
            .unnest('path')
            .with_columns(pl.format('{}_r2', 'day').alias('day'))
            .drop('index')
            .collect()
            .pivot('day', index='building', values='r2', sort_columns=True)
        )

        accuracy = (
            pl.read_parquet(files)
            .with_columns(error=pl.col('Ep') - pl.col('energy'))
            .group_by(['building', 'dataset'])
            .agg(
                pl.mean('energy'),
                (pl.col('error') - pl.col('energy') / pl.col('energy'))
                .abs()
                .mean()
                .alias('MAPE'),
                pl.col('error').pow(2).mean().sqrt().alias('RMSE'),
            )
            .with_columns(pl.col('RMSE').truediv('energy').alias('CV(RMSE)'))
            .join(model, on='building', how='full', validate='m:1', coalesce=True)
            .sort(pl.all())
        )

        output = self.conf.path.validation / f'{self.param}-error.xlsx'
        accuracy.write_excel(output, column_widths=120)

        fig = Figure()
        axes = fig.subplots(1, 2, sharex=True, sharey=True)

        ax: Axes
        for ax, day in zip(axes, ('workday', 'holiday'), strict=True):
            sns.scatterplot(
                accuracy,
                x=f'{day}_r2',
                y='CV(RMSE)',
                hue='dataset',
                ax=ax,
                alpha=0.8,
                legend=day == 'holiday',
            )
            utils.mpl.equal_scale(ax)
            ax.set_title(day.title(), loc='left', weight=500)
            ax.set_xlabel('r²')
            ax.axhline(0.25, ls='--', color='gray', alpha=0.5)
            ax.axhline(0.3, ls='--', color='gray', alpha=0.5)

            if legend := ax.get_legend():
                legend.set_title('')

        fig.savefig(output.with_suffix('.png'))

    def __call__(self):
        if self.fit:
            self._fit()

        self._eval()


if __name__ == '__main__':
    utils.terminal.LogHandler.set()
    utils.mpl.MplTheme('paper').grid().apply()
    utils.mpl.MplConciseDate().apply()
    app()

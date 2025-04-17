from __future__ import annotations

import dataclasses as dc
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import fastexcel
import matplotlib.pyplot as plt
import pint
import polars as pl
import rich
import seaborn as sns
from cmap import Colormap
from loguru import logger
from whenever import LocalDateTime

import scripts.exp.experiment as exp
from greenbutton import utils
from greenbutton.utils import App

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dc.dataclass
class _PublicInstitutionCpr:
    min_r2: float = 0.4
    max_anomaly_threshold: float = 4
    categories: Literal['all', 'office', 'public'] = 'office'

    models_source: str = 'PublicInstitutionCPR.parquet'

    _dirs: exp.Dirs = dc.field(init=False)

    def scan_models(self):
        lf = pl.scan_parquet(self._dirs.database / self.models_source).filter(
            pl.col('r2') >= self.min_r2,
        )

        match self.categories:
            case 'office':
                lf = lf.filter(
                    pl.col('category')
                    .is_in(['국립대학병원 등', '국립대학 및 공립대학'])
                    .not_()
                )
            case 'public':
                lf = lf.filter(pl.col('category').str.starts_with('공공기관'))

        return lf


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'keit'

    pi_cpr: _PublicInstitutionCpr = dc.field(default_factory=_PublicInstitutionCpr)

    def __post_init__(self):
        self.pi_cpr._dirs = self.dirs  # noqa: SLF001


app = App(
    config=cyclopts.config.Toml('config/.experiment.toml', use_commands_as_keys=False),
)


@app.command
def init(*, conf: Config):
    conf.dirs.mkdir()


def _read_heat_excel(source: str | bytes | Path, sheet: int | str = 0):
    reader = fastexcel.read_excel(source)
    raw = reader.load_sheet(sheet, header_row=None, dtypes='string').to_polars()

    # 변수명으로 열 이름 지정 (모든 행, 표에서 열 구성 같다고 가정)
    row1 = (
        pl.Series('first', raw.row(1))
        .fill_null(strategy='forward')
        .replace({'구분': 'day'})
    )
    row2 = [x or '' for x in raw.row(2)]
    columns = [f'{v}-{u}'.strip('-') for u, v in zip(row1, row2, strict=True)]
    columns = [f'{i}_{c}' for i, c in enumerate(columns)]
    raw.columns = columns

    data = (
        raw.with_row_index('row')
        .unpivot(index='row')
        .with_columns(
            pl.col('variable').str.extract_groups(r'(?<col>\d+)_(?<variable>.*)'),
            pl.col('value')
            .str.extract_groups(r'(?<year>\d+)년\W?(?<month>\d+)월')
            .alias('ym'),
        )
        .unnest('variable', 'ym')
        .with_columns(
            pl.col('col').cast(pl.UInt64),
            pl.col('year').cast(pl.UInt16),
            pl.col('month').cast(pl.UInt8),
        )
    )

    # --------------------------------------------------------------------------
    # 각 데이터에 year, month 할당
    ym = (
        data.select('row', 'col', 'year', 'month')
        .drop_nulls(['year', 'month'])
        .sort('row', 'col')
        .with_columns(
            pl.col('row').rank('dense').alias('row_rank'),
            pl.col('col').rank('dense').alias('col_rank'),
        )
    )

    # 제목(e.g. 난방적산열량계(Gcal),사용량(ton)  2016년 1월)이 포함된 행, 열
    title_rows = ym['row'].unique().sort().to_list()
    title_cols = ym['col'].unique().sort().to_list()

    def rank(expr: pl.Expr, indices: list[int]):
        return expr.cut(
            indices,
            left_closed=True,
            labels=[str(x) for x in range(len(indices) + 1)],
        ).cast(pl.UInt32)

    data = (
        data.with_columns(
            rank(pl.col('row'), title_rows).alias('row_rank'),
            rank(pl.col('col'), title_cols).alias('col_rank'),
        )
        .drop('year', 'month')
        .join(
            ym.select('row_rank', 'col_rank', 'year', 'month'),
            on=['row_rank', 'col_rank'],
            how='left',
        )
    )

    # --------------------------------------------------------------------------
    # day 할당
    row_day = (
        data.filter(pl.col('variable') == 'day')
        .rename({'value': 'day'})
        .select('year', 'month', 'row', pl.col('day').cast(pl.UInt8, strict=False))
        .drop_nulls('day')
        .unique()
        .sort('year', 'month', 'row')
    )

    return (
        data.filter(pl.col('variable') != 'day')
        .with_columns(pl.col('value').cast(pl.Float64, strict=False))
        .drop_nulls('value')
        .join(row_day, on=['year', 'month', 'row'], how='left', validate='m:1')
        .with_columns(
            pl.format('{}-{}-{}', 'year', 'month', 'day')
            .str.to_date(strict=False)
            .alias('date')
        )
        .drop_nulls('date')
        .pivot('variable', index='date', values='value', sort_columns=True)
        .sort('date')
    )


app.command(App('db', help='사용량 엑셀/AMI 데이터'))


@app['db'].command
def db_parse_heat(
    *,
    file: str = 'KEIT 열에너지 사용량(지역난방).xls',
    max_date: str = '2025-02-28',
    conf: Config,
):
    d = conf.dirs.database
    path = d / file
    sheets = fastexcel.read_excel(path).sheet_names

    md = LocalDateTime.strptime(max_date, '%Y-%m-%d').date().py_date()

    for sheet in sheets:
        data = _read_heat_excel(path, sheet=sheet).filter(pl.col('date') <= md)
        s = sheet.strip()
        data.write_parquet(d / f'0000.{s}.parquet')
        data.write_excel(d / f'0000.{s}.xlsx', column_widths=150)


@app['db'].command
def db_parse_elec(*, file: str = 'KEIT 전력사용량.xlsx', conf: Config):
    # NOTE 2023년까지는 순사용량(한전사용량)이 '전체사용량'으로 기입되어 있음
    variables = ['사용량', '태양광발전량', '총사용량', 'dummy']
    columns = ['day', *[f'{m}월 {v}' for m in range(1, 13) for v in variables]]
    raw = (
        fastexcel.read_excel(conf.dirs.database / file)
        .load_sheet(
            0, header_row=None, column_names=columns, skip_rows=1, dtypes='string'
        )
        .to_polars()
    )
    rich.print(raw)

    data = (
        raw.with_columns(
            year=pl.col('day').str.extract(r'(\d+)년도').forward_fill().cast(pl.UInt16)
        )
        .with_columns(pl.col('day').str.extract(r'(\d+)일').cast(pl.UInt8))
        .drop_nulls('day')
        .filter(pl.any_horizontal(pl.all().exclude('day', 'year').is_not_null()))
        .unpivot(index=['year', 'day'])
        .with_columns(
            pl.col('variable').str.extract_groups(r'^(?<month>\d+)월 (?<variable>.*)$'),
            pl.col('value').cast(pl.Float64),
        )
        .unnest('variable')
        .filter(pl.col('variable') != 'dummy')
        .with_columns(
            pl.format('{}-{}-{}', 'year', 'month', 'day')
            .str.to_date(strict=False)
            .alias('date')
        )
        .drop_nulls('date')
        .pivot('variable', index='date', values='value', sort_columns=True)
        .filter(pl.any_horizontal(pl.all().exclude('date').is_not_null()))
        .sort('date')
    )

    rich.print(data)
    data.write_parquet(conf.dirs.database / '0000.전력.parquet')
    data.write_excel(conf.dirs.database / '0000.전력.xlsx', column_widths=150)


@app['db'].command
def db_ami(
    iid: str = 'DB_B7AE8782-9689-8EED-E050-007F01001D51',
    *,
    ami_dir: str = 'AMI/PublicInstitution/0001.data',
    conf: Config,
):
    d = conf.root.parent / ami_dir
    d.stat()

    data = (
        pl.scan_parquet(list(d.glob('AMI*.parquet')))
        .filter(pl.col('기관ID') == iid)
        .sort('datetime')
        .collect()
    )

    rich.print(data)

    if data.select(pl.col('사용량').eq(pl.col('보정사용량')).all()).item():
        logger.info('KEIT AMI 사용량과 보정사용량 일치')
        data = data.drop('보정사용량')
    else:
        logger.info('KEIT AMI 사용량과 보정사용량 불일치')

    data.write_parquet(conf.dirs.database / '0000.AMI.parquet')


@dc.dataclass
class _EnergyCompare:
    conf: Config

    heat: pl.DataFrame = dc.field(init=False)
    elec: pl.DataFrame = dc.field(init=False)
    ami: pl.DataFrame = dc.field(init=False)

    def __post_init__(self):
        d = self.conf.dirs.database

        self.heat = (
            pl.scan_parquet(
                list(d.glob('0000.*적산열량계.parquet')),
                include_file_paths='path',
            )
            .with_columns(
                mode=pl.col('path')
                .str.extract(r'.*\\(.*)\.parquet')
                .str.strip_prefix('0000.')
            )
            .drop('path')
            .select('date', 'mode', '사용량-Gcal', '사용량-톤')
            .unpivot(index=['date', 'mode'], variable_name='unit')
            .with_columns(
                pl.col('unit').str.extract(r'사용량\-(.*)$').replace({'톤': 'ton'})
            )
            .with_columns(
                pl.format(
                    '{}({})',
                    pl.col('mode').str.extract('([냉난]방)'),
                    pl.col('unit'),
                ).alias('variable')
            )
            .collect()
        )
        self.elec = (
            pl.read_parquet(d / '0000.전력.parquet')
            .rename({'전체사용량': '전력 순사용량', '태양광발전량': '태양광 발전량'})
            .drop('총사용량')
        )
        self.ami = (
            pl.scan_parquet(d / '0000.AMI.parquet')
            .group_by_dynamic('datetime', every='1d')
            .agg(pl.sum('사용량'))
            .sort('datetime')
            .select(
                pl.col('datetime').dt.date().alias('date'),
                pl.col('사용량').alias('AMI'),
            )
            .collect()
        )

    def line(self):
        data = pl.concat(
            [
                self.heat.filter(pl.col('unit') == 'Gcal')
                .select(
                    'date',
                    pl.lit('열에너지').alias('variable'),
                    pl.col('variable').alias('hue'),
                    'value',
                )
                .sort(pl.col('hue').replace_strict({'냉방(Gcal)': 0, '난방(Gcal)': 1})),
                self.elec.unpivot(index='date', variable_name='hue').with_columns(
                    pl.lit('전력').alias('variable')
                ),
                self.ami.rename({'AMI': 'value'}).with_columns(
                    pl.lit('AMI').alias('variable'),
                    pl.lit('AMI').alias('hue'),
                ),
            ],
            how='diagonal',
        )

        grid = (
            sns.FacetGrid(
                data, row='variable', sharey=False, aspect=3 * 16 / 9, height=2
            )
            .map_dataframe(sns.lineplot, x='date', y='value', hue='hue', alpha=0.8)
            .set_xlabels('')
            .set_titles('')
            .set_titles('{row_name}', loc='left', weight=500)
        )

        for ax, ylabel in zip(
            grid.axes_dict.values(),
            ['열에너지 [Gcal]', '전력 [kWh]', 'AMI 전력량 [kWh]'],
            strict=True,
        ):
            ax.legend()
            ax.set_ylabel(ylabel)

        return grid

    def _unit_conversion(self, unit: str = 'kWh'):
        ur = pint.UnitRegistry()

        def convert(src: str, dst: str):
            return float(ur.Quantity(1, src).to(dst).m)

        heat = self.heat.filter(pl.col('unit') == 'Gcal').select(
            'date',
            pl.col('variable').str.replace(r'(.*)\(Gcal\)', '열에너지 사용량 ($1)'),
            pl.col('value') * convert('Gcal', unit),
        )
        elec = self.elec.unpivot(index='date').with_columns(
            pl.col('value') * convert('kWh', unit)
        )
        return pl.concat([heat, elec]).drop_nulls('value').sort('date', 'variable')

    def line_unit(self, unit: str = 'kWh', ax: Axes | None = None):
        data = self._unit_conversion(unit=unit)
        grouped = (
            data.lazy()
            .group_by_dynamic('date', every='1w', group_by='variable')
            .agg(pl.sum('value'))
            .collect()
        )

        if ax is None:
            ax = plt.gca()

        sns.lineplot(
            grouped,
            x='date',
            y='value',
            hue='variable',
            ax=ax,
            hue_order=[
                '열에너지 사용량 (냉방)',
                '열에너지 사용량 (난방)',
                '전력 순사용량',
                '태양광 발전량',
            ],
            alpha=0.8,
        )
        ax.set_xmargin(0.02)
        ax.set_xlabel('')
        ax.set_ylim(0, 42000)
        ax.set_ylabel(f'에너지 [{unit}]')
        legend = ax.get_legend()
        legend.set_title('')
        for line in legend.get_lines():
            line.set_linewidth(2)

        return ax, data

    def line_unit2(self, unit: str = 'kWh'):
        data = self._unit_conversion(unit=unit)

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
        variables = [
            ['열에너지 사용량 (냉방)', '열에너지 사용량 (난방)'],
            ['전력 순사용량', '태양광 발전량'],
        ]
        palette = Colormap('tol:bright-alt')([0, 1, 2, 3]).reshape([2, 2, 4])

        ax: Axes
        for idx, ax in enumerate(axes):
            vars_ = variables[idx]
            sns.lineplot(
                data.filter(pl.col('variable').is_in(vars_)),
                x='date',
                y='value',
                hue='variable',
                hue_order=vars_,
                palette=list(palette[idx]),
                ax=ax,
                alpha=0.75,
                lw=0.5,
            )
            legend = ax.get_legend()
            legend.set_title('')
            for line in legend.get_lines():
                line.set_linewidth(2)

            ax.set_xmargin(0.02)
            ax.set_xlabel('')
            ax.set_ylabel(f'에너지 [{unit}]')

        return fig

    def pair(self):
        data = (
            self.heat.group_by('date', 'unit')
            .agg(pl.sum('value'))
            .filter(pl.col('value') > 0)
            .with_columns(pl.format('열에너지 ({})', 'unit').alias('variable'))
            .pivot('variable', index='date', values='value')
            .join(self.elec, on='date', how='full', coalesce=True)
            .join(self.ami, on='date', how='full', coalesce=True)
        )
        rich.print('pair data', data)

        return (
            sns.PairGrid(data.drop('date'), height=1.5)
            .map_lower(sns.scatterplot, alpha=0.25)
            .map_upper(sns.kdeplot)
            .map_diag(sns.histplot)
        )


@app['db'].command
def db_plot_compare(*, unit: str = 'kWh', pair: bool = False, conf: Config):
    utils.MplTheme(0.8, palette='tol:bright').tick('x', 'both').grid().apply()
    compare = _EnergyCompare(conf=conf)

    grid = compare.line()
    grid.savefig(conf.dirs.analysis / '0000.사용량 비교 line.png')
    plt.close(grid.figure)

    fig, ax = plt.subplots()
    _, data = compare.line_unit(unit=unit, ax=ax)
    fig.savefig(conf.dirs.analysis / f'0000.사용량 비교 line ({unit}).png')
    data.write_excel(
        conf.dirs.analysis / f'0000.사용량 비교 line ({unit}).xlsx', column_widths=200
    )
    plt.close(fig)

    fig = compare.line_unit2(unit=unit)
    fig.savefig(conf.dirs.analysis / f'0000.사용량 비교 line ({unit})-split.png')
    plt.close(fig)

    if pair:
        grid = compare.pair()
        grid.savefig(conf.dirs.analysis / '0000.사용량 비교 pair.png')


@app['db'].command
def db_plot_compare_elec(*, scale: float = 0.6, conf: Config):
    """AMI와 자체 기록 전력 사용량 비교."""
    elec = (
        pl.scan_parquet(conf.dirs.database / '0000.전력.parquet')
        .select(
            'date',
            pl.col('전체사용량').alias('순사용량'),
            (pl.col('전체사용량') + pl.col('태양광발전량')).alias('총사용량'),
        )
        .collect()
    )
    ami = (
        pl.scan_parquet(conf.dirs.database / '0000.AMI.parquet')
        .with_columns(pl.col('datetime').dt.date().alias('date'))
        .group_by('date')
        .agg(pl.sum('사용량').alias('AMI'))
        .collect()
    )
    data = elec.join(ami, on='date')
    rich.print(data)

    avg_ami = data.select(pl.mean('AMI')).item()
    vrange = (
        data.unpivot(index='date')
        .select(vmin=pl.min('value'), vmax=pl.max('value'))
        .to_numpy()
        .ravel()
    )
    datalim = [vrange, vrange[::-1]]
    agg = (
        data.unpivot(index=['date', 'AMI'])
        .group_by('variable')
        .agg(
            error=(pl.col('value') - pl.col('AMI')).pow(2).mean().sqrt(),
            r=pl.corr('value', 'AMI'),
        )
    )

    utils.MplTheme(scale, palette='tol:bright').grid().apply()
    fig, axes = plt.subplots(1, 2)

    ax: Axes
    for y, ax in zip(['순사용량', '총사용량'], axes, strict=True):
        sns.scatterplot(data, x='AMI', y=y, ax=ax, alpha=0.25, s=8)
        ax.set_xlabel('AMI 일간 사용량 [kWh]')
        ax.set_ylabel(f'일간 {y} [kWh]')

        ax.dataLim.update_from_data_xy(datalim, ignore=False)
        ax.axline((0, 0), slope=1, c='k', alpha=0.1)
        ax.set_box_aspect(1)

        r = agg.filter(pl.col('variable') == y)['r'].item()
        e = agg.filter(pl.col('variable') == y)['error'].item()
        ax.text(
            0.02,
            0.98,
            f'r = {r:.3f}\nRMSE = {e:.1f} kWh\nCV(RMSE) = {e / avg_ami:.2%}',
            va='top',
            weight=500,
            transform=ax.transAxes,
        )

    fig.savefig(conf.dirs.analysis / '0001.전력 사용량 비교.png')


@app['db'].command
def db_plot_heat(*, conf: Config):
    data = (
        pl.read_parquet(
            list(conf.dirs.database.glob('0000.*적산열량계.parquet')),
            include_file_paths='path',
        )
        .with_columns(
            mode=pl.col('path')
            .str.extract(r'.*\\(.*)\.parquet')
            .str.strip_prefix('0000.')
        )
        .drop('path')
    )
    rich.print(data)

    unpivot = data.unpivot(['사용량-Gcal', '사용량-톤'], index=['date', 'mode'])

    utils.MplTheme(0.8).grid().tick('x', 'both').apply()
    grid = (
        sns.relplot(
            unpivot,
            x='date',
            y='value',
            hue='mode',
            hue_order=['냉방적산열량계', '난방적산열량계'],
            row='variable',
            kind='line',
            facet_kws={'sharey': False},
            height=2.5,
            aspect=2 * 16 / 9,
            alpha=0.8,
        )
        .set_xlabels('')
        .set_titles('{row_name}')
    )

    grid.savefig(conf.dirs.analysis / '0000.냉난방 적산열량.png')


@app['db'].command
def db_plot_ami(*, conf: Config):
    data = (
        pl.scan_parquet(conf.dirs.database / '0000.AMI.parquet')
        .group_by_dynamic('datetime', every='1d')
        .agg(pl.sum('사용량'))
        .collect()
    )

    fig, ax = plt.subplots()
    sns.lineplot(
        data.unpivot(index='datetime'),
        x='datetime',
        y='value',
        hue='variable',
        ax=ax,
        alpha=0.8,
    )
    fig.savefig(conf.dirs.analysis / '0000.AMI.png')


if __name__ == '__main__':
    utils.MplTheme().grid(lw=0.75, alpha=0.5).apply()
    utils.MplConciseDate().apply()
    utils.LogHandler.set()

    app()

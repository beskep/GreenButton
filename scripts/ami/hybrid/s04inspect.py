import dataclasses as dc
import functools
from pathlib import Path  # noqa: TC003

import cyclopts
import polars as pl
import polars.selectors as cs
import structlog

from greenbutton.utils.cli import App

logger = structlog.stdlib.get_logger()
app = App(
    config=cyclopts.config.Toml(
        'config/.ami.toml', root_keys='hybrid', use_commands_as_keys=False
    )
)


@app.command
@dc.dataclass
class CompareModels:
    root: Path

    _: dc.KW_ONLY
    sigfig: int = 4

    poor: tuple[str, ...] = ('0011.공공.김해시도시개발공사',)
    models: tuple[str, ...] = (
        'elec.Te',
        'total.Te',
        'total.Te+Pv',
        'total.Te+I',
        'total.Te+Pv+I',
        'total.Te+Mon+Fri',
        'total.Te+Mon+Fri+Pv+I',
    )

    @functools.cached_property
    def data(self):
        models = {v: i for i, v in enumerate(self.models)}
        return (
            pl
            .scan_parquet(self.root / '03.CPM/CPM.parquet')
            .with_columns(
                pl
                .concat_str('endog', 'exog', separator='.')
                .replace_strict(models, default=-1)
                .alias('index.model'),
                pl.col('n.heating').truediv(pl.col('n.obs')).alias('ratio.heating'),
                pl.col('n.cooling').truediv(pl.col('n.obs')).alias('ratio.cooling'),
            )
            .collect()
        )

    @staticmethod
    def reorder(df: pl.DataFrame):
        head = (
            'stat',
            'bldg.type',
            'index.model',
            'endog',
            'exog',
            'r2',
            'r2adj',
            'AIC',
            'BIC',
            'Th',
            'Tc',
            'ratio.heating',
            'ratio.cooling',
        )
        cols = df.columns
        cols = [*(x for x in head if x in cols), *(x for x in cols if x not in head)]
        return df.select(cols)

    def agg(self, *, sigfig: int = 4, drop_poor: bool = False):
        data = (
            self.data.filter(pl.col('bldg.case').is_in(self.poor).not_())
            if drop_poor
            else self.data
        )
        data = pl.concat([data, data.with_columns(pl.lit('bldg').alias('bldg.type'))])

        group = ('bldg.type', 'index.model', 'endog', 'exog')
        v = pl.col('value')

        stat = (
            data
            .unpivot(cs.float(), index=group)
            .group_by([*group, 'variable'])
            .agg(
                v.mean().alias('mean'),
                v.median().alias('median'),
                v.std().alias('std'),
                (v - v.median()).abs().median().alias('MAD'),
            )
            .unpivot(index=[*group, 'variable'], variable_name='stat')
            .with_columns(v.round_sig_figs(sigfig))
            .pivot(
                'variable',
                index=['stat', *group],
                values='value',
                sort_columns=True,
            )
            .sort(pl.all())
        )

        table = (
            data
            .drop('p-value')
            .unpivot(cs.float(), index=group)
            .group_by([*group, 'variable'])
            .agg(
                pl.concat_str(
                    v.mean().round_sig_figs(self.sigfig),
                    v.std().round_sig_figs(self.sigfig),
                    separator='±',
                ).alias('param'),
                pl.format(
                    '{}({})',
                    v.median().round_sig_figs(self.sigfig),
                    (v - v.median()).abs().median().round_sig_figs(self.sigfig),
                ).alias('non-param'),
            )
            .unpivot(index=[*group, 'variable'], variable_name='stat')
            .pivot(
                'variable',
                index=['stat', *group],
                values='value',
                sort_columns=True,
            )
            .sort(pl.all())
        )

        return self.reorder(stat), self.reorder(table)

    def __call__(self):
        output = self.root / '04.inspect'
        output.mkdir(exist_ok=True)

        for drop in (False, True):
            stat, table = self.agg(drop_poor=drop)

            suffix = f'{".drop" if drop else ""}.sigfig={self.sigfig}'
            stat.write_csv(output / f'01.stat{suffix}.csv', include_bom=True)
            table.write_csv(output / f'01.table{suffix}.csv', include_bom=True)


if __name__ == '__main__':
    app()

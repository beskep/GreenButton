"""eCPM 분석을 위한 실증지 데이터 통합.

- 2026-03-25 한전파주, 한국에너지공단, 산기평 세 건물 우선 통합
"""

import dataclasses as dc
from pathlib import Path  # noqa: TC003
from typing import ClassVar, Literal

import cyclopts
import polars as pl
import structlog

import scripts.exp.experiment as exp
from greenbutton.utils.cli import App

Delta = Literal['day', 'hour']

logger = structlog.stdlib.get_logger()
app = App(
    config=[
        cyclopts.config.Toml(f'config/{x}.toml', use_commands_as_keys=False)
        for x in ['.experiment', 'experiment']
    ],
)


@cyclopts.Parameter(name='*')
@dc.dataclass
class Config(exp.BaseConfig):
    BUILDING: ClassVar[str] = 'integration'

    def bldg_dirs(self, bldg: str):
        return exp.Dirs(root=self.root / self.buildings[bldg])


@app.command
@dc.dataclass(frozen=True)
class Kepco:
    conf: Config

    def get_db_data(self, delta: Delta):
        d = self.conf.bldg_dirs('kepco_paju').database / '0002.binary'

        src = [
            *d.glob(f'**/*T_BELO_FACILITY_{delta.upper()}*.parquet'),
            *d.glob(f'**/*T_BELO_ELEC_{delta.upper()}*.parquet'),
        ]
        lfs = [
            pl
            .scan_parquet(x)
            .with_columns(pl.col('tagValue').cast(pl.Float64))
            .select(
                pl.col('updateDate').alias('datetime'),
                pl.col('[tagName]').alias('variable'),
                pl.col('tagValue').cast(pl.Float64).alias('value'),
            )
            for x in src
        ]

        data = (
            pl
            .concat(lfs)
            .filter(
                pl.col('variable').is_in([
                    '실외온습도계.외기온도',
                    '실외온습도계.실내온도',
                    '전기.전체전력량',
                ])
            )
            .with_columns(
                pl.col('variable').replace_strict({
                    '실외온습도계.외기온도': 'temp_external',
                    '실외온습도계.실내온도': 'temp_internal',
                    '전기.전체전력량': 'electricity_kWh',
                })
            )
            .unique(['datetime', 'variable'])
            .collect()
        )
        return (
            (data)
            .pivot('variable', index='datetime', values='value', sort_columns=True)
            .sort('datetime')
        )

    def __call__(self):
        self.conf.dirs.mkdir()

        for d in ('hour', 'day'):
            logger.info('delta=%s', d)

            data = self.get_db_data(d)
            path = self.conf.dirs.analysis / f'01.KEPCO-BEMS-{d}.parquet'
            data.write_parquet(path)
            data.write_excel(path.with_suffix('.xlsx'), column_widths=150)


@app.command
@dc.dataclass(frozen=True)
class Kea:
    conf: Config
    iid: str = 'DB_B7AE8782-40AF-8EED-E050-007F01001D51'

    def energy(self):
        return (
            (pl)
            .scan_parquet(self.conf.dirs.database / f'AMI_{self.iid}.parquet')
            .select('datetime', pl.col('사용량').alias('electicity_kWh'))
        )

    def environment(self):
        d = self.conf.bldg_dirs('kea').database / '03.parsed'
        files = [*d.glob('*실내온도.parquet'), *d.glob('*실내습도.parquet')]
        return (
            pl
            .scan_parquet(files, include_file_paths='path')
            .select(
                'datetime',
                pl
                .col('path')
                .str.extract(r'.*?_(.* 실내[온습]도)\.parquet')
                .alias('variable'),
                pl.col('parsed_value').alias('value'),
            )
            .with_columns(
                pl.col('variable').str.extract_groups(
                    r'(?<location>.*?)\s+(?<variable>실내[온습]도)'
                )
            )
            .unnest('variable')
        )

    def weather(self):
        paths = list(self.conf.dirs.database.glob('ASOS_울산/*.csv'))
        return (
            pl
            .concat(
                pl.read_csv(x, encoding='korean', infer_schema=False) for x in paths
            )
            .select(
                pl.col('일시').str.to_datetime().alias('datetime'),
                pl.col('기온(°C)').cast(pl.Float64).alias('temp_external'),
            )
            .sort('datetime')
        )

    def __call__(self):
        d = self.conf.dirs.analysis

        environment = self.environment()
        raw = (
            environment
            .sort('datetime')
            .group_by_dynamic('datetime', every='1h', group_by=['location', 'variable'])
            .agg(pl.median('value'))
            .collect()
        )
        raw.write_parquet(d / '02.KEA-BEMS-raw-hourly.parquet')

        energy = self.energy().collect()
        weather = self.weather()

        for delta in ('hour', 'day'):
            logger.info('delta=%s', d)

            if delta == 'hour':
                e = energy
                w = weather
            else:
                e = (
                    (energy)
                    .group_by_dynamic('datetime', every='1d')
                    .agg(pl.all().mean())
                )
                w = (
                    (weather)
                    .group_by_dynamic('datetime', every='1d')
                    .agg(pl.mean('temp_external'))
                )

            t = (
                environment
                .filter(pl.col('variable') == '실내온도')
                .sort(pl.col('datetime'))
                .group_by_dynamic(
                    'datetime',
                    every={'day': '1d', 'hour': '1h'}[delta],
                )
                .agg(pl.col('value').median().alias('temp_internal'))
                .collect()
            )

            data = (
                e
                .join(t, on='datetime', how='full', coalesce=True)
                .join(w, on='datetime', how='left', coalesce=True)
                .sort('datetime')
            )
            data.write_parquet(d / f'02.KEA-BEMS-{delta}.parquet')
            data.write_excel(d / f'02.KEA-BEMS-{delta}.xlsx', column_widths=150)


@app.command
@dc.dataclass(frozen=True)
class Keit:
    conf: Config

    def temperature(self):
        return (
            pl
            .read_csv(
                self.conf.dirs.database / 'OBS_ASOS_DD_대구_143.csv',
                encoding='korean',
                infer_schema=False,
            )
            .select(
                pl.col('일시').str.to_date().alias('date'),
                pl.col('평균기온(°C)').cast(pl.Float64).alias('temp_external'),
            )
            .sort('date')
        )

    def _heat(self, t: Literal['cooling', 'heating'], /):
        kr = {'cooling': '냉방', 'heating': '난방'}[t]
        return (
            pl
            .scan_parquet(self.conf.dirs.database / f'KEIT_{kr}적산열량계.parquet')
            .select(
                'date',
                pl.col('금일지침-Gcal').alias(f'{t}_Gcal'),
                pl.col('금일지침-톤').alias(f'{t}_ton'),
            )
            .collect()
        )

    def energy(self):
        return (
            pl
            .scan_parquet(self.conf.dirs.database / 'KEIT_전력.parquet')
            .drop_nulls('사용량')
            .select(
                'date',
                pl.sum_horizontal('사용량', '태양광발전량').alias('electricity_kWh'),
            )
            .collect()
            .join(self._heat('cooling'), on='date', how='full', coalesce=True)
            .join(self._heat('heating'), on='date', how='full', coalesce=True)
        )

    def __call__(self):
        data = self.energy().join(self.temperature(), on='date', how='left')
        data.write_parquet(self.conf.dirs.analysis / '03.KEIT.parquet')
        data.write_excel(self.conf.dirs.analysis / '03.KEIT.xlsx', column_widths=150)


@app.command
@dc.dataclass(frozen=True)
class Pmv:
    conf: Config

    @staticmethod
    def _prep(path: Path):
        output = path.parent
        stem = path.stem

        date = 'measurement_start_date'
        data = pl.read_parquet(path).rename({'date': date})

        for (d,), df in data.group_by(date):
            logger.info(d)

            df.write_excel(output / f'{stem}_{d}.xlsx', column_widths=125)
            (
                df
                .with_columns(
                    col=pl.format(
                        '{} [{}]', 'variable', pl.col('unit').fill_null('')
                    ).str.strip_suffix(' []')
                )
                .pivot(
                    'col',
                    index=['space', 'floor', 'datetime'],
                    values='value',
                    sort_columns=True,
                )
                .drop_nulls('PMV')
                .sort(['space', 'floor', 'datetime'])
                .write_excel(output / f'{stem}_{d}_wide.xlsx', column_widths=125)
            )

    def __call__(self):
        dir_ = self.conf.dirs.root / '04.PMV'
        for path in dir_.glob('*.parquet'):
            logger.info(path)
            self._prep(path)


if __name__ == '__main__':
    app()

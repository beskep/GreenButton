from pathlib import Path

import cyclopts
import polars as pl
import seaborn as sns
import structlog
from matplotlib.figure import Figure
from tqdm.rich import tqdm

WORKSPACE = Path('work/misc/TRNSYS')

app = cyclopts.App(result_action='print_return_zero')
logger = structlog.stdlib.get_logger()


@app.command
def prep():
    data = (
        pl
        .scan_parquet(WORKSPACE / '00.raw.parquet')
        .with_columns(
            pl.col('source').str.extract_groups(
                r'^simulation/(?P<period>\w+)/OUTPUT_(?P<t_set>[\d.]+).txt$'
            )
        )
        .unnest('source')
        .with_columns(pl.col('t_set').cast(pl.Float64))
        .rename({'TIME__HOURS': 'time'})
        .unpivot(index=['period', 't_set', 'time'])
        .drop_nulls('value')
        .collect()
    )

    data.write_parquet(WORKSPACE / '01.data.parquet')
    data.write_csv(WORKSPACE / '01.data.csv')

    return data


@app.command
def timeline():
    data = pl.read_parquet(WORKSPACE / '01.data.parquet')

    output = WORKSPACE / 'timeline'
    output.mkdir(exist_ok=True)

    for df in tqdm(data.partition_by('variable')):
        v = str(df['variable'].first())
        logger.info(v)

        fig = Figure()
        ax = fig.subplots()
        sns.lineplot(
            df.drop_nulls('value').to_dict(),
            x='time',
            y='value',
            hue='t_set',
            style='period',
            ax=ax,
        )
        ax.set_xlabel('Time [hr]')
        ax.set_ylabel(v)
        fig.savefig(output / f'{v}.png')

    return data


@app.command
def cooling():
    weather = (
        pl
        .scan_csv(WORKSPACE / 'WeatherCooling.csv')
        .select(
            pl.col('Time_Hour').cast(pl.Float64).alias('time'),
            pl.col('TAMB').alias('t_amb'),
        )
        .with_columns()
    )
    data = (
        pl
        .scan_parquet(WORKSPACE / '01.data.parquet')
        .filter(pl.col('period') == 'cooling', pl.col('variable') == 'P_CHILLER_W__W')
        .rename({'value': 'delivered'})
        .join(weather, on='time', how='left')
        .collect()
    )

    fig = Figure()
    ax = fig.subplots()
    sns.scatterplot(data.to_dict(), x='t_amb', y='delivered', hue='t_set', ax=ax)
    fig.savefig(WORKSPACE / '02.t-d.png')

    return data


if __name__ == '__main__':
    app()

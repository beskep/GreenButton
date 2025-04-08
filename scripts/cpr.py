from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING, Literal, Self

import matplotlib.dates as mdates
import msgspec
import numpy as np
import plotly.graph_objects as go
import polars as pl
import polars.selectors as cs
from cyclopts import App
from plotly.express import colors

from greenbutton import cpr

if TYPE_CHECKING:
    from collections.abc import Iterable


class Observations(msgspec.Struct):
    temperature: list[float]
    energy: list[float] | None = None
    datetime: list[str] | None = None

    def convert(self):
        data = cpr.CprData.create(
            x=self.temperature,
            y=self.energy or np.repeat(np.nan, len(self.temperature)),
            datetime=self.datetime,
        ).dataframe

        if not self.energy:
            data = data.drop('energy')

        return data


class SearchRange(msgspec.Struct):
    min_: float = 0.05
    max_: float = 0.95
    delta: float = 1.0  # °C

    def convert(self) -> cpr.RelativeSearchRange:
        return cpr.RelativeSearchRange(self.min_, self.max_, delta=self.delta)


class Option(msgspec.Struct):
    method: Literal['brute', 'numerical'] = 'brute'
    operation: Literal['h', 'c', 'hc', 'best'] = 'best'


class Inputs(msgspec.Struct):
    observations: Observations
    search_range: SearchRange = msgspec.field(default_factory=SearchRange)
    option: Option = msgspec.field(default_factory=Option)


class Model(msgspec.Struct):
    baseline: float
    sensitivity: dict[str, float]
    change_points: dict[str, float]

    validity: int = 1
    linear_model: dict[str, float | list] = {}

    @staticmethod
    def _convert_linear_model(
        model: cpr.CprModel,
    ) -> Iterable[tuple[str, float | list]]:
        v: float | list
        for key, value in model.model_dict.items():
            if key in model.OBSERVATIONS:
                continue

            match value:
                case list():
                    v = value
                case np.ndarray():
                    v = value.tolist()
                case _:
                    v = float(value)  # type: ignore[arg-type]

            yield key, v

    @classmethod
    def convert(cls, model: cpr.CprModel, decimals: int = 8) -> Self:
        def drop_nan(it: Iterable[tuple[str, float]]) -> dict[str, float]:
            # msgspec 인코딩을 위해 nan은 제외
            return {k: float(np.round(v, decimals)) for k, v in it if not np.isnan(v)}

        change_points = drop_nan(zip(['HDD', 'CDD'], model.change_points, strict=True))
        coef = drop_nan(model.coef.items())
        baseline = coef.pop('Intercept')

        return cls(
            validity=int(model.validity),
            baseline=baseline,
            sensitivity=coef,
            change_points=change_points,
            linear_model=dict(cls._convert_linear_model(model)),
        )

    def cpr_model(self):
        model_dict = {
            'names': ['Intercept', 'HDD', 'CDD'],
            'coef': [
                self.baseline,
                self.sensitivity.get('HDD', 0),
                self.sensitivity.get('CDD', 0),
            ],
        }
        return cpr.CprModel(
            change_points=(
                self.change_points.get('HDD', np.nan),
                self.change_points.get('CDD', np.nan),
            ),
            model_dict=model_dict,  # type: ignore[arg-type]
            validity=cpr.Validity(self.validity),
            optimize_method='brute',  # 사용 안함
            optimize_result=None,
        )


class Predicted(msgspec.Struct):
    temperature: list[float]
    hdd: list[float]
    cdd: list[float]
    epb: list[float]
    eph: list[float]
    epc: list[float]
    ep: list[float]

    energy: list[float] | None = None
    edb: list[float] | None = None
    edh: list[float] | None = None
    edc: list[float] | None = None

    datetime: list[float] | None = None

    @classmethod
    def predict(
        cls,
        model: cpr.CprModel,
        data: pl.DataFrame,
        sig_figs: int = 8,
    ) -> Self:
        predicted = (
            model.disaggregate(data)
            if 'energy' in data.columns
            else model.predict(data)
        ).with_columns(cs.numeric().round_sig_figs(sig_figs))
        return cls(**{
            k.lower(): v for k, v in predicted.to_dict(as_series=False).items()
        })


class Output(msgspec.Struct):
    model: Model
    predicted: Predicted
    plot: str | None = None


class _Plotter:
    @staticmethod
    def scatter(data: pl.DataFrame):
        if 'datetime' not in data.columns:
            color = None
            ticks = []
        else:
            color = 'epoch'
            data = data.with_columns(pl.col('datetime').dt.epoch('d').alias(color))
            dtr = (min(data['datetime']), max(data['datetime']))
            ticks = mdates.AutoDateLocator().get_locator(*dtr).tick_values(*dtr)

        if not color:
            marker = None
        else:
            marker = go.scatter.Marker(
                color=data[color],
                colorscale=colors.sequential.matter,
                colorbar=go.scatter.marker.ColorBar(
                    tickmode='array',
                    tickvals=ticks,
                    ticktext=[x.strftime('%Y-%m-%d') for x in mdates.num2date(ticks)],
                    len=0.6,
                ),
            )

        return go.Scatter(
            x=data['temperature'],
            y=data['energy'],
            marker=marker,
            opacity=0.8,
            mode='markers',
            name='에너지 사용량',
        )

    @staticmethod
    def line(model: cpr.CprModel, data: pl.DataFrame):
        temp = data['temperature'].to_numpy()
        segments = model.segments(temp.min(), temp.max()).drop_nans('temperature')
        return go.Scatter(
            x=segments['temperature'],
            y=segments['Ep'],
            mode='lines',
            name='CPR 모델',
            opacity=0.5,
            marker={'color': 'gray'},
        )

    @staticmethod
    def change_points(fig: go.Figure, model: cpr.CprModel):
        for t, hc in zip(model.change_points, ['난방', '냉방'], strict=True):
            if np.isnan(t):
                continue

            pos = 'bottom left' if hc == '난방' else 'bottom right'
            fig.add_vline(
                t,
                line_dash='dash',
                line_color='gray',
                opacity=0.5,
                annotation={'text': f'{hc} 균형점 온도 ({t:.1f}°C)'},
                annotation_position=pos,
            )

    @staticmethod
    def annotate(fig: go.Figure, model: cpr.CprModel, unit: str):
        def annot(k: str, v: float):
            u = unit if '/' in unit else f'{unit}/'
            match k:
                case 'Intercept':
                    return f'기저부하: {v:.3g} {unit}'
                case 'HDD':
                    return f'난방 민감도: {v:.3g} {u}°C'
                case 'CDD':
                    return f'냉방 민감도: {v:.3g} {u}°C'
                case _:
                    raise ValueError

        fig.add_annotation(
            text='<br>'.join(starmap(annot, model.coef.items())),
            xref='paper',
            yref='paper',
            x=0.01,
            y=0.01,
            xanchor='left',
            yanchor='bottom',
            align='left',
            showarrow=False,
        )

    @classmethod
    def plot(cls, model: cpr.CprModel, data: pl.DataFrame, unit: str = 'kWh/m²'):
        fig = go.Figure()

        if 'energy' in data.columns:
            fig.add_trace(cls.scatter(data=data))

        fig.add_trace(cls.line(model=model, data=data))
        cls.change_points(fig=fig, model=model)
        cls.annotate(fig=fig, model=model, unit=unit)

        return (
            fig.update_layout({'template': 'plotly_white'})
            .update_xaxes(title='평균 기온 [°C]')
            .update_yaxes(range=[0, None], title=f'에너지 사용량 [{unit}]')
        )


def _output(
    model: cpr.CprModel | Model,
    observations: Observations | pl.DataFrame,
    plot: Literal['json', 'html'] | None = None,
):
    if isinstance(model, cpr.CprModel):
        cm = model
        m = Model.convert(model)
    else:
        cm = model.cpr_model()
        m = model

    data = (
        observations.convert()
        if isinstance(observations, Observations)
        else observations
    )

    pred = Predicted.predict(model=cm, data=data)

    match plot:
        case 'html' | 'json':
            fig = _Plotter.plot(model=cm, data=data)
            p = fig.to_html() if plot == 'html' else fig.to_json()
        case None:
            p = None

    return Output(model=m, predicted=pred, plot=p)


app = App(help_format='markdown')


@app.command
def analyze(
    data: str,
    plot: Literal['json', 'html'] | None = None,
    mode: Literal['stdout', 'return'] = 'stdout',
) -> Output | None:
    """
    건물 에너지 냉난방민감도 분석.

    Parameters
    ----------
    data : str
        관측된 외부 기온, 에너지 사용량, 분석 옵션 등 (Examples 참조).
    plot : Literal['json', 'html'] | None
        그래프 출력 옵션.
    mode : Literal['stdout', 'return']
        결과 출력 옵션.

    Returns
    -------
    Output | None

    Examples
    --------
    >>> import json

    >>> # 전체 데이터, 옵션 입력.
    >>> # 냉난방 모드는 냉방, 난방, 냉난방 중 최적 모델 선택({'operation': 'best'}).
    >>> data = {
    ...     'observations': {
    ...         'temperature': [0, 1, 2, 3, 4, 5, 6, 7],
    ...         'energy': [3, 2, 1, 1, 1, 1, 3, 5],
    ...         'datetime': [
    ...             '2000-01-01',
    ...             '2000-01-02',
    ...             '2000-01-03',
    ...             '2000-01-04',
    ...             '2000-01-05',
    ...             '2000-01-06',
    ...             '2000-01-07',
    ...             '2000-01-08',
    ...         ],
    ...     },
    ...     'search_range': {'min': 0.05, 'max': 0.95, 'delta': 0.1},
    ...     'option': {'method': 'brute', 'operation': 'best'},
    ... }
    >>> analyze(json.dumps(data))  # doctest: +ELLIPSIS
    {"model":...,"predicted":...}

    >>> # 온도, 에너지 사용량과 냉난방 모드만 입력.
    >>> # 냉방 모델만 해석 ({'operation': 'c'}).
    >>> data = {
    ...     'observations': {
    ...         'temperature': list(range(100)),
    ...         'energy': [*([1] * 50), *range(1, 51)],
    ...     },
    ...     'option': {'operation': 'c'},
    ... }
    >>> analyze(json.dumps(data))  # doctest: +ELLIPSIS
    {"model":...,"predicted":...}
    """
    inputs = msgspec.json.decode(data, type=Inputs)
    obs = inputs.observations
    estimator = cpr.CprEstimator(x=obs.temperature, y=obs.energy, datetime=obs.datetime)

    search_range = inputs.search_range.convert()
    model = estimator.fit(
        heating=search_range,
        cooling=search_range,
        method=inputs.option.method,
        operation=inputs.option.operation,
    )

    output = _output(model=model, observations=estimator.data.dataframe, plot=plot)

    if mode == 'stdout':
        print(msgspec.json.encode(output).decode())
        return None

    return output


@app.command
def predict(
    model: str,
    observations: str,
    plot: Literal['json', 'html'] | None = None,
    mode: Literal['stdout', 'return'] = 'stdout',
) -> Output | None:
    """
    분석된 모델을 통해 새 데이터의 냉방, 난방, 기저 에너지 사용량 추정.

    Parameters
    ----------
    model : str
        `analyze`를 통해 분석된 모델 정보.
        'baseline', 'sensitivity', 'change_points'만 지정하면 분석 가능.
    observations : str
        예측에 사용할 데이터. 외기온('temperature')만 입력하면 예상 전체(ep),
        기저 (epb), 난방 (eph), 냉방 (epc) 사용량 예측.
        실사용량('energy')도 입력하면 기저 (edb), 난방 (edh), 냉방 (edc) 분리
        결과도 반환.
    mode : Literal['stdout', 'return']
        결과 출력 옵션.

    Returns
    -------
    Output | None

    Examples
    --------
    >>> import json

    >>> # 모델 정보, 외기온(temperature)만 입력
    >>> model = {
    ...     'baseline': 1.0,
    ...     'sensitivity': {'HDD': 1, 'CDD': 2},
    ...     'change_points': {'HDD': 2, 'CDD': 5},
    ... }
    >>> observations = {'temperature': [0, 1, 2, 3, 4, 5, 6, 7]}
    >>> predict(json.dumps(model), json.dumps(observations))  # doctest: +ELLIPSIS
    {"model":...,"predicted":{...,"ep":[...],...,"edb":null,...}

    >>> # 모델 정보, 외기온 (temperature), 실제 사용량 (energy), 시간 (datetime) 입력
    >>> observations = {
    ...     'temperature': [0, 1, 2, 3, 4, 5, 6, 7],
    ...     'energy': [3, 2, 1, 1, 1, 1, 3, 5],
    ...     'datetime': [
    ...         '2000-01-01',
    ...         '2000-02-01',
    ...         '2000-03-01',
    ...         '2000-04-01',
    ...         '2000-05-01',
    ...         '2000-06-01',
    ...         '2000-07-01',
    ...         '2000-08-01',
    ...     ],
    ... }
    >>> predict(json.dumps(model), json.dumps(observations))  # doctest: +ELLIPSIS
    {"model":...,"predicted":{...,"ep":[...],...,"edb":[...],...}
    """
    m = msgspec.json.decode(model, type=Model).cpr_model()
    obs = msgspec.json.decode(observations, type=Observations)
    output = _output(model=m, observations=obs, plot=plot)

    if mode == 'stdout':
        print(msgspec.json.encode(output).decode())
        return None

    return output


if __name__ == '__main__':
    app()

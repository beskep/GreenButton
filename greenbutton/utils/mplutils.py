from __future__ import annotations

import dataclasses as dc
import datetime
import inspect
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, overload

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.units as munits
import numpy as np
import seaborn as sns
from matplotlib.legend import Legend

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from matplotlib.colors import ListedColormap
    from matplotlib.typing import ColorType


Context = Literal['paper', 'notebook', 'talk', 'poster']
Style = Literal['darkgrid', 'whitegrid', 'dark', 'white', 'ticks'] | None
MathFont = Literal['dejavusans', 'cm', 'stix', 'stixsans', 'custom']
FigSizeUnit = Literal['cm', 'inch']
WidthHeight = tuple[float | None, float | None]
WidthHeightAspect = tuple[float | None, float | None, float]


TOL_ORDER: dict[str, tuple[int, ...]] = {
    'tol:bright': (0, 4, 2, 3, 1, 5, 6),
    'tol:high_contrast': (2, 0, 1),
    'tol:light': (0, 6, 5, 7, 1, 2, 3, 4, 8),
    'tol:medium_contrast': (2, 5, 0, 4, 3, 1),
    'tol:muted': (6, 0, 5, 3, 1, 7, 2, 4, 8),
    'tol:vibrant': (3, 0, 1, 5, 4, 2, 6),
}


def get_palette(palette: str, *, from_seaborn=True, from_cmap=True, reorder_tol=True):
    if from_seaborn:
        try:
            return sns.color_palette(palette)
        except ValueError:
            pass

    if from_cmap:
        try:
            from cmap import Colormap  # noqa: PLC0415

            cm = Colormap(palette)
        except (ImportError, ValueError):
            pass
        else:
            return (
                cm(TOL_ORDER[palette])
                if reorder_tol and palette.startswith('tol:')
                else cm.color_stops.color_array
            )

    return None


class SeabornPlottingContext:
    BASE_CONTEXT: ClassVar[dict[str, float]] = {
        'axes.linewidth': 1.25,
        'grid.linewidth': 1,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'patch.linewidth': 1,
        'xtick.major.width': 1.25,
        'ytick.major.width': 1.25,
        'xtick.minor.width': 1,
        'ytick.minor.width': 1,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 4,
        'ytick.minor.size': 4,
    }
    TEXTS_CONTEXT: ClassVar[dict[str, float]] = {
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12,
    }
    SCALE: ClassVar[dict[str, float]] = {
        'paper': 0.8,
        'notebook': 1,
        'talk': 1.5,
        'poster': 2,
    }

    @classmethod
    def rc(cls, context: float | Context, font_scale: float = 1.0):
        scale = context if isinstance(context, float | int) else cls.SCALE[context]
        rc = {k: v * scale for k, v in (cls.BASE_CONTEXT | cls.TEXTS_CONTEXT).items()}

        if font_scale != 1:
            rc |= {k: v * font_scale for k, v in rc.items() if k in cls.TEXTS_CONTEXT}

        return rc


class _MplFont(TypedDict, total=False):
    family: str
    sans: Collection[str]
    serif: Collection[str]
    math: MathFont


@dc.dataclass
class MplFont:
    family: str = 'sans-serif'
    sans: Collection[str] = ('Noto Sans KR', 'Source Han Sans KR', 'sans-serif')
    serif: Collection[str] = ('Noto Serif KR', 'Source Han Serif KR', 'serif')
    math: MathFont = 'custom'


@dc.dataclass
class MplFigSize:
    width: float | None = 16
    height: float | None = 9
    aspect: float = 9 / 16
    unit: FigSizeUnit = 'cm'

    INCH: ClassVar[float] = 2.54

    def __post_init__(self):
        self.update()

    def update(self):
        for field in ['width', 'height', 'aspect']:
            v = getattr(self, field)
            if v is not None and v <= 0:
                msg = f'{field}=={v} <= 0'
                raise ValueError(msg)

        match self.width, self.height, self.aspect:
            case None, None, _:
                return
            case (w, None, a) if w is not None:
                self.height = float(w * a)
            case (None, h, a) if h is not None:
                self.width = float(h / a)

    def cm(self):
        self.update()

        if self.width is None or self.height is None:
            return None

        if self.unit == 'cm':
            return (self.width, self.height)

        return (self.width * self.INCH, self.height * self.INCH)

    def inch(self):
        self.update()

        if self.width is None or self.height is None:
            return None

        if self.unit == 'inch':
            return (self.width, self.height)

        return (self.width / self.INCH, self.height / self.INCH)


@dc.dataclass
class MplTheme:
    context: float | Context | None = 'notebook'
    font: MplFont | _MplFont = dc.field(default_factory=MplFont)
    font_scale: float = 1.0

    style: Style = 'whitegrid'
    palette: str | Sequence[ColorType] | None = 'tol:light'

    constrained: bool | None = True
    fig_size: MplFigSize | WidthHeight | WidthHeightAspect = dc.field(
        default_factory=MplFigSize
    )
    fig_dpi: float = 150
    save_dpi: float = 300

    rc: dict[str, object] = dc.field(default_factory=dict)

    def __post_init__(self):
        self.update()

    def update(self):
        if not isinstance(self.fig_size, MplFigSize):
            self.fig_size = MplFigSize(*self.fig_size)
        if not isinstance(self.font, MplFont):
            self.font = MplFont(**self.font)

        rc = {
            'font.family': self.font.family,
            'font.sans-serif': self.font.sans,
            'font.serif': self.font.serif,
            'mathtext.fontset': self.font.math,
            'figure.dpi': self.fig_dpi,
            'savefig.dpi': self.save_dpi,
        }

        if figsize := self.fig_size.inch():
            rc['figure.figsize'] = figsize

        if self.constrained is not None:
            rc['figure.constrained_layout.use'] = self.constrained

        self.rc |= rc
        return self

    def grid(self, *, show=True, color='.8', ls='-', lw=1, alpha=0.25):
        self.rc.update({
            'axes.grid': show,
            'grid.color': color,
            'grid.linestyle': ls,
            'grid.linewidth': lw,
            'grid.alpha': alpha,
        })
        return self

    def tick(
        self,
        axis: Literal['x', 'y', 'xy'] = 'xy',
        which: Literal['major', 'minor', 'both', 'neither'] = 'major',
        *,
        color='.2',
        labelcolor='k',
        direction: Literal['in', 'out', 'inout'] = 'out',
    ):
        major = which in {'major', 'both'}
        minor = which in {'minor', 'both'}

        rc: dict[str, object] = {}
        if 'x' in axis:
            rc |= {
                'xtick.bottom': major,
                'xtick.color': color,
                'xtick.labelcolor': labelcolor,
                'xtick.direction': direction,
                'xtick.minor.visible': minor,
            }
        if 'y' in axis:
            rc |= {
                'ytick.left': major,
                'ytick.color': color,
                'ytick.labelcolor': labelcolor,
                'ytick.direction': direction,
                'ytick.minor.visible': minor,
            }

        self.rc |= rc
        return self

    def rc_params(self):
        self.update()
        context = (
            {}
            if self.context is None
            else SeabornPlottingContext.rc(self.context, self.font_scale)
        )
        style = sns.axes_style(self.style)
        return context | style | self.rc

    def _palette(self):
        if isinstance(self.palette, str):
            return get_palette(self.palette)

        return self.palette

    def apply(self, rc: dict | None = None):
        rc_ = self.rc_params() | (rc or {})
        mpl.rcParams.update(rc_)

        if (p := self._palette()) is not None:
            sns.set_palette(p)

    @contextmanager
    def rc_context(self, rc: dict | None = None):
        prev = dict(mpl.rcParams.copy())
        prev.pop('backend', None)

        try:
            self.apply(rc)
            yield mpl.rcParams
        finally:
            mpl.rcParams.update(prev)


@dc.dataclass
class MplConciseDate:
    formats: Collection[str] = (
        '%Y',
        '%m월',
        '%d일',
        '%H:%M',
        '%H:%M',
        '%S.%f',
    )
    zero_formats: Collection[str] = (
        '',
        '%Y',
        '%m월',
        '%d일',
        '%H:%M',
        '%H:%M',
    )
    offset_formats: Collection[str] = (
        '',
        '%Y',
        '%Y-%m',
        '%Y-%m',
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M',
    )
    show_offset: bool = True
    interval_multiples: bool = True

    matplotlib_default: bool = False
    bold_zero_format: bool = False

    _N_FORMAT: ClassVar[int] = 6

    def __post_init__(self):
        for field in dc.fields(self):
            n = field.name
            if n.endswith('formats') and len(getattr(self, n)) != self._N_FORMAT:
                msg = f'len({n})!={self._N_FORMAT}'
                raise ValueError(msg)

    def converter_kwargs(self):
        kwargs = dc.asdict(self)
        default = kwargs.pop('matplotlib_default')
        bold_zero = kwargs.pop('bold_zero_format')

        if default:
            kwargs = {k: v for k, v in kwargs.items() if 'formats' not in k}
        elif bold_zero:
            kwargs['zero_formats'] = [
                rf'$\mathbf{{{x}}}$' if x else '' for x in kwargs['zero_formats']
            ]

        return kwargs

    def apply(self):
        kwargs = self.converter_kwargs()
        converter = mdates.ConciseDateConverter(**kwargs)
        munits.registry[np.datetime64] = converter
        munits.registry[datetime.date] = converter
        munits.registry[datetime.datetime] = converter


class ColWrap:
    N2NCOLS: ClassVar[dict[int, int]] = {1: 1, 2: 2, 3: 3, 4: 2}

    def __init__(self, n: int, *, ratio=9 / 16, ceil=False) -> None:
        if n <= 0:
            msg = f'{n=} <= 0'
            raise ValueError(msg)

        if not (ncols := self.N2NCOLS.get(int(n), 0)):
            c = np.sqrt(n / ratio)
            ncols = np.ceil(c) if ceil else np.round(c)

        self._ncols = int(ncols)
        self._nrows = int(np.ceil(n / ncols))

    def __int__(self):
        return self._ncols

    @property
    def nrows(self):
        return self._nrows

    @property
    def ncols(self):
        return self._ncols


def text_color(bg_color, threshold=0.25, dark='1', bright='w'):
    return dark if sns.utils.relative_luminance(bg_color) >= threshold else bright


def move_legend_fig_to_ax(fig, ax, loc, bbox_to_anchor=None, **kwargs):
    # https://github.com/mwaskom/seaborn/issues/2994
    if fig.legends:
        old_legend = fig.legends[-1]
    else:
        msg = 'Figure has no legend attached.'
        raise ValueError(msg)

    old_boxes = old_legend.get_children()[0].get_children()

    legend_kws = inspect.signature(Legend).parameters
    props = {k: v for k, v in old_legend.properties().items() if k in legend_kws}

    props.pop('bbox_to_anchor')
    title = props.pop('title')
    if 'title' in kwargs:
        title.set_text(kwargs.pop('title'))

    title_kwargs = {k: v for k, v in kwargs.items() if k.startswith('title_')}
    for key, val in title_kwargs.items():
        title.set(**{key[6:]: val})
        kwargs.pop(key)

    kwargs.setdefault('frameon', old_legend.legendPatch.get_visible())

    # Remove the old legend and create the new one
    props.update(kwargs)
    fig.legends = []
    new_legend = ax.legend([], [], loc=loc, bbox_to_anchor=bbox_to_anchor, **props)
    new_legend.get_children()[0].get_children().extend(old_boxes)


def move_grid_legend(grid: sns.FacetGrid, loc='center'):
    figinv = grid.figure.transFigure.inverted()  # display -> figure coord
    r = [(0, 0), (1, 1)]

    # 오른쪽 위 ax, 마지막 ax의 figure 좌표 [[xmin, ymin], [xmax, ymax]]
    xy0 = figinv.transform(grid.axes[grid._ncol - 1].transAxes.transform(r))  # noqa: SLF001 # pyright: ignore[reportAttributeAccessIssue]
    xy1 = figinv.transform(grid.axes[-1].transAxes.transform(r))

    # legend가 위치할 bounding box
    bbox = (
        xy0[0, 0],  # x
        xy1[0, 1],  # y,
        xy0[1, 0] - xy0[0, 0],  # w
        xy1[1, 1] - xy1[0, 1],  # h
    )

    sns.move_legend(grid, loc=loc, bbox_to_anchor=bbox)


@dc.dataclass
class Cubehelix:
    start: float = 0.5
    rot: float = -1.5
    hue: float = 1.2
    light: float = 0.2
    dark: float = 0.8

    def __str__(self) -> str:
        return (
            f'ch:s={self.start},r={self.rot},h={self.hue},l={self.light},d={self.dark}'
        )

    @overload
    def palette(self, n: None) -> ListedColormap: ...

    @overload
    def palette(self, n: int) -> list[tuple[float, float, float]]: ...

    def palette(self, n: int | None = None):
        kwargs = {'n_colors': n or 6, 'as_cmap': n is None}
        return sns.cubehelix_palette(**kwargs, **dc.asdict(self))

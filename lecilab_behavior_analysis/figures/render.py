"""
Drawing figures made of panels.

A figure is a Layout: rows of panels, each panel with a relative width.
A Panel is a small function that draws one plot on the axes it is given,
from the data in a FigureData, and says which columns it needs.

render() draws every panel of a layout. It never stops because one panel
fails: the panel shows a red "Could not generate ..." box, the error is
logged, and it is returned in the list of problems, so that a report can
list the panels that failed. A panel that has nothing to show for this data
(e.g. no visual trials) raises NotApplicable, and shows a grey message
instead (not a problem).

Example::

    @panel("reaction times", needs={"reaction_time", "year_month_day"})
    def reaction_times(data, ax):
        ...

    LAYOUT = Layout([Row([(summary, 1), (reaction_times, 2)])], width=15, height=4)
    fig, problems = render(LAYOUT, FigureData(prepared_df))
"""
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# style shared by every figure made with render()
STYLE = {
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

ERROR_COLOR = "firebrick"
MESSAGE_COLOR = "gray"


class NotApplicable(Exception):
    """Raised by a panel that has nothing to show for this data (not an error)."""


@dataclass
class PanelProblem:
    """A panel that could not be drawn."""

    panel: str
    error: str
    message: str

    def __str__(self) -> str:
        return f"{self.panel}: {self.error}: {self.message}"


class FigureData:
    """
    The data a figure is drawn from: a prepared dataframe (see
    preprocessing.prepare) and options (perf_window, events_df...).

    Panels can share intermediate results with cached(), so that a table
    used by several panels is computed once per figure.
    """

    def __init__(self, df: pd.DataFrame, **options: Any):
        self.df = df
        self.options = options
        self._cache: Dict[str, Any] = {}

    def option(self, name: str, default: Any = None) -> Any:
        value = self.options.get(name)
        return default if value is None else value

    def cached(self, key: str, compute: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]


@dataclass
class Panel:
    """One plot of a figure. draw(data, ax) draws it on ax."""

    name: str
    draw: Callable[[FigureData, Axes], Any]
    needs: frozenset = frozenset()
    projection: Optional[str] = None

    def __call__(self, data: FigureData, ax: Axes) -> Any:
        return self.draw(data, ax)


def panel(name: str, needs: Iterable[str] = (), projection: Optional[str] = None):
    """Decorator that turns a function draw(data, ax) into a Panel."""

    def wrap(draw: Callable[[FigureData, Axes], Any]) -> Panel:
        return Panel(name=name, draw=draw, needs=frozenset(needs), projection=projection)

    return wrap


@dataclass
class Row:
    """Panels of one row, as (panel, relative width). None leaves a blank space."""

    cells: Sequence[Tuple[Optional[Panel], float]]
    height: float = 1.0


@dataclass
class Layout:
    rows: List[Row]
    width: float = 15
    height: float = 10
    panels: List[Panel] = field(init=False)

    def __post_init__(self):
        self.panels = [p for row in self.rows for p, _ in row.cells if p is not None]


def split_cell(ax: Axes, nrows: int = 1, ncols: int = 1, **gridspec_kw) -> List[Axes]:
    """
    Replaces ax by a grid of new axes in the same space (for panels made of
    several plots). Returns the new axes, row by row.
    """
    fig = ax.figure
    spec = ax.get_subplotspec()
    ax.remove()
    grid = spec.subgridspec(nrows, ncols, **gridspec_kw)
    return [fig.add_subplot(grid[i, j]) for i in range(nrows) for j in range(ncols)]


def show_message(ax: Axes, text: str, color: str = MESSAGE_COLOR) -> Axes:
    """Writes a message in the middle of an empty panel."""
    ax.clear()
    ax.text(0.5, 0.5, text, color=color, ha="center", va="center", wrap=True,
            fontsize=8, transform=ax.transAxes)
    ax.set_axis_off()
    return ax


def _replace_with_message(fig: Figure, spec, axes_before: set, text: str, color: str,
                          projection: Optional[str]) -> None:
    # remove whatever the panel created (twin axes, split cells...)
    for axis in list(fig.axes):
        if axis not in axes_before:
            axis.remove()
    show_message(fig.add_subplot(spec), text, color)


def draw_panel(fig: Figure, spec, item: Panel, data: FigureData) -> Optional[PanelProblem]:
    """Draws one panel in the space spec of fig. Returns the problem, if any."""
    axes_before = set(fig.axes)
    ax = fig.add_subplot(spec, projection=item.projection)
    missing = sorted(item.needs - set(data.df.columns))
    if data.df.empty and item.needs:
        _replace_with_message(fig, spec, axes_before, "No data", MESSAGE_COLOR, item.projection)
        return None
    if missing:
        text = f"Could not generate {item.name}\nmissing columns: {', '.join(missing)}"
        _replace_with_message(fig, spec, axes_before, text, ERROR_COLOR, item.projection)
        logger.warning("Panel '%s' needs missing columns: %s", item.name, missing)
        return PanelProblem(item.name, "MissingColumns", ", ".join(missing))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            item.draw(data, ax)
    except NotApplicable as reason:
        _replace_with_message(fig, spec, axes_before, str(reason), MESSAGE_COLOR, item.projection)
        return None
    except Exception as error:  # a broken panel must not break the figure
        text = f"Could not generate {item.name}\n{type(error).__name__}: {error}"
        _replace_with_message(fig, spec, axes_before, text, ERROR_COLOR, item.projection)
        logger.warning("Panel '%s' failed: %s: %s", item.name, type(error).__name__, error)
        logger.debug("Panel '%s' traceback", item.name, exc_info=True)
        return PanelProblem(item.name, type(error).__name__, str(error))
    return None


def render(layout: Layout, data: FigureData, width: Optional[float] = None,
           height: Optional[float] = None, title: Optional[str] = None) -> Tuple[Figure, List[PanelProblem]]:
    """
    Draws every panel of the layout. Returns the figure and the panels that
    failed (empty list if all worked). Does not raise because of the data.
    """
    problems: List[PanelProblem] = []
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(width or layout.width, height or layout.height), layout="constrained")
        outer = fig.add_gridspec(len(layout.rows), 1, height_ratios=[r.height for r in layout.rows])
        for i, row in enumerate(layout.rows):
            inner = outer[i].subgridspec(1, len(row.cells), width_ratios=[w for _, w in row.cells])
            for j, (item, _) in enumerate(row.cells):
                if item is None:
                    continue
                problem = draw_panel(fig, inner[0, j], item, data)
                if problem is not None:
                    problems.append(problem)
        if title:
            fig.suptitle(title)
    fig.panel_problems = problems
    return fig, problems

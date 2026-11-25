"""
Draws the example figures of the documentation from the trials in
tests/data (real data of one mouse), so that they always show what the
current code draws. Run by conf.py at the start of every build; the images
go to source/_static/examples/ (not tracked by git).
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "_static" / "examples"
sys.path.insert(0, str(REPO))


def main() -> None:
    from lecilab_behavior_analysis.figures import (
        FigureData,
        Layout,
        NotApplicable,
        Row,
        panel,
        render,
        session_summary,
        subject_progress,
    )
    from lecilab_behavior_analysis.figures import panels as p
    from lecilab_behavior_analysis.metrics.performance import get_performance_through_trials
    from lecilab_behavior_analysis.preprocessing import prepare
    from lecilab_behavior_analysis.reports import overview

    logging.getLogger("lecilab_behavior_analysis").setLevel(logging.ERROR)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REPO / "tests" / "data" / "NUO003_excerpt.csv.gz", sep=";", low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = prepare(get_performance_through_trials(df, window=50))

    def save(fig, name):
        fig.savefig(OUT / f"{name}.png", dpi=80)
        matplotlib.pyplot.close(fig)

    fig, _ = subject_progress(df, perf_window=100)
    save(fig, "subject_progress")
    fig, _ = session_summary(df[df["year_month_day"] == "2026-01-05"], perf_window=50)
    save(fig, "session_summary")
    save(overview.cohort_trials_and_performance_figure(df), "cohort")
    save(overview.box_occupancy_figure(df), "box_occupancy")

    # the example of figures.md: a new panel, a panel that fails, and one with nothing to show
    @panel("trials per day", needs={"year_month_day"})
    def trials_per_day(data, ax):
        counts = data.df.groupby("year_month_day").size()
        ax.bar(pd.to_datetime(counts.index), counts.values)
        ax.set_ylabel("Trials")

    @panel("broken plot", needs={"year_month_day"})
    def broken(data, ax):
        raise ValueError("this panel has a bug")

    @panel("optogenetics", needs={"year_month_day"})
    def optogenetics(data, ax):
        if "opto" not in data.df.columns:
            raise NotApplicable("No optogenetics in these sessions")

    layout = Layout(
        [
            Row([(p.subject_summary, 1), (trials_per_day, 2)]),
            Row([(broken, 1), (optogenetics, 1), (p.subject_performance, 1)]),
        ],
        width=12,
        height=6,
    )
    fig, problems = render(layout, FigureData(df, perf_window=100))
    save(fig, "custom_layout")

    make_gallery(df)
    export_tutorial()


SOURCE = Path(__file__).resolve().parent
GALLERY_PAGE = SOURCE / "gallery.md"
GALLERY_FOLDER = SOURCE / "gallery"  # a page per figure
GALLERY_INTRO = """---
layout: simple
content_max_width: 1240px
---

# Gallery

Every plot of the package. Hover over a figure to see what it shows, and
click it to see the code that draws it (the names in the code link to their
documentation). The figures are drawn from the example data in `tests/data`
(one mouse, NUO003) by the current code each time this documentation is
built.
"""
GALLERY_INPUTS = """## The inputs of the examples

Every example draws from three inputs. With your data:

```python
import pandas as pd

from lecilab_behavior_analysis.io.loading import load_project_logs, load_project_trials

# the trials of a mouse, prepared for analysis (and its performance through trials)
trials = load_project_trials("COT_cannula_data", ["NUO003"])
# its trials of one day
day = trials[trials["year_month_day"] == "2025-10-15"]
# the sessions summaries of the project's setups (with the settings of each session)
_, summary, _ = load_project_logs("COT_cannula_data")
```

The [tutorial](tutorial.md) goes through these step by step.
"""


def example_code(function) -> str:
    """The body of an example function, without its def line and its final return."""
    import inspect
    import textwrap

    lines = inspect.getsource(function).splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("def ")) + 1
    while not lines[start].rstrip().endswith(":") and lines[start - 1].rstrip().endswith(","):
        start += 1  # a def line split over several lines
    body = textwrap.dedent("\n".join(lines[start:])).strip().splitlines()
    if body and body[-1].startswith("return fig"):
        body = body[:-1]
    elif body and body[-1].startswith("return "):
        body[-1] = "fig = " + body[-1][len("return ") :]
    return "\n".join(body).strip()


def plotly_preview(fig, path: Path) -> None:
    """A still image of a plotly figure (its lines and markers), for the grid of the gallery."""
    import matplotlib.pyplot as plt

    preview, ax = plt.subplots(figsize=(7, 3.5))
    for trace in fig.data:
        if getattr(trace, "x", None) is not None and getattr(trace, "y", None) is not None:
            ax.plot(list(trace.x), list(trace.y), linewidth=1)
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else ""
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    preview.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(preview)


def size_class(section: str, image: Path) -> str:
    """How much room a figure takes in the grid: the ready-made figures and wide figures are bigger."""
    from PIL import Image

    if section == "Ready-made figures":
        return "large"
    with Image.open(image) as picture:
        width, height = picture.size
    return "wide" if width / height >= 1.8 else ""


def make_gallery(trials: pd.DataFrame) -> None:
    """
    Runs every example of gallery_examples.py and writes the gallery: a grid
    of their images (gallery.md) and a page per example with its code
    (gallery/<name>.md).
    """
    import html
    import shutil

    import gallery_examples
    import matplotlib.pyplot as plt

    summary = pd.read_csv(REPO / "tests" / "data" / "NUO003_sessions_summary.csv", sep=";")
    day = trials[trials["year_month_day"] == gallery_examples.TRAINING_DAY]
    shutil.rmtree(GALLERY_FOLDER, ignore_errors=True)
    GALLERY_FOLDER.mkdir()
    cards = []
    for item in gallery_examples.EXAMPLES:
        name = item.function.__name__.removesuffix("_example")
        image = OUT / f"{name}.png"
        failure = None
        try:
            fig = item.function(trials, day, summary)
            if item.interactive:
                fig.write_html(OUT / f"{name}.html", include_plotlyjs="cdn", full_html=True)
                plotly_preview(fig, image)
            else:
                fig.savefig(image, dpi=80, bbox_inches="tight")
                plt.close(fig)
        except Exception as error:  # shown on the page; the tests run every example
            failure = f"{type(error).__name__}: {error}"

        # its card in the grid
        title, description = html.escape(item.title), html.escape(item.description)
        badge = '<span class="gallery-badge">interactive</span>' if item.interactive else ""
        picture = (
            f'<img src="_static/examples/{name}.png" alt="{title}" loading="lazy">'
            if failure is None
            else f"<span>This example failed: {html.escape(failure)}</span>"
        )
        size = size_class(item.section, image) if failure is None else ""
        cards.append(
            f'<a class="gallery-item {size}" href="gallery/{name}.html">{picture}{badge}'
            f'<div class="gallery-overlay"><strong>{title}</strong><span>{description}</span>'
            f"<em>Click to see the code</em></div></a>"
        )

        # its page
        if failure is not None:
            figure = f":::{{warning}}\nThis example failed: {failure}\n:::"
        elif item.interactive:
            figure = (
                f'<iframe src="../_static/examples/{name}.html" width="100%" height="560" frameborder="0"></iframe>'
            )
        else:
            figure = f'<div class="gallery-figure"><img src="../_static/examples/{name}.png" alt="{title}"></div>'
        (GALLERY_FOLDER / f"{name}.md").write_text(
            # orphan: reached from the gallery, not listed in the navigation; wide, without sidebars
            "---\norphan: true\nlayout: simple\ncontent_max_width: 1100px\n---\n\n"
            f"# {item.title}\n\n{item.description}\n\n{figure}\n\n"
            f"```python\n{example_code(item.function)}\n```\n\n"
            "`trials`, `day` and `summary` are [the inputs of the examples](../gallery.md#the-inputs-of-the-examples). "
            "[Back to the gallery](../gallery.md)\n"
        )

    grid = '<div class="gallery-grid">\n' + "\n".join(cards) + "\n</div>\n"
    GALLERY_PAGE.write_text("\n".join([GALLERY_INTRO, grid, GALLERY_INPUTS]))


def export_tutorial() -> None:
    """The tutorial notebook (notebooks/tutorial.py), run and saved as HTML for tutorial.md."""
    import shutil
    import subprocess

    target = OUT / "tutorial.html"
    marimo = shutil.which("marimo")
    if marimo is None:
        target.write_text("<p>The tutorial could not be built here (marimo is not installed).</p>")
        return
    result = subprocess.run(
        [marimo, "export", "html", str(REPO / "notebooks" / "tutorial.py"), "-o", str(target)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0 or not target.exists():
        target.write_text(f"<p>The tutorial could not be built:</p><pre>{result.stderr[-2000:]}</pre>")


if __name__ == "__main__":
    main()

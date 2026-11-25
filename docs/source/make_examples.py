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


GALLERY_PAGE = Path(__file__).resolve().parent / "gallery.md"
GALLERY_INTRO = """# Gallery

Every plot of the package, with the code that draws it. The images are drawn
from the example data in `tests/data` (one mouse, NUO003) by the current code
each time the documentation is built: the code under each image is the code
that drew it.

Each example uses three inputs. With your data:

```python
from lecilab_behavior_analysis.io.loading import load_project_logs, load_project_trials

# the trials of a mouse, prepared for analysis (and its performance through trials)
trials = load_project_trials("COT_cannula_data", ["NUO003"])
# its trials of one day
day = trials[trials["year_month_day"] == "2025-10-15"]
# the sessions summaries of the project's setups (with the settings of each session)
_, summary, _ = load_project_logs("COT_cannula_data")
```

See also the [tutorial](tutorial.md), which goes through these step by step.
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


def make_gallery(trials: pd.DataFrame) -> None:
    """Runs every example of gallery_examples.py and writes gallery.md with their images and code."""
    import gallery_examples
    import matplotlib.pyplot as plt

    summary = pd.read_csv(REPO / "tests" / "data" / "NUO003_sessions_summary.csv", sep=";")
    day = trials[trials["year_month_day"] == gallery_examples.TRAINING_DAY]
    sections: dict = {}
    for item in gallery_examples.EXAMPLES:
        name = item.function.__name__.removesuffix("_example")
        try:
            fig = item.function(trials, day, summary)
            if item.interactive:
                fig.write_html(OUT / f"{name}.html", include_plotlyjs="cdn", full_html=True)
                image = (
                    f'<iframe src="_static/examples/{name}.html" width="100%" height="480" frameborder="0"></iframe>'
                )
            else:
                fig.savefig(OUT / f"{name}.png", dpi=80, bbox_inches="tight")
                plt.close(fig)
                image = f"![{item.title}](_static/examples/{name}.png)"
        except Exception as error:  # shown on the page; the tests run every example
            image = f":::{{warning}}\nThis example failed: {type(error).__name__}: {error}\n:::"
        entry = f"### {item.title}\n\n{item.description}\n\n{image}\n\n```python\n{example_code(item.function)}\n```\n"
        sections.setdefault(item.section, []).append(entry)
    page = [GALLERY_INTRO]
    for section, entries in sections.items():
        page.append(f"## {section}\n")
        page.extend(entries)
    GALLERY_PAGE.write_text("\n".join(page))


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

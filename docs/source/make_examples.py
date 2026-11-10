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


if __name__ == "__main__":
    main()

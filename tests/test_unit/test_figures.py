"""
Every figure is drawn on real data, and no panel may fail.

The daily report (village-report-generator) draws these figures for every
subject every day: a panel that fails there shows a red box. These tests
catch it before.
"""

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

import lecilab_behavior_analysis.figure_maker as fm
from lecilab_behavior_analysis.figures import FigureData, Layout, NotApplicable, Row, panel, render
from lecilab_behavior_analysis.figures.layouts import SESSION_SUMMARY, SUBJECT_PROGRESS

DAYS = ["2025-09-05", "2025-09-25", "2025-10-15", "2025-11-11", "2025-12-18", "2026-01-05", "2026-04-04", "2026-06-02"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _problems(fig):
    return [str(p) for p in fig.panel_problems]


def test_subject_progress(prepared_trials):
    fig = fm.subject_progress_figure(prepared_trials, perf_window=100)
    assert _problems(fig) == []
    # every panel of the layout is drawn (no "missing columns" box)
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert not any(t.startswith("Could not generate") for t in texts)


def test_subject_progress_with_summary_matrix(prepared_trials):
    fig = fm.subject_progress_figure(prepared_trials, summary_matrix_plot=True)
    assert _problems(fig) == []


def test_subject_progress_from_raw_data(raw_trials):
    # the figure must not depend on what the caller ran before
    fig = fm.subject_progress_figure(raw_trials)
    assert _problems(fig) == []


@pytest.mark.parametrize("day", DAYS)
def test_session_summary(prepared_trials, day):
    fig = fm.session_summary_figure(prepared_trials[prepared_trials.year_month_day == day], perf_window=50)
    assert _problems(fig) == []


@pytest.mark.parametrize("day", ["2025-09-05", "2025-10-15", "2026-01-05"])
def test_session_summary_from_raw_data(raw_trials, day):
    fig = fm.session_summary_figure(raw_trials[raw_trials.date.dt.strftime("%Y-%m-%d") == day])
    assert _problems(fig) == []


def _new_format_day(raw_trials: pd.DataFrame) -> pd.DataFrame:
    """A visual hard / auditory day converted to the format saved since 2026-11-09."""
    df = raw_trials[raw_trials.date.dt.strftime("%Y-%m-%d") == "2025-10-15"].copy()
    df["date"] = df["date"] + (pd.Timestamp("2026-11-10") - pd.Timestamp("2025-10-15"))
    df["stimulus_modality"] = df["stimulus_modality"].replace({"auditory": "auditory-cot"})
    df["current_training_stage"] = df["current_training_stage"].str.replace("auditory", "auditory-cot")

    def left_right(row):
        if not isinstance(row["visual_stimulus"], str):
            return row["visual_stimulus"]
        correct, incorrect = eval(row["visual_stimulus"])
        return str((correct, incorrect) if row["correct_side"] == "left" else (incorrect, correct))

    df["visual_stimulus"] = df.apply(left_right, axis=1)
    # some timeouts
    df["correct"] = df["correct"].astype(object)
    df.loc[df.index[::10], "correct"] = None
    return df


def test_figures_with_new_format(raw_trials):
    df = _new_format_day(raw_trials)
    assert _problems(fm.session_summary_figure(df)) == []
    assert _problems(fm.subject_progress_figure(df)) == []


def test_mixed_old_and_new_format(raw_trials):
    df = pd.concat([raw_trials, _new_format_day(raw_trials)], ignore_index=True)
    assert _problems(fm.subject_progress_figure(df)) == []


def test_engagement_panel_with_events(prepared_trials):
    # a "back home" event 60 s after the end of each session
    ends = prepared_trials.groupby("date")["TRIAL_END"].max() - prepared_trials.groupby("date")["TRIAL_START"].min()
    events = pd.DataFrame(
        {
            "date": [(d + pd.Timedelta(seconds=float(s) + 60)).strftime("%Y-%m-%d %H:%M:%S") for d, s in ends.items()],
            "type": "INFO",
            "subject": "NUO003",
            "description": "Subject back home: NUO003",
        }
    )
    fig = fm.subject_progress_figure(prepared_trials, events_df=events)
    assert _problems(fig) == []
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "No events_df provided" not in texts


def test_empty_data_does_not_raise():
    empty = pd.DataFrame(columns=["subject", "session", "date", "trial", "correct"])
    fm.session_summary_figure(empty)
    fm.subject_progress_figure(empty)


def test_layouts_list_their_panels():
    assert len(SUBJECT_PROGRESS.panels) == 10
    assert len(SESSION_SUMMARY.panels) == 10


# render ---------------------------------------------------------------------
@panel("broken panel", needs={"correct"})
def _broken(data, ax):
    raise RuntimeError("boom")


@panel("needs a column that is not there", needs={"no_such_column"})
def _needs_missing(data, ax):
    ax.plot([0, 1])


@panel("nothing to show", needs={"correct"})
def _not_applicable(data, ax):
    raise NotApplicable("No trials in this modality")


@panel("working panel", needs={"correct"})
def _working(data, ax):
    ax.plot(np.arange(len(data.df)))


def test_a_failing_panel_does_not_stop_the_figure():
    data = FigureData(pd.DataFrame({"correct": [True, False, True]}))
    layout = Layout([Row([(_broken, 1), (_needs_missing, 1), (_not_applicable, 1), (_working, 1)])], width=8, height=2)
    fig, problems = render(layout, data)
    assert [(p.panel, p.error) for p in problems] == [
        ("broken panel", "RuntimeError"),
        ("needs a column that is not there", "MissingColumns"),
    ]
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "Could not generate broken panel\nRuntimeError: boom" in texts
    assert "No trials in this modality" in texts
    # the working panel was drawn
    assert any(len(ax.lines) == 1 for ax in fig.axes)

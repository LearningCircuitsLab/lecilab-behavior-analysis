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


# psychometric panels: one per modality ----------------------------------------
def _psychometric_axes(fig):
    return [ax for ax in fig.axes if ax.get_title().startswith("choices on")]


def _sound_location_day(raw_trials, modality="auditory_locaPT", with_level=True, date="2026-01-05"):
    """A day of real trials made into sound location trials (old name), with medium and hard trials."""
    day = raw_trials[raw_trials["date"].dt.strftime("%Y-%m-%d") == date].copy()
    rng = np.random.default_rng(0)
    level = rng.uniform(40, 70, len(day))
    day["stimulus_modality"] = modality
    day["current_training_stage"] = f"TwoAFC_{modality}_hard"
    day["difficulty"] = np.where(level > 55, "medium", "hard")
    if with_level:
        day["auditory_stimulus"] = [str({"frequency_hz": 1000, "sound_db_spl": round(v, 1)}) for v in level]
    else:
        day["auditory_stimulus"] = [str({"frequency_hz": 1000}) for _ in level]
    return day


def test_old_sound_location_name_is_read_as_the_current_one(raw_trials):
    from lecilab_behavior_analysis.preprocessing import prepare

    prepared = prepare(_sound_location_day(raw_trials))
    assert set(prepared["stimulus_modality"]) == {"auditory-local-wn"}
    assert set(prepared["current_training_stage"]) == {"TwoAFC_auditory-local-wn_hard"}


def test_one_psychometric_panel_for_one_modality(raw_trials):
    from lecilab_behavior_analysis.figures.layouts import session_modalities, session_summary_layout

    fig = fm.session_summary_figure(_sound_location_day(raw_trials))
    assert _problems(fig) == []
    psychometric = _psychometric_axes(fig)
    assert [ax.get_title() for ax in psychometric] == ["choices on auditory-local-wn trials"]
    assert psychometric[0].get_xlabel().startswith("Sound level")
    assert len(psychometric[0].lines) + len(psychometric[0].collections) > 0
    # the repeat / alternate panel takes the place of the second psychometric panel
    layout = session_summary_layout(["auditory-local-wn"])
    widths = dict((cell.name, width) for cell, width in layout.rows[1].cells)
    assert widths["repeat or alternate plot"] == 2 and len(layout.rows[1].cells) == 4
    assert session_modalities(
        pd.DataFrame({"task": ["TwoAFC"] * 2, "stimulus_modality": ["visual", "auditory-cot"]})
    ) == ["visual", "auditory-cot"]


def test_sound_location_without_saved_level(raw_trials):
    fig = fm.session_summary_figure(_sound_location_day(raw_trials, with_level=False))
    assert _problems(fig) == []
    assert len(_psychometric_axes(fig)) == 1


def test_two_psychometric_panels_for_two_modalities(raw_trials):
    day = raw_trials[raw_trials["date"].dt.strftime("%Y-%m-%d") == "2026-01-05"].copy()
    day["stimulus_modality"] = np.where(np.arange(len(day)) % 2 == 0, "visual", "auditory")
    fig = fm.session_summary_figure(day)
    assert sorted(ax.get_title() for ax in _psychometric_axes(fig)) == [
        "choices on auditory-cot trials",
        "choices on visual trials",
    ]


def test_first_sound_location_name_is_read_as_the_current_one():
    from lecilab_behavior_analysis.data_format_changes import normalize_modality, normalize_stage

    assert normalize_modality("auditory_loca") == "auditory-local-wn"
    assert normalize_stage("TwoAFC_auditory_loca_easy") == "TwoAFC_auditory-local-wn_easy"


def test_sound_location_psychometric_has_three_points_per_side(raw_trials):
    from lecilab_behavior_analysis.metrics.choices import two_sided_bin_edges

    edges = two_sided_bin_edges(pd.Series([-70, -55, -40, 40, 52, 70]), 3)
    assert len(edges) == 8  # 3 bins per side, and the gap between them
    assert edges[3] < -39 and edges[4] > 39
    fig = fm.session_summary_figure(_sound_location_day(raw_trials, date="2025-10-15"))  # 291 trials
    ax = _psychometric_axes(fig)[0]
    points = [line for line in ax.lines if line.get_marker() == "o"]
    assert len(points) == 1 and len(points[0].get_xdata()) == 6

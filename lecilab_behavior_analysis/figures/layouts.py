"""
The figures of the package, as layouts of panels (see render.py and panels.py).

subject_progress(df) and session_summary(df) prepare the data, draw every
panel, and return (figure, problems): problems lists the panels that could
not be drawn, so that a report can show them instead of failing.
"""

from typing import List, Optional, Tuple

import pandas as pd
from matplotlib.figure import Figure

import lecilab_behavior_analysis.figures.panels as p
from lecilab_behavior_analysis.figures.render import FigureData, Layout, PanelProblem, Row, render
from lecilab_behavior_analysis.preprocessing import prepare

SUBJECT_PROGRESS = Layout(
    rows=[
        Row([(p.subject_summary, 1), (p.training_calendar, 3), (p.training_times, 1)], height=0.8),
        Row([(p.subject_performance, 1.5), (p.trials_and_water, 3)]),
        Row([(p.performance_by_decision, 1.5), (p.bias_evolution, 1), (p.center_holds, 2.5)]),
        Row([(p.reaction_times_by_day, 1), (p.engagement, 1)]),
    ],
    width=15,
    height=10,
)

SUMMARY_MATRIX_ROW = Row([(p.summary_matrix, 1)])

SESSION_SUMMARY = Layout(
    rows=[
        Row([(p.session_summary_text, 2), (p.session_performance, 2), (p.correct_responses, 1)]),
        Row(
            [
                (p.repeat_or_alternate, 1),
                (p.psychometric("visual"), 1),
                (p.psychometric("auditory-cot"), 1),
                (p.center_pokes_histogram, 1),
                (p.center_hold_histogram, 1),
            ]
        ),
        Row([(p.trial_start_times, 1), (p.reaction_times_in_session, 1)]),
    ],
    width=15,
    height=10,
)


def subject_progress(
    df: pd.DataFrame,
    perf_window: int = 50,
    events_df: Optional[pd.DataFrame] = None,
    summary_matrix_plot: bool = False,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> Tuple[Figure, List[PanelProblem]]:
    """
    Training history of one subject. Returns (figure, panels that failed).

    Args:
        df: trials of one subject (raw or already prepared).
        perf_window: number of trials of the performance window.
        events_df: events of the setup, for the engagement panel (optional).
        summary_matrix_plot: add a row with the choices by difficulty per session.
    """
    if df.subject.nunique() > 1:
        raise ValueError(f"The dataframe contains more than one subject: {df.subject.unique()}.")
    layout = SUBJECT_PROGRESS
    if summary_matrix_plot:
        layout = Layout(
            rows=SUBJECT_PROGRESS.rows + [SUMMARY_MATRIX_ROW], width=layout.width, height=layout.height * 1.25
        )
    data = FigureData(prepare(df), perf_window=perf_window, events_df=events_df)
    return render(layout, data, width=width, height=height)


def session_summary(
    df: pd.DataFrame,
    perf_window: int = 50,
    perf_hue: Optional[str] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> Tuple[Figure, List[PanelProblem]]:
    """
    Summary of some trials of one subject, usually one day. Returns
    (figure, panels that failed).

    Args:
        df: trials (raw or already prepared; a day of prepared data keeps the
            trial numbers of the whole history).
        perf_window: number of trials of the performance window.
        perf_hue: column that colors the performance (default: training stage
            if there are several, otherwise stimulus modality).
    """
    data = FigureData(prepare(df), perf_window=perf_window, perf_hue=perf_hue)
    return render(SESSION_SUMMARY, data, width=width, height=height)

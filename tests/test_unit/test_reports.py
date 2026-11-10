"""Report helpers return a figure and never raise."""

import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure

import lecilab_behavior_analysis.reports.overview as overview
from lecilab_behavior_analysis.figures.render import PanelProblem


def test_tables(prepared_trials):
    assert isinstance(overview.trials_by_stage_table(prepared_trials), go.Figure)
    assert isinstance(overview.last_days_table(prepared_trials, n_days=3), go.Figure)
    table = overview.trials_by_stage(prepared_trials)
    assert table.drop(columns="subject").to_numpy().sum() == len(prepared_trials)


def test_last_days_counts_timeouts_as_errors():
    df = pd.DataFrame(
        {
            "subject": ["m"] * 4,
            "year_month_day": ["2026-11-10"] * 4,
            "trial": [1, 2, 3, 4],
            "correct": [True, None, True, False],
        }
    )
    table = overview.last_days(df, n_days=1)
    assert table.loc[0, "performance_2026-11-10"] == "0.500"
    assert table.loc[0, "trials_2026-11-10"] == 4


def test_figures(prepared_trials):
    assert isinstance(overview.performance_over_trials_figure(prepared_trials), go.Figure)
    assert isinstance(overview.box_occupancy_figure(prepared_trials), Figure)
    assert isinstance(overview.cohort_trials_and_performance_figure(prepared_trials), Figure)


def test_problems_table():
    assert isinstance(overview.problems_table({}), go.Figure)
    problems = {"NUO003": [PanelProblem("reaction time plot", "ValueError", "boom")]}
    assert isinstance(overview.problems_table(problems), go.Figure)


def test_cells_never_raise():
    broken = pd.DataFrame({"nothing": [1]})
    fig = overview.cohort_trials_and_performance_figure(broken)
    assert "Could not generate" in fig.axes[0].texts[0].get_text()
    fig = overview.trials_by_stage_table(broken)
    assert "Could not generate" in fig.layout.annotations[0].text


def test_failed_cells_are_recorded():
    overview.reset_failed_cells()
    overview.cohort_trials_and_performance_figure(pd.DataFrame({"nothing": [1]}))
    failed = overview.failed_cells()
    assert [p.panel for p in failed] == ["Trials and performance of the cohort"]
    overview.reset_failed_cells()
    assert overview.failed_cells() == []


def test_tv_report_without_logs():
    overview.reset_failed_cells()
    fig = overview.tv_report_table(None, None)
    assert "No events" in fig.layout.annotations[0].text
    assert overview.failed_cells() == []

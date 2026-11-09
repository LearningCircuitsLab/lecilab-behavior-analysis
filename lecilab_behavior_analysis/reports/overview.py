"""
Tables and figures of the overview page of the daily report
(village-report-generator), one function per report cell.

They take the trials of all the subjects of a project, prepared with
preprocessing.prepare (or analyze_df). None of them raises: if something
fails, they return a figure that says what failed (and log the error), so
that one broken cell does not stop the report.

Performance counts timeouts (correct = None) as errors, as in the figures
(see data_format_changes.py).
"""
import functools
import logging
from typing import Callable, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import lecilab_behavior_analysis.data_format_changes as dfc
from lecilab_behavior_analysis.figures.render import PanelProblem
from lecilab_behavior_analysis.metrics.occupancy import (
    get_daily_occupancy_percentages,
    get_occupancy_heatmap,
    get_start_and_end_of_sessions_df,
)
from lecilab_behavior_analysis.metrics.performance import get_performance_through_trials
from lecilab_behavior_analysis.plots.occupancy import (
    plot_percentage_of_occupancy_per_day,
    plot_training_times_clock_heatmap,
)
from lecilab_behavior_analysis.plots.training import plot_table_from_df
from lecilab_behavior_analysis.reports.tables import generate_tv_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Never raise
# ---------------------------------------------------------------------------
def _plotly_message(title: str, text: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, font=dict(color="firebrick"),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(title_text=title, xaxis_visible=False, yaxis_visible=False, height=200)
    return fig


def _matplotlib_message(title: str, text: str) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, text, color="firebrick", ha="center", va="center", wrap=True)
    ax.set_title(title)
    ax.set_axis_off()
    return fig


def never_raises(title: str, kind: str) -> Callable:
    """
    Decorator: if the function raises, log the error and return a figure
    (kind "plotly" or "matplotlib") with the message instead.
    """

    def decorate(function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as error:
                logger.warning("Report cell '%s' failed: %s: %s", title, type(error).__name__, error)
                text = f"Could not generate {title}<br>{type(error).__name__}: {error}"
                if kind == "plotly":
                    return _plotly_message(title, text)
                return _matplotlib_message(title, text.replace("<br>", "\n"))

        return wrapper

    return decorate


def _correct_numeric(df: pd.DataFrame) -> pd.Series:
    return df["correct"].apply(dfc.correct_as_number)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
@never_raises("TV report", "plotly")
def tv_report_table(events_df: pd.DataFrame, sessions_summary_df: pd.DataFrame, hours: int = 24) -> go.Figure:
    """Detections, sessions, water and weight of each subject in the last hours."""
    report_df, last_date = generate_tv_report(events_df.copy(), sessions_summary_df.copy(), hours)
    return plot_table_from_df(report_df, title=f"TV Report - Last {hours} Hours from {last_date}")


def trials_by_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Number of trials of each subject in each training stage (stages in order of appearance)."""
    stages = df["current_training_stage"].dropna().unique().tolist()
    table = df.groupby(["subject", "current_training_stage"]).size().unstack(fill_value=0).reset_index()
    return table.reindex(columns=["subject"] + stages)


@never_raises("Trials by Training Stage", "plotly")
def trials_by_stage_table(df: pd.DataFrame) -> go.Figure:
    return plot_table_from_df(trials_by_stage(df), title="Trials by Training Stage")


def last_days(df: pd.DataFrame, n_days: int = 3) -> pd.DataFrame:
    """
    Trials and performance of each subject in the last n_days days with data,
    one column per day ("trials_<day>", "performance_<day>").
    """
    df = df.assign(correct_numeric=_correct_numeric(df))
    table = df.groupby(["subject", "year_month_day"]).agg(
        trials=("trial", "count"), performance=("correct_numeric", "mean")
    ).reset_index()
    days = sorted(table["year_month_day"].unique())[-n_days:]
    table = table[table["year_month_day"].isin(days)]
    table["performance"] = table["performance"].map(lambda x: f"{x:.3f}")
    table = table.pivot(index="subject", columns="year_month_day", values=["trials", "performance"])
    table.columns = [f"{name}_{day}" for name, day in table.columns]
    return table.reset_index()


@never_raises("Trials and Performance of last days", "plotly")
def last_days_table(df: pd.DataFrame, n_days: int = 3) -> go.Figure:
    return plot_table_from_df(last_days(df, n_days), title="Trials and Performance of last days")


@never_raises("Panels that could not be generated", "plotly")
def problems_table(problems: Dict[str, List[PanelProblem]]) -> go.Figure:
    """
    The figure panels that failed, e.g. {"NUO003": fig.panel_problems, ...}.
    Shows "All panels were generated" if there are none.
    """
    rows = [
        {"figure": name, "panel": p.panel, "error": p.error, "message": p.message[:200]}
        for name, figure_problems in problems.items()
        for p in figure_problems
    ]
    if not rows:
        return _plotly_message("Panels that could not be generated", "All panels were generated")
    return plot_table_from_df(pd.DataFrame(rows), title="Panels that could not be generated")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
@never_raises("Performance over trials", "plotly")
def performance_over_trials_figure(df: pd.DataFrame, window: int = 50) -> go.Figure:
    """Performance through the trials of every subject (interactive)."""
    if "performance_w" not in df.columns:
        df = pd.concat(
            [get_performance_through_trials(d, window=window) for _, d in df.groupby("subject")],
            ignore_index=True,
        )
    return px.line(df, x="total_trial", y="performance_w", color="subject")


@never_raises("Box occupancy and training times", "matplotlib")
def box_occupancy_figure(df: pd.DataFrame) -> Figure:
    """Percentage of each day the box was used, and the hours of the day it was used."""
    occupancy = get_start_and_end_of_sessions_df(df)
    fig = plt.figure(figsize=(9, 4), layout="constrained")
    ax_days = fig.add_subplot(1, 2, 1)
    ax_clock = fig.add_subplot(1, 2, 2, projection="polar")
    plot_percentage_of_occupancy_per_day(get_daily_occupancy_percentages(occupancy), ax=ax_days)
    plot_training_times_clock_heatmap(get_occupancy_heatmap(occupancy, window_size=30), ax=ax_clock)
    return fig


def daily_trials_and_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Trials, performance and misses of each subject per day and training stage."""
    df = df.assign(correct_numeric=_correct_numeric(df))
    keys = ["subject", "year_month_day", "current_training_stage"]
    table = df.groupby(keys).agg(
        trials=("trial", "size"),
        correct=("correct_numeric", "mean"),
        miss_trial=("miss_trial", "mean"),
    ).reset_index()
    table["current_training_stage"] = table["current_training_stage"].replace({"not saved": "Habituation"})
    table["day"] = pd.to_datetime(table["year_month_day"])
    return table.sort_values("day")


@never_raises("Trials and performance of the cohort", "matplotlib")
def cohort_trials_and_performance_figure(df: pd.DataFrame) -> Figure:
    """Number of trials and performance per day of the whole cohort, by training stage."""
    table = daily_trials_and_performance(df)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
    sns.lineplot(data=table, x="day", y="trials", hue="current_training_stage", ax=axs[0])
    axs[0].set_title("Number of trials per day")
    sns.lineplot(data=table, x="day", y="correct", hue="current_training_stage", ax=axs[1], legend=False)
    axs[1].set_title("Correct percentage per day")
    for ax in axs:
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=45)
        ax.spines[["top", "right"]].set_visible(False)
    if axs[0].get_legend() is not None:
        axs[0].get_legend().set_title("")
    return fig


__all__ = [
    "never_raises",
    "tv_report_table",
    "trials_by_stage",
    "trials_by_stage_table",
    "last_days",
    "last_days_table",
    "problems_table",
    "performance_over_trials_figure",
    "box_occupancy_figure",
    "daily_trials_and_performance",
    "cohort_trials_and_performance_figure",
]

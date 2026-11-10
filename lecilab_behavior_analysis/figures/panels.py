"""
Panels of the subject progress and session summary figures.

Each panel draws one plot from a FigureData (a prepared dataframe plus
options, see render.py). It lists the columns it needs; the data is
prepared once before drawing (preprocessing.prepare), so panels only read
columns and never depend on what the caller ran before.

To add a panel: write a function draw(data, ax) with the @panel decorator,
and put it in a layout (layouts.py). Raise NotApplicable("message") when
there is nothing to show for this data.
"""
import pandas as pd

import lecilab_behavior_analysis.data_format_changes as dfc
import lecilab_behavior_analysis.plots as plots
from lecilab_behavior_analysis.figures.render import FigureData, NotApplicable, panel, split_cell
from lecilab_behavior_analysis.metrics.choices import (
    get_bias_evolution_df,
    get_performance_by_difficulty_ratio,
    points_to_lines_for_bias_evolution,
)
from lecilab_behavior_analysis.metrics.engagement import add_engagement_column, get_box_usage_df
from lecilab_behavior_analysis.metrics.occupancy import get_occupancy_heatmap, get_start_and_end_of_sessions_df
from lecilab_behavior_analysis.metrics.performance import (
    get_performance_by_decision_df,
    get_performance_through_trials,
    get_repeat_or_alternate_performance,
    get_repeat_or_alternate_series,
    get_training_summary_matrix,
)
from lecilab_behavior_analysis.metrics.timing import get_center_hold_df, get_reaction_times_by_date_df
from lecilab_behavior_analysis.metrics.trial_events import (
    add_time_from_session_start,
    side_and_difficulty_to_numeric,
    transform_side_choice_to_numeric,
)
from lecilab_behavior_analysis.metrics.timing import adjust_trials_and_time_of_start_to_first_session


# ---------------------------------------------------------------------------
# Intermediate results shared by several panels (computed once per figure)
# ---------------------------------------------------------------------------
def with_numeric_correct(data: FigureData) -> pd.DataFrame:
    """The trials with correct as 1.0 / 0.0 (timeouts count as errors, see data_format_changes.py)."""
    def compute():
        df = data.df.copy()
        df["correct"] = df["correct"].apply(dfc.correct_as_number)
        return df
    return data.cached("numeric_correct", compute)


def performance_df(data: FigureData) -> pd.DataFrame:
    """Trials sorted by session and trial, with performance_w (rolling window perf_window)."""
    return data.cached(
        "performance",
        lambda: get_performance_through_trials(data.df, window=data.option("perf_window", 50)),
    )


def session_changes(data: FigureData) -> pd.Index:
    """Index (rows of performance_df) of the first trial of each session."""
    def compute():
        df = performance_df(data)
        return df[df.session != df.session.shift(1)].index
    return data.cached("session_changes", compute)


def by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a day column (datetime) from year_month_day."""
    df = df.copy()
    df["day"] = pd.to_datetime(df["year_month_day"])
    return df


def day_limits(data: FigureData):
    """First and last day of the data, so that every "by day" panel shares the x axis."""
    def compute():
        days = pd.to_datetime(pd.Series(data.df["year_month_day"].unique()))
        return days.min() - pd.Timedelta(days=1), days.max() + pd.Timedelta(days=1)
    return data.cached("day_limits", compute)


def _date_axis(data: FigureData, ax) -> None:
    ax.set_xlim(*day_limits(data))
    plots.format_date_axis(ax)


# ---------------------------------------------------------------------------
# Subject progress panels
# ---------------------------------------------------------------------------
@panel("subject summary", needs={"subject", "session", "water", "year_month_day"})
def subject_summary(data: FigureData, ax) -> None:
    plots.summary_text_plot(data.df, kind="subject", ax=ax, fontsize=12)


@panel("calendar plot", needs={"year_month_day"})
def training_calendar(data: FigureData, ax) -> None:
    daily = data.df.groupby("year_month_day").size()
    daily.index = pd.to_datetime(daily.index)
    years = sorted(daily.index.year.unique())
    axes = split_cell(ax, nrows=len(years), ncols=1)
    plots.calendar_plot(daily, axes)
    axes[0].set_title("Trials per day")


@panel("training time plot", needs={"subject", "session", "trial", "TRIAL_START", "TRIAL_END"}, projection="polar")
def training_times(data: FigureData, ax) -> None:
    occupancy = get_start_and_end_of_sessions_df(data.df)
    heatmap = get_occupancy_heatmap(occupancy, window_size=30)
    plots.plot_training_times_clock_heatmap(heatmap, ax=ax)


@panel("trials and water plot", needs={"year_month_day", "session", "current_training_stage", "water"})
def trials_and_water(data: FigureData, ax) -> None:
    df = data.df
    sessions = (
        df.groupby(["year_month_day", "session", "current_training_stage"], sort=False)
        .size().reset_index(name="trials")
    )
    sessions = by_day(sessions)
    plots.trials_by_day_stacked_plot(sessions, ax=ax)
    water = df.groupby("year_month_day")["water"].sum()
    water.index = pd.to_datetime(water.index)
    plots.water_by_day_plot(water.sort_index(), ax)
    _date_axis(data, ax)


@panel("performance plot", needs={"session", "trial", "correct", "subject", "current_training_stage"})
def subject_performance(data: FigureData, ax) -> None:
    df = performance_df(data)
    plots.performance_vs_trials_plot(df, ax=ax, legend=False)
    ax.set_title(f"Performance (window of {data.option('perf_window', 50)} trials)")


@panel("performance-by-decision plot", needs={"correct_side", "correct", "total_trial", "subject"})
def performance_by_decision(data: FigureData, ax) -> None:
    df = performance_df(data).copy()
    df["correct"] = df["correct"].apply(dfc.correct_as_number)
    df = get_repeat_or_alternate_performance(df)
    df = get_performance_by_decision_df(df)
    plots.performance_by_decision_plot(df, ax=ax)
    ax.set_title("Performance by trial type")


@panel("bias evolution plot", needs={"roa_choice_numeric", "total_trial", "subject"})
def bias_evolution(data: FigureData, ax) -> None:
    df = data.df[["subject", "total_trial", "roa_choice_numeric"]].copy()
    trial_group_size = max(df.shape[0] // 40, 1)
    df["trial_group"] = df["total_trial"] // trial_group_size * trial_group_size
    if df["trial_group"].nunique() > 1:
        df = df[df["trial_group"] < df["trial_group"].max() * 0.7]
    evolution = get_bias_evolution_df(df, groupby="trial_group")
    evolution = points_to_lines_for_bias_evolution(evolution, groupby="trial_group")
    plots.plot_decision_evolution_triangle(evolution, ax=ax, hue="trial_group")
    ax.set_title("Bias evolution")


@panel("center hold plot", needs={"year_month_day", "port2_holds"})
def center_holds(data: FigureData, ax) -> None:
    daily = by_day(get_center_hold_df(data.df))
    plots.mean_and_ci_by_day_plot(daily, "number_of_pokes", ax=ax, color="tab:green", label="center pokes per trial")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    plots.mean_and_ci_by_day_plot(daily, "hold_time", ax=ax2, color="tab:red", label="center hold time (s)")
    _date_axis(data, ax)
    ax.set_title("Center port")


@panel("reaction time plot", needs={"year_month_day", "reaction_time", "time_between_trials"})
def reaction_times_by_day(data: FigureData, ax) -> None:
    daily = by_day(get_reaction_times_by_date_df(data.df))
    plots.mean_and_ci_by_day_plot(daily, "reaction_time", ax=ax, color="tab:blue", ylog=True, label="reaction time (s)")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    plots.mean_and_ci_by_day_plot(daily, "time_between_trials", ax=ax2, color="tab:orange", ylog=True,
                                  label="time between trials (s)")
    _date_axis(data, ax)
    ax.set_title("Reaction time and time between trials")


@panel("engagement plot", needs={"date", "subject", "trial_duration", "TRIAL_START", "TRIAL_END", "correct"})
def engagement(data: FigureData, ax) -> None:
    events_df = data.option("events_df")
    if events_df is None:
        raise NotApplicable("No events_df provided")
    df = add_engagement_column(with_numeric_correct(data), engagement_sd_criteria=2)
    usage = get_box_usage_df(df, events_df, verbose=False)
    if usage.empty:
        raise NotApplicable("No session ends found in the events")
    usage["day"] = pd.to_datetime(pd.to_datetime(usage["date"]).dt.strftime("%Y-%m-%d"))
    hue_order = ["engaged_time", "disengaged_time", "time_to_complete_first_trial", "time_to_exit_box"]
    for time_type in hue_order:
        rows = usage[usage["time_type"] == time_type].groupby("day")["percentage_of_time"].mean()
        rows = rows.asfreq("D") if len(rows) else rows  # break the line over days without data
        ax.plot(rows.index, rows.values, label=time_type.replace("_", " "))
    ax.set_ylabel("Percentage of time (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    _date_axis(data, ax)


@panel("summary matrix plot", needs={"session", "difficulty", "correct", "correct_side", "date", "water"})
def summary_matrix(data: FigureData, ax) -> None:
    matrix, info = get_training_summary_matrix(with_numeric_correct(data))
    plots.summary_matrix_plot(mat_df=matrix, mat_df_metadata=info, mouse_name="", ax=ax,
                              training_stage_inlegend=False)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, ncol=1)


# ---------------------------------------------------------------------------
# Session summary panels
# ---------------------------------------------------------------------------
@panel("session summary", needs={"subject", "session", "date", "current_training_stage", "correct", "water"})
def session_summary_text(data: FigureData, ax) -> None:
    plots.summary_text_plot(data.df, kind="session", ax=ax, fontsize=9)


def _performance_hue(data: FigureData) -> str:
    hue = data.option("perf_hue")
    if hue is None:
        hue = "current_training_stage" if data.df["current_training_stage"].nunique() > 1 else "stimulus_modality"
    return hue


@panel("performance plot", needs={"session", "trial", "correct", "subject", "current_training_stage", "stimulus_modality"})
def session_performance(data: FigureData, ax) -> None:
    plots.performance_vs_trials_plot(performance_df(data), ax, legend=True,
                                     session_changes=session_changes(data), hue=_performance_hue(data))
    ax.set_title(f"Performance (window of {data.option('perf_window', 50)} trials)")


@panel("correct responses plot", needs={"correct_side", "correct"})
def correct_responses(data: FigureData, ax) -> None:
    plots.number_of_correct_responses_plot(data.df, ax)


@panel("repeat or alternate plot", needs={"correct_side", "correct", "session", "trial", "total_trial"})
def repeat_or_alternate(data: FigureData, ax) -> None:
    df = performance_df(data).copy()
    df["correct"] = df["correct"].apply(dfc.correct_as_number)
    df["repeat_or_alternate"] = get_repeat_or_alternate_series(df.correct_side)
    df = get_repeat_or_alternate_performance(df, window=data.option("perf_window", 50))
    plots.repeat_or_alternate_performance_plot(df, ax, session_changes=session_changes(data))


def psychometric(modality: str):
    """Panel with the choices on the trials of one modality."""

    @panel(f"{modality} psychometric plot",
           needs={"task", "stimulus_modality", "difficulty", "correct_side", "first_choice"})
    def draw(data: FigureData, ax) -> None:
        trials = data.df[data.df["task"] != "Habituation"]
        if len(trials) == 0:
            raise NotApplicable("Habituation phase")
        if modality not in trials["stimulus_modality"].unique():
            raise NotApplicable(f"No trials in {modality}")
        df = trials[trials["stimulus_modality"] == modality].copy()
        if df["difficulty"].nunique() == 1 and df["difficulty"].unique()[0] == "easy":
            # only easy trials: choices by side and difficulty
            df["side_difficulty"] = df.apply(side_and_difficulty_to_numeric, axis=1)
            df["first_choice_numeric"] = df["first_choice"].apply(transform_side_choice_to_numeric)
            plots.choice_by_difficulty_plot(df, ax=ax)
        else:
            if modality == "visual":
                x, value_type = "visual_stimulus_ratio", "discrete"
            elif modality == "auditory-cot":
                x, value_type = "total_evidence_strength", "continue"
            else:
                raise NotApplicable(f"No psychometric plot for {modality}")
            psych = get_performance_by_difficulty_ratio(df)
            plots.psychometric_plot(psych, x=x, y="first_choice_numeric", ax=ax, valueType=value_type)
        ax.set_title(f"choices on {modality} trials")
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    return draw


@panel("port 2 pokes histogram", needs={"port2_holds"})
def center_pokes_histogram(data: FigureData, ax) -> None:
    plots.plot_number_of_pokes_histogram(data.df, port_number=2, ax=ax)


@panel("port 2 hold time histogram", needs={"port2_holds"})
def center_hold_histogram(data: FigureData, ax) -> None:
    plots.plot_port_holding_time_histogram(data.df, port_number=2, ax=ax)


@panel("trial start time plot", needs={"subject", "session", "trial", "TRIAL_START"})
def trial_start_times(data: FigureData, ax) -> None:
    df = add_time_from_session_start(data.df.copy())
    df = adjust_trials_and_time_of_start_to_first_session(df)
    plots.plot_trial_time_of_start(df, ax=ax, session_changes=session_changes(data))


@panel("reaction time plot", needs={"time_between_trials", "reaction_time", "total_trial"})
def reaction_times_in_session(data: FigureData, ax) -> None:
    axes = split_cell(ax, nrows=1, ncols=3, width_ratios=[1, 6, 1], wspace=0)
    plots.time_between_trials_and_reaction_time_plot(data.df, axes, session_changes=session_changes(data))

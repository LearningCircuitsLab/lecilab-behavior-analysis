import pandas as pd
from lecilab_behavior_analysis.df_transforms import (
    get_dates_df,
    get_water_df,
    get_repeat_or_alternate_series,
    get_performance_through_trials,
    get_repeat_or_alternate_performance,
)
from lecilab_behavior_analysis.plots import (
    rasterized_calendar_plot,
    trials_by_date_plot,
    trials_by_session_hist,
    water_by_date_plot,
    performance_vs_trials_plot,
    session_summary_text,
    correct_left_and_right_plot,
    repeat_or_alternate_performance_plot,
)
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec


def subject_progress_figure(df: pd.DataFrame, title: str, **kwargs) -> Figure:
    """
    Information about the trials done in a session and the water consumption
    """
    # TODO: add the heatmap that I did in postdoc
    
    # create the main figure with GridSpec
    fig = plt.figure(figsize=(10, 6))
    rows_gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
    # Create separate inner grids for each row with different width ratios
    top_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=rows_gs[0])
    med_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=rows_gs[1], width_ratios=[1.5, 3]
    )
    # bot_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=rows_gs[2], width_ratios=[1, 1, 1])
    # Create the top axis spanning both columns
    ax_cal = fig.add_subplot(top_gs[0, 0])
    # Create the medium axes
    ax_bar = fig.add_subplot(med_gs[0, 1])
    # ax_hist = fig.add_subplot(med_gs[0, 1])
    # Create the bottom axis
    ax_perf = fig.add_subplot(med_gs[0, 0])
    # change the width of the ax_perf
    # ax_perf.set_position([0.1, 0.1, 0.2, 0.2])

    # generate the dates dataframe
    dates_df = get_dates_df(df)

    # generate the calendar plot
    cal_image = rasterized_calendar_plot(dates_df)
    # paste the calendar plot
    ax_cal.imshow(cal_image)
    ax_cal.axis("off")

    # do the barplot
    ax_bar = trials_by_date_plot(dates_df, ax=ax_bar)
    # get the legend and move it to the top right, and change the text size
    ax_bar.legend(
        bbox_to_anchor=(1.1, 1.2),
        loc="upper right",
        borderaxespad=0.0,
        fontsize=7,
        ncol=len(dates_df.current_training_stage.unique()),
    )
    # remove box
    ax_bar.get_legend().get_frame().set_linewidth(0.0)

    # Add a vertical histogram
    # ax_hist = trials_by_session_hist(dates_df, ax=ax_hist, ylim=ax_bar.get_ylim())

    # overlay water consumption in the bar plot
    water_df = get_water_df(df)
    ax_bar = water_by_date_plot(water_df, ax=ax_bar)

    # Add the performance vs trials plot
    if "perf_window" in kwargs:
        window = kwargs["perf_window"]
    else:
        window = 50
    df = get_performance_through_trials(df, window=window)
    print(df.columns)
    ax_perf = performance_vs_trials_plot(df, ax=ax_perf, legend=False)

    # Add title within the figure
    fig.suptitle(title, fontsize=12, y=0.90)

    return fig


def session_summary_figure(df: pd.DataFrame, mouse_name: str, **kwargs) -> Figure:
    """
    summary of a particular session
    """
    # if more than one session is there, raise an error
    if df.session.nunique() > 1 or df.date.nunique() > 1:
        raise ValueError("The dataframe contains more than one session")

    # create the main figure with GridSpec
    fig = plt.figure(figsize=(10, 6))
    rows_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    # Create separate inner grids for each row with different width ratios
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=rows_gs[0], width_ratios=[2, 2, 1]
    )
    bot_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=rows_gs[1], width_ratios=[1, 1, 1]
    )

    text_ax = fig.add_subplot(top_gs[0, 0])
    perf_ax = fig.add_subplot(top_gs[0, 1])
    lrc_ax = fig.add_subplot(top_gs[0, 2])
    roap_ax = fig.add_subplot(bot_gs[0, 0])
    # TODO: Response-time by trial (scatter with histogram) For side and trial start
    # TODO: Psychometric if available, separating optogenetic and control if available
    # TODO: Performance by trial with blocks if available

    text_ax = session_summary_text(df, text_ax, mouse_name)
    # Add the performance vs trials plot
    if "perf_window" in kwargs:
        window = kwargs["perf_window"]
    else:
        window = 50
    df = get_performance_through_trials(df, window=window)
    df.columns
    perf_ax = performance_vs_trials_plot(df, perf_ax, legend=False)
    lrc_ax = correct_left_and_right_plot(df, lrc_ax)
    df["repeat_or_alternate"] = get_repeat_or_alternate_series(df.correct_side)
    df = get_repeat_or_alternate_performance(df, window=window)
    roap_ax = repeat_or_alternate_performance_plot(df, roap_ax)

    fig.tight_layout()

    return fig

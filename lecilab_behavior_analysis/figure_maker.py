import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from lecilab_behavior_analysis.df_transforms import (
    get_dates_df, get_performance_by_difficulty,
    get_performance_through_trials, get_repeat_or_alternate_performance,
    get_repeat_or_alternate_series, get_water_df, get_training_summary_matrix, calculate_time_between_trials_and_rection_time)
from lecilab_behavior_analysis.plots import (
    correct_left_and_right_plot, performance_vs_trials_plot, psychometric_plot,
    rasterize_plot, repeat_or_alternate_performance_plot, session_summary_text,
    summary_matrix_plot, training_calendar_plot, trials_by_date_plot,
    trials_by_session_hist, water_by_date_plot, plot_time_between_trials_and_reaction_time)


def subject_progress_figure(df: pd.DataFrame, title: str, **kwargs) -> Figure:
    """
    Information about the trials done in a session and the water consumption
    """
    # TODO: add a plot to show the evolution of weight
    
    # create the main figure with GridSpec
    fig = plt.figure(figsize=(10, 9))
    rows_gs = gridspec.GridSpec(3, 1, height_ratios=[.5, 1, .7])
    # Create separate inner grids for each row with different width ratios
    top_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=rows_gs[0])
    med_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=rows_gs[1], width_ratios=[1.5, 3]
    )
    bot_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=rows_gs[2])
    # Create the top axis spanning both columns
    ax_cal = fig.add_subplot(top_gs[0, 0])
    # Create the medium axes
    ax_bar = fig.add_subplot(med_gs[0, 1])
    # ax_hist = fig.add_subplot(med_gs[0, 1])
    # Create the bottom axis
    ax_perf = fig.add_subplot(med_gs[0, 0])
    # change the width of the ax_perf
    # ax_perf.set_position([0.1, 0.1, 0.2, 0.2])
    ax_summat = fig.add_subplot(bot_gs[0, 0])

    # add a vertical space between the medium and bottom row
    fig.subplots_adjust(hspace=.5)

    # generate the dates dataframe
    dates_df = get_dates_df(df)

    # generate the calendar plot
    cal_image = rasterize_plot(training_calendar_plot(dates_df))
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
    ax_perf = performance_vs_trials_plot(df, ax=ax_perf, legend=False)

    # Summary plot
    summat_df, summat_info = get_training_summary_matrix(df)
    ax_summat = summary_matrix_plot(
        mat_df=summat_df,
        mat_df_metadata=summat_info,
        mouse_name="",
        ax=ax_summat,
        training_stage_inlegend=False,
    )
    # put legend to the right in one column
    ax_summat.legend(
        bbox_to_anchor=(1.35, .9),
        # loc="upper right",
        borderaxespad=0.0,
        fontsize=7,
        ncol=1,
    )

    # Add title within the figure
    fig.suptitle(title, fontsize=12, y=0.90)

    # fig.tight_layout()

    return fig


def session_summary_figure(df: pd.DataFrame, mouse_name: str = "", **kwargs) -> plt.Figure:
    """
    Summary of a particular session.
    """
    # if more than one session is there, raise an error
    if df.date.nunique() > 1:
        raise ValueError("The dataframe contains more than one session")

    # create the main figure with GridSpec
    width = kwargs.get("width", 10)
    height = kwargs.get("height", 6)
    fig = plt.figure(figsize=(width, height))
    rows_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    # Create separate inner grids for each row with different width ratios
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=rows_gs[0], width_ratios=[2, 2, 1]
    )
    bot_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=rows_gs[1], width_ratios=[1, 1, 2]
    )

    text_ax = fig.add_subplot(top_gs[0, 0])
    perf_ax = fig.add_subplot(top_gs[0, 1])
    lrc_ax = fig.add_subplot(top_gs[0, 2])
    roap_ax = fig.add_subplot(bot_gs[0, 0])
    psych_ax = fig.add_subplot(bot_gs[0, 1])

    # TODO: Psychometric with actual values and fit
    # TODO: separate optogenetic and control if available in several plots
    # TODO: Performance by trial with blocks if available

    text_ax = session_summary_text(df, text_ax, mouse_name)
    # Add the performance vs trials plot
    window = kwargs.get("perf_window", 50)
    df = get_performance_through_trials(df, window=window)
    perf_ax = performance_vs_trials_plot(df, perf_ax, legend=False)
    lrc_ax = correct_left_and_right_plot(df, lrc_ax)
    df["repeat_or_alternate"] = get_repeat_or_alternate_series(df.correct_side)
    df = get_repeat_or_alternate_performance(df, window=window)
    roap_ax = repeat_or_alternate_performance_plot(df, roap_ax)
    psych_df = get_performance_by_difficulty(df)
    psych_ax = psychometric_plot(psych_df, psych_ax)
    df = calculate_time_between_trials_and_rection_time(df, window=window)
    fig.tight_layout()
    reaction_time_image = rasterize_plot(plot_time_between_trials_and_reaction_time(df), dpi=600)
    # Manually place the image using add_axes with figure-relative coordinates
    # Coordinates in (x0, y0, width, height) format, relative to the figure
    x0, y0 = 0.53, 0.015  # Positioning the image in the bottom-right corner
    img_width = 0.5  # Image width (50% of the figure width)
    img_height = 0.5  # Image height (50% of the figure height)

    # Add an axes to the figure where the image will be placed
    ax_image = fig.add_axes([x0, y0, img_width, img_height])
    
    # Place the rasterized image within the defined axes
    ax_image.imshow(reaction_time_image, aspect='auto')
    
    # Turn off the axis for clean presentation
    ax_image.axis("off")
    return fig
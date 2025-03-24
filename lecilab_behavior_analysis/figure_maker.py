import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import lecilab_behavior_analysis.df_transforms as dft
import lecilab_behavior_analysis.plots as plots
import lecilab_behavior_analysis.utils as utils

def subject_progress_figure(df: pd.DataFrame, title: str, **kwargs) -> Figure:
    """
    Information about the trials done in a session and the water consumption
    """
    # TODO: add a plot to show the evolution of weight
    # TODO: add a "day plot" with the favourite entries in the day
    
    # create the main figure with GridSpec
    width = kwargs.get("width", 10)
    height = kwargs.get("height", 6)
    summary_matrix_plot_requested = kwargs.get("summary_matrix_plot", False)
    fig = plt.figure(figsize=(width, height))
    # Create a GridSpec with 3 rows and 1 column
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
    if summary_matrix_plot_requested:
        ax_summat = fig.add_subplot(bot_gs[0, 0])

    # add a vertical space between the medium and bottom row
    fig.subplots_adjust(hspace=.5)

    # fill information in df if it is missing
    df = dft.fill_missing_data(df)

    # add a column with the date for the day
    df = dft.add_day_column_to_df(df)

    # generate the calendar plot
    dates_df = dft.get_dates_df(df)
    cal_image = plots.rasterize_plot(plots.training_calendar_plot(dates_df))
    # paste the calendar plot
    ax_cal.imshow(cal_image)
    ax_cal.axis("off")

    # do the barplot
    bp_df = df.groupby(["year_month_day", "date", "current_training_stage"]).count()
    # pivot the dataframe so that the individual "date" indexes are stacked
    bp_df = bp_df.pivot_table(index="year_month_day", columns=["date", "current_training_stage"], values="trial")
    # Plot the stacked barplot, coloring by current_training_stage
    ax_bar = plots.trials_by_day_plot(bp_df, ax=ax_bar, cmap="tab10")
    # ax_bar = plots.trials_by_session_plot(dates_df, ax=ax_bar)
    # get the legend and move it to the top right, and change the text size
    # ax_bar.legend(
    #     bbox_to_anchor=(1.1, 1.2),
    #     loc="upper right",
    #     borderaxespad=0.0,
    #     fontsize=7,
    #     ncol=len(dates_df.current_training_stage.unique()),
    # )
    # remove box
    ax_bar.get_legend().get_frame().set_linewidth(0.0)

    # Add a vertical histogram
    # ax_hist = trials_by_session_hist(dates_df, ax=ax_hist, ylim=ax_bar.get_ylim())

    # overlay water consumption in the bar plot
    water_df = dft.get_water_df(df, grouping_column="year_month_day")
    ax_bar = plots.water_by_date_plot(water_df, ax=ax_bar)

    # Add the performance vs trials plot
    if "perf_window" in kwargs:
        window = kwargs["perf_window"]
    else:
        window = 50
    df = dft.get_performance_through_trials(df, window=window)
    ax_perf = plots.performance_vs_trials_plot(df, ax=ax_perf, legend=False)

    # Summary plot
    if summary_matrix_plot_requested:
        summat_df, summat_info = dft.get_training_summary_matrix(df)
        ax_summat = plots.summary_matrix_plot(
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

    return fig


def session_summary_figure(df: pd.DataFrame, mouse_name: str = "", **kwargs) -> plt.Figure:
    """
    Summary of a particular session.
    """
    # if more than one session is there, raise an error
    # if df.date.nunique() > 1:
    #     raise ValueError("The dataframe contains more than one session")

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
    # reaction_time_ax = fig.add_subplot(bot_gs[0, 2])
    fig.tight_layout()
    # original_pos = reaction_time_ax.get_position()
    #Altering the bottom right subplot bbox to remove the padding between subplots and the figure border to adapt the rasterized image later that already includes axis
    #The default pading = 0.2 and so the pading between subplots is 0.05
    # reaction_time_ax.set_position(pos=[original_pos.x0-0.05, original_pos.y0-0.075, original_pos.width+0.05, original_pos.height+0.075])
    
    # TODO: Psychometric with actual values and fit
    # TODO: separate optogenetic and control if available in several plots
    # TODO: Performance by trial with blocks if available

    text_ax = plots.session_summary_text(df, text_ax, mouse_name)
    # Add the performance vs trials plot
    window = kwargs.get("perf_window", 50)
    df = dft.get_performance_through_trials(df, window=window)
    perf_ax = plots.performance_vs_trials_plot(df, perf_ax, legend=False)
    lrc_ax = plots.correct_left_and_right_plot(df, lrc_ax)
    df["repeat_or_alternate"] = dft.get_repeat_or_alternate_series(df.correct_side)
    df = dft.get_repeat_or_alternate_performance(df, window=window)
    roap_ax = plots.repeat_or_alternate_performance_plot(df, roap_ax)
    psych_df = dft.get_performance_by_difficulty(df)
    psych_ax = plots.psychometric_plot(psych_df, psych_ax)

    # df = dft.calculate_time_between_trials_and_reaction_time(df, window=window)
    # reaction_time_image = plots.rasterize_plot(plots.plot_time_between_trials_and_reaction_time(df), dpi=300)
    # reaction_time_ax.imshow(reaction_time_image, aspect='auto')
    # # Turn off the axis for clean presentation
    # reaction_time_ax.axis("off")

    return fig
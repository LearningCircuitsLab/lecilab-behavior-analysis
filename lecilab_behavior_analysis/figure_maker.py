import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import lecilab_behavior_analysis.df_transforms as dft
import lecilab_behavior_analysis.plots as plots
import lecilab_behavior_analysis.utils as utils

def subject_progress_figure(df: pd.DataFrame, **kwargs) -> Figure:
    """
    Information about the trials done in a session and the water consumption
    """
    # TODO: add a plot to show the evolution of weight

    # make sure there is only one subject in the dataframe
    if df.subject.nunique() > 1:
        raise ValueError(
            f"The dataframe contains more than one subject: {df.subject.unique()}."
        )
    
    # create the main figure with GridSpec
    width = kwargs.get("width", 15)
    height = kwargs.get("height", 10)
    summary_matrix_plot_requested = kwargs.get("summary_matrix_plot", False)
    fig = plt.figure(figsize=(width, height))
    # Create a GridSpec with 3 rows and 1 column
    rows_gs = gridspec.GridSpec(4, 1, height_ratios=[.7, 1, 1, 1])
    # Create separate inner grids for each row with different width ratios
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=rows_gs[0], width_ratios=[1, 3, 1])
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=rows_gs[1], width_ratios=[1.5, 3])
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=rows_gs[2], width_ratios=[1.5, 1, 2, 1])
    gs4 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=rows_gs[3])
    # Create the top axis
    ax_name = fig.add_subplot(gs1[0, 0])
    ax_cal = fig.add_subplot(gs1[0, 1])
    ax_training_time = fig.add_subplot(gs1[0, 2])
    # Create the medium axes
    ax_bar = fig.add_subplot(gs2[0, 1])
    # ax_hist = fig.add_subplot(med_gs[0, 1])
    # Create the bottom axis
    ax_perf = fig.add_subplot(gs2[0, 0])
    # change the width of the ax_perf
    # ax_perf.set_position([0.1, 0.1, 0.2, 0.2])
    # performance by decision
    ax_pbd = fig.add_subplot(gs3[0, 0])
    # evolution of the bias
    ax_bias = fig.add_subplot(gs3[0, 1])
    # holding time in center port
    ax_htime = fig.add_subplot(gs3[0, 2])

    # summary plot if requested
    if summary_matrix_plot_requested:
        ax_summat = fig.add_subplot(gs4[0, 0])

    # add a vertical space between the medium and bottom row
    # fig.subplots_adjust(hspace=.5)

    # fill information in df if it is missing
    df = dft.fill_missing_data(df)

    # add a column with the date for the day
    df = dft.add_day_column_to_df(df)

    # write the subject name and some info
    ax_name = plots.summary_text_plot(df, kind="subject", ax=ax_name, fontsize=15)

    # generate the calendar plot
    dates_df = dft.get_dates_df(df)
    cal_image = plots.rasterize_plot(plots.training_calendar_plot(dates_df), dpi=600)
    # paste the calendar plot filling the entire axis
    ax_cal.imshow(cal_image)
    ax_cal.axis("off")

    # add the training time plot
    # Plot the percentage of time that the task is running per day and the heatmap of the occupancy during the day
    occupancy_df = dft.get_start_and_end_of_sessions_df(df)
    occupancy_heatmap = dft.get_occupancy_heatmap(occupancy_df, window_size=30)
    fig.delaxes(ax_training_time)  # Remove the default second subplot
    ax_training_time = fig.add_subplot(gs1[0, 2], projection='polar')  # Add a polar subplot
    plots.plot_training_times_clock_heatmap(occupancy_heatmap, ax=ax_training_time)

    # do the barplot
    bp_df = df.groupby(["year_month_day", "date", "current_training_stage"]).count()
    # pivot the dataframe so that the individual "date" indexes are stacked
    bp_df = bp_df.pivot_table(index="year_month_day", columns=["date", "current_training_stage"], values="trial")
    # Plot the stacked barplot, coloring by current_training_stage
    ax_bar = plots.trials_by_day_plot(bp_df, ax=ax_bar, cmap="tab10")

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

    df = dft.get_repeat_or_alternate_performance(df)
    df_pbd = dft.get_performance_by_decision_df(df)
    ax_pbd = plots.performance_by_decision_plot(df_pbd, ax=ax_pbd)

    # add the bias plot showing 20 steps
    # get what the animal is doing, if it is alternating or repeating to the left or to the right
    df = dft.add_mouse_first_choice(df)
    df = dft.add_mouse_last_choice(df)
    df = dft.add_port_where_animal_comes_from(df)
    # get a metric to see the bias in choices (including alternation)
    df['roa_choice_numeric'] = df.apply(utils.get_repeat_or_alternate_to_numeric, axis=1)
    # evolution of bias by trial group
    trial_group_size = df.shape[0] // 40
    df['trial_group'] = df['total_trial'] // trial_group_size * trial_group_size
    # remove the last group if it is smaller than the others
    if df['trial_group'].nunique() > 1:
        df = df[df['trial_group'] < df['trial_group'].max() * 0.7]
    df_bias_evolution = dft.get_bias_evolution_df(df, groupby='trial_group')
    df_bias_evolution = dft.points_to_lines_for_bias_evolution(df_bias_evolution, groupby='trial_group')
    ax_bias = plots.plot_decision_evolution_triangle(df_bias_evolution, ax=ax_bias, hue='trial_group')

    # holding time in the center port
    df_ch = dft.get_center_hold_df(df)
    ax_htime = plots.plot_poke_number_by_date(df_ch, ax=ax_htime, color='tab:green')
    ax_htime = plots.plot_poke_speed_by_date(df_ch, ax=ax_htime, color='tab:red')


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

    plt.tight_layout()

    return fig


def session_summary_figure(df: pd.DataFrame, **kwargs) -> plt.Figure:
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
    rows_gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
    # Create separate inner grids for each row with different width ratios
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=rows_gs[0], width_ratios=[2, 2, 1]
    )
    mid_gs = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=rows_gs[1], width_ratios=[1, 1, 1, 1]
    )
    bot_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=rows_gs[2], width_ratios=[1, 1]
    )

    text_ax = fig.add_subplot(top_gs[0, 0])
    perf_ax = fig.add_subplot(top_gs[0, 1])
    lrc_ax = fig.add_subplot(top_gs[0, 2])
    roap_ax = fig.add_subplot(mid_gs[0, 0])
    psych_ax = fig.add_subplot(mid_gs[0, 1])
    p2hn_ax = fig.add_subplot(mid_gs[0, 2])
    p2ht_ax = fig.add_subplot(mid_gs[0, 3])
    # reaction_time_ax = fig.add_subplot(mid_gs[0, 2])
    bias_ax = fig.add_subplot(bot_gs[0, 0])

    # fig.tight_layout()
    # original_pos = reaction_time_ax.get_position()
    #Altering the bottom right subplot bbox to remove the padding between subplots and the figure border to adapt the rasterized image later that already includes axis
    #The default pading = 0.2 and so the pading between subplots is 0.05
    # reaction_time_ax.set_position(pos=[original_pos.x0-0.05, original_pos.y0-0.075, original_pos.width+0.05, original_pos.height+0.075])
    
    # TODO: Psychometric with actual values and fit
    # TODO: separate optogenetic and control if available in several plots
    # TODO: Performance by trial with blocks if available

    text_ax = plots.summary_text_plot(df, kind="session", ax=text_ax)
    # Add the performance vs trials plot
    window = kwargs.get("perf_window", 50)
    df = dft.get_performance_through_trials(df, window=window)
    # find the index of the session changes
    session_changes = df[df.session != df.session.shift(1)].index
    # add a vertical line to the performance plot
    perf_ax = plots.performance_vs_trials_plot(df, perf_ax, legend=False, session_changes=session_changes)
    lrc_ax = plots.correct_left_and_right_plot(df, lrc_ax)
    df["repeat_or_alternate"] = dft.get_repeat_or_alternate_series(df.correct_side)
    df = dft.get_repeat_or_alternate_performance(df, window=window)
    roap_ax = plots.repeat_or_alternate_performance_plot(df, roap_ax)
    psych_df = dft.get_performance_by_difficulty(df)
    psych_ax = plots.psychometric_plot(psych_df, psych_ax)
    df["port2_holds"] = df.apply(lambda row: utils.get_trial_port_hold(row, 2), axis=1)
    p2hn_ax = plots.plot_number_of_pokes_histogram(df, port_number=2, ax=p2hn_ax)
    p2ht_ax = plots.plot_port_holding_time_histogram(df, port_number=2, ax=p2ht_ax)


    # df = dft.calculate_time_between_trials_and_reaction_time(df, window=window)
    # reaction_time_image = plots.rasterize_plot(plots.plot_time_between_trials_and_reaction_time(df), dpi=300)
    # reaction_time_ax.imshow(reaction_time_image, aspect='auto')
    # # Turn off the axis for clean presentation
    # reaction_time_ax.axis("off")

    # add the bias plot
    # get the first choice of the mouse
    df = dft.add_mouse_first_choice(df)
    # is it repeating or alternating?
    df["roa_choice"] = dft.get_repeat_or_alternate_series(df.first_choice)
    # turn bias into a number (-1 for left, 1 for right, 0 for alternating)
    df['bias'] = df.apply(utils.get_repeat_or_alternate_to_numeric, axis=1)
    bias_ax = plots.bias_vs_trials_plot(df, bias_ax)

    fig.tight_layout()

    return fig
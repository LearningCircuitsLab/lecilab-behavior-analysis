import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go

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
    rows_gs = gridspec.GridSpec(5, 1, height_ratios=[.7, 1, 1, 1, 1])
    # Create separate inner grids for each row with different width ratios
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=rows_gs[0], width_ratios=[1, 3, 1])
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=rows_gs[1], width_ratios=[1.5, 3])
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=rows_gs[2], width_ratios=[1.5, 1, 2, 1])
    gs4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=rows_gs[3], width_ratios=[1, 1])
    gs5 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=rows_gs[4])
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
    # reaction time
    ax_rt = fig.add_subplot(gs4[0, 0])
    # engagement plot
    ax_eng = fig.add_subplot(gs4[0, 1])

    # summary plot if requested
    if summary_matrix_plot_requested:
        ax_summat = fig.add_subplot(gs5[0, 0])

    # add a vertical space between the medium and bottom row
    # fig.subplots_adjust(hspace=.5)

    # fill information in df if it is missing
    df = dft.fill_missing_data(df)

    # add a column with the date for the day
    df = dft.add_day_column_to_df(df)

    # write the subject name and some info
    ax_name = plots.summary_text_plot(df, kind="subject", ax=ax_name, fontsize=15)

    # if the dataframe is empty, return the figure with the text
    if df.empty:
        return fig

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
    # reduce the number of x ticks to a maximum of 16
    ax_bar.xaxis.set_major_locator(plt.MaxNLocator(16))

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
        df_evol = df[df['trial_group'] < df['trial_group'].max() * 0.7]
    else:
        df_evol = df.copy()
    df_bias_evolution = dft.get_bias_evolution_df(df_evol, groupby='trial_group')
    df_bias_evolution = dft.points_to_lines_for_bias_evolution(df_bias_evolution, groupby='trial_group')
    ax_bias = plots.plot_decision_evolution_triangle(df_bias_evolution, ax=ax_bias, hue='trial_group')

    # holding time in the center port
    df_ch = dft.get_center_hold_df(df)
    ax_htime = plots.plot_mean_and_cis_by_date(df_ch, item_to_show="number_of_pokes", group_trials_by="year_month_day", ax=ax_htime, color='tab:green')
    ax_htime = plots.plot_mean_and_cis_by_date(df_ch, item_to_show="hold_time", group_trials_by="year_month_day", ax=ax_htime, color='tab:red')
    # remove x label
    ax_htime.set_xlabel("")
    # reduce the number of x ticks to a maximum of 16
    ax_htime.xaxis.set_major_locator(plt.MaxNLocator(16))

    # add the reaction time plot
    rt_df = dft.get_reaction_times_by_date_df(df)
    ax_rt = plots.plot_mean_and_cis_by_date(rt_df, item_to_show="reaction_time", group_trials_by="year_month_day", ax=ax_rt, color='tab:blue', ylog=True)
    ax_rt = plots.plot_mean_and_cis_by_date(rt_df, item_to_show="time_between_trials", group_trials_by="year_month_day", ax=ax_rt, color='tab:orange', ylog=True)
    ax_rt.set_xlabel("")
    # reduce the number of x ticks to a maximum of 16
    ax_rt.xaxis.set_major_locator(plt.MaxNLocator(16))
    
    # add the engagement plot. Make sure that events_df is in kwargs
    if "events_df" not in kwargs:
        # print the warning in the plot
        ax_eng.text(0.5, 0.5, "No events_df provided", fontsize=12, color='red', ha='center', va='center')
    else:
        events_df = kwargs["events_df"]
        df = dft.add_trial_duration_column_to_df(df)
        df = dft.add_engagement_column(df, engagement_sd_criteria=2)
        sbu_df = dft.get_box_usage_df(df, events_df, verbose=False)
        sbu_df = dft.add_day_column_to_df(sbu_df)
        ax_eng = plots.plot_box_usage_by_date(sbu_df, ax=ax_eng)
        ax_eng.set_xlabel("")
        ax_eng.xaxis.set_major_locator(plt.MaxNLocator(16))
    
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
    width = kwargs.get("width", 15)
    height = kwargs.get("height", 10)
    fig = plt.figure(figsize=(width, height))
    rows_gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
    # Create separate inner grids for each row with different width ratios
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=rows_gs[0], width_ratios=[2, 2, 1]
    )
    mid_gs = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=rows_gs[1], width_ratios=[1, 1, 1, 1, 1]
    )
    bot_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=rows_gs[2], width_ratios=[1, 1]
    )

    text_ax = fig.add_subplot(top_gs[0, 0])
    perf_ax = fig.add_subplot(top_gs[0, 1])
    lrc_ax = fig.add_subplot(top_gs[0, 2])
    roap_ax = fig.add_subplot(mid_gs[0, 0])
    # psych_ax = fig.add_subplot(mid_gs[0, 1])
    visual_psych_by_difficulty_ratio_ax = fig.add_subplot(mid_gs[0, 1])
    auditory_psych_by_difficulty_ratio_ax = fig.add_subplot(mid_gs[0, 2])
    p2hn_ax = fig.add_subplot(mid_gs[0, 3])
    p2ht_ax = fig.add_subplot(mid_gs[0, 4])
    tst_ax = fig.add_subplot(bot_gs[0, 0])
    ax_rt = fig.add_subplot(bot_gs[0, 1])


    # fig.tight_layout()
    # original_pos = reaction_time_ax.get_position()
    #Altering the bottom right subplot bbox to remove the padding between subplots and the figure border to adapt the rasterized image later that already includes axis
    #The default pading = 0.2 and so the pading between subplots is 0.05
    # reaction_time_ax.set_position(pos=[original_pos.x0-0.05, original_pos.y0-0.075, original_pos.width+0.05, original_pos.height+0.075])
    
    # TODO: separate optogenetic and control if available in several plots
    # TODO: Performance by trial with blocks if available

    text_ax = plots.summary_text_plot(df, kind="session", ax=text_ax)
    # if the dataframe is empty, return the figure with the text
    if df.empty:
        return fig
    # Add the performance vs trials plot
    window = kwargs.get("perf_window", 50)
    df = dft.get_performance_through_trials(df, window=window)
    # find the index of the session changes and add as vertical lines to the performance plot
    session_changes = df[df.session != df.session.shift(1)].index
    # define the hue based on stimulus_modality or current_training_stage if there are more than one
    perf_hue = kwargs.get("perf_hue", None)
    if perf_hue is None:
        if df.current_training_stage.nunique() > 1:
            perf_hue = "current_training_stage"
        else:
            perf_hue = "stimulus_modality"
    perf_ax = plots.performance_vs_trials_plot(df,
                                               perf_ax,
                                               legend=True,
                                               session_changes=session_changes,
                                               hue=perf_hue)
    lrc_ax = plots.number_of_correct_responses_plot(df, lrc_ax)
    df["repeat_or_alternate"] = dft.get_repeat_or_alternate_series(df.correct_side)
    df = dft.get_repeat_or_alternate_performance(df, window=window)
    roap_ax = plots.repeat_or_alternate_performance_plot(df, roap_ax, session_changes=session_changes)
    # psych_df = dft.get_performance_by_difficulty(df)
    # psych_ax = plots.psychometric_plot(psych_df, psych_ax)


    # TODO: For Nuo:
    """
    What we need is for these functions (e.g. session_summary_figure) to be able to take any set of trials
    and work regardless of what they contain. For instance, your changes work now if within the trials there are
    some that correspond to "current_training_stage" == "TwoAFC_visual_hard", but if there are no such trials,
    the psychometric plot will not be generated (try running plot_testing.ipynb). We need to make sure that the function can handle
    all possible cases. These are:
    - All trials are of one modality (visual and auditory) and all are easy (no need for psychometric plot)
    - All trials are of one modality (visual and auditory) and all are hard (psychometric plot)
    - All trials are of one modality (visual and auditory) and some are easy and some are hard (psychometric plot, but including all training stages)
        ***Thinking about this, it would be very nice to overlay a trial count (a histogram or something like that) on top of the psychometric plot.
            These plots can be tricky as they require to make a separate figure, and then rasterize it to paste it in the main figure. Leave this for later.
    - Same as above but we have mixed modalities. In this case, we need to generate two psychometric plots, one for each modality.

    The trick here is to think of a robust logic to do all this without errors. 
    
    """

    df_task = df[df['task'] != 'Habituation']
    for mod in ['visual', 'auditory']:
        ax_name = eval(mod + '_psych_by_difficulty_ratio_ax')
        if len(df_task) > 0:
            if mod in df_task['stimulus_modality'].unique():
                df_mod = df_task[df_task["stimulus_modality"] == mod].copy()
                if df_mod['difficulty'].nunique() == 1 and df_mod['difficulty'].unique()[0] == 'easy':
                        # if all trials are easy, do the normal plot by difficulty
                        df_mod["side_difficulty"] = df_mod.apply(lambda row: utils.side_and_difficulty_to_numeric(row), axis=1)
                        df_mod = dft.add_mouse_first_choice(df_mod)
                        df_mod['first_choice_numeric'] = df_mod['first_choice'].apply(utils.transform_side_choice_to_numeric)
                        plots.choice_by_difficulty_plot(df_mod, ax=ax_name)
                else:
                    if mod == 'visual':
                        xvar = 'visual_stimulus_ratio'
                        value_type = 'discrete'
                    elif mod == 'auditory':
                        xvar = 'total_evidence_strength'
                        value_type = 'continue'
                    psych_df = dft.get_performance_by_difficulty_ratio(df_mod)
                    plots.psychometric_plot(psych_df, x = xvar, y = 'first_choice_numeric', ax = ax_name, valueType=value_type)
                # set the title of the axis
                ax_name.set_title("choices on " + mod + " trials", fontsize=10)
                # remove legend if it exists
                if ax_name.get_legend() is not None:
                    ax_name.get_legend().remove()
            else:
                ax_name.text(0.1, 0.5, "No trials in " + mod, fontsize=10, color='k')
        else:
            ax_name.text(0.1, 0.5, "Habituation phase", fontsize=10, color='k')


    # for mod in ['visual', 'auditory']:
    #     ax_name = eval(mod + '_psych_by_difficulty_ratio_ax')
    #     if mod in df['stimulus_modality'].unique():
    #         df_mod = df[df["stimulus_modality"] == mod]
    #         if 'hard' in df_mod["difficulty"].unique():
    #             df_mod_hard = df_mod[df_mod["difficulty"] == 'hard']
    #             psych_df = dft.get_performance_by_difficulty_ratio(df_mod_hard)
    #             if mod == 'visual':
    #                 plots.psychometric_plot(psych_df, x = 'visual_stimulus_ratio', y = 'first_choice_numeric', ax = ax_name)
    #             elif mod == 'auditory':
    #                 plots.psychometric_plot(psych_df, x = 'total_evidence_strength', y = 'first_choice_numeric', ax = ax_name, valueType='continue')
    #             ax_name.set_title(mod + " psychometric plot", fontsize=10)
    #         else:
    #              ax_name.text(0.1, 0.5, "No hard trials in " + mod, fontsize=10, color='k')
    #     else:
    #         ax_name.text(0.1, 0.5, "No trials in " + mod, fontsize=10, color='k')

    df["port2_holds"] = df.apply(lambda row: utils.get_trial_port_hold(row, 2), axis=1)
    p2hn_ax = plots.plot_number_of_pokes_histogram(df, port_number=2, ax=p2hn_ax)
    p2ht_ax = plots.plot_port_holding_time_histogram(df, port_number=2, ax=p2ht_ax)

    # TODO: think about the best way to represent the bias
    # # add the bias plot
    # # get the first choice of the mouse
    # df = dft.add_mouse_first_choice(df)
    # # is it repeating or alternating?
    # df["roa_choice"] = dft.get_repeat_or_alternate_series(df.first_choice)
    # # turn bias into a number (-1 for left, 1 for right, 0 for alternating)
    # df['bias'] = df.apply(utils.get_repeat_or_alternate_to_numeric, axis=1)
    # bias_ax = plots.bias_vs_trials_plot(df, bias_ax)

    # add the speed of doing trials
    df = utils.add_time_from_session_start(df)
    sdf = dft.adjust_trials_and_time_of_start_to_first_session(df)
    tst_ax = plots.plot_trial_time_of_start(sdf, ax=tst_ax, session_changes=session_changes)

    df = dft.calculate_time_between_trials_and_reaction_time(df)
    reaction_time_image = plots.rasterize_plot(plots.plot_time_between_trials_and_reaction_time(df), dpi=300)
    ax_rt.imshow(reaction_time_image, aspect='auto')
    # Turn off the axis for clean presentation
    ax_rt.axis("off")

    fig.tight_layout()

    return fig


def make_temperature_humidity_figure(df):
    fig = go.Figure()

    # Temperature (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["temperature"],
            mode="lines",
            name="Temperature",
            line=dict(width=2, color="#1f77b4")  # blue
        )
    )

    # Humidity (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["humidity"],
            mode="lines",
            name="Humidity",
            line=dict(width=2, color="#ff7f0e"),  # orange
            yaxis="y2"
        )
    )

    fig.update_layout(
        title="Temperature and Humidity in the Past Week",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Temperature", range=[20, 24]),
        yaxis2=dict(
            title="Humidity",
            overlaying="y",
            side="right",
            range=[20, 90],
        ),
        template="plotly_white",
    )

    return fig

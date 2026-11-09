"""Plots of timing: reaction times, center port holds, trial start times, box usage."""
from typing import List

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lecilab_behavior_analysis.common import column_checker


def plot_time_between_trials_and_reaction_time(df_in: pd.DataFrame, **kwargs) -> plt.figure:
    """
    Plot Time Between Trials (TBT) and Reaction Time (RT) on the same plot with histograms on the y-axes.
    """
    # Check if the dataframe has the necessary columns
    column_checker(df_in, required_columns={"time_between_trials", "reaction_time", "total_trial"})

    # Create a copy of the dataframe to avoid modifying the original
    df = df_in.copy()
    tts = []
    if "session_changes" in kwargs and len(kwargs["session_changes"]) > 1:
        for sc in kwargs["session_changes"][1:]:
            tts.append(df.loc[sc]["total_trial"])
    # Drop NaN or Inf values to avoid errors in plotting
    df = df.dropna(subset=['reaction_time', 'time_between_trials'])
    df = df[(df['reaction_time'] != float('inf')) & (df['time_between_trials'] != float('inf'))]
    
    # Create a 1x3 grid layout for two histograms and the main plot in the center
    fig = plt.figure(figsize=(14, 8))  # Increased figure size for better visibility
    grid = plt.GridSpec(1, 3, width_ratios=[0.5, 5, 0.5], wspace=0.0)  # Adjust width ratios and wspace to align the plots

    # Right plot for Reaction Time (RT) KDE plot
    ax_right = fig.add_subplot(grid[0, 2])
    if df['reaction_time'].nunique() > 1:
        sns.kdeplot(y=df['reaction_time'], ax=ax_right, color="tab:orange", fill=True)
    else:
        sns.histplot(y=df['reaction_time'], ax=ax_right, color="tab:orange")

    ax_right.invert_xaxis()  # Flip the x-axis to make the plot point towards the main plot
    ax_right.set_xlabel("")
    ax_right.yaxis.set_label_position('right')  # Force y-label on the right
    ax_right.set_ylabel("Reaction times (s)", rotation=270, fontsize=28, labelpad=30)

    ax_right.tick_params(right=True, left=False, labelleft=False, labelright=True, bottom=False, labelbottom=False, labelsize=25, width=2)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['bottom'].set_linewidth(2)

    # Center plot for the main Time Between Trials and Reaction Time plot
    ax_center = fig.add_subplot(grid[0, 1])
    ax2_center = ax_center.twinx()

    # Plot Time Between Trials
    ax_center.plot(df["total_trial"], df["time_between_trials"], color="tab:blue", label="Time Between Trials")
    ax_center.set_xlabel("Trial number", fontsize=28, labelpad=10)
    ax_center.tick_params(labelsize=28, width=2, length=10)
    for tt in tts:
        ax_center.axvline(tt, linestyle="--", color="gray")

    # Plot Reaction Time
    ax2_center.plot(df["total_trial"], df["reaction_time"], color="tab:orange", label="Reaction Time")
    ax2_center.set_ylabel("")
    ax2_center.tick_params(left=False, right=False, top=False, bottom=False, labelleft=False, labelright=False, labeltop=False, labelbottom=False, labelsize=20)    

    # Add legends
    handles1, labels1 = ax_center.get_legend_handles_labels()
    handles2, labels2 = ax2_center.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = ["TBT", "RT"]
    ax_center.legend(handles, labels, bbox_to_anchor=(0.5, 1.05), loc="upper center", ncol=2, borderaxespad=0.0, frameon=False, fontsize=20)

    # Remove lateral and topspines for the center plot
    ax_center.spines["top"].set_visible(False)
    ax_center.spines["right"].set_visible(False)
    ax_center.spines["left"].set_visible(False)
    ax_center.spines["bottom"].set_linewidth(2)
    ax2_center.spines["top"].set_visible(False)
    ax2_center.spines["right"].set_visible(False)
    ax2_center.spines["left"].set_visible(False)

    # Left plot for Time Between Trials (TBT) KDE plot
    ax_left = fig.add_subplot(grid[0, 0])
    if df['time_between_trials'].nunique() > 1:
        sns.kdeplot(y=df['time_between_trials'], ax=ax_left, color="tab:blue", fill=True)
    else:
        sns.histplot(y=df['time_between_trials'], ax=ax_left, color="tab:blue")

    ax_left.set_xlabel("")
    ax_left.set_ylabel("Time between trials (s)", fontsize=28)
    ax_left.spines['left'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['bottom'].set_linewidth(2)
    ax_left.tick_params(left=True, right=False, labelright=False, labelleft=True, bottom=False, labelbottom=False, labelsize=25, width=2)

    # Make all y lower limits to 0
    ax_left.set_ylim(bottom=0)
    ax_center.set_ylim(bottom=0)
    ax_right.set_ylim(bottom=0)
    ax2_center.set_ylim(bottom=0)

    # higher y lims to show 97% of data
    topy_tbt = df["time_between_trials"].quantile(0.97)
    ax_left.set_ylim(top=topy_tbt)
    ax_center.set_ylim(top=topy_tbt)
    topy_rt = df["reaction_time"].quantile(0.97)
    ax_right.set_ylim(top=topy_rt)
    ax2_center.set_ylim(top=topy_rt)

    # Ensure layout is adjusted
    plt.tight_layout()

    return fig


def plot_number_of_pokes_histogram(df: pd.DataFrame, port_number: int, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    port_hold_column = f"port{port_number}_holds"
    column_checker(df, required_columns={port_hold_column})
    # plot the histogram of the number of pokes
    npokes = df[port_hold_column].apply(lambda x: len(x))
    sns.histplot(npokes,
                 ax=ax,
                 bins=np.nanmax(npokes) - np.nanmin(npokes),
                 color="gray")
    ax.set_xlabel(f"Number of pokes in Port{port_number}")
    ax.set_ylabel("Frequency")
    # remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.set_title("Number of pokes histogram")
    return ax


def plot_port_holding_time_histogram(df: pd.DataFrame,
                                     port_number: int,
                                     ax: plt.Axes = None,
                                     **kwargs) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    port_hold_column = f"port{port_number}_holds"
    column_checker(df, required_columns={port_hold_column})
    # plot the histogram of the holding times
    port_holds = df[port_hold_column].tolist()
    port_holds = [item for sublist in port_holds for item in sublist]
    if "bins" in kwargs:
        bins = kwargs["bins"]
    else:
        bins = 100

    sns.histplot(port_holds,
                 ax=ax,
                 bins=bins,
                 color="gray",
                 stat="probability",
                 kde=True)
    ax.set_xlabel(f"Port{port_number} holding time (s)")
    ax.set_ylabel("Frequency")
    # remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.set_title("Holding time histogram")
    return ax


def plot_mean_and_cis_by_date(df: pd.DataFrame, item_to_show: str, group_trials_by: str, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    mean_col = f"{item_to_show}_mean"
    bot95_col = f"{item_to_show}_bot95"
    top95_col = f"{item_to_show}_top95"
    column_checker(df, required_columns={group_trials_by, mean_col, bot95_col, top95_col})
    
    if ax is None:
        ax = plt.gca()
    # check if there is someting plotted in the axis already
    items_in_axis = (
        len(ax.patches) + len(ax.lines) + len(ax.collections) + len(ax.texts)
    )
    if items_in_axis > 0:
        # create a new y axis on the right
        ax2 = ax.twinx()
        ax_to_use = ax2
        add_stuff = False
    else:
        ax_to_use = ax
        add_stuff = True

    # plot a thick line with the mean speed of holding time per trial
    # and a shaded area with the CIs
    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = "black"
    if "ticksize" in kwargs:
        ticksize = kwargs["ticksize"]
    else:
        ticksize = 8
    if "axislabelsize" in kwargs:
        axislabelsize = kwargs["axislabelsize"]
    else:
        axislabelsize = 10
    if "ylim_bottom" in kwargs:
        ylim_bottom = kwargs["ylim_bottom"]
    else:
        ylim_bottom = None
    if "ylim_top" in kwargs:
        ylim_top = kwargs["ylim_top"]
    else:
        ylim_top = None

    sns.lineplot(data=df, x=group_trials_by, y=mean_col, ax=ax_to_use, color=color, linewidth=2)
    ax_to_use.fill_between(df[group_trials_by], df[bot95_col], df[top95_col], color=color, alpha=0.2)
    ax_to_use.set_ylabel(f"{item_to_show}", color=color)
    ax_to_use.spines["top"].set_visible(False)

    # increase the font size of the ticks
    ax_to_use.tick_params(axis='x', labelsize=ticksize)
    ax_to_use.tick_params(axis='y', labelsize=ticksize)
    # increase the size of the y axis labels
    ax_to_use.yaxis.label.set_size(axislabelsize)

    if "ylog" in kwargs and kwargs["ylog"]:
        ax_to_use.set_yscale("log")
    
    ax_to_use.set_ylim(bottom=ylim_bottom, top=ylim_top)

    if not add_stuff:
        ax_to_use.set_xlabel(group_trials_by)
        # remove the top and right spines
        ax_to_use.spines["right"].set_visible(False)
        ax_to_use.xaxis.label.set_size(axislabelsize)
    # rotate the x-axis labels and align them to the end
    for label in ax_to_use.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
    
    return ax


def plot_trial_time_of_start(df: pd.DataFrame, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """
    Plot the time of start of each trial.
    """
    column_checker(df, required_columns={"time_from_start", "trial"})
    if ax is None:
        ax = plt.gca()
    
    sns.scatterplot(data=df, x="time_from_start", y="trial", ax=ax, s=4)

    if "session_changes" in kwargs and len(kwargs["session_changes"]) > 1:
        for sc in kwargs["session_changes"][1:]:
            tt = df.loc[sc]["time_from_start"]
            ax.axvline(tt, linestyle="--", color="gray")

    ax.set_ylabel("Trial number")
    ax.set_xlabel("Trial start time (s)")
    # remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    return ax


def plot_box_usage_by_date(df: pd.DataFrame, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """
    Plot the box usage by date.
    """
    column_checker(df, required_columns={"year_month_day", "percentage_of_time", "time_type"})
    if ax is None:
        ax = plt.gca()
    
    hue_order = ['engaged_time', 'disengaged_time', 'time_to_complete_first_trial', 'time_to_exit_box']
    sns.lineplot(data=df, x='year_month_day', y='percentage_of_time', hue='time_type', ax=ax, hue_order=hue_order)
    # despine
    ax.spines[["top", "right"]].set_visible(False)
    # legend to top in 2 columns
    ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=8)
    # rotate the x-axis labels and align them to the end
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percentage of time (%)")
    
    return ax


# ---------------------------------------------------------------------------
# Plots used by the figures (figures/panels.py)
# ---------------------------------------------------------------------------
def mean_and_ci_by_day_plot(daily: pd.DataFrame, item: str, ax: plt.Axes, color: str = "black",
                            ylog: bool = False, label: str = None) -> plt.Axes:
    """
    Mean (line) and 5-95 percentiles (shade) of item per day.

    Args:
        daily: one row per day, with columns day (datetime), <item>_mean,
            <item>_bot95 and <item>_top95.
    """
    column_checker(daily, required_columns={"day", f"{item}_mean", f"{item}_bot95", f"{item}_top95"})
    # one row per calendar day, so that lines break where there is no data
    daily = daily.sort_values("day").set_index("day")
    daily = daily[~daily.index.duplicated()].asfreq("D").reset_index()
    ax.plot(daily["day"], daily[f"{item}_mean"], color=color, linewidth=1.5)
    ax.fill_between(daily["day"], daily[f"{item}_bot95"], daily[f"{item}_top95"], color=color, alpha=0.2, linewidth=0)
    ax.set_ylabel(label or item.replace("_", " "), color=color)
    if ylog:
        ax.set_yscale("log")
    return ax


def time_between_trials_and_reaction_time_plot(df: pd.DataFrame, axes: List[plt.Axes],
                                               session_changes: pd.Index = None) -> List[plt.Axes]:
    """
    Time between trials (left distribution, blue) and reaction time (right
    distribution, orange) through the trials of a session (center).
    Same content as plot_time_between_trials_and_reaction_time, drawn directly
    on three axes (left, center, right) instead of in a separate figure.
    """
    column_checker(df, required_columns={"time_between_trials", "reaction_time", "total_trial"})
    ax_left, ax_center, ax_right = axes
    lines = []
    if session_changes is not None and len(session_changes) > 1:
        lines = [df.loc[sc, "total_trial"] for sc in session_changes[1:] if sc in df.index]
    df = df[["total_trial", "time_between_trials", "reaction_time"]].astype(float)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["reaction_time", "time_between_trials"])
    if df.empty:
        raise ValueError("no trials with both reaction time and time between trials")
    top_tbt = df["time_between_trials"].quantile(0.97)
    top_rt = df["reaction_time"].quantile(0.97)

    ax_rt = ax_center.twinx()
    ax_center.plot(df["total_trial"], df["time_between_trials"], color="tab:blue", linewidth=0.8, label="TBT")
    ax_rt.plot(df["total_trial"], df["reaction_time"], color="tab:orange", linewidth=0.8, label="RT")
    for x in lines:
        ax_center.axvline(x, linestyle="--", color="gray")
    ax_center.set_xlabel("Trial number")
    ax_center.set_ylim(0, top_tbt)
    ax_rt.set_ylim(0, top_rt)
    ax_center.tick_params(left=False, labelleft=False)
    ax_rt.tick_params(right=False, labelright=False)
    ax_center.spines["left"].set_visible(False)
    ax_rt.spines[["left", "right", "top"]].set_visible(False)
    ax_center.legend(handles=ax_center.lines[:1] + ax_rt.lines[:1], loc="lower center",
                     bbox_to_anchor=(0.5, 1.0), ncol=2)

    for ax, column, color, top in [(ax_left, "time_between_trials", "tab:blue", top_tbt),
                                   (ax_right, "reaction_time", "tab:orange", top_rt)]:
        values = df[column]
        if values.nunique() > 1:
            sns.kdeplot(y=values, ax=ax, color=color, fill=True, clip=(0, top))
        else:
            sns.histplot(y=values, ax=ax, color=color)
        ax.set_ylim(0, top)
        ax.set_xlabel("")
        ax.tick_params(bottom=False, labelbottom=False)
    ax_left.set_ylabel("Time between trials (s)", color="tab:blue")
    ax_right.invert_xaxis()
    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position("right")
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_visible(True)
    ax_right.set_ylabel("Reaction time (s)", color="tab:orange")
    return [ax_left, ax_center, ax_right]

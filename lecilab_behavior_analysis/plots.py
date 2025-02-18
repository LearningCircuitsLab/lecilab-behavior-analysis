import calplot
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from lecilab_behavior_analysis.utils import (column_checker, get_text_from_df)


def training_calendar_plot(dates_df: pd.DataFrame) -> plt.Figure:
    # make the calendar plot and convert it to an image
    column_checker(dates_df, required_columns={"trial"})
    cpfig, _ = calplot.calplot(
        data=dates_df.trial, yearlabel_kws={"fontname": "sans-serif"}
    )

    return cpfig


def rasterize_plot(plot: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(plot)
    canvas.draw()
    width, height = plot.get_size_inches() * plot.get_dpi()
    # Get ARGB buffer and reshape
    argb = np.frombuffer(canvas.tostring_argb(), dtype="uint8").reshape(
        int(height), int(width), 4  # 4 channels (A, R, G, B)
    )
    # Remove alpha channel by keeping only R, G, B
    rgb = argb[:, :, 1:]  # Drop the first (alpha) channel
    plt.close(plot)
    return rgb


def trials_by_date_plot(dates_df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    column_checker(dates_df, required_columns={"trial"})
    if dates_df.index.name != "date":
        raise ValueError("The dataframe must have [date] as index")
    if ax is None:
        ax = plt.gca()
    if "current_training_stage" in dates_df.columns:
        hue = "current_training_stage"
    else:
        hue = None
    sns.barplot(data=dates_df, x="date", y="trial", hue=hue, ax=ax)
    ax.legend(bbox_to_anchor=(0.5, 1.25), loc="upper center", ncol=3, borderaxespad=0.0)
    ax.get_legend().get_frame().set_linewidth(0.0)
    ax.set_ylabel("Number of trials")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def trials_by_session_hist(
    dates_df: pd.DataFrame, ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    column_checker(dates_df, required_columns={"trial"})
    dates_df["trial"].hist(ax=ax, orientation="horizontal", bins=15, color="gray")
    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])
    # remove the axis
    ax.axis("off")
    # flip x axis
    ax.invert_xaxis()
    return ax


def performance_vs_trials_plot(
    df: pd.DataFrame, ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    """
    Plot the performance vs the number of trials
    """
    # check if the dataframe has the necessary columns
    column_checker(df, required_columns={"total_trial", "performance_w"})

    if ax is None:
        ax = plt.gca()

    # if there is no "current_training_stage" column, generate a fake one
    if "current_training_stage" not in df.columns:
        df["current_training_stage"] = "All"

    # plot the performance
    for stage in df["current_training_stage"].unique():
        stage_df = df[df["current_training_stage"] == stage]
        sns.lineplot(
            data=stage_df, x="total_trial", y="performance_w", ax=ax, label=stage
        )
    ax.set_xlim(left=0)
    ax.set_ylim(40, 100)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Performance")
    # horizontal line at 50%
    ax.axhline(50, linestyle="--", color="gray")
    # remove box
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if "legend" in kwargs:
        if kwargs["legend"] == False:
            ax.get_legend().remove()

    return ax


def water_by_date_plot(water_df: pd.Series, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    if water_df.index.name != "date":
        raise ValueError("The dataframe must have [date] as index")
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

    water_df.plot(ax=ax_to_use, color="black")
    ax_to_use.set_ylabel("Water consumed (μl)")
    ax_to_use.spines["top"].set_visible(False)
    if add_stuff:
        ax_to_use.set_xlabel("Date")
        ax_to_use.tick_params(axis="x", rotation=45)
        for label in ax_to_use.get_xticklabels():
            label.set_horizontalalignment("right")
        ax_to_use.spines["right"].set_visible(False)
    ax_to_use.set_ylim(bottom=0)
    return ax_to_use


def session_summary_text(
    df: pd.DataFrame, ax: plt.Axes = None, mouse_name: str = ""
) -> plt.Axes:
    """
    summary of a particular session
    """
    if ax is None:
        ax = plt.gca()
    text = get_text_from_df(df, mouse_name)
    ax.text(0, 0, text, fontsize=10)
    ax.axis("off")
    return ax


def correct_left_and_right_plot(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    column_checker(df, required_columns={"correct_side", "correct"})
    sns.countplot(data=df, x="correct_side", hue="correct", ax=ax, hue_order=[False, True])
    ax.set_xlabel("Correct side")
    ax.set_ylabel("Number of trials")
    ax.set_ylim(top=ax.get_ylim()[1]*1.1)
    ax.legend(["Error", "Correct"], bbox_to_anchor=(0.5, 1.05), loc="upper center", ncol=1, borderaxespad=0.0)
    ax.get_legend().get_frame().set_linewidth(0.0)
    ax.get_legend().set_frame_on(False)
    # change labels of the legend
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def repeat_or_alternate_performance_plot(
    df: pd.DataFrame, ax: plt.Axes = None
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    column_checker(
        df,
        required_columns={
            "repeat_or_alternate",
            "repeat_or_alternate_performance",
            "total_trial",
        },
    )
    sns.lineplot(
        data=df,
        x="total_trial",
        y="repeat_or_alternate_performance",
        hue="repeat_or_alternate",
        ax=ax,
    )
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Performance")
    ax.axhline(50, linestyle="--", color="gray")
    ax.spines[["top", "right"]].set_visible(False)
    # x axis always starts at 0
    ax.set_xlim(left=0)
    ax.set_ylim(40, 100)
    # put the legend at the top in one row
    ax.legend(bbox_to_anchor=(0.5, 1.05), loc="upper center", ncol=2, borderaxespad=0.0)
    # remove box
    ax.get_legend().get_frame().set_linewidth(0.0)
    ax.get_legend().set_frame_on(False)

    return ax


def psychometric_plot(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    column_checker(df, required_columns={"leftward_evidence", "leftward_choices"})
    sns.scatterplot(
        data=df,
        x="leftward_evidence",
        y="leftward_choices",
        # hue="difficulty",
        ax=ax,
    )
    ax.set_xlabel("Leftward evidence")
    ax.set_ylabel("P(Left)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-.1, 1.1)
    ax.axhline(0.5, linestyle="--", color="gray")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def summary_matrix_plot(
    mat_df: pd.DataFrame,
    mat_df_metadata: dict,
    mouse_name: str,
    ax: plt.Axes = None,
    top_labels=["stimulus_modality"],
    training_stage_inlegend=True,
) -> plt.Axes:
    """
    Generates a matrix plot with information regarding
    each particular session on the top
    """
    if ax is None:
        ax = plt.gca()
    sns.set_theme(style="white")
    sp = sns.heatmap(
        mat_df,
        linewidth=0.001,
        square=True,
        cmap="coolwarm",
        cbar_kws={"shrink": 0.2, "label": "% Leftward choices"},
        ax=ax,
        vmin=0,
        vmax=1,
    )
    # TODO: check that the size is proportional (area vs radius)

    # The protocols is the default that gets plotted,
    # with the size of the dots proportional to the number of trials
    training_stages = [
        mat_df_metadata[session]["current_training_stage"]
        for session in mat_df.columns
    ]
    trials_list = [
        mat_df_metadata[session]["n_trials"] for session in mat_df.columns
    ]
    evidence_levels = mat_df.index
    shift_up = 0.5
    training_stages_done = []
    for tr_stg in training_stages:
        if tr_stg in training_stages_done:
            continue
        tr_stg_idx = [i for i, x in enumerate(training_stages) if x == tr_stg]
        if training_stage_inlegend:
            label=tr_stg
        else:
            label=None
        ax.scatter(
            [x + 0.5 for x in tr_stg_idx],
            np.repeat(len(evidence_levels) + shift_up, len(tr_stg_idx)),
            marker="o",
            s=[trials_list[x] / 5 for x in tr_stg_idx],
            label=label,
        )
        training_stages_done.append(tr_stg)
    shift_up += 1

    # label the rest of the sessions as given in the input
    marker_list = ["*", "P", "h", "D", "X"]
    for n_lab, t_label in enumerate(top_labels):
        t_label_uniques = [
            mat_df_metadata[session][t_label] for session in mat_df.columns
        ]
        for tlu in np.unique(t_label_uniques):
            tlu_idx = [i for i, x in enumerate(t_label_uniques) if x == tlu]
            ax.scatter(
                [x + 0.5 for x in tlu_idx],
                np.repeat(len(evidence_levels) + shift_up, len(tlu_idx)),
                marker=marker_list[n_lab],
                s=100,
                label=tlu,
            )
        shift_up += 1

    ax.legend(loc=(0, 1), borderaxespad=0.0, ncol=5, frameon=True)
    ax.set_ylim([0, len(evidence_levels) + shift_up - 0.5])
    ax.set_ylabel("Evidence level")
    ax.set_xlabel("Session")
    sp.set_yticklabels(sp.get_yticklabels(), rotation=0)
    sp.set_xticklabels(
        sp.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    sp.set_title(
        mouse_name + "\n\n", fontsize=20, fontweight=0
    )

    return ax


def side_correct_performance_plot(df: pd.DataFrame, ax: plt.Axes, trials_to_show: int = 50) -> plt.Axes:
    # used for real time plotting
    column_checker(df, required_columns={"trial", "correct_side", "correct"})
    ax.clear()
    # select only the last X trials
    df = df.tail(trials_to_show)
    sns.scatterplot(data=df, x="trial", y="correct_side", hue="correct", ax=ax)
    # make sure the y axis ticks are ascending, inverting the y axis
    ax.invert_yaxis()
    # plot the mean of the last 10 trials
    ax.plot(pd.Series([int(x) for x in df.correct]).rolling(10).mean(), "r")
    # add a horizontal line at 50%
    ax.axhline(0.5, linestyle="--", color="gray")
    # make the x axis to be at least 100 trials
    ax.set_xlim(left=max(0, df.trial.min() - 1), right=max(trials_to_show, df.trial.max() + 1))

    return ax

def plot_time_between_trials_and_reaction_time(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    """
    Plot Time Between Trials (TBT) and Reaction Time (RT) on the same plot with histograms on the y-axes.
    """
    # Check if the dataframe has the necessary columns
    column_checker(df, required_columns={"Time_Between_Trials", "Reaction_Time"})

    # Drop NaN or Inf values to avoid errors in plotting
    df = df.dropna(subset=['Reaction_Time', 'Time_Between_Trials'])
    df = df[(df['Reaction_Time'] != float('inf')) & (df['Time_Between_Trials'] != float('inf'))]

    # Create a 1x3 grid layout for two histograms and the main plot in the center
    fig = plt.figure(figsize=(14, 8))  # Increased figure size for better visibility
    grid = plt.GridSpec(1, 3, width_ratios=[0.5, 5, 0.5], wspace=0.0)  # Adjust width ratios and wspace to align the plots

    # Right plot for Reaction Time (RT) KDE plot
    ax_right = fig.add_subplot(grid[0, 2])
    if df['Reaction_Time'].nunique() > 1:
        sns.kdeplot(y=df['Reaction_Time'], ax=ax_right, color="tab:orange", fill=True)
    else:
        sns.histplot(y=df['Reaction_Time'], ax=ax_right, color="tab:orange")

    ax_right.invert_xaxis()  # Flip the x-axis to make the plot point towards the main plot
    ax_right.set_xlabel("")
    ax_right.yaxis.set_label_position('right')  # Force y-label on the right
    ax_right.set_ylabel("Reaction Time (RT) [ms]", rotation=270)

    ax_right.tick_params(right=True, left=False, labelleft=False, labelright=True, bottom=False, labelbottom=False)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['top'].set_visible(False)

    # Center plot for the main Time Between Trials and Reaction Time plot
    ax_center = fig.add_subplot(grid[0, 1])
    ax2_center = ax_center.twinx()

    # Plot Time Between Trials
    ax_center.plot(df["trial"], df["Time_Between_Trials"], color="tab:blue", label="Time Between Trials")
    ax_center.set_xlabel("Trial number")

    # Plot Reaction Time
    ax2_center.plot(df["trial"], df["Reaction_Time"], color="tab:orange", label="Reaction Time")
    ax2_center.set_ylabel("")
    ax2_center.tick_params(left=False, right=False, top=False, bottom=False, labelleft=False, labelright=False, labeltop=False, labelbottom=False)

    # Add legends
    handles1, labels1 = ax_center.get_legend_handles_labels()
    handles2, labels2 = ax2_center.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = ["TBT", "RT"]
    ax_center.legend(handles, labels, bbox_to_anchor=(0.5, 1.05), loc="upper center", ncol=2, borderaxespad=0.0, frameon=False)

    # Remove lateral and topspines for the center plot
    ax_center.spines["top"].set_visible(False)
    ax_center.spines["right"].set_visible(False)
    ax_center.spines["left"].set_visible(False)
    ax2_center.spines["top"].set_visible(False)
    ax2_center.spines["right"].set_visible(False)
    ax2_center.spines["left"].set_visible(False)

    # Left plot for Time Between Trials (TBT) KDE plot
    ax_left = fig.add_subplot(grid[0, 0])
    if df['Time_Between_Trials'].nunique() > 1:
        sns.kdeplot(y=df['Time_Between_Trials'], ax=ax_left, color="tab:blue", fill=True)
    else:
        sns.histplot(y=df['Time_Between_Trials'], ax=ax_left, color="tab:blue")

    ax_left.set_xlabel("")
    ax_left.set_ylabel("Time Between Trials (TBT) [ms]")
    ax_left.spines['left'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.spines['top'].set_visible(False)
    ax_left.tick_params(left=True, right=False, labelright=False, labelleft=True, bottom=False, labelbottom=False)

    # Ensure layout is adjusted
    plt.tight_layout()

    return ax_center, ax2_center, ax_right, ax_left

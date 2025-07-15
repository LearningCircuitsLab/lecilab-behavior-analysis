import calplot
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import lecilab_behavior_analysis.utils as utils
from lecilab_behavior_analysis.utils import (column_checker, get_text_from_subset_df, list_to_colors, get_text_from_subject_df)


def training_calendar_plot(dates_df: pd.DataFrame) -> plt.Figure:
    # make the calendar plot and convert it to an image
    column_checker(dates_df, required_columns={"trial"})
    cpfig, _ = calplot.calplot(
        data=dates_df.trial, yearlabel_kws={"fontname": "sans-serif"}
    )

    return cpfig


def rasterize_plot(plot: plt.Figure, dpi: int = 300) -> np.ndarray:
    plot.set_dpi(dpi)
    canvas = FigureCanvasAgg(plot)
    canvas.draw()
    width, height = plot.get_size_inches() * dpi
    # Get ARGB buffer and reshape
    argb = np.frombuffer(canvas.tostring_argb(), dtype="uint8").reshape(
        int(height), int(width), 4  # 4 channels (A, R, G, B)
    )
    # Remove alpha channel by keeping only R, G, B
    rgb = argb[:, :, 1:]  # Drop the first (alpha) channel
    plt.close(plot)
    return rgb


def trials_by_session_plot(dates_df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
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


def trials_by_day_plot(
    dates_df: pd.DataFrame, ax: plt.Axes = None, cmap: str = "tab10"
) -> plt.Axes:
    array_of_training_stages = np.array([x[1] for x in dates_df.columns])
    color_list, color_dict = list_to_colors(ids=array_of_training_stages, cmap=cmap)
    ax = dates_df.plot(kind="bar", stacked=True, edgecolor="black", color=color_list, ax=ax)
    ax.set_ylabel("Number of trials")
    ax.set_xlabel("")
    # remove the legend
    ax.get_legend().remove()
    # create a new legend
    labels, leg_cols = zip(*color_dict.items())
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in leg_cols]
    ax.legend(handles, labels, bbox_to_anchor=(0.5, 1.25), loc="upper center", ncol=3, borderaxespad=0.0)
    ax.get_legend().get_frame().set_linewidth(0.0)
    
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

    # if there is no "current_training_stage" column, generate a fake one
    if "current_training_stage" not in df.columns:
        df["current_training_stage"] = "All"

    # plot the performance
    # for stage in df["current_training_stage"].unique():
    #     stage_df = df[df["current_training_stage"] == stage]
    #     sns.lineplot(
    #         data=stage_df, x="total_trial", y="performance_w", ax=ax, label=stage
    #     )
    sns.lineplot(
        data=df,
        x="total_trial",
        y="performance_w",
        hue="current_training_stage",
        ax=ax,
    )
    if "session_changes" in kwargs and len(kwargs["session_changes"]) > 1:
        for sc in kwargs["session_changes"][1:]:
            tt = df.loc[sc]["total_trial"]
            ax.axvline(tt, linestyle="--", color="gray")
    ax.set_xlim(left=0)
    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])
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
    # if water_df.index.name != "date":
    #     raise ValueError("The dataframe must have [date] as index")
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


def summary_text_plot(
    df: pd.DataFrame, kind: str = "session", ax: plt.Axes = None, **kwargs
) -> plt.Axes:
    """
    summary of a particular session or subject
    """
    if ax is None:
        ax = plt.gca()
    if kind == "session":
        text = get_text_from_subset_df(df)
    elif kind == "subject":
        text = get_text_from_subject_df(df)
    else:
        raise ValueError("kind must be 'session' or 'subject'")
    if "fontsize" in kwargs:
        fontsize = kwargs["fontsize"]
    else:
        fontsize = 10
    ax.text(0, 0, text, fontsize=fontsize)
    ax.axis("off")
    return ax


def number_of_correct_responses_plot(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    if "stimulus_modality" not in df.columns:
        return correct_left_and_right_plot(df, ax)
    elif df['stimulus_modality'].nunique() == 1:
        # if there is only one modality, plot the correct left and right plot
        return correct_left_and_right_plot(df, ax)
    else:
        return correct_left_and_right_plot_multisensory(df, ax)


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


def correct_left_and_right_plot_multisensory(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    column_checker(df, required_columns={"correct_side", "correct", "stimulus_modality"})
    
    # Group the data by 'stimulus_modality', 'correct_side', and 'correct' and count occurrences
    grouped_df = (
        df.groupby(["stimulus_modality", "correct_side", "correct"])
        .size()
        .reset_index(name="count")
    )
    
    # Create a new column combining 'correct_side' and 'stimulus_modality' with a line break for the x-axis
    grouped_df["x"] = grouped_df["correct_side"] + "\n" + grouped_df["stimulus_modality"]
    
    # Create a bar plot using the grouped data
    sns.barplot(
        data=grouped_df,
        x="x",
        y="count",
        hue="correct",
        ax=ax,
        hue_order=[False, True],
        errorbar=None,
    )
    
    ax.set_xlabel("")
    ax.set_ylabel("Number of Trials")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Correct", loc="upper right")
    
    return ax


def repeat_or_alternate_performance_plot(
    df: pd.DataFrame, ax: plt.Axes = None, **kwargs
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
        hue_order=["repeat", "alternate"],
        ax=ax,
    )
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Performance")
    ax.axhline(50, linestyle="--", color="gray")
    ax.spines[["top", "right"]].set_visible(False)
    # x axis always starts at 0
    ax.set_xlim(left=0)
    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])
    # put the legend at the top in one row
    ax.legend(bbox_to_anchor=(0.5, 1.05), loc="upper center", ncol=2, borderaxespad=0.0)
    # remove box
    ax.get_legend().get_frame().set_linewidth(0.0)
    ax.get_legend().set_frame_on(False)

    return ax


# def psychometric_plot(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
#     if ax is None:
#         ax = plt.gca()
#     column_checker(df, required_columns={"leftward_evidence", "leftward_choices"})
#     sns.scatterplot(
#         data=df,
#         x="leftward_evidence",
#         y="leftward_choices",
#         # hue="difficulty",
#         ax=ax,
#     )
#     ax.set_xlabel("Leftward evidence")
#     ax.set_ylabel("P(Left)")
#     ax.set_xlim(-1.1, 1.1)
#     ax.set_ylim(-.1, 1.1)
#     ax.axhline(0.5, linestyle="--", color="gray")
#     ax.spines[["top", "right"]].set_visible(False)
#     return ax

def psychometric_plot(df: pd.DataFrame, x, y, ax: plt.Axes = None, 
                                   point_kwargs=None,
                                   line_kwargs=None, valueType = 'discrete') -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    
    default_point_kwargs = {
            'color': 'blue',
            'markers': 'o',
            'errorbar': ("ci", 95),
            'label': 'Observed Choices',
            'native_scale': True,
            'linestyles': '',
            'markersize': 5,
            'capsize': 0.01
        }
    if point_kwargs is None:
        point_kwargs = {}
    point_kwargs = {**default_point_kwargs, **point_kwargs}
    
    default_line_kwargs = {
            'color': 'red',
            'label': 'Lapse Logistic Fit (Independent)',
            'linestyle': '-'
        }
    if line_kwargs is None:
        line_kwargs = {}
    line_kwargs = {**default_line_kwargs, **line_kwargs}

    column_checker(df, required_columns={x, y})
    df_copy = df.copy(deep=True)
    if valueType == 'discrete':
        if 0 in df_copy[x].unique():
            # if there are 0s in the data, we need to add a small value to avoid log(0)
            df_copy[x + "_fit"] = df_copy[x]
            df_copy[x + "_fit"].replace(0, 0.0001, inplace=True)
            ax.set_xlabel(x)
        else:
            df_copy[x + "_fit"] = np.sign(df_copy[x]) * (np.log(abs(df_copy[x])).round(4))
            ax.set_xlabel("log_"+x)
    else:
        # bin the continuous values when valueType is not discrete
        bins = pd.cut(df_copy[x], bins = 6)
        labels = df_copy[x].groupby(bins).mean()
        df_copy[x + "_fit"] = pd.cut(df_copy[x], bins = 6, labels = labels).astype(float)
        df_copy[x + "_fit"] = np.sign(df_copy[x + "_fit"]) * (np.log(abs(df_copy[x + "_fit"]*10)).round(4))
        ax.set_xlabel("log_"+x + "*10")
    sns.pointplot(
        x=x + "_fit",
        y=y,
        data=df_copy,
        estimator=lambda x: np.mean(x),
        ax=ax,
        **point_kwargs
    )
    xs = np.linspace(df_copy[x + "_fit"].min(), df_copy[x + "_fit"].max(), 100).reshape(-1, 1)
    p_left, fitted_params = utils.fit_lapse_logistic_independent(df_copy[x + "_fit"], df_copy[y])
    ax.plot(xs, p_left, **line_kwargs)
    if y == "first_choice_numeric":
        ax.set_ylabel("P(Leftward Choices)")
    elif y == "correct_choice_numeric":
        ax.set_ylabel("P(Correct Choices)")
    ax.set_ylim(0, 1)
    ax.legend()
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
    ax_right.set_ylabel("Reaction Times (RT) [ms]", rotation=270, fontsize=28, labelpad=30)

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
    ax_left.set_ylabel("Time Between Trials (TBT) [ms]", fontsize=28)
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


def bias_vs_trials_plot(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    column_checker(df, required_columns={"total_trial", "bias"})
    sns.lineplot(data=df, x="total_trial", y="bias", ax=ax)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Bias")
    # set y ticks label to be -1, 0, 1
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["left", "alternate", "right"])
    ax.axhline(0, linestyle="--", color="gray")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def performance_by_decision_plot(df: pd.DataFrame, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    column_checker(df, required_columns={"correct", "correct_side_repeat_or_alternate", "trial_group"})
    # if kwargs has a color_dict, use it to set the colors
    if "color_dict" in kwargs:
        color_dict = kwargs["color_dict"]
    else:
        color_dict = {
            "left_repeat": (0.121, 0.466, 0.705, 1),  # tab:blue with alpha=0.8
            "left_alternate": (0.121, 0.466, 0.705, 0.4),  # tab:orange with alpha=0.8
            "right_repeat": (1.0, 0.498, 0.054, 1),  # tab:purple with alpha=0.8
            "right_alternate": (1.0, 0.498, 0.054, 0.4),  # tab:red with alpha=0.8
        }

    # Create a custom palette using the color_dict
    palette = {key: color for key, color in color_dict.items()}

    sns.lineplot(data=df, x='trial_group', y='correct', hue='correct_side',
                 style="repeat_or_alternate",
                 errorbar=None, ax=ax)#, palette=palette)
    # rotate the x-axis labels and align them to the end
    # for label in ax.get_xticklabels():
    #     label.set_rotation(45)
    #     label.set_horizontalalignment('right')
    # ax.set_title('Performance by "decision"', pad=20)
    ax.set_xlabel('Trials')
    ax.set_ylabel('Performance')
    # Filter legend to only include entries with handles (skip titles like "correct_side")
    handles, labels = ax.get_legend_handles_labels()
    # clean handles and labels to only include the ones that are not titles
    for handle, label in zip(handles, labels):
        if label not in ["left", "right", "repeat", "alternate"]:
            handles.remove(handle)
            labels.remove(label)
    # Recreate legend with only handle-associated labels
    ax.legend(handles=handles, labels=labels, frameon=False, title='')

    # remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # start the x axis at min value
    ax.set_xlim(left=df["trial_group"].min())

    return ax


def triangle_polar_plot(df_bias: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    column_checker(df_bias, required_columns={"subject", "bias_angle", "percentage"})
    if ax is None:
        ax = plt.subplot(111, polar=True)
    # Plot each subject in a polar plot
    for subject in df_bias['subject'].unique():
        subject_data = df_bias[df_bias['subject'] == subject]
        
        # Extract the angles and distances
        angles = subject_data['bias_angle'].values
        percentages = subject_data['percentage'].values
        
        # Close the triangle by repeating the first point
        angles = np.append(angles, angles[0])
        percentages = np.append(percentages, percentages[0])
        
        # Plot the triangle
        ax.plot(angles, percentages, marker='o', linestyle='-', label=subject)
        
    # Set title and formatting
    # ax.set_title(f"Animal biases", va='bottom', fontsize=16)
    ax.set_rlabel_position(-22.5)  # Adjust label position
    ax.grid(True)

    # make the circular 0.33 grid thicker
    ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
    # plot the 0.33 grid
    ax.set_yticks([0.25, 0.5])

    ax.set_xticks([2*np.pi/4, 5*np.pi/4, 7*np.pi/4])
    ax.set_xticklabels(['Alternate', 'Left', 'Right'], fontsize=14)

    # Show the legend outside the plot
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, .8), ncol=1)
    # remove border of legend box
    plt.setp(ax.get_legend().get_frame(), linewidth=0)  # Set legend frame color and linewidth

    return ax


def plot_decision_evolution_triangle(df: pd.DataFrame, hue: str, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    column_checker(df, required_columns={"left_right_bias", "alternating_bias", hue})
    if ax is None:
        ax = plt.gca()
    if "palette" in kwargs:
        palette = kwargs["palette"]
    else:
        palette = sns.color_palette("viridis", as_cmap=True)
    # plot the trajectory of the left_right_bias and alternating_bias through the sessions,
    # without averaging the same values
    sns.lineplot(data=df, x='left_right_bias', y='alternating_bias', ax=ax,
                 hue=hue, palette=palette, legend=False,
                 linewidth=3, alpha=0.8)
    # connect the points in the order of the session
    # plt.plot(df_bias_pivot['xs'], df_bias_pivot['ys'], marker='o', linestyle='-')

    # plot a triangle between -1 and 1 in the x axis and 0 and 1 in the y axis
    triangle = np.array([[0, 1], [1, 0], [-1, 0], [0, 1]])
    ax.plot(triangle[:, 0], triangle[:, 1], color='black', linestyle='--')
    # draw a dashed circle in 0, 0.5 of radius 0.1, no linestyle, fill it with black and alpha 0.1
    circle = plt.Circle((0, 0.5), 0.1, color='black', fill=True, alpha=0.1)
    ax.add_artist(circle)
    # label the triangle vertices
    ax.text(0, 1.05, 'Alternating', ha='center', va='center')
    ax.text(1.05, -0.05, 'Right', ha='center', va='center')
    ax.text(-1.05, -0.05, 'Left', ha='center', va='center')
    # remove spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    return ax


def plot_percentage_of_occupancy_per_day(daily_percentages: pd.Series, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    # Plot the percentage of time for each day
    daily_percentages.plot(kind='line', ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Percentage of time (%)')
    ax.set_title('Percentage of time that task is running per day')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 100)
    ax.grid(True)
    return ax


def plot_training_times_heatmap(heatmap_vector: np.ndarray, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    # Plot the heatmap vector
    sns.heatmap(heatmap_vector.reshape(1, -1), cmap='YlGnBu', cbar=True,
                xticklabels=60, vmin=0, vmax=np.max(heatmap_vector), ax=ax)
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Event Occurrences')
    ax.set_title('Training times')
    # Set the x-ticks to show every hour
    ax.set_xticks(np.arange(0, 1440, 60))
    ax.set_xticklabels([f'{i//60:02d}:00' for i in range(0, 1440, 60)], rotation=45)
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.grid(False)
    plt.tight_layout()
    
    return ax


def plot_training_times_clock_heatmap(heatmap_vector: np.ndarray, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3), subplot_kw={'projection': 'polar'})
    # Create a polar plot
    theta = np.linspace(0, 2 * np.pi, 1440)  # 1440 minutes in a day

    # Plot the heatmap as a polar bar plot
    ax.bar(theta, heatmap_vector, width=2 * np.pi / 1440, color=plt.cm.YlGnBu(heatmap_vector / np.max(heatmap_vector)), edgecolor='none')

    # shade in light gray from 20:00 to 8:00 and put it behind the bars
    # get the maximum value of the plot radius
    max_radius = ax.get_ylim()[1]
    ax.fill_between(theta, 0, max_radius, where=(theta >= 20 * np.pi / 12) | (theta <= 8 * np.pi / 12), color='lightgray', alpha=0.5, zorder=0)

    # Customize the plot to resemble a clock
    try:
        ax.set_theta_zero_location('N')  # Set midnight at the top
        ax.set_theta_direction(-1)  # Set clockwise direction
    except AttributeError:
        print("initialize your figure with subplot_kw={'projection': 'polar'} to make it circular")
    
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))  # Hourly ticks
    ax.set_xticklabels([f'{i}:00' for i in range(0, 24, 2)])  # Label every 2 hours
    ax.set_yticks([])  # Remove radial ticks
    ax.set_title('Training Times', pad=25)
    
    return ax


def plot_transition_matrix(transition_matrix: pd.DataFrame, figsize: tuple = (5, 5)) -> plt.Figure:
    # plot the heatmap of the transition matrix with clustering
    # if ax is None:
    #     ax = plt.gca()

    # sns.heatmap(transition_matrix, annot=True, fmt='d', cmap='YlGnBu')
    fig = sns.clustermap(transition_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar=True,
                   xticklabels=transition_matrix.columns, yticklabels=transition_matrix.index,
                   figsize=figsize, dendrogram_ratio=(0.2, 0.2), cbar_kws={"shrink": .8})
    # get the axis of the clustermap
    ax = fig.ax_heatmap
    # set the title and labels
    ax.set_title('Transition Matrix Heatmap', fontsize=16)
    ax.set_xlabel('To Item')
    ax.set_ylabel('From Item')

    return fig


def plot_transition_network_with_curved_edges(transition_matrix: pd.DataFrame, threshold: int = 0, figsize: tuple = (10, 10)) -> plt.Figure:
    # Create a directed graph from the transition matrix
    G = nx.from_pandas_adjacency(transition_matrix, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency

    # Calculate total edge weight and filter edges above the threshold
    total_weight = sum(G[u][v]['weight'] for u, v in G.edges())
    min_weight = threshold * total_weight / 100  # Calculate the minimum weight based on % threshold

    # Edge attributes: Filter edges above the threshold and scale them
    filtered_edges = [(u, v) for u, v in G.edges() if G[u][v]['weight'] >= min_weight]
    edge_weights = [G[u][v]['weight'] for u, v in filtered_edges]
    
    # Normalize edge properties for filtered edges
    if edge_weights:
        max_weight = max(edge_weights)
    else:
        max_weight = 1  # Prevent division by zero if no edges remain
    
    edge_widths = [2 + (5 * G[u][v]['weight'] / max_weight) for u, v in filtered_edges]
    edge_alphas = [0.3 + (0.7 * G[u][v]['weight'] / max_weight) for u, v in filtered_edges]
    
    # Apply a colormap to the edges based on normalized weights
    norm = mcolors.Normalize(vmin=min(edge_weights, default=0), vmax=max_weight)
    edge_colors = [cm.YlGnBu(norm(G[u][v]['weight'])) for u, v in filtered_edges]
    
    # Create a figure
    plt.figure(figsize=figsize)

    # Draw larger, fully opaque nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue', alpha=1.0)
    
    # Draw curved edges with varying properties, terminating early at the nodes
    for i, (u, v) in enumerate(filtered_edges):
        # Curve edges if they're bidirectional
        if G.has_edge(v, u) and (u, v) != (v, u):
            curve_scale = 0.2  # Add curvature for bidirectional edges
        else:
            curve_scale = 0
        
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], width=edge_widths[i], alpha=edge_alphas[i], edge_color=[edge_colors[i]],
            connectionstyle=f"arc3,rad={curve_scale}", arrows=True, arrowsize=15, arrowstyle='-|>',
            min_source_margin=10, min_target_margin=10,  # Terminate arrows earlier at the node edges
        )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap='YlGnBu', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Transition Frequency', fontsize=12)

    plt.title(f'Transition Network (Edges above {threshold}% of total weight)', fontsize=16)
    
    return plt.gcf()


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


def plot_port_holding_time_histogram(df: pd.DataFrame, port_number: int, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    port_hold_column = f"port{port_number}_holds"
    column_checker(df, required_columns={port_hold_column})
    # plot the histogram of the holding times
    port_holds = df[port_hold_column].tolist()
    port_holds = [item for sublist in port_holds for item in sublist]

    sns.histplot(port_holds,
                 ax=ax,
                 bins=100,
                 color="gray",
                 stat="probability",
                 kde=True)
    ax.set_xlabel(f"Port{port_number} holding time (ms)")
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
    sns.lineplot(data=df, x=group_trials_by, y=mean_col, ax=ax_to_use, color=color, linewidth=2)
    ax_to_use.fill_between(df[group_trials_by], df[mean_col] - df[bot95_col],
                            df[mean_col] + df[top95_col], color=color, alpha=0.2)
    ax_to_use.set_ylabel(f"{item_to_show}", color=color)
    ax_to_use.spines["top"].set_visible(False)

    if not add_stuff:
        ax_to_use.set_xlabel(group_trials_by)
        # remove the top and right spines
        ax_to_use.spines["right"].set_visible(False)
    # rotate the x-axis labels and align them to the end
    for label in ax_to_use.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
    
    if "ylog" in kwargs and kwargs["ylog"]:
        ax_to_use.set_yscale("log")
    return ax


def plot_filter_model_variables(corr_mat_list:list, norm_contribution_df:pd.DataFrame, **kwargs) -> plt.Axes:
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    X = corr_mat_list[0].index
    corr_mat_mean = np.mean(np.stack(corr_mat_list), axis=0)
    # # Create a mask for the upper triangle
    # mask = np.triu(np.ones_like(corr_mat_mean, dtype=bool), k=1)
    sns.heatmap(corr_mat_mean, ax=ax[0], cmap='coolwarm', annot=True, fmt=".2f")
    ax[0].set_xticklabels(X, rotation=16, ha="right", rotation_mode="anchor")
    ax[0].set_yticklabels(X, rotation=8, ha="right", rotation_mode="anchor")
    ax[0].set_title("Mean Correlation Matrix")

    norm_contribution_df.mean(axis=1).sort_values().plot(kind='barh', ax=ax[1])
    ax[1].set_xlabel('Mean Contribution')
    plt.tight_layout()
    plt.show()
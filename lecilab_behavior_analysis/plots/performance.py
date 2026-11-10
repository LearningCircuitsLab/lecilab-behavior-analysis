"""Plots of performance: through trials, by side, repeat or alternate, by decision."""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import lecilab_behavior_analysis.data_format_changes as dfc
from lecilab_behavior_analysis.common import column_checker


def performance_vs_trials_plot(df: pd.DataFrame, ax: plt.Axes = None, **kwargs) -> plt.Axes:
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

    if "hue" in kwargs:
        hue = kwargs["hue"]
    else:
        hue = "current_training_stage"
    if "palette" in kwargs:
        palette = kwargs["palette"]
    else:
        # standard palette
        palette = sns.color_palette("tab10", n_colors=df[hue].nunique())

    # Sort data by trial number to ensure proper line plotting
    df_sorted = df.sort_values("total_trial").reset_index(drop=True)

    # Find break points where hue column changes
    hue_changes = df_sorted[hue] != df_sorted[hue].shift(1)
    break_indices = df_sorted[hue_changes].index.tolist()

    # Add start and end indices for completeness
    if 0 not in break_indices:
        break_indices.insert(0, 0)
    break_indices.append(len(df_sorted))

    # Keep track of which hue values have been labeled
    labeled_hues = set()

    # Plot segments between hue changes
    for i in range(len(break_indices) - 1):
        start_idx = break_indices[i]
        end_idx = break_indices[i + 1]
        segment = df_sorted.iloc[start_idx:end_idx]

        if len(segment) > 0:
            hue_value = segment[hue].iloc[0]
            color_idx = list(df_sorted[hue].unique()).index(hue_value)

            # Only label the first occurrence of each hue value
            label = hue_value if hue_value not in labeled_hues else ""
            if hue_value not in labeled_hues:
                labeled_hues.add(hue_value)

            if len(segment) > 1:
                ax.plot(segment["total_trial"], segment["performance_w"], color=palette[color_idx], label=label)
            else:
                # Single point - plot as scatter
                ax.scatter(segment["total_trial"], segment["performance_w"], color=palette[color_idx], label=label)
    if "session_changes" in kwargs and len(kwargs["session_changes"]) > 1:
        for sc in kwargs["session_changes"][1:]:
            tt = df.loc[sc]["total_trial"]
            ax.axvline(tt, linestyle="--", color="gray")
    # ax.set_xlim(left=0)
    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Performance")
    # horizontal line at 50%
    ax.axhline(50, linestyle="--", color="gray")
    # remove box
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend for the different hue values
    handles, labels = ax.get_legend_handles_labels()
    if handles:  # Only create legend if there are handles
        ax.legend()

    if "legend" in kwargs:
        if kwargs["legend"] is False:
            legend = ax.get_legend()
            if legend:
                legend.remove()

    return ax


def _correct_as_bool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copy of the dataframe with "correct" as True / False. Timeouts (correct = None)
    are shown as errors (see data_format_changes.py).
    """
    df = df.copy()
    df["correct"] = df["correct"].apply(dfc.correct_as_number) == 1
    return df


def number_of_correct_responses_plot(df_in: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    column_checker(df_in, required_columns={"correct_side", "correct"})
    df_in = _correct_as_bool(df_in)
    if "stimulus_modality" not in df_in.columns or df_in["stimulus_modality"].nunique() == 1:
        # create a copy of the df
        df = df_in.copy()
    else:
        # create a copy of the df
        df = df_in.copy()
        # Create a new column combining 'correct_side' and 'stimulus_modality' with a line break for the x-axis
        df["column_name_to_substitute"] = df["stimulus_modality"] + "\n" + df["correct_side"]
        # drop correct_side and rename x
        df = df.drop(columns=["correct_side"])
        df = df.rename(columns={"column_name_to_substitute": "correct_side"})
    columns_to_plot = ["correct", "correct_side"]
    if "early_pokeout" in df.columns:
        columns_to_plot.append("early_pokeout")
        # trials where it could not be computed (e.g. habituation) count as successful holds
        df["early_pokeout"] = df["early_pokeout"].astype(object).where(df["early_pokeout"].notna(), False).astype(bool)
        # Prepare data for plotting
        grouped = df.groupby(columns_to_plot).size().unstack(fill_value=0)
        # Call the new function
        return partially_stacked_barplot(grouped, ax=ax)
    else:
        return correct_left_and_right_plot(df, ax=ax)


def correct_left_and_right_plot(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    column_checker(df, required_columns={"correct_side", "correct"})
    df = _correct_as_bool(df)
    # Define colors
    colors = {True: "forestgreen", False: "firebrick"}
    sns.countplot(data=df, x="correct_side", hue="correct", ax=ax, hue_order=[False, True], palette=colors)
    ax.set_xlabel("Correct side")
    ax.set_ylabel("Number of trials")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
    ax.legend(["Error", "Correct"], bbox_to_anchor=(0.5, 1), loc="upper center", ncol=2, borderaxespad=0.0)
    ax.get_legend().get_frame().set_linewidth(0.0)
    ax.get_legend().set_frame_on(False)
    # change labels of the legend
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def partially_stacked_barplot(grouped: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    # Define colors
    colors = {True: "forestgreen", False: "firebrick"}

    # Plot bars (a side with no correct, or no incorrect, trials gets an empty bar)
    sides = grouped.index.get_level_values("correct_side").unique()
    for i, correct in enumerate([True, False]):
        if correct in grouped.index.get_level_values("correct"):
            base = grouped.xs(correct, level="correct")
        else:
            base = pd.DataFrame(0, index=sides, columns=grouped.columns)
        base = base.reindex(sides, fill_value=0)

        base.plot(
            kind="bar",
            stacked=True,
            color=[colors[correct] if not early else "none" for early in base.columns],
            edgecolor=colors[correct],
            ax=ax,
            position=i,
            width=0.4,
            legend=False,
        )

    # Customize plot
    ax.set_xlabel("Correct side")
    ax.set_ylabel("Number of trials")
    ax.set_xticks(range(len(grouped.index.get_level_values("correct_side").unique())))
    ax.set_xticklabels(grouped.index.get_level_values("correct_side").unique(), rotation=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust xlim to ensure bars are centered properly
    ax.set_xlim(-0.5, len(grouped.index.get_level_values("correct_side").unique()) - 0.5)

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="forestgreen", markersize=10, label="Correct"),
        Line2D([0], [0], marker="o", color="firebrick", markersize=10, label="Incorrect"),
        Patch(facecolor="gray", edgecolor="gray", label="Successful center hold"),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, 1), loc="lower center", ncol=2, frameon=False)

    return ax


def repeat_or_alternate_performance_plot(df: pd.DataFrame, ax: plt.Axes = None, **kwargs) -> plt.Axes:
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
    if "session_changes" in kwargs and len(kwargs["session_changes"]) > 1:
        for sc in kwargs["session_changes"][1:]:
            tt = df.loc[sc]["total_trial"]
            ax.axvline(tt, linestyle="--", color="gray")
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
    # ax.set_xlim(left=0)
    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])
    # put the legend at the top in one row
    ax.legend(bbox_to_anchor=(0.5, 1.05), loc="upper center", ncol=2, borderaxespad=0.0)
    # remove box
    ax.get_legend().get_frame().set_linewidth(0.0)
    ax.get_legend().set_frame_on(False)

    return ax


def side_correct_performance_plot(df: pd.DataFrame, ax: plt.Axes, trials_to_show: int = 50) -> plt.Axes:
    # used for real time plotting
    column_checker(df, required_columns={"trial", "correct_side", "correct"})
    ax.clear()
    # select only the last X trials
    df = _correct_as_bool(df.tail(trials_to_show))
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


def performance_by_decision_plot(df: pd.DataFrame, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    column_checker(df, required_columns={"correct", "correct_side_repeat_or_alternate", "trial_group"})
    sns.lineplot(
        data=df, x="trial_group", y="correct", hue="correct_side", style="repeat_or_alternate", errorbar=None, ax=ax
    )
    # rotate the x-axis labels and align them to the end
    # for label in ax.get_xticklabels():
    #     label.set_rotation(45)
    #     label.set_horizontalalignment('right')
    # ax.set_title('Performance by "decision"', pad=20)
    ax.set_xlabel("Trials")
    ax.set_ylabel("Performance")
    # Filter legend to only include entries with handles (skip titles like "correct_side")
    handles, labels = ax.get_legend_handles_labels()
    # clean handles and labels to only include the ones that are not titles
    for handle, label in zip(handles, labels):
        if label not in ["left", "right", "repeat", "alternate"]:
            handles.remove(handle)
            labels.remove(label)
    # Recreate legend with only handle-associated labels
    ax.legend(handles=handles, labels=labels, frameon=False, title="")

    # remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # start the x axis at min value
    try:
        ax.set_xlim(left=df["trial_group"].min())
    except Exception as e:
        print(f"Error setting x-axis limits: {e}")

    return ax

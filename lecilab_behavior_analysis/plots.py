import calplot
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from lecilab_behavior_analysis.utils import column_checker


def rasterized_calendar_plot(dates_df: pd.DataFrame) -> np.ndarray:
    # make the calendar plot and convert it to an image
    column_checker(dates_df, required_columns={"trial"})
    cpfig, _ = calplot.calplot(
        data=dates_df.trial, yearlabel_kws={"fontname": "sans-serif"}
    )
    canvas = FigureCanvasAgg(cpfig)
    canvas.draw()
    width, height = cpfig.get_size_inches() * cpfig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
        int(height), int(width), 3
    )
    plt.close(cpfig)
    return image


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
    return ax_to_use


def session_summary_text(
    df: pd.DataFrame, ax: plt.Axes = None, mouse_name: str = ""
) -> plt.Axes:
    """
    summary of a particular session
    """
    # if more than one session is there, raise an error
    if df.date.nunique() > 1:
        raise ValueError("The dataframe contains more than one session")
    if ax is None:
        ax = plt.gca()
    # get the date
    date = df.date.iloc[0]
    # get the current training stage
    current_training_stage = df.current_training_stage.iloc[0]
    # get the number of trials
    n_trials = df.shape[0]
    # get the number of correct trials
    n_correct = df.correct.sum()
    # get the performance
    performance = n_correct / n_trials * 100
    # get the water consumed
    water = df.water.sum()

    # write the text
    text = f"""\
    Mouse: {mouse_name}
    Date: {date}
    Training stage: {current_training_stage}
    Number of trials: {n_trials}
    Number of correct trials: {n_correct}
    Performance: {performance:.2f}%
    Water consumed: {water} μl
    """
    ax.text(0, 0, text, fontsize=10)
    ax.axis("off")
    return ax


def correct_left_and_right_plot(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    column_checker(df, required_columns={"correct_side", "correct"})
    sns.countplot(data=df, x="correct_side", hue="correct", ax=ax)
    ax.set_xlabel("Correct side")
    ax.set_ylabel("Number of trials")
    ax.legend(title="Correct")
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
    return ax

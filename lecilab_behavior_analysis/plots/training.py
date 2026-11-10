"""Plots of the training history: calendar, trials and water per day, text summaries."""

from typing import List, Optional

import calplot
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from lecilab_behavior_analysis.common import column_checker, list_to_colors
from lecilab_behavior_analysis.metrics.performance import get_text_from_subject_df, get_text_from_subset_df


def training_calendar_plot(dates_df: pd.DataFrame) -> plt.Figure:
    # make the calendar plot and convert it to an image
    column_checker(dates_df, required_columns={"trial"})
    cpfig, _ = calplot.calplot(data=dates_df.trial, yearlabel_kws={"fontname": "sans-serif"})

    return cpfig


def rasterize_plot(plot: plt.Figure, dpi: int = 300) -> np.ndarray:
    plot.set_dpi(dpi)
    canvas = FigureCanvasAgg(plot)
    canvas.draw()
    width, height = plot.get_size_inches() * dpi
    # Get ARGB buffer and reshape
    argb = np.frombuffer(canvas.tostring_argb(), dtype="uint8").reshape(
        int(height),
        int(width),
        4,  # 4 channels (A, R, G, B)
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


def trials_by_day_plot(dates_df: pd.DataFrame, ax: plt.Axes = None, cmap: str = "tab10") -> plt.Axes:
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
    # rotate the x-axis labels and align them to the end
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    return ax


def trials_by_session_hist(dates_df: pd.DataFrame, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    column_checker(dates_df, required_columns={"trial"})
    dates_df["trial"].hist(ax=ax, orientation="horizontal", bins=15, color="gray")
    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])
    # remove the axis
    ax.axis("off")
    # flip x axis
    ax.invert_xaxis()
    return ax


def water_by_date_plot(water_df: pd.Series, ax: plt.Axes = None) -> plt.Axes:
    # if water_df.index.name != "date":
    #     raise ValueError("The dataframe must have [date] as index")
    # check if there is something plotted in the axis already
    items_in_axis = len(ax.patches) + len(ax.lines) + len(ax.collections) + len(ax.texts)
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


def summary_text_plot(df: pd.DataFrame, kind: str = "session", ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """
    summary of a particular session or subject
    """
    if ax is None:
        ax = plt.gca()
    # if the dataframe is empty
    if df.empty:
        text = "No data available"
    elif kind == "session":
        text = get_text_from_subset_df(df, reduced=True)
    elif kind == "subject":
        text = get_text_from_subject_df(df)
    else:
        raise ValueError("kind must be 'session' or 'subject'")
    if "fontsize" in kwargs:
        fontsize = kwargs["fontsize"]
    else:
        fontsize = 10
    ax.text(0, 0.5, text, fontsize=fontsize, va="center", transform=ax.transAxes)
    ax.axis("off")
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
    training_stages = [mat_df_metadata[session]["current_training_stage"] for session in mat_df.columns]
    trials_list = [mat_df_metadata[session]["n_trials"] for session in mat_df.columns]
    evidence_levels = mat_df.index
    shift_up = 0.5
    training_stages_done = []
    for tr_stg in training_stages:
        if tr_stg in training_stages_done:
            continue
        tr_stg_idx = [i for i, x in enumerate(training_stages) if x == tr_stg]
        if training_stage_inlegend:
            label = tr_stg
        else:
            label = None
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
        t_label_uniques = [mat_df_metadata[session][t_label] for session in mat_df.columns]
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
    sp.set_xticklabels(sp.get_xticklabels(), rotation=45, horizontalalignment="right")
    sp.set_title(mouse_name + "\n\n", fontsize=20, fontweight=0)

    return ax


def plot_table_from_df(df: pd.DataFrame, **kwargs) -> go.Figure:
    # Create table
    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[80] + [60] * len(df.columns),  # Adjust column widths
                header=dict(
                    values=[f"<b>{col.replace('_', '<br>')}</b>" for col in df.columns],
                    align="center",
                    font=dict(size=14),
                    height=40,
                    line_color="darkslategray",
                    fill_color="lightgray",
                ),
                cells=dict(
                    values=[df[col].tolist() for col in df.columns],
                    align="center",
                    font=dict(size=12),
                    height=35,
                    line_color="darkslategray",
                    fill_color="white",
                ),
            )
        ]
    )

    # Update layout
    if "title" in kwargs:
        fig.update_layout(title_text=kwargs["title"], title_x=0.5)
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        height=max(250, 30 * len(df)),  # Ensure a minimum height of 100
    )

    return fig


# ---------------------------------------------------------------------------
# Plots used by the figures (figures/panels.py). Days are real dates on the
# x axis, so that gaps in training are visible and every "by day" panel of a
# figure shares the same axis.
# ---------------------------------------------------------------------------
def format_date_axis(ax: plt.Axes) -> plt.Axes:
    """Readable dates on the x axis, whatever the number of days."""
    import matplotlib.dates as mdates

    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    left, right = ax.get_xlim()
    if right - left < 10:  # a few days: one tick per day, never hours
        locator = mdates.DayLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlabel("")
    return ax


def calendar_plot(daily_trials: pd.Series, axes: List[plt.Axes], cmap: str = "viridis") -> List[plt.Axes]:
    """
    Calendar of the number of trials per day, one row of axes per year
    (axes must have one axis per year in daily_trials).

    Args:
        daily_trials: number of trials, indexed by day (datetime).
    """
    years = sorted(daily_trials.index.year.unique())
    if len(axes) != len(years):
        raise ValueError(f"calendar_plot needs one axis per year ({len(years)})")
    vmax = daily_trials.max()
    for ax, year in zip(axes, years):
        calplot.yearplot(
            daily_trials, year=year, ax=ax, vmin=0, vmax=vmax, cmap=cmap, linewidth=0.5, dayticks=[0, 2, 4, 6]
        )
        ax.set_ylabel(str(year))
        ax.tick_params(labelsize=6)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap=cmap)
    axes[0].figure.colorbar(mappable, ax=axes, shrink=0.8, aspect=10, label="trials")
    return axes


def trials_by_day_stacked_plot(
    sessions: pd.DataFrame, ax: plt.Axes = None, cmap: str = "tab10", stage_order: Optional[List[str]] = None
) -> plt.Axes:
    """
    Number of trials per day, one stacked bar segment per session, colored by
    training stage.

    Args:
        sessions: one row per session, with columns day (datetime),
            current_training_stage and trials.
    """
    column_checker(sessions, required_columns={"day", "current_training_stage", "trials"})
    if ax is None:
        ax = plt.gca()
    sessions = sessions.sort_values("day", kind="stable")
    stages = stage_order or list(pd.unique(sessions["current_training_stage"]))
    colormap = plt.get_cmap(cmap)
    colors = {stage: colormap(i % colormap.N) for i, stage in enumerate(stages)}
    bottoms = sessions.groupby("day")["trials"].cumsum() - sessions["trials"]
    ax.bar(
        sessions["day"],
        sessions["trials"],
        bottom=bottoms,
        width=0.8,
        color=[colors[s] for s in sessions["current_training_stage"]],
        edgecolor="black",
        linewidth=0.2,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in stages]
    ax.legend(handles, stages, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(stages), 4))
    ax.set_ylabel("Number of trials")
    format_date_axis(ax)
    return ax


def water_by_day_plot(water: pd.Series, ax: plt.Axes) -> plt.Axes:
    """Water consumed per day (index: datetime) as a line on a right-hand axis."""
    ax2 = ax.twinx()
    # one value per calendar day, so that the line breaks where there is no data
    water = water.sort_index()
    water = water[~water.index.duplicated()].asfreq("D")
    ax2.plot(water.index, water.values, color="black", linewidth=1)
    ax2.set_ylabel("Water consumed (μl)")
    ax2.set_ylim(bottom=0)
    ax2.spines["right"].set_visible(True)
    return ax2

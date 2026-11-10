"""Plots of choices: psychometric curves, choices by difficulty, bias."""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from lecilab_behavior_analysis.common import column_checker
from lecilab_behavior_analysis.models.psychometric import fit_lapse_logistic_independent


def psychometric_plot(
    df: pd.DataFrame, x, y, ax: plt.Axes = None, point_kwargs=None, line_kwargs=None, valueType="discrete", bins=6
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    default_point_kwargs = {
        "color": "cornflowerblue",
        "markers": "o",
        "errorbar": ("ci", 95),
        "label": "Observed Choices",
        "native_scale": True,
        "linestyles": "",
        "markersize": 5,
        "capsize": 0.01,
        "alpha": 1.0,
    }
    if point_kwargs is None:
        point_kwargs = {}
    point_kwargs = {**default_point_kwargs, **point_kwargs}

    default_line_kwargs = {
        "color": "lightcoral",
        "label": "Lapse Logistic Fit (Independent)",
        "linestyle": "-",
        "alpha": 1.0,
    }
    if line_kwargs is None:
        line_kwargs = {}
    line_kwargs = {**default_line_kwargs, **line_kwargs}

    column_checker(df, required_columns={x, y})
    df_copy = df.copy(deep=True)
    if valueType == "discrete":
        if 0 in df_copy[x].unique():
            # if there are 0s in the data, we need to add a small value to avoid log(0)
            df_copy[x + "_fit"] = df_copy[x]
            df_copy[x + "_fit"].replace(0, 0.0001, inplace=True)
        df_copy[x + "_fit"] = np.sign(df_copy[x]) * (np.log(abs(df_copy[x])).round(4))
        ax.set_xlabel("log_" + x)
    else:
        # bin the continuous values when valueType is not discrete
        # each value is replaced by the mean of its bin (bins may be empty,
        # e.g. sound levels never close to 0)
        bin_groups = pd.cut(df_copy[x], bins=bins)
        df_copy[x + "_fit"] = df_copy[x].groupby(bin_groups, observed=True).transform("mean").astype(float)
        # df_copy[x + "_fit"] = np.sign(df_copy[x + "_fit"]) * (np.log(abs(df_copy[x + "_fit"]*10)).round(4))
        # ax.set_xlabel("log_" + x + "*10")
        ax.set_xlabel(x + " (binned)")
    sns.pointplot(x=x + "_fit", y=y, data=df_copy, estimator=lambda x: np.mean(x), ax=ax, **point_kwargs)
    xs = np.linspace(df_copy[x + "_fit"].min(), df_copy[x + "_fit"].max(), 100).reshape(-1, 1)
    p_left, _ = fit_lapse_logistic_independent(df_copy[x + "_fit"], df_copy[y])
    ax.plot(xs, p_left, **line_kwargs)
    if y == "first_choice_numeric":
        ax.set_ylabel("P(Leftward Choices)")
    elif y == "correct_choice_numeric":
        ax.set_ylabel("P(Correct Choices)")
    ax.set_ylim(0, 1)
    ax.legend()
    # remove legend box
    ax.get_legend().get_frame().set_linewidth(0.0)
    # despine
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def choice_by_difficulty_plot(df: pd.DataFrame, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """
    Plot the performance by difficulty of each trial.
    """
    column_checker(df, required_columns={"side_difficulty", "first_choice_numeric"})
    if ax is None:
        ax = plt.gca()

    sns.pointplot(
        x="side_difficulty",
        y="first_choice_numeric",
        data=df,
        estimator=lambda x: np.nanmean(x),
        ax=ax,
        # remove line
        linestyles="",
        # do categorical
        native_scale=False,
        **kwargs,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Leftward Choices")
    ax.set_xlabel("Leftward Evidence Level")
    # despine
    ax.spines[["top", "right"]].set_visible(False)

    return ax


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


def triangle_polar_plot(df_bias: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    column_checker(df_bias, required_columns={"subject", "bias_angle", "percentage"})
    if ax is None:
        ax = plt.subplot(111, polar=True)
    # Plot each subject in a polar plot
    for subject in df_bias["subject"].unique():
        subject_data = df_bias[df_bias["subject"] == subject]

        # Extract the angles and distances
        angles = subject_data["bias_angle"].values
        percentages = subject_data["percentage"].values

        # Close the triangle by repeating the first point
        angles = np.append(angles, angles[0])
        percentages = np.append(percentages, percentages[0])

        # Plot the triangle
        ax.plot(angles, percentages, marker="o", linestyle="-", label=subject)

    # Set title and formatting
    # ax.set_title(f"Animal biases", va='bottom', fontsize=16)
    ax.set_rlabel_position(-22.5)  # Adjust label position
    ax.grid(True)

    # make the circular 0.33 grid thicker
    ax.yaxis.grid(True, color="gray", linestyle="--", linewidth=0.5)
    # plot the 0.33 grid
    ax.set_yticks([0.25, 0.5])

    ax.set_xticks([2 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
    ax.set_xticklabels(["Alternate", "Left", "Right"], fontsize=14)

    # Show the legend outside the plot
    plt.legend(loc="upper right", bbox_to_anchor=(1.4, 0.8), ncol=1)
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
    sns.lineplot(
        data=df,
        x="left_right_bias",
        y="alternating_bias",
        ax=ax,
        hue=hue,
        palette=palette,
        legend=False,
        linewidth=3,
        alpha=0.8,
    )
    # connect the points in the order of the session
    # plt.plot(df_bias_pivot['xs'], df_bias_pivot['ys'], marker='o', linestyle='-')

    # plot a triangle between -1 and 1 in the x axis and 0 and 1 in the y axis
    triangle = np.array([[0, 1], [1, 0], [-1, 0], [0, 1]])
    ax.plot(triangle[:, 0], triangle[:, 1], color="black", linestyle="--")
    # draw a dashed circle in 0, 0.5 of radius 0.1, no linestyle, fill it with black and alpha 0.1
    circle = plt.Circle((0, 0.5), 0.1, color="black", fill=True, alpha=0.1)
    ax.add_artist(circle)
    # label the triangle vertices
    ax.text(0, 1.05, "Alternating", ha="center", va="center")
    ax.text(1.05, -0.05, "Right", ha="center", va="center")
    ax.text(-1.05, -0.05, "Left", ha="center", va="center")
    # remove spines and ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    return ax


def plot_filter_model_variables(corr_mat_list: list, norm_contribution_df: pd.DataFrame, **kwargs) -> plt.Axes:
    """Plot the mean correlation matrix and the mean contribution of each variable.
    corr_mat_list: list of correlation matrices
    norm_contribution_df: DataFrame with the normalized contribution of each variable
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    # get the variable names from the first correlation matrix
    X = corr_mat_list[0].index
    # Calculate the mean correlation matrix
    corr_mat_mean = np.mean(np.stack(corr_mat_list), axis=0)
    # # Create a mask for the upper triangle
    # mask = np.triu(np.ones_like(corr_mat_mean, dtype=bool), k=1)
    sns.heatmap(corr_mat_mean, ax=ax[0], cmap="coolwarm", annot=True, fmt=".2f")
    ax[0].set_xticklabels(X, rotation=16, ha="right", rotation_mode="anchor")
    ax[0].set_yticklabels(X, rotation=8, ha="right", rotation_mode="anchor")
    ax[0].set_title("Mean Correlation Matrix")
    # Plot the mean contribution of each variable
    norm_contribution_df.mean(axis=1).sort_values().plot(kind="barh", ax=ax[1])
    ax[1].set_xlabel("Mean Contribution")
    plt.tight_layout()
    plt.show()


def pointplot_stimulus_dual_dependence(
    df,
    x_variable,
    column_name,
    hue_variable,
    ax=None,
    folded=False,
    x_label=None,
    y_label=None,
    estimator=np.mean,
):
    """Plot a pointplot showing the dependence of a column on the stimulus evidence,
    with different lines for a different variable (e.g., overall dB)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    x = f"{x_variable}_abs" if folded else x_variable
    x_label = x_variable if x_label is None else x_label
    x_label = f"{x_label} (abs)" if folded else x_label
    ax = sns.pointplot(
        data=df,
        x=x,
        y=column_name,
        hue=hue_variable,
        dodge=True,
        markers="o",
        linestyles="-",
        palette="coolwarm",
        native_scale=True,
        ax=ax,
        estimator=estimator,
    )
    ax.set_xlabel(x_label, size=14)
    ax.set_ylabel(y_label if y_label is not None else column_name, size=14)
    # change the size of the ticks using tick_params
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    # plt.xticks(rotation=45, ha='right')
    # despine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # plt.tight_layout()
    return ax

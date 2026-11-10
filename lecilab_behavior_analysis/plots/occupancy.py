"""Plots of when the box was used, and transitions between events."""

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_percentage_of_occupancy_per_day(daily_percentages: pd.Series, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    # Plot the percentage of time for each day
    daily_percentages.plot(kind="line", ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Percentage of time (%)")
    ax.set_title("Percentage of time that task is running per day")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 100)
    ax.grid(True)
    return ax


def plot_training_times_heatmap(heatmap_vector: np.ndarray, ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    # Plot the heatmap vector
    sns.heatmap(
        heatmap_vector.reshape(1, -1),
        cmap="YlGnBu",
        cbar=True,
        xticklabels=60,
        vmin=0,
        vmax=np.max(heatmap_vector),
        ax=ax,
    )
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Event Occurrences")
    ax.set_title("Training times")
    # Set the x-ticks to show every hour
    ax.set_xticks(np.arange(0, 1440, 60))
    ax.set_xticklabels([f"{i//60:02d}:00" for i in range(0, 1440, 60)], rotation=45)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.grid(False)
    plt.tight_layout()

    return ax


def plot_training_times_clock_heatmap(heatmap_vector: np.ndarray, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3), subplot_kw={"projection": "polar"})
    # Create a polar plot
    theta = np.linspace(0, 2 * np.pi, 1440)  # 1440 minutes in a day

    # Plot the heatmap as a polar bar plot
    ax.bar(
        theta,
        heatmap_vector,
        width=2 * np.pi / 1440,
        color=plt.cm.YlGnBu(heatmap_vector / np.max(heatmap_vector)),
        edgecolor="none",
    )

    # change the limit of the radius if specified
    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])

    # shade in light gray from 20:00 to 8:00 and put it behind the bars
    # get the maximum value of the plot radius
    max_radius = ax.get_ylim()[1]
    ax.fill_between(
        theta,
        0,
        max_radius,
        where=(theta >= 20 * np.pi / 12) | (theta <= 8 * np.pi / 12),
        color="lightgray",
        alpha=0.5,
        zorder=0,
    )

    # Customize the plot to resemble a clock
    try:
        ax.set_theta_zero_location("N")  # Set midnight at the top
        ax.set_theta_direction(-1)  # Set clockwise direction
    except AttributeError:
        print("initialize your figure with subplot_kw={'projection': 'polar'} to make it circular")

    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))  # Hourly ticks
    ax.set_xticklabels([f"{i}:00" for i in range(0, 24, 2)])  # Label every 2 hours
    ax.set_yticks([])  # Remove radial ticks
    ax.set_title("Training Times", pad=25)

    return ax


def plot_transition_matrix(transition_matrix: pd.DataFrame, figsize: tuple = (5, 5)) -> plt.Figure:
    # plot the heatmap of the transition matrix with clustering
    # if ax is None:
    #     ax = plt.gca()

    # sns.heatmap(transition_matrix, annot=True, fmt='d', cmap='YlGnBu')
    fig = sns.clustermap(
        transition_matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=True,
        xticklabels=transition_matrix.columns,
        yticklabels=transition_matrix.index,
        figsize=figsize,
        dendrogram_ratio=(0.2, 0.2),
        cbar_kws={"shrink": 0.8},
    )
    # get the axis of the clustermap
    ax = fig.ax_heatmap
    # set the title and labels
    ax.set_title("Transition Matrix Heatmap", fontsize=16)
    ax.set_xlabel("To Item")
    ax.set_ylabel("From Item")

    return fig


def plot_transition_network_with_curved_edges(
    transition_matrix: pd.DataFrame, threshold: int = 0, figsize: tuple = (10, 10)
) -> plt.Figure:
    # Create a directed graph from the transition matrix
    G = nx.from_pandas_adjacency(transition_matrix, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency

    # Calculate total edge weight and filter edges above the threshold
    total_weight = sum(G[u][v]["weight"] for u, v in G.edges())
    min_weight = threshold * total_weight / 100  # Calculate the minimum weight based on % threshold

    # Edge attributes: Filter edges above the threshold and scale them
    filtered_edges = [(u, v) for u, v in G.edges() if G[u][v]["weight"] >= min_weight]
    edge_weights = [G[u][v]["weight"] for u, v in filtered_edges]

    # Normalize edge properties for filtered edges
    if edge_weights:
        max_weight = max(edge_weights)
    else:
        max_weight = 1  # Prevent division by zero if no edges remain

    edge_widths = [2 + (5 * G[u][v]["weight"] / max_weight) for u, v in filtered_edges]
    edge_alphas = [0.3 + (0.7 * G[u][v]["weight"] / max_weight) for u, v in filtered_edges]

    # Apply a colormap to the edges based on normalized weights
    norm = mcolors.Normalize(vmin=min(edge_weights, default=0), vmax=max_weight)
    edge_colors = [cm.YlGnBu(norm(G[u][v]["weight"])) for u, v in filtered_edges]

    # Create a figure
    plt.figure(figsize=figsize)

    # Draw larger, fully opaque nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="lightblue", alpha=1.0)

    # Draw curved edges with varying properties, terminating early at the nodes
    for i, (u, v) in enumerate(filtered_edges):
        # Curve edges if they're bidirectional
        if G.has_edge(v, u) and (u, v) != (v, u):
            curve_scale = 0.2  # Add curvature for bidirectional edges
        else:
            curve_scale = 0

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=edge_widths[i],
            alpha=edge_alphas[i],
            edge_color=[edge_colors[i]],
            connectionstyle=f"arc3,rad={curve_scale}",
            arrows=True,
            arrowsize=15,
            arrowstyle="-|>",
            min_source_margin=10,
            min_target_margin=10,  # Terminate arrows earlier at the node edges
        )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Transition Frequency", fontsize=12)

    plt.title(f"Transition Network (Edges above {threshold}% of total weight)", fontsize=16)

    return plt.gcf()

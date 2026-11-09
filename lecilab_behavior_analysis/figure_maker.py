"""
The figures used by the daily report (village-report-generator), the
visualizer and the notebooks.

subject_progress_figure and session_summary_figure keep their old names and
arguments. They are drawn from panels (see the figures package): a panel
that fails shows a red "Could not generate ..." box instead of stopping the
figure, and the failures are listed in fig.panel_problems.
"""
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from lecilab_behavior_analysis.figures import session_summary, subject_progress


def subject_progress_figure(df: pd.DataFrame, **kwargs) -> Figure:
    """
    Training history of one subject.

    Keyword arguments: width, height, perf_window (default 50),
    summary_matrix_plot (default False), events_df (for the engagement panel).
    The panels that failed are listed in fig.panel_problems.
    """
    fig, _ = subject_progress(
        df,
        perf_window=kwargs.get("perf_window", 50),
        events_df=kwargs.get("events_df"),
        summary_matrix_plot=kwargs.get("summary_matrix_plot", False),
        width=kwargs.get("width"),
        height=kwargs.get("height"),
    )
    return fig


def session_summary_figure(df: pd.DataFrame, **kwargs) -> plt.Figure:
    """
    Summary of some trials of one subject (usually one day).

    Keyword arguments: width, height, perf_window (default 50), perf_hue.
    The panels that failed are listed in fig.panel_problems.
    """
    fig, _ = session_summary(
        df,
        perf_window=kwargs.get("perf_window", 50),
        perf_hue=kwargs.get("perf_hue"),
        width=kwargs.get("width"),
        height=kwargs.get("height"),
    )
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
        yaxis=dict(title="Temperature", range=[18, 26]),
        yaxis2=dict(
            title="Humidity",
            overlaying="y",
            side="right",
            range=[20, 90],
        ),
        template="plotly_white",
    )

    return fig

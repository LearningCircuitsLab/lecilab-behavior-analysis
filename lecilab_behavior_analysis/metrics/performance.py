"""Performance: per session or day, through trials, by difficulty, by decision,
repeat or alternate, and the text summaries of a subject or session.
"""

from typing import Tuple

import numpy as np
import pandas as pd

import lecilab_behavior_analysis.data_format_changes as dfc
from lecilab_behavior_analysis.common import column_checker
from lecilab_behavior_analysis.metrics.trial_events import side_and_difficulty_to_numeric


def get_session_performance(df, session: int) -> float:
    """
    This method calculates the performance of a session.
    """

    return df[df.session == session].correct.mean()


def get_day_performance(df, day: str) -> float:
    """
    This method calculates the performance of a day.
    """

    return df[df.year_month_day == day].correct.mean()


def get_session_number_of_trials(df, session: int) -> int:
    """
    This method calculates the number of trials in a session.
    """

    return df[df.session == session].shape[0]


def get_day_number_of_trials(df, day: str) -> int:
    """
    This method calculates the number of trials in a day.
    """
    return df[df.year_month_day == day].shape[0]


def get_text_from_subset_df(df: pd.DataFrame, reduced: bool = False) -> str:
    # make sure that there is only one subject in the dataframe
    if df.subject.nunique() > 1:
        raise ValueError("The dataframe contains more than one subject.")
    # get the session
    session = df.session.unique()
    # get the date
    date = df.date.unique()
    # get the current training stage
    current_training_stage = df.current_training_stage.unique()
    # get the number of trials
    n_trials = df.shape[0]
    # get the number of correct trials
    # (timeouts, correct = None, count as errors: see data_format_changes.py)
    n_correct = int(df.correct.apply(dfc.correct_as_number).sum())
    # get the performance
    try:
        performance = n_correct / n_trials * 100
    except ZeroDivisionError:
        performance = np.nan
    # get the water consumed
    water = df.water.sum()
    # get the subject
    mouse_name = df.subject.unique()[0]

    def _join(values) -> str:
        return ", ".join(str(v) for v in values)

    if reduced and len(session) > 3:
        session_text = f"{session[0]} ... {session[-1]}"
        date_text = f"{date[0]} ... {date[-1]}"
    else:
        session_text = _join(session)
        date_text = _join(date)

    # write the text
    text = "\n".join(
        [
            f"Mouse: {mouse_name}",
            f"Sessions: {session_text}",
            f"Dates: {date_text}",
            f"Training stages: {_join(current_training_stage)}",
            f"Number of trials: {n_trials}",
            f"Number of correct trials: {n_correct}",
            f"Performance: {performance:.2f}%",
            f"Water consumed: {water:g} μl",
        ]
    )

    return text


def get_text_from_subject_df(df: pd.DataFrame) -> str:
    # make sure that there is only one subject in the dataframe
    if df.subject.nunique() != 1:
        raise ValueError("The dataframe contains more than one subject.")
    column_checker(df, {"subject", "session", "water", "year_month_day"})
    # get the subject
    mouse_name = df.subject.unique()[0]
    # get the average water consumed per day
    days_of_training = df.year_month_day.nunique()
    avg_water_per_day = df.water.sum() / days_of_training if days_of_training > 0 else 0
    # get the number of trials per day on average
    avg_trials_per_day = df.shape[0] / days_of_training if days_of_training > 0 else 0
    # get the number of sessions per day on average
    n_sessions = df.session.nunique()
    avg_sessions_per_day = n_sessions / days_of_training if days_of_training > 0 else 0

    # write the text
    text = "\n".join(
        [
            f"{mouse_name}",
            f"{avg_sessions_per_day:.2f} sessions per day",
            f"{avg_trials_per_day:.0f} trials per day",
            f"{avg_water_per_day:.0f} μl of water per day",
        ]
    )

    return text


def get_repeat_or_alternate_series(series: pd.Series) -> pd.Series:
    """
    Given a series of values, return a series of the same length
    with "repeat" or "alternate" depending on whether the value
    is the same as the previous one or not.

    Meant to be used for the trial side column (correct_side)
    Or the columns with the first choice of the mouse
    """
    prev_choices = series.shift(1, fill_value=np.nan)
    return repeat_or_alternate_series_comparison(series, prev_choices)


def repeat_or_alternate_series_comparison(series: pd.Series, comparison_series: pd.Series) -> pd.Series:
    """
    Given a series of values, return a series of the same length
    with "repeat" or "alternate" depending on whether the value
    is the same between the two series or not.
    """
    repeat_or_alternate = np.where(
        series.isna() | comparison_series.isna(),
        None,  # None instead of np.nan to keep dtype the same TODO: check!
        np.where(series == comparison_series, "repeat", "alternate"),
    )

    return pd.Series(repeat_or_alternate, index=series.index)


def get_performance_through_trials(df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    column_checker(df, required_columns={"session", "trial", "correct"})
    # make sure only one subject is present
    if df["subject"].nunique() > 1:
        raise ValueError("The dataframe should contain only one subject.")
    # sort the df by "session" and "trial"
    df = df.sort_values(["session", "trial"])
    # add a column with the total number of trials if it doesn't exist
    if "total_trial" not in df.columns:
        df["total_trial"] = np.arange(1, df.shape[0] + 1)
    # calculate the performance as a mean of the last X trials
    # timeouts (correct = None) count as errors (see data_format_changes.py)
    df["performance_w"] = df.correct.apply(dfc.correct_as_number).rolling(window=window).mean() * 100

    return df


def get_repeat_or_alternate_performance(df_in: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    df = df_in.copy()  # Create a copy to avoid modifying the original DataFrame
    if "repeat_or_alternate" not in df.columns:
        df["repeat_or_alternate"] = get_repeat_or_alternate_series(df.correct_side)

    df["repeat_or_alternate_performance"] = df.groupby("repeat_or_alternate")["correct"].transform(
        lambda x: x.rolling(window=window).mean() * 100
    )
    return df


def get_evidence_ratio(df):
    # raise error
    raise ValueError("Not implemented yet")
    return df


def get_performance_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    column_checker(df, required_columns={"difficulty", "correct", "correct_side"})
    pbd_df = df.groupby(["difficulty", "correct_side"]).correct.mean().unstack().reset_index()
    # melt the dataframe
    pbd_df = pbd_df.melt(id_vars=["difficulty"])
    # assign a numeric value to the side and the difficulty for plotting purposes
    pbd_df["leftward_evidence"] = pbd_df.apply(side_and_difficulty_to_numeric, axis=1)
    pbd_df["leftward_choices"] = np.where(pbd_df["correct_side"] == "left", pbd_df["value"], 1 - pbd_df["value"])
    return pbd_df


def get_training_summary_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    column_checker(df, required_columns={"session"})
    # Initialize lists to save important data
    leftward_evidence_list = []
    leftward_choices_list = []
    # Initialize a dicctionary to save information about the sessions
    session_info = {}

    # process data from all sessions
    for session in df.session.unique():
        pbd_df = get_performance_by_difficulty(df[df.session == session])
        leftward_evidence_list.append(pbd_df.leftward_evidence.to_list())
        leftward_choices_list.append(pbd_df.leftward_choices.to_list())
        session_info[session] = {
            "date": df[df.session == session].date.unique()[0],
            "current_training_stage": df[df.session == session].current_training_stage.unique()[0],
            "n_trials": df[df.session == session].shape[0],
            "n_correct": int(df[df.session == session].correct.sum()),
            "performance": int(df[df.session == session].correct.sum()) / df[df.session == session].shape[0] * 100,
            "water": df[df.session == session].water.sum(),
            "stimulus_modality": df[df.session == session].stimulus_modality.unique()[0],
        }

    # get difficulty levels
    ev_levels = np.unique(np.concatenate(leftward_evidence_list).ravel())
    # Initialize the matrix
    mat_df = np.full([len(ev_levels), len(leftward_evidence_list)], np.nan)
    # Loop to fill it
    for i, evidence in enumerate(ev_levels):
        for j, choice in enumerate(leftward_choices_list):
            if evidence in leftward_evidence_list[j]:
                idx = np.where(leftward_evidence_list[j] == evidence)[0][0]
                mat_df[i, j] = choice[idx]

    # Transform to dataframe
    mat_df = pd.DataFrame(mat_df)
    mat_df = mat_df.set_index(ev_levels)
    mat_df.columns = df.session.unique()

    return mat_df, session_info


def get_performance_by_decision_df(df: pd.DataFrame, trial_group_size: int = 500) -> pd.DataFrame:
    """
    Get the performance by decision dataframe.
    """
    # check that there is only one subject
    if df["subject"].nunique() > 1:
        raise ValueError("The dataframe should contain only one subject.")
    column_checker(df, required_columns={"correct_side", "repeat_or_alternate", "correct", "total_trial"})
    # bin the data into groups of trial_group_size
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    df["trial_group"] = df["total_trial"] // trial_group_size * trial_group_size
    # group by the trial group and calculate the mean of the correct column
    df_binned = df.groupby(["correct_side", "repeat_or_alternate", "trial_group"])[["correct"]].mean().reset_index()
    # performance in between 0 and 100
    df_binned["correct"] = df_binned["correct"] * 100
    # get the different combinations of correct side and repeat or alternate
    df_binned["correct_side_repeat_or_alternate"] = df_binned["correct_side"] + "_" + df_binned["repeat_or_alternate"]
    # get only the 4 combinations and remove nans etc
    df_binned = df_binned[
        df_binned["correct_side_repeat_or_alternate"].isin(
            ["left_repeat", "left_alternate", "right_repeat", "right_alternate"]
        )
    ]

    return df_binned

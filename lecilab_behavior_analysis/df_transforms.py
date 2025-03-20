import pandas as pd
import numpy as np
from typing import Tuple
from lecilab_behavior_analysis.utils import column_checker

def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing data in the dataframe.
    """
    columns_to_fill = ["stimulus_modality", "current_training_stage", "difficulty", "correct_side"]
    for column in columns_to_fill:
        try:
            df[column] = df[column].fillna("not saved")
        except KeyError:
            df[column] = "not saved"
    # Fill missing values in the "correct" column
    try:
        df["correct"] = df["correct"].infer_objects(copy=False)
    except KeyError:
        df["correct"] = True

    return df


def get_dates_df(df: pd.DataFrame) -> pd.DataFrame:
    column_checker(df, required_columns={"date", "current_training_stage"})
    dates_df = df.groupby(["date", "current_training_stage"]).count().reset_index()
    # set as index the date
    dates_df.set_index("date", inplace=True)
    dates_df.index = pd.to_datetime(dates_df.index)
    return dates_df


def get_water_df(df: pd.DataFrame, grouping_column: str = "date") -> pd.DataFrame:
    column_checker(df, required_columns={grouping_column, "water"})
    water_df = df.groupby(grouping_column)["water"].sum()
    return water_df


def get_repeat_or_alternate_series(series: pd.Series) -> pd.Series:
    """
    Given a series of values, return a series of the same length
    with "repeat" or "alternate" depending on whether the value
    is the same as the previous one or not.

    Meant to be used for the trial side column (correct_side)
    """
    prev_choices = series.shift(1, fill_value=np.nan)
    repeat_or_alternate = np.where(series == prev_choices, "repeat", "alternate")
    return pd.Series(repeat_or_alternate, index=series.index)


def get_performance_through_trials(df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    column_checker(df, required_columns={"session", "trial", "correct"})
    # sort the df by "session" and "trial"
    df = df.sort_values(["session", "trial"])
    # add a column with the total number of trials
    df["total_trial"] = np.arange(1, df.shape[0] + 1)
    # calculate the performance as a mean of the last X trials
    df["performance_w"] = df.correct.rolling(window=window).mean() * 100

    return df


def get_repeat_or_alternate_performance(
    df: pd.DataFrame, window: int = 25
) -> pd.DataFrame:
    if "repeat_or_alternate" not in df.columns:
        df["repeat_or_alternate"] = get_repeat_or_alternate_series(df.correct_side)

    df["repeat_or_alternate_performance"] = df.groupby("repeat_or_alternate")[
        "correct"
    ].transform(lambda x: x.rolling(window=window).mean() * 100)
    return df


def get_performance_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    column_checker(df, required_columns={"difficulty", "correct", "correct_side"})
    pbd_df = df.groupby(["difficulty", "correct_side"]).correct.mean().unstack().reset_index()
    # melt the dataframe
    pbd_df = pbd_df.melt(id_vars=["difficulty"])
    # assing a numeric value to the side and the difficulty for plotting purposes
    pbd_df["leftward_evidence"] = pbd_df.apply(side_and_difficulty_to_numeric, axis=1)
    pbd_df["leftward_choices"] = np.where(pbd_df["correct_side"] == "left", pbd_df["value"], 1 - pbd_df["value"])
    return pbd_df


def side_and_difficulty_to_numeric(row: pd.Series) -> float:
    match row.difficulty:
        case "easy":
            numval = 3
        case "medium":
            numval = 2
        case "hard":
            numval = 1
        case _:
            numval = 0
    match row.correct_side:
        case "left":
            pass
        case "right":
            numval *= -1
        case _:
            numval = 0
    
    return round(numval / 3, 3)



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


def calculate_time_between_trials_and_rection_time(df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """
    Calculate Time Between Trials and Reaction Time.
    """
    # Check if the required columns are present
    column_checker(df, required_columns={"Port1In", "Port1Out", "Port2In", "Port2Out", "Port3In", "Port3Out"})
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    df['Time_Between_Trials'] = df.groupby('session')['Port2Out'].diff()
    df['Reaction_Time'] = df.apply(
        lambda row: row['Port1In'] - row['Port2Out'] if pd.notna(row['Port1In']) else row['Port3In'] - row['Port2Out'],
        axis=1
    )

    return df


def add_day_column_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a day column to the dataframe.
    """
    # Check if the required columns are present
    column_checker(df, required_columns={"date"})
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    df['year_month_day'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return df


def add_trial_of_day_column_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a trial of the day column to the dataframe.
    """
    # Check if the required columns are present
    column_checker(df, required_columns={"year_month_day", "trial"})
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    df['trial_of_day'] = df.groupby('year_month_day')['trial'].transform(lambda x: x - x.min() + 1)
    return df

# if __name__ == "__main__":
#     from lecilab_behavior_analysis.utils import load_example_data
#     df = load_example_data("mouse1")
#     summary_matrix_df = summary_matrix(df)
#     print(summary_matrix_df)

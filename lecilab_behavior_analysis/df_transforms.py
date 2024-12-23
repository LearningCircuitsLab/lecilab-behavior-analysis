import pandas as pd
import numpy as np


def get_dates_df(df: pd.DataFrame) -> pd.DataFrame:
    # raise an error if the date column is not present
    if "date" not in df.columns:
        raise ValueError("The dataframe must have a date column")
    if "current_training_stage" not in df.columns:
        raise ValueError("The dataframe must have a current_training_stage column")
    dates_df = df.groupby(["date", "current_training_stage"]).count().reset_index()
    # set as index the date
    dates_df.set_index("date", inplace=True)
    dates_df.index = pd.to_datetime(dates_df.index)
    return dates_df


def get_water_df(df: pd.DataFrame) -> pd.DataFrame:
    # raise an error if the date column is not present
    if "date" not in df.columns:
        raise ValueError("The dataframe must have a date column")
    if "water" not in df.columns:
        raise ValueError("The dataframe must have a water column")
    water_df = df.groupby("date")["water"].sum()
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
    # raise an error if the session column is not present
    if "session" not in df.columns:
        raise ValueError("The dataframe must have a session column")
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

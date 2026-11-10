"""When the box was used: sessions per day, occupancy through the day, water per day."""

import numpy as np
import pandas as pd

from lecilab_behavior_analysis.common import column_checker


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


def get_start_and_end_of_sessions_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per session: subject, session, start_time (start of its first
    trial), end_time (end of its last trial), duration (minutes) and date.
    """
    column_checker(df, required_columns={"subject", "session", "trial", "TRIAL_START", "TRIAL_END"})
    groups = df.groupby(["subject", "session"], sort=False)["trial"]
    first = df.loc[groups.idxmin()]
    last = df.loc[groups.idxmax()]
    occupancy_df = pd.DataFrame(
        {
            "subject": first["subject"].to_numpy(),
            "session": first["session"].to_numpy(),
            "start_time": pd.to_datetime(first["TRIAL_START"].to_numpy(dtype=float), unit="s"),
            "end_time": pd.to_datetime(last["TRIAL_END"].to_numpy(dtype=float), unit="s"),
        }
    )
    # Calculate the duration of each event in minutes
    occupancy_df["duration"] = (occupancy_df["end_time"] - occupancy_df["start_time"]).dt.total_seconds() / 60
    # Group by date and calculate the total duration of events for each day
    occupancy_df["date"] = occupancy_df["start_time"].dt.date
    return occupancy_df


def get_daily_occupancy_percentages(occupancy_df: pd.DataFrame) -> pd.DataFrame:
    daily_durations = occupancy_df.groupby("date")["duration"].sum()
    # Calculate the percentage of the day that the events occupy (1440 minutes in a day)
    return (daily_durations / 1440) * 100


def get_occupancy_heatmap(occupancy_df: pd.DataFrame, window_size: int = 30) -> np.ndarray:
    column_checker(occupancy_df, required_columns={"start_time", "end_time"})
    # Create a vector to represent the heatmap (1440 minutes in a day)
    heatmap_vector = np.zeros(1440)

    # Populate the heatmap vector with event occurrences
    for start, end in zip(occupancy_df["start_time"], occupancy_df["end_time"]):
        start_minute_of_day = start.hour * 60 + start.minute
        end_minute_of_day = end.hour * 60 + end.minute

        if start_minute_of_day <= end_minute_of_day:
            heatmap_vector[start_minute_of_day:end_minute_of_day] += 1
        else:
            heatmap_vector[start_minute_of_day:] += 1
            heatmap_vector[:end_minute_of_day] += 1

    # normalize the heatmap vector to the number of days
    heatmap_vector /= len(occupancy_df.date.unique())

    # add the window size to the beginning and end of the vector
    heatmap_vector = np.concatenate(
        (heatmap_vector[len(heatmap_vector) - window_size :], heatmap_vector, heatmap_vector[:window_size])
    )
    # apply the moving average
    heatmap_vector = np.convolve(heatmap_vector, np.ones(window_size) / window_size, mode="same")
    # cut the vector to the original size
    heatmap_vector = heatmap_vector[window_size : len(heatmap_vector) - window_size]

    return heatmap_vector


def get_occupancy_matrix(occupancy_df: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
    # do the same as in the get_occupancy_heatmap function, but for each day of training
    column_checker(occupancy_df, required_columns={"start_time", "end_time"})
    # get all the possible dates using both start and end times
    all_dates = pd.concat([occupancy_df["start_time"].dt.date, occupancy_df["end_time"].dt.date]).unique()
    # generate the matrix
    occupancy_matrix = pd.DataFrame(index=all_dates, columns=np.arange(0, 1440, 1))
    # fill the matrix with zeros
    occupancy_matrix.fillna(0, inplace=True)
    # Populate the matrix with event occurrences
    # for each row of the dataframe
    counter = 0
    for _, row in occupancy_df.iterrows():
        start_minute_of_day = row["start_time"].hour * 60 + row["start_time"].minute
        end_minute_of_day = row["end_time"].hour * 60 + row["end_time"].minute
        date = row["start_time"].date()

        if start_minute_of_day <= end_minute_of_day:
            occupancy_matrix.loc[date, start_minute_of_day:end_minute_of_day] += 1
        else:
            occupancy_matrix.loc[date, start_minute_of_day:] += 1
            # get the next day
            next_day = date + pd.Timedelta(days=1)
            occupancy_matrix.loc[next_day, :end_minute_of_day] += 1

        counter += 1
        if counter % 100 == 0:
            print(f"Processed {counter} rows out of {occupancy_df.shape[0]}")
    return occupancy_matrix

"""Engagement: how the animal used its time in the box during each session."""
from typing import Tuple
from typing import Union
import numpy as np
import pandas as pd

from lecilab_behavior_analysis.common import column_checker


def find_next_end_task_time_in_events(events_df: pd.DataFrame, date: str, subject: str) -> Tuple[Union[str, None], Union[float, None]]:
    """
    Find the end task time in the events dataframe for a given date.
    """
    # filter events
    filtered_events = events_df[np.logical_or(
        events_df['description'] == "The subject has returned home.",
        events_df['description'].str.startswith("Subject back home:")
    )]

    # get the first event after the given date
    dates_in_datetime = pd.to_datetime(filtered_events['date'])
    end_event = filtered_events[dates_in_datetime > pd.to_datetime(date)]
    # if there are no events after the given date, return None
    if end_event.empty:
        print(f"No end event found after date {date}.")
        return "No date", None
    # calculate the duration from the given date to the end event
    duration = (pd.to_datetime(end_event['date'].iloc[0]) - pd.to_datetime(date)).total_seconds()
    # get the subject as well
    subject_of_event = end_event['subject'].iloc[0]

    if subject_of_event != subject:
        print(f"Subject mismatch for {date}: expected {subject}, found {subject_of_event}.")
        return None, None

    # return the date of the first event
    return end_event['date'].iloc[0], duration


def get_session_box_usage(session_df: pd.DataFrame, session_duration: float) -> pd.DataFrame:

    if session_df.date.unique().size != 1:
        raise ValueError("Session dataframe must contain data for a single date.")

    #TODO: do the column checker


    date = session_df.date.unique()[0]
    subject = session_df.subject.unique()[0]
    time_to_complete_first_trial = session_df.iloc[0].trial_duration
    start_of_first_trial = session_df.iloc[0].TRIAL_START
    last_trial_completed_time = session_df.iloc[-1].TRIAL_END - start_of_first_trial
    time_to_exit_box = session_duration - last_trial_completed_time
    # add accuracy as well
    accuracy = session_df['correct'].mean() * 100

    # add the time of engagement and disengagement, removing the first trial
    session_df = session_df.iloc[1:]  # remove the first trial for engagement calculation
    engaged_time = session_df[session_df['engaged'] == True]['trial_duration'].sum()
    disengaged_time = session_df[session_df['engaged'] == False]['trial_duration'].sum()

    unaccounted_time = session_duration - (time_to_complete_first_trial + time_to_exit_box +
                                           engaged_time + disengaged_time)
    
    total_session_time = time_to_complete_first_trial + time_to_exit_box + engaged_time + disengaged_time + unaccounted_time

    return pd.DataFrame({
        "date": [date] * 5,
        "subject": [subject] * 5,
        "time_type": [
            "time_to_complete_first_trial",
            "time_to_exit_box",
            "engaged_time",
            "disengaged_time",
            "unaccounted_time"
        ],
        "absolute_time": [
            time_to_complete_first_trial,
            time_to_exit_box,
            engaged_time,
            disengaged_time,
            unaccounted_time
        ],
        "percentage_of_time": [
            time_to_complete_first_trial / total_session_time * 100,
            time_to_exit_box / total_session_time * 100,
            engaged_time / total_session_time * 100,
            disengaged_time / total_session_time * 100,
            unaccounted_time / total_session_time * 100
        ],
        "accuracy": [accuracy] * 5,
    })


def add_engagement_column(df_in: pd.DataFrame, engagement_sd_criteria: float = 2) -> pd.DataFrame:
    column_checker(df_in, required_columns={"trial_duration"})
    # make sure there is only one subject
    if df_in['subject'].nunique() > 1:
        raise ValueError("The dataframe should contain only one subject.")
    # make a copy of the dataframe to avoid modifying the original
    df = df_in.copy()
    # get the median and std of the log of the trial_duration
    df['td_log'] = df['trial_duration'].copy().apply(lambda x: np.log(x))
    median_td_log = df['td_log'].median()
    std_td_log = df['td_log'].std()

    # classify trials as engaged or not engaged depending on the td_log
    df['engaged'] = df['td_log'] < (median_td_log + engagement_sd_criteria * std_td_log)

    return df


def get_box_usage_df(df: pd.DataFrame, events_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    gsbu_dfs = []
    # keep only the "back home" events, with dates read once (instead of once per session)
    events_df = events_df[np.logical_or(
        events_df['description'] == "The subject has returned home.",
        events_df['description'].str.startswith("Subject back home:")
    )].copy()
    events_df['date'] = pd.to_datetime(events_df['date'])
    for date, df_session in df.groupby('date', sort=False):
        subject = df_session['subject'].unique()[0]
        _, session_duration = find_next_end_task_time_in_events(events_df, date, subject)
        if session_duration is None:
            if verbose:
                print(f"Session duration not found for {subject} on {date}. Skipping this session.")
            continue
        gsbu_dfs.append(get_session_box_usage(df_session, session_duration=session_duration))
    return pd.concat(gsbu_dfs, ignore_index=True)

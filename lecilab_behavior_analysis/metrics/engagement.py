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


BOX_USAGE_COLUMNS = ["date", "subject", "time_type", "absolute_time", "percentage_of_time", "accuracy"]


MAX_SESSION_HOURS = 6


def get_session_ends(df: pd.DataFrame, events_df: pd.DataFrame, max_hours: float = MAX_SESSION_HOURS) -> pd.DataFrame:
    """
    End of each session (subject, date) of df: the first time the same subject
    returned home ("Subject back home: ..." or "The subject has returned
    home.") after the session started, before its next session started and
    at most max_hours later. Columns: subject, date, end_date,
    session_duration (seconds). Sessions without such an event (e.g. events
    missing) are left out.

    Matching the subject (instead of taking the next event of any subject)
    lets events of several boxes be used together.
    """
    home = events_df[np.logical_or(
        events_df["description"] == "The subject has returned home.",
        events_df["description"].str.startswith("Subject back home:"),
    )]
    home = pd.DataFrame({"subject": home["subject"], "end_date": pd.to_datetime(home["date"])})
    home = home.dropna().sort_values("end_date")
    if home.empty:
        return pd.DataFrame(columns=["subject", "date", "end_date", "session_duration"])
    sessions = df[["subject", "date"]].drop_duplicates().copy()
    sessions["start"] = pd.to_datetime(sessions["date"])
    sessions = sessions.sort_values("start")
    sessions["next_start"] = sessions.groupby("subject")["start"].shift(-1)
    ends = pd.merge_asof(sessions, home, left_on="start", right_on="end_date", by="subject",
                         direction="forward", allow_exact_matches=False)
    ends = ends[ends["end_date"].notna()
                & ~(ends["end_date"] >= ends["next_start"])
                & (ends["end_date"] - ends["start"] <= pd.Timedelta(hours=max_hours))]
    ends["session_duration"] = (ends["end_date"] - ends["start"]).dt.total_seconds()
    return ends[["subject", "date", "end_date", "session_duration"]].reset_index(drop=True)


def get_box_usage_df(df: pd.DataFrame, events_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Time of each session spent completing the first trial, engaged,
    disengaged, unaccounted and exiting the box (see get_session_box_usage),
    with the session end taken from events_df (see get_session_ends). Sessions
    without an end in the events are skipped. Empty (with the columns) if
    there are none.
    """
    ends = get_session_ends(df, events_df)
    durations = {(row.subject, row.date): row.session_duration for row in ends.itertuples()}
    gsbu_dfs = []
    for (subject, date), df_session in df.groupby(["subject", "date"], sort=False):
        session_duration = durations.get((subject, date))
        if session_duration is None:
            if verbose:
                print(f"Session duration not found for {subject} on {date}. Skipping this session.")
            continue
        gsbu_dfs.append(get_session_box_usage(df_session, session_duration=session_duration))
    if not gsbu_dfs:
        return pd.DataFrame(columns=BOX_USAGE_COLUMNS)
    return pd.concat(gsbu_dfs, ignore_index=True)

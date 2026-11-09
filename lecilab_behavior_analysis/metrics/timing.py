"""Timing: reaction time, time between trials, center port holds, trial duration,
misses and early pokeouts.
"""
import ast
from typing import Optional

import numpy as np
import pandas as pd

from lecilab_behavior_analysis.common import column_checker
from lecilab_behavior_analysis.metrics.trial_events import (
    EventColumns,
    early_pokeout_from_events,
    get_previous_row_index,
    initiation_time_from_events,
    miss_from_events,
    port_holds_from_events,
    reaction_time_from_events,
)


def add_time_between_trials(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time_between_trials: time from the start of the previous trial of the
    same session (NaN for the first trial of each session).
    """
    column_checker(df_in, required_columns={"TRIAL_START", "date", "session"})
    df = df_in.copy()
    keys = [c for c in ["subject", "date", "session"] if c in df.columns]
    df["time_between_trials"] = df.groupby(keys, sort=False)["TRIAL_START"].diff()
    return df


def add_reaction_time(df_in: pd.DataFrame, events: Optional[EventColumns] = None) -> pd.DataFrame:
    """
    Adds reaction_time: time from the last center poke out to the next side poke.
    """
    column_checker(df_in, required_columns={"Port1In", "Port2Out", "Port3In"})
    df = df_in.copy()
    events = events if events is not None else EventColumns(df)
    df["reaction_time"] = [
        reaction_time_from_events(p2out, p1, p3)
        for p2out, p1, p3 in zip(events["Port2Out"], events["Port1In"], events["Port3In"])
    ]
    return df


def calculate_time_between_trials_and_reaction_time(in_df: pd.DataFrame, events: Optional[EventColumns] = None) -> pd.DataFrame:
    """
    Calculate Time Between Trials and Reaction Time.
    """
    # Check if the required columns are present
    column_checker(in_df, required_columns={"Port1In", "Port1Out", "Port2In", "Port2Out", "Port3In", "Port3Out", "TRIAL_START"})
    df = add_time_between_trials(in_df)
    return add_reaction_time(df, events)


def add_inter_trial_interval_column_to_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Add an inter-trial interval column to the dataframe.
    """
    # Check if the required columns are present
    column_checker(df_in, required_columns={"Port2In", "TRIAL_END"})
    df = df_in.copy()  # Make a copy to avoid modifying the original DataFrame
    for date in pd.unique(df['date']):
        session_df = df[df['date'] == date].copy()
        # Calculate the inter-trial interval
        # shift the trial end time by one
        trial_end_shifted = session_df['TRIAL_END'].shift(1)
        # Animals can do multiple port2Ins in each trial. It could happen that the
        # animal gives up in the middle of these pokes. We will consider the last port2In as the one that counts
        # TODO: changed to the min
        port2ins_last = session_df['Port2In'].apply(lambda x: np.min(ast.literal_eval(x)) if isinstance(x, str) else np.min(x))
        iti_vector = port2ins_last - trial_end_shifted
        # fill the first value with NaN
        iti_vector.iloc[0] = np.nan
        # assign the new column to the original dataframe
        df.loc[df['date'] == date, 'inter_trial_interval'] = iti_vector
    
    return df


def add_trial_initiation_time_to_df(df_in: pd.DataFrame, events: Optional[EventColumns] = None) -> pd.DataFrame:
    """
    Add a trial initiation time column to the dataframe.
    """
    # Check if the required columns are present
    column_checker(df_in, required_columns={"TRIAL_START", "Port2In"})
    df = df_in.copy()  # Make a copy to avoid modifying the original DataFrame
    events = events if events is not None else EventColumns(df)
    df['trial_initiation_time'] = [
        initiation_time_from_events(p2in, start) for p2in, start in zip(events["Port2In"], df["TRIAL_START"])
    ]
    return df


def add_trial_duration_column_to_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Add a trial duration column to the dataframe.
    """
    df = df_in.copy()  # Make a copy to avoid modifying the original DataFrame
    # Check if the required columns are present
    column_checker(df, required_columns={"TRIAL_START", "TRIAL_END"})
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    trial_duration_date = df['TRIAL_END'] - df['TRIAL_START']
    df['trial_duration'] = trial_duration_date

    return df


def add_trial_misses(df_in: pd.DataFrame, events: Optional[EventColumns] = None) -> pd.DataFrame:
    """
    Add a trial misses column to the dataframe.
    """
    # Check if the required columns are present
    column_checker(df_in, required_columns={"STATE_stimulus_state_END", "TRIAL_END"})
    df = df_in.copy()  # Make a copy to avoid modifying the original DataFrame
    events = events if events is not None else EventColumns(df)
    df["miss_trial"] = [
        miss_from_events(stim_end, end) for stim_end, end in zip(events["STATE_stimulus_state_END"], df["TRIAL_END"])
    ]
    return df


def add_port_holds(df_in: pd.DataFrame, port_number: int = 2, events: Optional[EventColumns] = None) -> pd.DataFrame:
    """
    Adds port<N>_holds: the duration of every poke in that port during the
    trial (including the ITI), as an array per trial.
    """
    column_checker(df_in, required_columns={f"Port{port_number}In", f"Port{port_number}Out"})
    df = df_in.copy()
    events = events if events is not None else EventColumns(df)
    df[f"port{port_number}_holds"] = [
        port_holds_from_events(ins, outs)
        for ins, outs in zip(events[f"Port{port_number}In"], events[f"Port{port_number}Out"])
    ]
    return df


def _mean_and_cis(values: np.ndarray) -> tuple:
    """Mean, and 5th and 95th percentiles, ignoring NaNs (NaN if no values)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    bottom, top = np.percentile(values, [5, 95])
    return values.mean(), bottom, top


def get_center_hold_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Per day: mean and 5-95 percentiles of the number of center pokes per trial
    and of the duration of those pokes. Uses port2_holds if it is already there.
    """
    column_checker(df_in, required_columns={"year_month_day"})
    df = df_in if "port2_holds" in df_in.columns else add_port_holds(df_in, 2)
    rows = []
    for day, holds in df.groupby("year_month_day", sort=True)["port2_holds"]:
        holds = list(holds)
        n_mean, n_bot, n_top = _mean_and_cis([len(h) for h in holds])
        flat = np.concatenate([np.asarray(h, dtype=float) for h in holds]) if holds else np.array([])
        s_mean, s_bot, s_top = _mean_and_cis(flat)
        rows.append({
            "year_month_day": day,
            "number_of_pokes_mean": n_mean,
            "number_of_pokes_bot95": n_bot,
            "number_of_pokes_top95": n_top,
            "hold_time_mean": s_mean,
            "hold_time_bot95": s_bot,
            "hold_time_top95": s_top,
        })
    return pd.DataFrame(rows)


def get_reaction_times_by_date_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Per day: mean and 5-95 percentiles of the reaction time and of the time
    between trials. Uses those columns if they are already there.
    """
    column_checker(df_in, required_columns={"year_month_day"})
    df = df_in
    if "time_between_trials" not in df.columns:
        df = add_time_between_trials(df)
    if "reaction_time" not in df.columns:
        df = add_reaction_time(df)
    rows = []
    for day, group in df.groupby("year_month_day", sort=True):
        rt = _mean_and_cis(group["reaction_time"].to_numpy(dtype=float))
        tbt = _mean_and_cis(group["time_between_trials"].to_numpy(dtype=float))
        rows.append({
            "year_month_day": day,
            "reaction_time_mean": rt[0],
            "reaction_time_bot95": rt[1],
            "reaction_time_top95": rt[2],
            "time_between_trials_mean": tbt[0],
            "time_between_trials_bot95": tbt[1],
            "time_between_trials_top95": tbt[2],
        })
    return pd.DataFrame(rows)


def adjust_trials_and_time_of_start_to_first_session(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts the 'TRIAL_START' and 'trial' columns to start from the first session.
    """
    df = df_in.copy()

    # get the indeces of session changes
    session_changes = df[df.session != df.session.shift(1)].index[1:]
    # adjust "trial" and "time_from_start" columns
    for i in range(len(session_changes)):
        idx = session_changes[i]
        # find the index of the previous entry in the df
        prev_idx = get_previous_row_index(df, idx)
        past_trial = df.loc[prev_idx, 'trial']
        past_time_from_start = df.loc[prev_idx, 'time_from_start']
        if i < len(session_changes) - 1:
            next_idx = session_changes[i + 1] - 1
        else:
            next_idx = df.index[-1]
        df.loc[idx:next_idx, 'trial'] = df.loc[idx:next_idx, 'trial'] + past_trial
        df.loc[idx:next_idx, 'time_from_start'] = df.loc[idx:next_idx, 'time_from_start'] + past_time_from_start
    return df


def add_early_pokeouts(df_in: pd.DataFrame, events: Optional[EventColumns] = None) -> pd.DataFrame:
    """
    Adds a boolean column 'early_pokeout' to the dataframe indicating whether there was an early pokeout
    in the center port before holding the required time. This is irrespective of whether the trial was correct or incorrect.
    """
    column_checker(df_in, required_columns={"STATE_ready_to_initiate_START", "Port2In", "Port2Out", "holding_time"})
    df = df_in.copy()  # Make a copy to avoid modifying the original DataFrame
    events = events if events is not None else EventColumns(df)
    df['early_pokeout'] = [
        early_pokeout_from_events(ready, p2in, p2out, hold)
        for ready, p2in, p2out, hold in zip(
            events["STATE_ready_to_initiate_START"], events["Port2In"], events["Port2Out"], df["holding_time"]
        )
    ]
    return df

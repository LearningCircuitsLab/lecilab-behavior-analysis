"""Preparing the data once, before any analysis or figure (see prepare())."""
import ast
import numpy as np
import pandas as pd

import lecilab_behavior_analysis.data_format_changes as dfc
from lecilab_behavior_analysis.common import column_checker
from lecilab_behavior_analysis.metrics.choices import (
    add_mouse_first_choice,
    add_mouse_last_choice,
    add_port_where_animal_comes_from,
    add_repeat_or_alternate_numeric,
)
from lecilab_behavior_analysis.metrics.timing import (
    add_early_pokeouts,
    add_port_holds,
    add_reaction_time,
    add_time_between_trials,
    add_trial_duration_column_to_df,
    add_trial_misses,
)
from lecilab_behavior_analysis.metrics.trial_events import EventColumns


def fill_missing_data(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing data in the dataframe.
    """
    df = df_in.copy()  # Create a copy to avoid modifying the original DataFrame
    columns_to_fill = ["stimulus_modality", "current_training_stage", "difficulty", "correct_side"]
    for column in columns_to_fill:
        try:
            df[column] = df[column].fillna("not saved")
        except KeyError:
            df[column] = "not saved"
    # Fill missing values in the "correct" column
    try:
        df["correct"] = df["correct"].infer_objects()
    except KeyError:
        df["correct"] = True
    # use the current modality and stage names (see data_format_changes.py)
    df = dfc.normalize_task_names(df)

    return df


def add_day_column_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a day column (year_month_day, text "YYYY-MM-DD") to the dataframe.
    """
    # Check if the required columns are present
    column_checker(df, required_columns={"date"})
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    # there are few different dates (one per session): convert each one once
    codes, uniques = pd.factorize(df["date"])
    days = pd.to_datetime(pd.Series(uniques)).dt.strftime("%Y-%m-%d").to_numpy(dtype=object)
    df["year_month_day"] = np.where(codes >= 0, days[np.maximum(codes, 0)] if len(days) else None, None)
    return df


def add_trial_of_day_column_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a trial of the day column to the dataframe.
    """
    # Check if the required columns are present
    column_checker(df, required_columns={"year_month_day", "trial"})
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    df['trial_of_day'] = df['trial'] - df.groupby('year_month_day')['trial'].transform('min') + 1
    return df


def reformat_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    df['visual_stimulus'] = df['visual_stimulus'].apply(ast.literal_eval)
    # TODO: fix issues with this when there are no values

    return df


def analyze_df(df: pd.DataFrame, include_early_pokeouts: bool = True) -> pd.DataFrame:
    """
    Analyze the dataframe adding new columns and filling missing data.
    Same as prepare() (kept with its old name).
    """
    return prepare(df, include_early_pokeouts=include_early_pokeouts)


# columns added by prepare(); a dataframe that has all of them was prepared already
PREPARED_COLUMNS = (
    "year_month_day",
    "total_trial",
    "trial_of_day",
    "miss_trial",
    "first_choice",
    "last_choice",
    "last_choice_before_stimulus",
    "previous_port_before_stimulus",
    "roa_choice",
    "roa_choice_numeric",
    "early_pokeout",
    "reaction_time",
    "time_between_trials",
    "trial_duration",
    "port2_holds",
)


def prepare(df_in: pd.DataFrame, include_early_pokeouts: bool = True) -> pd.DataFrame:
    """
    Prepares the trials of one or several subjects for any analysis or figure.

    Every analysis and figure of the package reads the columns added here, so
    they do not depend on what the caller ran before. It can be called on data
    that was prepared already (only the missing columns are added), and on a
    subset of prepared data (e.g. one day), keeping the values computed on the
    whole history (such as total_trial).

    It:
    - fills missing task information and converts old names to the current ones
      (fill_missing_data, see data_format_changes.py),
    - reads the event columns (Port1In, STATE_..._START...) once,
    - adds: year_month_day, total_trial (per subject), trial_of_day,
      miss_trial, first_choice, last_choice, last_choice_before_stimulus,
      previous_port_before_stimulus, roa_choice, roa_choice_numeric,
      early_pokeout (None if it cannot be computed), reaction_time,
      time_between_trials, trial_duration and port2_holds.

    Event columns are left as they are (text in the .csv).
    """
    df = fill_missing_data(df_in)
    if df.empty:
        return df
    events = EventColumns(df)
    columns = set(df.columns)

    if "year_month_day" not in columns and "date" in columns:
        df = add_day_column_to_df(df)
    if "total_trial" not in columns and "subject" in columns:
        # trials are counted in the order of the dataframe
        df["total_trial"] = df.groupby("subject", sort=False).cumcount() + 1
    if "trial_of_day" not in columns and {"year_month_day", "trial"} <= set(df.columns):
        df = add_trial_of_day_column_to_df(df)
    if "miss_trial" not in columns and {"STATE_stimulus_state_END", "TRIAL_END"} <= columns:
        df = add_trial_misses(df, events)
    if "first_choice" not in columns and {"Port1In", "Port3In", "STATE_stimulus_state_START"} <= columns:
        df = add_mouse_first_choice(df, events)
    if "last_choice" not in columns and {"Port1In", "Port3In"} <= columns:
        df = add_mouse_last_choice(df, events)
    if "roa_choice" not in columns and {"first_choice", "last_choice", "session", "subject"} <= set(df.columns):
        df = add_port_where_animal_comes_from(df, events)
    if "roa_choice_numeric" not in columns and {"roa_choice", "first_choice"} <= set(df.columns):
        df = add_repeat_or_alternate_numeric(df)
    if include_early_pokeouts and "early_pokeout" not in columns:
        if {"STATE_ready_to_initiate_START", "Port2In", "Port2Out", "holding_time"} <= columns:
            df = add_early_pokeouts(df, events)
        else:
            df["early_pokeout"] = None
    if "time_between_trials" not in columns and {"TRIAL_START", "date", "session"} <= columns:
        df = add_time_between_trials(df)
    if "reaction_time" not in columns and {"Port1In", "Port2Out", "Port3In"} <= columns:
        df = add_reaction_time(df, events)
    if "trial_duration" not in columns and {"TRIAL_START", "TRIAL_END"} <= columns:
        df = add_trial_duration_column_to_df(df)
    if "port2_holds" not in columns and {"Port2In", "Port2Out"} <= columns:
        df = add_port_holds(df, 2, events)
    return df

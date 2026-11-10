"""Reading the events of one trial (a row): first and last choice, reaction time,
center port holds, misses. Event columns (Port1In, STATE_*_START...) can be
lists or text such as "[1.2, 3.4]".
"""
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd




def get_start_and_end_times(df):
    """
    Get the start and end times of the session.
    """
    # ensure that there is only one session in the dataframe
    if df['session'].nunique() != 1:
        raise ValueError("The dataframe contains more than one session.")
    # get the start time of the session as the start of the first trial
    start_time = df.loc[df['trial'] == np.min(df.trial), 'TRIAL_START'].values[0]
    # get the end time of the session as the end of the last trial
    end_time = df.loc[df['trial'] == np.max(df.trial), 'TRIAL_END'].values[0]
    # convert the start and end times to datetime
    start_time = pd.to_datetime(start_time, unit='s')
    end_time = pd.to_datetime(end_time, unit='s')
    return start_time, end_time


# ---------------------------------------------------------------------------
# Reading event times
# ---------------------------------------------------------------------------
# Event and state columns (Port1In, STATE_stimulus_state_START...) hold the
# times of that event in the trial. In the .csv they are text such as
# "[1.2, 3.4]" ("[nan]" for a state that was not visited, empty if the event
# did not happen). parse_events reads any of these forms once, much faster
# than ast.literal_eval. NaN times are left out, so a state that was not
# visited gives an empty list.
_NUMPY_SCALAR = re.compile(r"(?:np|numpy)\.\w+\(([^)]*)\)")
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf", re.IGNORECASE)


def parse_events(value: Any) -> List[float]:
    """Times of one event in one trial, as a list of floats ([] if none)."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if not text.strip():
            return []
        try:
            times = [float(x) for x in text.split(",")]
        except ValueError:
            # e.g. "[np.float64(1.2)]"
            text = _NUMPY_SCALAR.sub(r"\1", text)
            times = [float(x) for x in _NUMBER.findall(text)]
        return [x for x in times if x == x]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value if x == x]
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return [] if value != value else [float(value)]
    return []


def parse_event_column(df: pd.DataFrame, column: str) -> List[List[float]]:
    """parse_events for every trial of a column ([] for every trial if the column is missing)."""
    if column not in df.columns:
        return [[] for _ in range(len(df))]
    return [parse_events(v) for v in df[column].to_numpy(dtype=object)]


def _side_of_first_poke(port1: List[float], port3: List[float], after: float) -> Optional[str]:
    left = [t for t in port1 if t > after]
    right = [t for t in port3 if t > after]
    if not left and not right:
        return None
    if not left:
        return "right"
    if not right:
        return "left"
    if min(left) < min(right):
        return "left"
    if min(right) < min(left):
        return "right"
    return None


def _side_of_last_poke(port1: List[float], port3: List[float]) -> Optional[str]:
    if not port1 and not port3:
        return None
    if not port1:
        return "right"
    if not port3:
        return "left"
    if max(port1) > max(port3):
        return "left"
    if max(port3) > max(port1):
        return "right"
    return None


def first_choice_from_events(stimulus_start: List[float], port1: List[float], port3: List[float]) -> Optional[str]:
    """Side of the first side poke after the stimulus started (None if no poke)."""
    if not stimulus_start:
        return None
    return _side_of_first_poke(port1, port3, min(stimulus_start))


def last_choice_before_stimulus_from_events(stimulus_start: List[float], port1: List[float], port3: List[float]) -> Optional[str]:
    """Side of the last side poke before the stimulus started (None if no poke)."""
    if not stimulus_start:
        return None
    start = min(stimulus_start)
    return _side_of_last_poke([t for t in port1 if t < start], [t for t in port3 if t < start])


def miss_from_events(stimulus_end: List[float], trial_end: float) -> bool:
    """True if the trial ended when the stimulus state ended (no choice)."""
    if not stimulus_end:
        return False
    return bool(np.round(max(stimulus_end), 6) == np.round(trial_end, 6))


def early_pokeout_from_events(ready_start: List[float], port2_in: List[float], port2_out: List[float], holding_time: float) -> Optional[bool]:
    """True if the first center poke was shorter than holding_time (None if unknown)."""
    if not ready_start or not port2_in or not port2_out:
        return None
    first_ready = min(ready_start)
    ins = [t for t in port2_in if t > first_ready]
    if not ins:
        return None
    first_in = min(ins)
    outs = [t for t in port2_out if t > first_in]
    if not outs:
        return None
    return bool((min(outs) - first_in) < holding_time)


def reaction_time_from_events(port2_out: List[float], port1_in: List[float], port3_in: List[float]) -> float:
    """Time from the last center poke out to the next side poke (NaN if none)."""
    if not port2_out:
        return np.nan
    out_time = max(port2_out)
    side_ins = [t for t in port1_in if t > out_time] + [t for t in port3_in if t > out_time]
    if not side_ins:
        return np.nan
    return min(side_ins) - out_time


def initiation_time_from_events(port2_in: List[float], trial_start: float) -> float:
    """Time from the start of the trial to the first center poke (NaN if none)."""
    if not port2_in:
        return np.nan
    return min(port2_in) - trial_start


def port_holds_from_events(ins: List[float], outs: List[float]) -> Union[list, np.ndarray]:
    """Duration of every poke in a port during the trial (including the ITI)."""
    if not ins or not outs:
        return []
    # if the first out is earlier than the first in, remove it
    if outs[0] < ins[0]:
        outs = outs[1:]
    # if they are different lengths, pad the shorter one with NaNs
    if len(ins) > len(outs):
        outs = list(outs) + [np.nan] * (len(ins) - len(outs))
    elif len(outs) > len(ins):
        ins = list(ins) + [np.nan] * (len(outs) - len(ins))
    return np.array(outs) - np.array(ins)


# ---------------------------------------------------------------------------
# The same, for one trial given as a row (kept for old code; prefer the
# columns that preprocessing.prepare adds)
# ---------------------------------------------------------------------------
def is_this_a_miss_trial(series: pd.Series) -> bool:
    """
    checks if STATE_stimulus_state_END is the last state
    """
    return miss_from_events(parse_events(series.get("STATE_stimulus_state_END")), series["TRIAL_END"])


def is_this_an_early_pokeout_trial(series: pd.Series) -> Union[bool, None]:
    """
    checks if there was an early pokeout in the center port before holding the required time
    """
    return early_pokeout_from_events(
        parse_events(series.get("STATE_ready_to_initiate_START")),
        parse_events(series.get("Port2In")),
        parse_events(series.get("Port2Out")),
        series["holding_time"],
    )


def trial_reaction_time(series: pd.Series) -> float:
    """
    This method calculates the reaction time of a trial.
    """
    return reaction_time_from_events(
        parse_events(series.get("Port2Out")),
        parse_events(series.get("Port1In")),
        parse_events(series.get("Port3In")),
    )


def trial_initiation_time(series: pd.Series) -> float:
    """
    This method calculates the initiation time of a trial.
    """
    return initiation_time_from_events(parse_events(series.get("Port2In")), series["TRIAL_START"])


def first_poke_after_stimulus_state(series: pd.Series) -> Union[str, None]:
    return first_choice_from_events(
        parse_events(series.get("STATE_stimulus_state_START")),
        parse_events(series.get("Port1In")),
        parse_events(series.get("Port3In")),
    )


def get_last_poke_of_trial(series: pd.Series) -> Union[str, None]:
    return _side_of_last_poke(parse_events(series.get("Port1In")), parse_events(series.get("Port3In")))


def get_last_poke_before_stimulus_state(series: pd.Series) -> Union[str, None]:
    """
    Sometimes the animal can poke in the center port and not wait long enough
    to get the stimulus state. If trial is not aborted due to selected settings,
    the animal can keep poking.
    """
    return last_choice_before_stimulus_from_events(
        parse_events(series.get("STATE_stimulus_state_START")),
        parse_events(series.get("Port1In")),
        parse_events(series.get("Port3In")),
    )


def get_dictionary_event_as_list(ser_dict: Dict, event: str) -> List:
    return parse_events(ser_dict.get(event))


def get_repeat_or_alternate_to_numeric(row: pd.Series) -> float:
    if row['roa_choice'] == 'alternate':
        return 0
    elif row['roa_choice'] == 'repeat':
        if row['first_choice'] == 'left':
            return -1
        elif row['first_choice'] == 'right':
            return 1
    else:
        return np.nan


def get_trial_port_hold(row, port_number):
    """
    This calculates everything, including those occurring in the ITIs for example.
    """
    if not type(row) == pd.Series:
        raise ValueError("row must be a pandas Series")
    return port_holds_from_events(
        parse_events(row["Port" + str(port_number) + "In"]),
        parse_events(row["Port" + str(port_number) + "Out"]),
    )


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


def add_time_from_session_start(df_in: pd.DataFrame) -> pd.DataFrame:
    # Group by both 'subject' and 'session' to calculate time from session start
    df_in['time_from_start'] = df_in.groupby(['subject', 'session'])['TRIAL_START'].transform(lambda x: x - x.iloc[0])
    return df_in


def get_previous_row_index(df: pd.DataFrame, current_index) -> int:
    # Get the position of the current index
    current_pos = df.index.get_loc(current_index)
    # Ensure it's not the first row
    if current_pos > 0:
        # Return the previous index
        return df.index[current_pos - 1]
    else:
        raise IndexError("No previous row exists for the given index.")


def transform_side_choice_to_numeric(side: str) -> Union[int, float]:
    match side:
        case "left":
            return 1
        case "right":
            return 0
        case _:
            return np.nan


class EventColumns:
    """
    Event columns of a dataframe, read with parse_events. Each column is read
    the first time it is used and kept, so that several functions can share it:

        events = EventColumns(df)
        df = add_mouse_first_choice(df, events)
        df = add_trial_misses(df, events)

    The lists follow the order of the rows of df.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._columns: Dict[str, List[List[float]]] = {}

    def __getitem__(self, column: str) -> List[List[float]]:
        if column not in self._columns:
            self._columns[column] = parse_event_column(self.df, column)
        return self._columns[column]

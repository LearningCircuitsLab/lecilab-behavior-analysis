"""Choices of the animal: first / last choice, where it comes from, bias,
psychometric data (visual and cloud of tones) and model regressors.
"""

import re
from typing import Optional, Union

import numpy as np
import pandas as pd

import lecilab_behavior_analysis.data_format_changes as dfc
from lecilab_behavior_analysis.common import _get_single_modality, column_checker
from lecilab_behavior_analysis.metrics.performance import repeat_or_alternate_series_comparison
from lecilab_behavior_analysis.metrics.sound import add_auditory_real_statistics
from lecilab_behavior_analysis.metrics.trial_events import (
    EventColumns,
    _side_of_last_poke,
    first_choice_from_events,
    last_choice_before_stimulus_from_events,
    transform_side_choice_to_numeric,
)


def previous_impact_on_time_kernel(series, max_lag=10, tau=5):
    # kernel = np.exp(-np.arange(max_lag+1, 1, -1) / tau)  # exponential decay kernel
    kernel = 1 / (1 + np.arange(max_lag + 1, 1, -1) / tau)  # hyperbolic decay kernel
    padded = np.concatenate([[0] * max_lag, series])
    return np.array([np.dot(kernel, padded[i - max_lag : i]) for i in range(max_lag, len(padded))])


def add_port_where_animal_comes_from(df_in: pd.DataFrame, events: Optional[EventColumns] = None) -> pd.DataFrame:
    # TODO: the name of this function is not great, as it adds also extra information
    """
    The previous side port can come from two sources.
    One, from the previous trial, where the animal was drinking water
    (note, this should not necessarily be true as the animal can go to the
    other port afterwards without going through the middle, but if a trial has
    started before that happens, this function will take care of it)
    Second, from side port pokes previous to the initiation of the stimulus. This
    second condition can happen from two sources. Initial pokes before poking on
    the center, or poking in the center just brief enough to not trigger the
    stimulus state. In some cases this is not punished.

    Adds last_choice_before_stimulus, previous_port_before_stimulus and
    roa_choice ("repeat" / "alternate": first choice compared with that port).
    """
    df = df_in.copy()  # Create a copy to avoid modifying the original DataFrame
    events = events if events is not None else EventColumns(df)
    # Get side port pokes before the stimulus state
    # TODO: I am mixing here choices and pokes... really bad
    df["last_choice_before_stimulus"] = [
        last_choice_before_stimulus_from_events(stim, p1, p3)
        for stim, p1, p3 in zip(events["STATE_stimulus_state_START"], events["Port1In"], events["Port3In"])
    ]
    # Get the last port in each trial
    if "last_choice" not in df.columns:
        df = add_mouse_last_choice(df, events)
    if "first_choice" not in df.columns:
        df = add_mouse_first_choice(df, events)

    # The port the animal comes from: its side pokes before the stimulus state
    # in this trial or, if there are none, its last choice in the previous
    # trial of the same session
    last_in_previous = df.groupby(["subject", "session"], sort=False)["last_choice"].shift(1)
    before = df["last_choice_before_stimulus"]
    previous_port = before.where(before.notna(), last_in_previous)
    df["previous_port_before_stimulus"] = previous_port.where(previous_port.notna(), np.nan)
    df["roa_choice"] = repeat_or_alternate_series_comparison(df["first_choice"], df["previous_port_before_stimulus"])
    return df


def add_repeat_or_alternate_numeric(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Adds roa_choice_numeric: 0 if the animal alternated, -1 if it repeated to
    the left, 1 if it repeated to the right (NaN otherwise). Same values as
    get_repeat_or_alternate_to_numeric, for the whole column at once.
    """
    column_checker(df_in, required_columns={"roa_choice", "first_choice"})
    df = df_in.copy()
    roa = df["roa_choice"]
    choice = df["first_choice"]
    df["roa_choice_numeric"] = np.select(
        [roa == "alternate", (roa == "repeat") & (choice == "left"), (roa == "repeat") & (choice == "right")],
        [0.0, -1.0, 1.0],
        default=np.nan,
    )
    return df


def get_left_choice(df):
    df = add_mouse_first_choice(df)
    df["left_choice"] = df["first_choice"].apply(transform_side_choice_to_numeric)
    return df


def get_performance_by_difficulty_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy(deep=True)
    modality = _get_single_modality(df_copy)
    if modality == "visual":
        # brighter / dimmer ratio, positive if the left port is the brighter one
        # (visual_stimulus is read in either format, see data_format_changes.py)
        df_copy = dfc.add_left_right_brightness(df_copy)
        brighter = df_copy[["left_brightness", "right_brightness"]].max(axis=1)
        dimmer = df_copy[["left_brightness", "right_brightness"]].min(axis=1)
        df_copy["visual_stimulus_ratio"] = (brighter / dimmer).where(
            df_copy["left_brightness"] > df_copy["right_brightness"],
            -(brighter / dimmer),
        )
        # df["visual_stimulus_ratio"] = df["visual_stimulus_ratio"].apply(np.log).round(4)
    elif modality == "auditory-cot":
        df_copy = add_auditory_real_statistics(df_copy)
    else:
        raise ValueError("modality must be either 'visual' or 'auditory-cot'")
    df_copy = add_mouse_first_choice(df_copy)
    df_copy["first_choice_numeric"] = df_copy["first_choice"].apply(transform_side_choice_to_numeric)

    return df_copy


def get_performance_by_difficulty_diff(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy(deep=True)
    if df_copy["current_training_stage"].str.contains("visual").any():
        # left minus right brightness
        # (visual_stimulus is read in either format, see data_format_changes.py)
        df_copy = dfc.add_left_right_brightness(df_copy)
        df_copy["visual_stimulus_diff"] = df_copy["left_brightness"] - df_copy["right_brightness"]
    elif df_copy["current_training_stage"].str.contains("auditory").any():
        stim_col = "auditory_stimulus"
        diff_col = "auditory_stimulus_diff"
        df_copy[diff_col] = df_copy[stim_col].apply(lambda x: abs(eval(x)[0] - eval(x)[1]))
        df_copy[diff_col] = df_copy.apply(
            lambda row: row[diff_col] if row["correct_side"] == "left" else -row[diff_col], axis=1
        )
    else:
        raise ValueError("modality must be one of 'visual' or 'auditory'")

    df_copy["first_choice_numeric"] = df_copy["first_choice"].apply(transform_side_choice_to_numeric)
    return df_copy


def add_mouse_first_choice(df: pd.DataFrame, events: Optional[EventColumns] = None) -> pd.DataFrame:
    """
    Add the first choice made by the mouse in each trial.
    """
    column_checker(df, required_columns={"Port1In", "Port3In", "STATE_stimulus_state_START"})
    df = df.copy()
    events = events if events is not None else EventColumns(df)
    df["first_choice"] = [
        first_choice_from_events(stim, p1, p3)
        for stim, p1, p3 in zip(events["STATE_stimulus_state_START"], events["Port1In"], events["Port3In"])
    ]
    return df


def add_mouse_last_choice(df: pd.DataFrame, events: Optional[EventColumns] = None) -> pd.DataFrame:
    """
    Add the last choice made by the mouse in each trial.
    """
    column_checker(df, required_columns={"Port1In", "Port3In"})
    df = df.copy()
    events = events if events is not None else EventColumns(df)
    df["last_choice"] = [_side_of_last_poke(p1, p3) for p1, p3 in zip(events["Port1In"], events["Port3In"])]
    return df


def get_triangle_polar_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    column_checker(df, required_columns={"roa_choice_numeric", "subject"})
    # get the bias for each animal
    df_bias = df.groupby(["subject"])["roa_choice_numeric"].value_counts().reset_index(name="count")
    # transform the bias to be in the range of 0 to 2pi
    df_bias["bias_angle"] = df_bias["roa_choice_numeric"].replace(
        {0: 2 * np.pi / 4, -1: 5 * np.pi / 4, 1: 7 * np.pi / 4}
    )
    # transform counts into percentages
    df_bias["percentage"] = df_bias["count"] / df_bias.groupby(["subject"])["count"].transform("sum")

    return df_bias


def get_bias_evolution_df(df: pd.DataFrame, groupby: Union[str, list[str]]) -> pd.DataFrame:
    """
    Gets how the bias of the animals (alternating, right bias, or left bias)
    evolves over time.

    Arguments:
    df: DataFrame with the data
    groupby: str or list, the column(s) to group by. Can be 'session' or 'trial_group'

    Returns:
    df: DataFrame with the bias evolution
    """
    groupby_items = ["subject"]
    if isinstance(groupby, str):
        groupby_items.append(groupby)
    elif isinstance(groupby, list):
        groupby_items.extend(groupby)
    column_checker(df, required_columns=set(groupby_items + ["roa_choice_numeric"]))
    df_anchev = df.copy()
    df_anchev = df_anchev.groupby(groupby_items)["roa_choice_numeric"].value_counts().reset_index(name="count")
    # transform counts into percentages
    df_anchev["percentage"] = df_anchev["count"] / df_anchev.groupby(groupby_items)["count"].transform("sum")

    # pivot or melt the dataframe so that each subject and session has a y value,
    # that will be the percentage when the bias
    # is 0, and the x value will be the differences between the percentages when the bias is 1 and -1
    df_bias_pivot = df_anchev.pivot(index=groupby_items, columns="roa_choice_numeric", values="percentage")

    # fill the NaN values with 0
    df_bias_pivot = df_bias_pivot.fillna(0)
    # calculate the difference between the percentages when the bias is 1 and -1
    df_bias_pivot["left_right_bias"] = df_bias_pivot[1] - df_bias_pivot[-1]
    df_bias_pivot["alternating_bias"] = df_bias_pivot[0]
    # reset index
    return df_bias_pivot.reset_index()


def points_to_lines_for_bias_evolution(df: pd.DataFrame, groupby: str) -> pd.DataFrame:
    # if more than one subject, raise an error
    if df["subject"].nunique() > 1:
        raise ValueError("The dataframe should contain only one subject.")
    column_checker(df, required_columns={groupby})
    # convert the xs and ys points to lines by duplicating the endpoint using the next trial group
    dfbps = df.copy()
    dfbps[groupby] = dfbps[groupby].shift(1)
    # add the two dataframes together
    df_bias_pivot_merged = pd.concat([df, dfbps], ignore_index=True)
    df_bias_pivot_merged.dropna(inplace=True)

    return df_bias_pivot_merged


def create_transition_matrix(events: list) -> pd.DataFrame:
    # Initialize a matrix with zeros
    items = list(set(events))
    items.sort()  # Sort the items to ensure consistent ordering
    n = len(items)
    transition_matrix = np.zeros((n, n), dtype=int)

    # Map items to their indices in the matrix
    item_index = {item: i for i, item in enumerate(items)}

    # Count transitions from one item to the next
    for i in range(len(events) - 1):
        from_item = events[i]
        to_item = events[i + 1]
        transition_matrix[item_index[from_item], item_index[to_item]] += 1

    # Return the transition matrix as a pandas DataFrame for better readability
    return pd.DataFrame(transition_matrix, index=items, columns=items)


def _difference_correct_minus_incorrect(pair) -> float:
    if pair is None:
        return np.nan
    return pair[0] - pair[1]


def add_visual_stimulus_difference(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()  # Create a copy to avoid modifying the original DataFrame
    column_checker(df_in, required_columns={"visual_stimulus", "correct_side", "date"})
    # correct minus incorrect port brightness
    # (visual_stimulus is read in either format, see data_format_changes.py)
    df["visual_stim_difference"] = df.apply(
        lambda row: _difference_correct_minus_incorrect(
            dfc.get_correct_incorrect_brightness(row["visual_stimulus"], row["correct_side"], row["date"])
        ),
        axis=1,
    )
    # bin the data every 0.1
    df["vis_stim_dif_bin"] = np.round((df["visual_stim_difference"] // 0.1) * 0.1, 1)
    return df


def get_left_auditory_stim(df: pd.DataFrame, param: str) -> pd.DataFrame:
    pbd_df = df.copy(deep=True)
    pbd_df[param + "_left"] = pbd_df.apply(
        lambda row: row[param + "_high"] if row["correct_side"] == "left" else row[param + "_low"], axis=1
    )
    return pbd_df


def get_choice_before(df):
    df_copy = df.copy(deep=True)
    column_checker(
        df_copy, required_columns={"first_choice", "last_choice", "previous_port_before_stimulus", "correct"}
    )

    for mouse in df_copy["subject"].unique():
        for session in df_copy[df_copy.subject == mouse]["session"].unique():
            df_mouse_session = df_copy[np.logical_and(df_copy["subject"] == mouse, df_copy["session"] == session)]
            # df_mouse_session['correct_side_in_previous'] = (
            #     df_mouse_session['correct_side'].shift(1, fill_value=np.nan))
            df_mouse_session["previous_correct"] = df_mouse_session["correct"].shift(1, fill_value=np.nan)
            df_mouse_session["previous_first_choice"] = df_mouse_session["first_choice"].shift(1, fill_value=np.nan)
            df_mouse_session["previous_last_choice"] = df_mouse_session["last_choice"].shift(1, fill_value=np.nan)
            df_mouse_session["previous_correct_side"] = df_mouse_session["correct_side"].shift(1, fill_value=np.nan)

            df_copy.loc[df_mouse_session.index, "previous_correct"] = df_mouse_session["previous_correct"]
            df_copy.loc[df_mouse_session.index, "previous_first_choice"] = df_mouse_session["previous_first_choice"]
            df_copy.loc[df_mouse_session.index, "previous_last_choice"] = df_mouse_session["previous_last_choice"]
            df_copy.loc[df_mouse_session.index, "previous_port_before_stimulus"] = df_mouse_session[
                "previous_port_before_stimulus"
            ]

            # Add the previous choice is left and correct as 1, or 0, remain NaN
            previous_left_choice_correct_numeric = np.where(
                df_mouse_session["previous_first_choice"].isna() | df_mouse_session["previous_correct"].isna(),
                None,
                (
                    (df_mouse_session["previous_first_choice"] == "left")
                    & df_mouse_session["previous_correct"].eq(True)
                ).astype(int),
            )
            df_copy.loc[df_mouse_session.index, "previous_left_choice_correct_numeric"] = (
                previous_left_choice_correct_numeric
            )

            # Add the previous choice is right and wrong as 1, or 0, remain NaN
            previous_right_choice_wrong_numeric = np.where(
                df_mouse_session["previous_first_choice"].isna() | df_mouse_session["previous_correct"].isna(),
                None,
                (
                    (df_mouse_session["previous_first_choice"] == "right")
                    & df_mouse_session["previous_correct"].eq(False)
                ).astype(int),
            )
            df_copy.loc[df_mouse_session.index, "previous_right_choice_wrong_numeric"] = (
                previous_right_choice_wrong_numeric
            )

            # Add the same and correct in choice from previous, where 1 is a same and correct choice and 0 is the rest
            previous_same_choice_correct_numeric = np.where(
                df_mouse_session["previous_first_choice"].isna() | df_mouse_session["previous_correct"].isna(),
                None,
                (
                    (
                        (df_mouse_session["previous_first_choice"] == "left")
                        & (df_mouse_session["first_choice"] == "left")
                        & df_mouse_session["previous_correct"].eq(True)
                    )
                    | (
                        (df_mouse_session["previous_first_choice"] == "right")
                        & (df_mouse_session["first_choice"] == "right")
                        & df_mouse_session["previous_correct"].eq(True)
                    )
                ).astype(int),
            )
            df_copy.loc[df_mouse_session.index, "previous_same_choice_correct_numeric"] = (
                previous_same_choice_correct_numeric
            )

            # Add the difference and wrong in choice from previous,
            # where 1 is a different and wrong choice and 0 is the rest
            previous_diff_choice_wrong_numeric = np.where(
                df_mouse_session["previous_first_choice"].isna() | df_mouse_session["previous_correct"].isna(),
                None,
                (
                    (
                        (df_mouse_session["previous_first_choice"] == "left")
                        & (df_mouse_session["first_choice"] == "right")
                        & df_mouse_session["previous_correct"].eq(False)
                    )
                    | (
                        (df_mouse_session["previous_first_choice"] == "right")
                        & (df_mouse_session["first_choice"] == "left")
                        & df_mouse_session["previous_correct"].eq(False)
                    )
                ).astype(int),
            )
            df_copy.loc[df_mouse_session.index, "previous_diff_choice_wrong_numeric"] = (
                previous_diff_choice_wrong_numeric
            )

            # Add the same choice as previous, where 1 is the same choice and 0 is a different choice
            previous_same_choice_numeric = np.where(
                df_mouse_session["previous_first_choice"].isna() | df_mouse_session["first_choice"].isna(),
                None,
                (
                    (
                        (df_mouse_session["previous_first_choice"] == "left")
                        & (df_mouse_session["first_choice"] == "left")
                    )
                    | (
                        (df_mouse_session["previous_first_choice"] == "right")
                        & (df_mouse_session["first_choice"] == "right")
                    )
                ).astype(int),
            )
            df_copy.loc[df_mouse_session.index, "previous_same_choice_numeric"] = previous_same_choice_numeric

    return df_copy


def parameters_for_fit(df):
    """
    Get the parameters for the fit
    """
    df_copy = df.copy(deep=True)
    df_copy = add_mouse_first_choice(df_copy)
    df_copy = add_mouse_last_choice(df_copy)
    df_copy = add_port_where_animal_comes_from(df_copy)
    modality = _get_single_modality(df)
    if modality == "visual":
        df_copy = get_performance_by_difficulty_ratio(df_copy)
        df_copy = get_performance_by_difficulty_diff(df_copy)
        df_copy["abs_visual_stimulus_ratio"] = df_copy["visual_stimulus_ratio"].abs()
        df_copy["visual_ratio_diff_interact"] = df_copy["visual_stimulus_ratio"] * (
            df_copy["visual_stimulus_diff"].abs()
        )
        # brightness of the left port, and of the incorrect port
        # (visual_stimulus is read in either format, see data_format_changes.py)
        df_copy["left_bright"] = df_copy["left_brightness"]
        df_copy["visual_ratio_bright_interact"] = df_copy["abs_visual_stimulus_ratio"] * df_copy["left_bright"]
        df_copy["wrong_bright"] = df_copy["left_brightness"].where(
            df_copy["correct_side"] == "right", df_copy["right_brightness"]
        )
    elif modality == "auditory-cot":
        df_copy = add_auditory_real_statistics(df_copy)
        df_copy = get_left_auditory_stim(df_copy, "total_percentage_of_tones")
        df_copy = get_left_auditory_stim(df_copy, "number_of_tones")
        df_copy = get_left_auditory_stim(df_copy, "percentage_of_timebins_with_evidence")
        df_copy["abs_total_evidence_strength"] = df_copy["total_evidence_strength"].abs()
        df_copy["high_tones"] = df_copy["auditory_stimulus"].apply(lambda row: eval(row)["high_tones"])
        df_copy["high_tones_amplitude_sum"] = df_copy["high_tones"].apply(lambda row: np.sum(pd.DataFrame(row).values))
        df_copy["low_tones"] = df_copy["auditory_stimulus"].apply(lambda row: eval(row)["low_tones"])
        df_copy["low_tones_amplitude_sum"] = df_copy["low_tones"].apply(lambda row: np.sum(pd.DataFrame(row).values))
        df_copy["amplitude_strength"] = (df_copy["high_tones_amplitude_sum"] / df_copy["number_of_tones_high"]) / (
            df_copy["low_tones_amplitude_sum"] / df_copy["number_of_tones_low"]
        )
        df_copy["left_tones_amplitude_sum"] = df_copy.apply(
            lambda row: row["high_tones_amplitude_sum"]
            if row["correct_side"] == "left"
            else row["low_tones_amplitude_sum"],
            axis=1,
        )
        df_copy["amplitude_strength_left_right"] = df_copy.apply(
            lambda row: (
                (
                    (row["high_tones_amplitude_sum"] / (row["number_of_tones_high"] or np.nan))
                    / (row["low_tones_amplitude_sum"] / (row["number_of_tones_low"] or np.nan))
                )
                if row["correct_side"] == "left"
                else (
                    (row["low_tones_amplitude_sum"] / (row["number_of_tones_low"] or np.nan))
                    / (row["high_tones_amplitude_sum"] / (row["number_of_tones_high"] or np.nan))
                )
            ),
            axis=1,
        )

    df_copy["previous_port_before_stimulus_numeric"] = df_copy["previous_port_before_stimulus"].apply(
        transform_side_choice_to_numeric
    )
    df_copy["roa_choice_numeric"] = df_copy["roa_choice"].apply(
        lambda x: 1 if x == "repeat" else 0 if x == "alternate" else np.nan
    )
    df_copy["last_choice_numeric"] = df_copy["last_choice"].apply(transform_side_choice_to_numeric)

    # Add the correct column as numeric, 1 for correct, 0 for incorrect
    df_copy["correct_numeric"] = df_copy["correct"].astype(int)

    # Add the past trials impact by time kernel

    df_copy = get_choice_before(df_copy)

    df_copy["previous_first_choice_numeric"] = df_copy["previous_first_choice"].apply(transform_side_choice_to_numeric)
    df_copy["previous_last_choice_numeric"] = df_copy["previous_last_choice"].apply(transform_side_choice_to_numeric)
    df_copy["first_choice_numeric"] = df_copy["first_choice"].apply(transform_side_choice_to_numeric)
    df_copy["previous_correct_numeric"] = df_copy["previous_correct"].astype("Int64")

    return df_copy


def get_time_kernel_impact(df: pd.DataFrame, y: str, max_lag, tau):
    df_copy = df.copy(deep=True)
    if y == "first_choice_numeric":
        df_copy = add_mouse_first_choice(df_copy)
        df_copy["first_choice_numeric"] = df_copy["first_choice"].apply(transform_side_choice_to_numeric)
        df_copy["time_kernel_impact_first_choice"] = previous_impact_on_time_kernel(
            df_copy[y], max_lag=max_lag, tau=tau
        )
    elif y == "correct_numeric":
        df_copy["correct_numeric"] = df_copy["correct"].astype(int)
        df_copy["time_kernel_impact_correct"] = previous_impact_on_time_kernel(df_copy[y], max_lag=max_lag, tau=tau)
    else:
        raise ValueError("impact should be either 'first_choice_numeric' and 'correct_numeric'")

    return df_copy


# keys of auditory_stimulus with the loudness of a sound location trial, in
# order of preference ("sound_db_spl" in the current task)
SOUND_LEVEL_KEYS = ("sound_db_spl", "sound_db", "db_spl", "db", "amplitude_db", "amplitude")


def _sound_level(stimulus) -> float:
    """Loudness of a sound location trial from its auditory_stimulus (dict or its text)."""
    import ast

    if isinstance(stimulus, str):
        # numbers saved by numpy 2 look like np.float64(59.1)
        text = re.sub(r"(?:np|numpy)\.\w+\(([^()]*)\)", r"\1", stimulus)
        try:
            stimulus = ast.literal_eval(re.sub(r"\bnan\b", "None", text))
        except (ValueError, SyntaxError):
            return np.nan
    if not isinstance(stimulus, dict):
        return np.nan
    for key in SOUND_LEVEL_KEYS:
        value = stimulus.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return np.nan


def add_sound_location_level(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sound_location_db: loudness of the sound of a sound location trial
    (auditory-local-wn), positive if it came from the left and negative if
    from the right (the side in auditory_output_side, or the correct side, as
    the sound plays on the side of the reward). NaN if it is not known.
    """
    column_checker(df_in, required_columns={"auditory_stimulus", "correct_side"})
    df = df_in.copy()
    level = df["auditory_stimulus"].apply(_sound_level)
    side = df["correct_side"]
    if "auditory_output_side" in df.columns:
        side = df["auditory_output_side"].where(df["auditory_output_side"].isin(["left", "right"]), side)
    sign = side.map({"left": 1.0, "right": -1.0})
    df["sound_location_db"] = level * sign
    return df

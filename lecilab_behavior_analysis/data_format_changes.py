"""
Changes in the format of the data saved by the tasks.

Why this file exists
--------------------
The behavioural tasks (repository auditory-task-python, run on the Training
Village) save one row per trial. Over time, the meaning or the values of some
columns have changed. Old rows cannot be rewritten, so the analysis code has to
read old and new rows correctly at the same time.

Each change is documented here, in one place, with:

- what changed, and from when,
- why it changed,
- the constant or function the rest of the analysis code must use to read
  both versions.

The rest of the package imports from this file instead of hard-coding dates or
names. When a task changes the format of a column again, add a new section
here (and use it where that column is read), rather than patching each
analysis function separately.

Rule for the rest of the package
--------------------------------
- The data is converted once, at the start: preprocessing.prepare (also
  called analyze_df) calls fill_missing_data, which calls
  normalize_task_names. Every figure (figure_maker, the figures package)
  prepares its data first. A new public figure or analysis function must do
  the same before using stimulus_modality or current_training_stage.
- Internal code then compares only with the current names
  ("auditory-cot", "TwoAFC_auditory-cot_easy"...), never with the old ones.
- Functions that may be called directly on raw data and need one specific
  modality normalize it themselves (e.g. common._get_single_modality).
- Columns whose meaning changed (visual_stimulus) are always read through the
  functions in this file, which use the date and the village (system_name)
  of each trial.
- The default is always the newest format: data of a village that is not in
  NEW_TASK_CODE_SINCE (e.g. a new village) is read in the newest format.

Current changes
---------------
1. visual_stimulus: (correct, incorrect) -> (left, right).
2. Modality and stage names: "auditory" / "auditory_cot" -> "auditory-cot",
   "auditory_local_wn" -> "auditory-local-wn".
3. correct: can be None (timeouts).
4. use_sound_location: cloud of tones from the side of the reward (an older,
   now deprecated option) is read as the modality "auditory-local-cot".

The first three come from the same revision of the task (branch hmv_claude of
auditory-task-python), which reorganised how stimuli are handled. Each
village started running it on a different day: see NEW_TASK_CODE_SINCE.
Only the first change needs that date (the others are read from the values
themselves).
"""

import ast
from typing import Any, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# When each village started running the new task code
# ---------------------------------------------------------------------------
# Rows saved by a village from this moment on use the new formats, and earlier
# rows the old ones. The value is a date ("2026-11-25", from midnight) or a
# date and time ("2026-11-25 15:30", if a village was updated during the day
# and had sessions before the update).
#
# - A village that is not here (e.g. a new village) is read in the newest
#   format: add a village only if it has data saved by the old task code.
# - No village ran the new code before the earliest date here, so earlier rows
#   are read in the old format whatever their village (also rows without
#   system_name).
# - Rows after it without system_name are read in the newest format.
#
# When a village is updated, add it here with the moment it was updated.
NEW_TASK_CODE_SINCE = {
    "village06": "2026-11-09",
    "village01": "2026-11-25 14:00",
    "village02": "2026-11-25 14:00",
    "village04": "2026-11-25 14:00",
    "village05": "2026-11-25 14:00",
}
FIRST_NEW_TASK_CODE = min(pd.Timestamp(since) for since in NEW_TASK_CODE_SINCE.values())


def uses_new_task_code(date: Any, village: Any = None) -> bool:
    """
    True if a trial saved on this date by this village (its system_name)
    comes from the new task code, i.e. uses the new formats.
    """
    date = pd.Timestamp(date)
    if date < FIRST_NEW_TASK_CODE:
        return False
    since = NEW_TASK_CODE_SINCE.get(village) if isinstance(village, str) else None
    if since is None:  # a village that is not listed, or no village: newest format
        return True
    return date >= pd.Timestamp(since)


# ---------------------------------------------------------------------------
# 1. visual_stimulus: (correct, incorrect) -> (left, right)
# ---------------------------------------------------------------------------
# Before: visual_stimulus = (brightness of the correct port,
#                            brightness of the incorrect port)
# After:  visual_stimulus = (brightness of the LEFT port,
#                            brightness of the RIGHT port)
#
# Why: the task can now light the side ports as a distractor (random light in
# trials whose cue is a sound). A distractor light is not related to the reward
# side, so "correct" and "incorrect" port stop making sense for it. Saving the
# brightness by side is valid for every trial, informative or not.
#
# How to tell old and new rows apart: the date and the village of the trial
# (see NEW_TASK_CODE_SINCE). No column marks the format. Pass the village
# (system_name) of the trial: without it, rows after FIRST_NEW_TASK_CODE are
# read in the newest format.


def _parse_pair(value: Any) -> Optional[Tuple[float, float]]:
    """Reads a (a, b) pair saved as text or as a sequence. None if empty."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        if value.strip() in ("", "None", "nan"):
            return None
        value = ast.literal_eval(value)
    if value is None:
        return None
    first, second = value
    return float(first), float(second)


def visual_stimulus_is_left_right(date: Any, village: Any = None) -> bool:
    """True if a trial saved on this date by this village saved visual_stimulus as (left, right)."""
    return uses_new_task_code(date, village)


def get_left_right_brightness(
    visual_stimulus: Any, correct_side: str, date: Any, village: Any = None
) -> Optional[Tuple[float, float]]:
    """
    Returns (left, right) brightness of the side ports for one trial, whatever
    the format it was saved in. None if the trial had no light stimulus.

    Args:
        visual_stimulus: the value of the visual_stimulus column (text or pair).
        correct_side: "left" or "right" (needed to read old rows).
        date: the date of the trial (decides the format, with the village).
        village: the system_name of the trial (see NEW_TASK_CODE_SINCE).
    """
    pair = _parse_pair(visual_stimulus)
    if pair is None:
        return None
    if visual_stimulus_is_left_right(date, village):
        return pair
    # old rows: (correct, incorrect)
    correct_brightness, incorrect_brightness = pair
    if correct_side == "left":
        return correct_brightness, incorrect_brightness
    return incorrect_brightness, correct_brightness


def get_correct_incorrect_brightness(
    visual_stimulus: Any, correct_side: str, date: Any, village: Any = None
) -> Optional[Tuple[float, float]]:
    """
    Returns (correct port, incorrect port) brightness for one trial, whatever the
    format it was saved in. None if the trial had no light stimulus.
    """
    left_right = get_left_right_brightness(visual_stimulus, correct_side, date, village)
    if left_right is None:
        return None
    left, right = left_right
    if correct_side == "left":
        return left, right
    return right, left


def add_left_right_brightness(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the columns left_brightness and right_brightness (NaN if the trial had
    no light stimulus), reading visual_stimulus in either format (from the
    date and the system_name of each row).
    """
    df = df_in.copy()
    pairs = df.apply(
        lambda row: get_left_right_brightness(
            row["visual_stimulus"], row["correct_side"], row["date"], row.get("system_name")
        ),
        axis=1,
    )
    df["left_brightness"] = pairs.apply(lambda p: p[0] if p is not None else float("nan"))
    df["right_brightness"] = pairs.apply(lambda p: p[1] if p is not None else float("nan"))
    return df


# ---------------------------------------------------------------------------
# 2. Modality and stage names
# ---------------------------------------------------------------------------
# Before: stimulus_modality "auditory" (main / new_tv versions of the task) or
#         "auditory_cot" and "auditory_local_wn" (nuo_dev version), and stages
#         such as "TwoAFC_auditory_easy" or "TwoAFC_auditory_cot_easy".
#         A version used in the project frequency_vs_locationPureTone (e.g.
#         NUO066-NUO077) called the sound location modality "auditory_locaPT"
#         (location, pure tone; a few first trials "auditory_loca"): it is the
#         current sound location modality, with the same auditory_stimulus
#         (sound_db_spl...).
# After:  "auditory-cot" (cloud of tones) and "auditory-local-wn" (white noise
#         from the side of the reward), and stages such as
#         "TwoAFC_auditory-cot_easy".
#
# Why: a second auditory modality (sound location with white noise) was added,
# so "auditory" alone became ambiguous. Hyphens are used inside modality names
# because underscores separate the parts of a stage name:
# TwoAFC_<modality>_<difficulty>. "visual" and "multisensory" did not change.
#
# How to read both: normalize the names with the functions below (done in
# preprocessing.fill_missing_data), and compare only with the new names.
LEGACY_MODALITY_NAMES = {
    "auditory": "auditory-cot",
    "auditory_cot": "auditory-cot",
    "auditory_local_wn": "auditory-local-wn",
    "auditory_locaPT": "auditory-local-wn",
    "auditory_loca": "auditory-local-wn",
}


def normalize_modality(name: Any) -> Any:
    """Returns the current name of a modality (other values are returned as they are)."""
    if isinstance(name, str):
        return LEGACY_MODALITY_NAMES.get(name, name)
    return name


def normalize_stage(stage: Any) -> Any:
    """Returns the current name of a training stage, e.g.
    "TwoAFC_auditory_cot_easy" or "TwoAFC_auditory_easy" -> "TwoAFC_auditory-cot_easy"."""
    if not isinstance(stage, str) or not stage.startswith("TwoAFC_"):
        return stage
    rest = stage[len("TwoAFC_") :]
    for difficulty in ("easy", "hard"):
        suffix = "_" + difficulty
        if rest.endswith(suffix):
            modality = rest[: -len(suffix)]
            return "TwoAFC_" + normalize_modality(modality) + suffix
    return stage


def normalize_task_names(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the stimulus_modality and current_training_stage columns, if
    present, and names the trials of use_sound_location (see section 4).
    """
    df = df_in.copy()
    if "stimulus_modality" in df.columns:
        df["stimulus_modality"] = df["stimulus_modality"].apply(normalize_modality)
        if "use_sound_location" in df.columns:
            located = df["use_sound_location"].apply(_is_true)
            if located.any():
                df["stimulus_modality"] = df["stimulus_modality"].astype(object)
                df.loc[located, "stimulus_modality"] = SOUND_LOCATION_COT
    if "current_training_stage" in df.columns:
        df["current_training_stage"] = df["current_training_stage"].apply(normalize_stage)
    return df


# ---------------------------------------------------------------------------
# 3. correct can be None (timeouts)
# ---------------------------------------------------------------------------
# Before: correct was True or False on every trial. A trial in which the mouse
#         did not choose a side counted as False.
# After:  correct is
#         - True / False: the first side poke was correct / incorrect,
#         - False: early withdrawal from the center port that ended the trial,
#         - None (empty in the .csv): timeout, the mouse did not choose a side.
#
# Why: a timeout is not a wrong choice. The task's anti-bias and the training
# protocol (which uses the mean of correct per day) leave timeouts out.
#
# How the analysis treats them (decision of 2026-11-09):
# - Session performance (correct trials / all trials, e.g. in the session
#   summary) keeps counting timeouts as errors.
# - Functions that take the mean of correct (e.g. get_day_performance) skip them,
#   as the training protocol does.
# - Plots must not fail on empty values; COUNT_TIMEOUTS_AS_ERRORS says how to
#   treat them when a number is needed.
COUNT_TIMEOUTS_AS_ERRORS = True


def correct_as_number(value: Any) -> float:
    """
    1.0 for a correct trial, 0.0 for an incorrect one. A timeout (None / NaN) is
    0.0 if COUNT_TIMEOUTS_AS_ERRORS, NaN otherwise. Accepts text ("True").
    """
    if isinstance(value, str):
        value = {"True": True, "False": False}.get(value.strip(), None)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0 if COUNT_TIMEOUTS_AS_ERRORS else float("nan")
    return 1.0 if bool(value) else 0.0


# ---------------------------------------------------------------------------
# 4. use_sound_location: cloud of tones from the side of the reward
# ---------------------------------------------------------------------------
# What: an option of the main branch of the task (2026-07-31, "option to follow
# where the sound comes from"), saved in the column use_sound_location. When
# True, the cloud of tones of an auditory trial was played only from the
# speaker on the side of the reward, with a random dominant frequency: the cue
# was where the sound came from, not its frequency. Those trials were saved as
# stimulus_modality "auditory" (read as "auditory-cot").
#
# Data: the project COT_unilateral_sound (ACV023-ACV033, village06,
# 2026-08-11 to 2026-09-07, Manual_training), about 108,600 trials; in all of
# them the sound came from the side of the reward.
#
# The option is deprecated (the new task code does not have it; its sound
# location modality is "auditory-local-wn", white noise).
#
# How to read it: the column marks each trial, so no date is needed. Trials
# with use_sound_location True are read as the modality "auditory-local-cot"
# (cloud of tones from one side), so that they are not analysed as a cloud of
# tones whose frequency is the cue. Done in normalize_task_names.
SOUND_LOCATION_COT = "auditory-local-cot"


def _is_true(value: Any) -> bool:
    """True for True (also numpy's) or "True" (as read from the .csv); False for anything else (False, empty...)."""
    if isinstance(value, str):
        return value.strip() == "True"
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(value == True)  # noqa: E712 (also numpy booleans)

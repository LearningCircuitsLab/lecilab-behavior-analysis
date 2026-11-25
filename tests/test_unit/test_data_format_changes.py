"""
Tests for the changes in the format of the saved data
(lecilab_behavior_analysis/data_format_changes.py) and the functions that use them.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

import lecilab_behavior_analysis.data_format_changes as dfc
import lecilab_behavior_analysis.df_transforms as dft
import lecilab_behavior_analysis.plots as plots

OLD_DATE = "2026-11-08 12:00:00"
NEW_DATE = "2026-11-09 08:00:00"


# 1. visual_stimulus -------------------------------------------------------
@pytest.mark.parametrize(
    "visual_stimulus, correct_side, date, expected",
    [
        # old rows: (correct, incorrect)
        ("(0.6, 0.1)", "left", OLD_DATE, (0.6, 0.1)),
        ("(0.6, 0.1)", "right", OLD_DATE, (0.1, 0.6)),
        # new rows: (left, right)
        ("(0.6, 0.1)", "left", NEW_DATE, (0.6, 0.1)),
        ("(0.1, 0.6)", "right", NEW_DATE, (0.1, 0.6)),
        # pairs that are not text
        ((0.6, 0.1), "right", OLD_DATE, (0.1, 0.6)),
        # no light stimulus
        (None, "left", NEW_DATE, None),
        (np.nan, "left", NEW_DATE, None),
        ("", "left", NEW_DATE, None),
        ("None", "left", NEW_DATE, None),
    ],
)
def test_get_left_right_brightness(visual_stimulus, correct_side, date, expected):
    assert dfc.get_left_right_brightness(visual_stimulus, correct_side, date) == expected


def test_change_date_is_inclusive():
    assert dfc.visual_stimulus_is_left_right("2026-11-09 00:00:00", "village06")
    assert not dfc.visual_stimulus_is_left_right("2026-11-08 23:59:59", "village06")
    assert dfc.visual_stimulus_is_left_right("2026-11-25 00:00:00", "village02")
    assert not dfc.visual_stimulus_is_left_right("2026-11-24 23:59:59", "village02")


@pytest.mark.parametrize(
    "date, village, new_format",
    [
        # village06 runs the new code since 2026-11-09
        ("2026-11-10 10:00", "village06", True),
        # the other villages since 2026-11-25
        ("2026-11-10 10:00", "village01", False),
        ("2026-11-10 10:00", "village02", False),
        ("2026-11-24 23:00", "village04", False),
        ("2026-11-25 10:00", "village05", True),
        ("2026-12-01 10:00", "village02", True),
        # a village that is not listed (e.g. a new one): the newest format
        ("2026-11-10 10:00", "village09", True),
        # rows without village: the newest format
        ("2026-11-10 10:00", None, True),
        ("2026-11-10 10:00", float("nan"), True),
        # nobody ran the new code before 2026-11-09
        ("2026-11-08 10:00", "village06", False),
        ("2026-11-08 10:00", "village09", False),
        ("2026-11-08 10:00", None, False),
    ],
)
def test_new_task_code_by_village(date, village, new_format):
    assert dfc.uses_new_task_code(date, village) is new_format


def test_every_listed_village_has_a_valid_date():
    for village, since in dfc.NEW_TASK_CODE_SINCE.items():
        assert village.startswith("village") and pd.Timestamp(since) >= dfc.FIRST_NEW_TASK_CODE


def test_left_right_brightness_uses_the_village_of_each_row():
    """The same day: village06 saved (left, right), village02 still (correct, incorrect)."""
    df = pd.DataFrame(
        {
            "visual_stimulus": ["(0.1, 0.6)", "(0.6, 0.1)"],
            "correct_side": ["right", "right"],
            "date": pd.to_datetime(["2026-11-20 10:00", "2026-11-20 10:00"]),
            "system_name": ["village06", "village02"],
        }
    )
    out = dfc.add_left_right_brightness(df)
    # both trials: the right port (correct) was the brighter one
    assert out["left_brightness"].tolist() == [0.1, 0.1]
    assert out["right_brightness"].tolist() == [0.6, 0.6]
    # the correct port is the brighter one when read through the other function too
    for _, row in df.iterrows():
        assert dfc.get_correct_incorrect_brightness(
            row["visual_stimulus"], row["correct_side"], row["date"], row["system_name"]
        ) == (0.6, 0.1)


def test_left_right_brightness_without_system_name_column():
    df = pd.DataFrame(
        {"visual_stimulus": ["(0.1, 0.6)"], "correct_side": ["right"], "date": pd.to_datetime(["2026-11-20"])}
    )
    out = dfc.add_left_right_brightness(df)  # newest format
    assert (out["left_brightness"].iloc[0], out["right_brightness"].iloc[0]) == (0.1, 0.6)


def test_get_correct_incorrect_brightness():
    # new row, right trial, right port brighter
    assert dfc.get_correct_incorrect_brightness("(0.1, 0.6)", "right", NEW_DATE) == (0.6, 0.1)
    # old row is already (correct, incorrect)
    assert dfc.get_correct_incorrect_brightness("(0.6, 0.1)", "right", OLD_DATE) == (0.6, 0.1)


def _visual_trials() -> pd.DataFrame:
    """Same four informative trials, saved in the old and in the new format."""
    rows = []
    for date, fmt in [(OLD_DATE, "old"), (NEW_DATE, "new")]:
        for side in ["left", "right"]:
            correct_brightness, incorrect_brightness = 0.6, 0.1
            if fmt == "old" or side == "left":
                stimulus = (correct_brightness, incorrect_brightness)
            else:
                stimulus = (incorrect_brightness, correct_brightness)
            rows.append(
                {
                    "date": date,
                    "correct_side": side,
                    "visual_stimulus": str(stimulus),
                    "stimulus_modality": "visual",
                    "current_training_stage": "TwoAFC_visual_hard",
                    "STATE_stimulus_state_START": "[1.0]",
                    "Port1In": "[2.0]" if side == "left" else "[]",
                    "Port3In": "[2.0]" if side == "right" else "[]",
                }
            )
    return pd.DataFrame(rows)


def test_visual_ratio_is_the_same_for_old_and_new_rows():
    df = dft.get_performance_by_difficulty_ratio(_visual_trials())
    # positive when the left port is the brighter one
    assert list(df["visual_stimulus_ratio"]) == pytest.approx([6, -6, 6, -6])


def test_visual_difference_is_the_same_for_old_and_new_rows():
    df = _visual_trials()
    df["first_choice"] = ["left", "right", "left", "right"]
    df = dft.get_performance_by_difficulty_diff(df)
    assert list(df["visual_stimulus_diff"]) == pytest.approx([0.5, -0.5, 0.5, -0.5])
    df = dft.add_visual_stimulus_difference(_visual_trials())
    assert list(df["visual_stim_difference"]) == pytest.approx([0.5] * 4)


# 2. modality and stage names ----------------------------------------------
@pytest.mark.parametrize(
    "old, new",
    [
        ("auditory", "auditory-cot"),
        ("auditory_cot", "auditory-cot"),
        ("auditory_local_wn", "auditory-local-wn"),
        ("visual", "visual"),
        ("multisensory", "multisensory"),
        ("auditory-cot", "auditory-cot"),
    ],
)
def test_normalize_modality(old, new):
    assert dfc.normalize_modality(old) == new


@pytest.mark.parametrize(
    "old, new",
    [
        ("TwoAFC_auditory_easy", "TwoAFC_auditory-cot_easy"),
        ("TwoAFC_auditory_cot_hard", "TwoAFC_auditory-cot_hard"),
        ("TwoAFC_auditory_local_wn_easy", "TwoAFC_auditory-local-wn_easy"),
        ("TwoAFC_visual_hard", "TwoAFC_visual_hard"),
        ("TwoAFC_auditory-cot_easy", "TwoAFC_auditory-cot_easy"),
        ("Habituation", "Habituation"),
        ("Manual_training", "Manual_training"),
    ],
)
def test_normalize_stage(old, new):
    assert dfc.normalize_stage(old) == new


def test_fill_missing_data_normalizes_names():
    df = pd.DataFrame(
        {
            "stimulus_modality": ["auditory", None],
            "current_training_stage": ["TwoAFC_auditory_easy", "Habituation"],
            "difficulty": ["easy", "easy"],
            "correct_side": ["left", "right"],
            "correct": [True, False],
        }
    )
    df = dft.fill_missing_data(df)
    assert list(df["stimulus_modality"]) == ["auditory-cot", "not saved"]
    assert list(df["current_training_stage"]) == ["TwoAFC_auditory-cot_easy", "Habituation"]


# 3. correct can be None ---------------------------------------------------
@pytest.mark.parametrize(
    "value, expected",
    [(True, 1.0), (False, 0.0), ("True", 1.0), ("False", 0.0), (None, 0.0), (np.nan, 0.0)],
)
def test_correct_as_number_counts_timeouts_as_errors(value, expected):
    assert dfc.correct_as_number(value) == expected


def _session_with_timeouts() -> pd.DataFrame:
    n = 12
    return pd.DataFrame(
        {
            "subject": ["mouse"] * n,
            "session": [1] * n,
            "trial": range(1, n + 1),
            "correct_side": ["left", "right"] * (n // 2),
            "correct": [True, None, False, True] * (n // 4),
        }
    )


def test_performance_through_trials_counts_timeouts_as_errors():
    df = dft.get_performance_through_trials(_session_with_timeouts(), window=4)
    assert df["performance_w"].iloc[-1] == pytest.approx(50)


def test_live_plots_accept_timeouts():
    df = _session_with_timeouts()
    fig, ax = plt.subplots()
    plots.side_correct_performance_plot(df, ax)
    plots.correct_left_and_right_plot(df, ax)
    plots.number_of_correct_responses_plot(df, ax)
    plt.close(fig)


# 4. use_sound_location ------------------------------------------------------
@pytest.mark.parametrize(
    "use_sound_location, expected",
    [
        (True, "auditory-local-cot"),
        ("True", "auditory-local-cot"),
        (np.True_, "auditory-local-cot"),
        (False, "auditory-cot"),
        ("False", "auditory-cot"),
        (np.nan, "auditory-cot"),
        (None, "auditory-cot"),
    ],
)
def test_use_sound_location_trials_are_sound_location(use_sound_location, expected):
    df = pd.DataFrame({"stimulus_modality": ["auditory"], "use_sound_location": [use_sound_location]})
    assert dfc.normalize_task_names(df)["stimulus_modality"].tolist() == [expected]


def test_use_sound_location_only_changes_its_rows():
    df = pd.DataFrame(
        {
            "stimulus_modality": ["auditory", "auditory", "visual", "auditory_cot"],
            "use_sound_location": ["True", "False", "False", None],
            "current_training_stage": ["Manual_training", "TwoAFC_auditory_easy", "TwoAFC_visual_easy", "x"],
        }
    )
    out = dfc.normalize_task_names(df)
    assert out["stimulus_modality"].tolist() == ["auditory-local-cot", "auditory-cot", "visual", "auditory-cot"]
    assert out["current_training_stage"].tolist() == [
        "Manual_training",
        "TwoAFC_auditory-cot_easy",
        "TwoAFC_visual_easy",
        "x",
    ]


def test_without_use_sound_location_column():
    df = pd.DataFrame({"stimulus_modality": ["auditory"]})
    assert dfc.normalize_task_names(df)["stimulus_modality"].tolist() == ["auditory-cot"]
